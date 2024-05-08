/**
 * This library includes software components derived from the following projects:
 * [ChatGPTApp] (https://github.com/scriptit-fr/ChatGPTApp)
 * [Google AI JavaScript SDK] (https://github.com/google/generative-ai-js/)
 * 
 * These components are licensed under the Apache License 2.0. 
 * A copy of the license can be found in the LICENSE file.
 */

/**
 * @license
 * Copyright 2024 Martin Hawksey
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * CoreFunctions for GenerativeModel and Chat.
 */
class _CoreFunctions {
  constructor() {}

  _countTokens(auth, model, params) {
    const url = new RequestUrl(model, Task.COUNT_TOKENS, auth, false, {});
    const response = this._makeRequest(url, JSON.stringify({ ...params, model }));
    return response;
  }

  _generateContent(auth, model, params, requestOptions) {
    const url = new RequestUrl(model, Task.GENERATE_CONTENT, auth, false, requestOptions);
    const responseJson = this._makeRequest(url, JSON.stringify(params));
    return { response: this._addHelpers(responseJson) };
  }

  _makeRequest(url, body) {
    const maxRetries = 5;
    let retries = 0;
    let success = false;

    let options = {
      'method': 'POST',
      'contentType': 'application/json',
      'muteHttpExceptions': true,
      'headers': {},
      'payload': body
    };

    if (url.apiKey) {
      options.headers = { 'X-Goog-Api-Key': url.apiKey };
    } else if (url._auth?.type === 'service_account') {
      const credentials = this._credentialsForVertexAI(url._auth);
      options.headers = { 'Authorization': `Bearer ${credentials.accessToken}` }
    } else {
      options.headers = { 'Authorization': `Bearer ${ScriptApp.getOAuthToken()}` };
    }

    let response;
    while (retries < maxRetries && !success) {
      response = UrlFetchApp.fetch(url.toString(), options);
      let responseCode = response.getResponseCode();

      if (responseCode === 200) {
        // The request was successful, exit the loop.
        response = JSON.parse(response.getContentText());
        success = true;
      } else if (responseCode === 429 || (responseCode >= 500 && responseCode < 600)) {
        console.warn(`Retrying after ${responseCode} response from the server.`);
        let delay = Math.pow(2, retries) * 1000; // Exponential backoff starting at 1 second.
        Utilities.sleep(delay);
        retries++;
      } else {
        // The request failed for another reason, log the error and exit the loop.
        console.error(`Request failed with response code ${responseCode} - ${response.getContentText()}`);
        break;
      }
    }

    if (!success) {
      throw new Error(`Failed to call API after ${retries} retries.`);
    }
    return response;
  }

  _credentialsForVertexAI(auth) {
    try {
      const service = OAuth2.createService("Vertex")
        .setTokenUrl('https://oauth2.googleapis.com/token')
        .setPrivateKey(auth.private_key)
        .setIssuer(auth.client_email)
        .setPropertyStore(PropertiesService.getScriptProperties())
        .setCache(CacheService.getScriptCache())
        .setScope("https://www.googleapis.com/auth/cloud-platform");
      return { accessToken: service.getAccessToken() };
    } catch (e) {
      console.error(e);
    }
  }

  _formatGenerateContentInput(params) {
    if (params.contents) {
      return params;
    } else {
      const content = this._formatNewContent(params);
      return { contents: [content] };
    }
  }

  _formatNewContent(request) {
    let newParts = [];
    if (typeof request === "string") {
      newParts = [{ text: request }];
    } else {
      for (const partOrString of request) {
        if (typeof partOrString === "string") {
          newParts.push({ text: partOrString });
        } else {
          newParts.push(partOrString);
        }
      }
    }
    return this._assignRoleToPartsAndValidateSendMessageRequest(newParts);
  }

  _assignRoleToPartsAndValidateSendMessageRequest(parts) {
    const userContent = { role: "user", parts: [] };
    const functionContent = { role: "function", parts: [] };
    let hasUserContent = false;
    let hasFunctionContent = false;
    for (const part of parts) {
      if ("functionResponse" in part) {
        functionContent.parts.push(part);
        hasFunctionContent = true;
      } else {
        userContent.parts.push(part);
        hasUserContent = true;
      }
    }

    if (hasUserContent && hasFunctionContent) {
      throw new Error("FunctionResponse cannot be mixed with other types of parts in the request for sending a chat message.");
    }

    if (!hasUserContent && !hasFunctionContent) {
      throw new Error("No content is provided for sending a chat message.");
    }

    if (hasUserContent) {
      return userContent;
    }

    return functionContent;
  }

  _addHelpers(response) {
    // Helper functions added to response object
    response.text = () => {
      if (response.candidates?.[0]?.content?.parts?.[0]?.text) {
        return response.candidates[0].content.parts.map(({ text }) => text).join("");
      } else {
        return "";
      }
    };

    response.getFunctionCall = () => {
      return response.candidates?.[0]?.content?.parts?.[0]?.functionCall || "";
    };

    return response;
  }
}

/**
 * Class representing _GoogleGenerativeAI
 * 
 * @constructor
 * @param {Object|string} options - Configuration options for the class instance.
 * @param {string} [options.apiKey] - API key for authentication.
 * @param {string} [options.region] - Region for the Vertex AI project.
 * @param {string} [options.project_id] - Project ID for the Vertex AI project
 * @param {string} [options.type] - Type of authentication (e.g., 'service_account').
 * @param {string} [options.private_key] - Private key for service account authentication.
 * @param {string} [options.client_email] - Client email for service account authentication .
 */
class _GoogleGenerativeAI extends _CoreFunctions {
  constructor(options) {
    super();
    this.model = options.model;
    this.auth = options.auth;
  }

  generateContent(params, requestOptions = {}) {
    const formattedParams = this._formatGenerateContentInput(params);
    return this._generateContent(this.auth, this.model, formattedParams, requestOptions);
  }
}

/**
 * Enum representing the task types.
 * @enum {string}
 */
const Task = {
  COUNT_TOKENS: "count_tokens",
  GENERATE_CONTENT: "generate_content"
};

/**
 * Enum representing the model types.
 * @enum {string}
 */
const Model = {
  DIALOGUE: "dialogue",
  CONTENT_CREATOR: "content-creator"
};

/**
 * Enum representing the model types.
 * @enum {string}
 */
const ModelVersion = {
  V1: "v1",
  V2: "v2"
};

/**
 * Class representing RequestUrl
 * 
 * @constructor
 * @param {Model} model - Model type for the request.
 * @param {Task} task - Task type for the request.
 * @param {Object} auth - Authentication parameters for the request.
 * @param {boolean} useSSL - Whether to use SSL for the request.
 * @param {Object} requestOptions - Additional options for the request.
 */
class RequestUrl {
  constructor(model, task, auth, useSSL, requestOptions) {
    this.model = model;
    this.task = task;
    this.auth = auth;
    this.useSSL = useSSL;
    this.requestOptions = requestOptions;
  }

  toString() {
    let url = `https://${this.model}.googleapis.com/${this.task}`;
    if (this.model === Model.CONTENT_CREATOR) {
      url += `?model_version=${this.requestOptions.modelVersion}`;
    }
    return url;
  }
}

/**
 * Function to generate content.
 * 
 * @param {Object} auth - Authentication parameters for the request.
 * @param {Model} model - Model type for the request.
 * @param {Object|string} params - Parameters for the request.
 * @param {Object} requestOptions - Additional options for the request.
 * @returns {Object} Response object containing generated content.
 */
function generateContent(auth, model, params, requestOptions = {}) {
  const generativeAI = new _GoogleGenerativeAI({ model, auth });
  return generativeAI.generateContent(params, requestOptions);
}
