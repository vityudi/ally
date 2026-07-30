//! `ModelBackend` implementation talking to a local (or remote) Ollama
//! daemon over HTTP. Was the original default before `LlamaCppBackend`
//! (`src/llama_cpp.rs`) took over that role — no longer what `Ally::new()`
//! gives you, but still here and still useful: point `Ally::with_model` at
//! `OllamaBackend::new(...)` if you'd rather run a bigger model than the
//! pinned default, or one already served by an existing `ollama serve`
//! instance, without recompiling anything.

use crate::{ChatMessage, ChatRequest, ChatResponse, ModelBackend, ModelError, ToolCall, ToolSpec};
use async_trait::async_trait;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::Value;

const DEFAULT_BASE_URL: &str = "http://localhost:11434";

/// Model used for `embed()` when a backend is constructed without an
/// explicit embedding model. Ollama only serves `/api/embeddings` for
/// models tagged with the `embedding` capability in its registry — chat
/// models like `qwen2.5` are not usable for this even though they load
/// fine for `chat`, so embeddings always need a separate, dedicated model.
/// `all-minilm` is small (~45 MB) and local, matching the embedded-device
/// friendliness called for in `docs/PRINCIPLES.md`.
pub const DEFAULT_EMBEDDING_MODEL: &str = "all-minilm";

pub struct OllamaBackend {
    http: reqwest::Client,
    base_url: String,
    model: String,
    embedding_model: String,
}

impl OllamaBackend {
    /// Connects to a local `ollama serve` instance on the default port,
    /// using `model` for chat and [`DEFAULT_EMBEDDING_MODEL`] for `embed`.
    pub fn new(model: impl Into<String>) -> Self {
        Self::with_base_url(DEFAULT_BASE_URL, model)
    }

    pub fn with_base_url(base_url: impl Into<String>, model: impl Into<String>) -> Self {
        Self::with_embedding_model(base_url, model, DEFAULT_EMBEDDING_MODEL)
    }

    /// Same as [`OllamaBackend::with_base_url`], but lets the embedding
    /// model be set explicitly instead of assuming [`DEFAULT_EMBEDDING_MODEL`]
    /// is pulled. Use this if you've pulled a different embedding model
    /// (e.g. `nomic-embed-text`) or want to point at one already in use
    /// elsewhere.
    pub fn with_embedding_model(
        base_url: impl Into<String>,
        model: impl Into<String>,
        embedding_model: impl Into<String>,
    ) -> Self {
        Self {
            http: reqwest::Client::new(),
            base_url: base_url.into(),
            model: model.into(),
            embedding_model: embedding_model.into(),
        }
    }

    fn url(&self, path: &str) -> String {
        format!("{}{path}", self.base_url.trim_end_matches('/'))
    }
}

#[derive(Serialize)]
struct OllamaTool<'a> {
    r#type: &'static str,
    function: OllamaFunction<'a>,
}

#[derive(Serialize)]
struct OllamaFunction<'a> {
    name: &'a str,
    description: &'a str,
    parameters: &'a Value,
}

#[derive(Serialize)]
struct OllamaChatRequest<'a> {
    model: &'a str,
    messages: &'a [ChatMessage],
    stream: bool,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<OllamaTool<'a>>,
    options: OllamaOptions,
}

/// Zero temperature (greedy decoding) for tool-calling reliability: with
/// default sampling, whether a small model like `qwen2.5:1.5b` actually
/// emits a tool call for the same message is highly non-deterministic —
/// confirmed by running identical prompts repeatedly and getting a
/// tool call maybe 1 time in 5. Greedy decoding won't fix a model that
/// truly can't handle a request, but it removes sampling noise as a
/// separate, compounding source of unreliability. Matches the "deterministic
/// wherever possible" principle already stated for the Planner.
#[derive(Serialize)]
struct OllamaOptions {
    temperature: f32,
}

impl Default for OllamaOptions {
    fn default() -> Self {
        Self { temperature: 0.0 }
    }
}

#[derive(Deserialize)]
struct OllamaChatMessage {
    role: String,
    #[serde(default)]
    content: String,
    #[serde(default)]
    tool_calls: Vec<OllamaToolCall>,
}

#[derive(Deserialize)]
struct OllamaToolCall {
    function: OllamaToolCallFunction,
}

#[derive(Deserialize)]
struct OllamaToolCallFunction {
    name: String,
    #[serde(default)]
    arguments: Value,
}

#[derive(Deserialize)]
struct OllamaChatChunk {
    #[serde(default)]
    message: Option<OllamaChatMessage>,
    #[serde(default)]
    done: bool,
}

#[derive(Serialize)]
struct OllamaEmbedRequest<'a> {
    model: &'a str,
    prompt: &'a str,
}

#[derive(Deserialize)]
struct OllamaEmbedResponse {
    embedding: Vec<f32>,
}

fn to_ollama_tools(tools: &[ToolSpec]) -> Vec<OllamaTool<'_>> {
    tools
        .iter()
        .map(|t| OllamaTool {
            r#type: "function",
            function: OllamaFunction {
                name: &t.name,
                description: &t.description,
                parameters: &t.parameters,
            },
        })
        .collect()
}

fn into_response(message: OllamaChatMessage) -> ChatResponse {
    ChatResponse {
        tool_calls: message
            .tool_calls
            .into_iter()
            .map(|c| ToolCall {
                name: c.function.name,
                arguments: c.function.arguments,
            })
            .collect(),
        message: ChatMessage {
            role: message.role,
            content: message.content,
        },
    }
}

#[async_trait]
impl ModelBackend for OllamaBackend {
    fn name(&self) -> &str {
        "ollama"
    }

    fn model_id(&self) -> String {
        self.model.clone()
    }

    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse, ModelError> {
        let body = OllamaChatRequest {
            model: &self.model,
            messages: &request.messages,
            stream: false,
            tools: to_ollama_tools(&request.tools),
            options: OllamaOptions::default(),
        };

        let response = self
            .http
            .post(self.url("/api/chat"))
            .json(&body)
            .send()
            .await
            .map_err(|e| ModelError::Backend(e.to_string()))?
            .error_for_status()
            .map_err(|e| ModelError::Backend(e.to_string()))?
            .json::<OllamaChatChunk>()
            .await
            .map_err(|e| ModelError::Backend(e.to_string()))?;

        let message = response
            .message
            .ok_or_else(|| ModelError::Backend("ollama returned no message".to_string()))?;
        Ok(into_response(message))
    }

    async fn chat_stream(
        &self,
        request: ChatRequest,
        mut on_token: Box<dyn for<'a> FnMut(&'a str) + Send>,
    ) -> Result<ChatResponse, ModelError> {
        let body = OllamaChatRequest {
            model: &self.model,
            messages: &request.messages,
            stream: true,
            tools: to_ollama_tools(&request.tools),
            options: OllamaOptions::default(),
        };

        let response = self
            .http
            .post(self.url("/api/chat"))
            .json(&body)
            .send()
            .await
            .map_err(|e| ModelError::Backend(e.to_string()))?
            .error_for_status()
            .map_err(|e| ModelError::Backend(e.to_string()))?;

        let mut stream = response.bytes_stream();
        let mut buffer = String::new();
        let mut role = "assistant".to_string();
        let mut content = String::new();
        let mut tool_calls = Vec::new();

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| ModelError::Backend(e.to_string()))?;
            buffer.push_str(&String::from_utf8_lossy(&chunk));

            while let Some(newline) = buffer.find('\n') {
                let line = buffer[..newline].trim().to_string();
                buffer.drain(..=newline);
                if line.is_empty() {
                    continue;
                }

                let parsed: OllamaChatChunk = serde_json::from_str(&line)
                    .map_err(|e| ModelError::Backend(e.to_string()))?;

                if let Some(message) = parsed.message {
                    role = message.role;
                    on_token(&message.content);
                    content.push_str(&message.content);
                    tool_calls.extend(message.tool_calls.into_iter().map(|c| ToolCall {
                        name: c.function.name,
                        arguments: c.function.arguments,
                    }));
                }

                if parsed.done {
                    break;
                }
            }
        }

        Ok(ChatResponse {
            message: ChatMessage { role, content },
            tool_calls,
        })
    }

    async fn embed(&self, text: &str) -> Result<Vec<f32>, ModelError> {
        let body = OllamaEmbedRequest { model: &self.embedding_model, prompt: text };

        let response = self
            .http
            .post(self.url("/api/embeddings"))
            .json(&body)
            .send()
            .await
            .map_err(|e| ModelError::Backend(e.to_string()))?
            .error_for_status()
            .map_err(|e| ModelError::Backend(e.to_string()))?
            .json::<OllamaEmbedResponse>()
            .await
            .map_err(|e| ModelError::Backend(e.to_string()))?;

        Ok(response.embedding)
    }
}
