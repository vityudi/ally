//! Model Runtime: a single interface over interchangeable inference backends
//! (llama.cpp, Ollama, ONNX Runtime, MLX, OpenAI/Anthropic-compatible APIs, PALM).
//!
//! `ModelBackend` covers every capability listed in `ARCHITECTURE.md`:
//! Chat, Tool Calling and Streaming are dedicated methods because their
//! request/response shapes differ; Summarization and Completion are
//! provided as default methods built on top of `chat` so a new backend
//! only has to implement `chat`, `chat_stream` and `embed` to be usable
//! end-to-end. Token Counting is approximate until a backend can report
//! exact counts from its own tokenizer.

mod ollama;

#[cfg(feature = "llama-cpp")]
mod llama_cpp;

pub use ollama::{OllamaBackend, DEFAULT_EMBEDDING_MODEL};

#[cfg(feature = "llama-cpp")]
pub use llama_cpp::LlamaCppBackend;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ModelError {
    #[error("backend error: {0}")]
    Backend(String),
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn user(content: impl Into<String>) -> Self {
        Self { role: "user".to_string(), content: content.into() }
    }

    pub fn system(content: impl Into<String>) -> Self {
        Self { role: "system".to_string(), content: content.into() }
    }

    pub fn tool(content: impl Into<String>) -> Self {
        Self { role: "tool".to_string(), content: content.into() }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self { role: "assistant".to_string(), content: content.into() }
    }
}

/// Describes a callable Tool to the model, in the shape most local
/// backends (Ollama, OpenAI-compatible) already expect. The Model Runtime
/// never executes a tool itself — see `ally_sdk::Ally::chat`, which
/// dispatches `ChatResponse::tool_calls` through the Tool Orchestrator.
#[derive(Debug, Clone, Serialize)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ToolCall {
    pub name: String,
    #[serde(default)]
    pub arguments: Value,
}

#[derive(Debug, Clone, Default)]
pub struct ChatRequest {
    pub messages: Vec<ChatMessage>,
    pub tools: Vec<ToolSpec>,
}

#[derive(Debug, Clone, Default)]
pub struct ChatResponse {
    pub message: ChatMessage,
    pub tool_calls: Vec<ToolCall>,
}

#[async_trait]
pub trait ModelBackend: Send + Sync {
    fn name(&self) -> &str;

    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse, ModelError>;

    /// Same contract as `chat`, but invokes `on_token` as content streams
    /// in before returning the final assembled response.
    async fn chat_stream(
        &self,
        request: ChatRequest,
        on_token: Box<dyn for<'a> FnMut(&'a str) + Send>,
    ) -> Result<ChatResponse, ModelError>;

    async fn embed(&self, text: &str) -> Result<Vec<f32>, ModelError>;

    /// Approximate token count (~4 characters per token). Backends with
    /// access to their own tokenizer should override this.
    fn count_tokens(&self, text: &str) -> usize {
        (text.len() / 4).max(1)
    }

    async fn summarize(&self, text: &str) -> Result<String, ModelError> {
        let request = ChatRequest {
            messages: vec![ChatMessage::user(format!(
                "Summarize the following text in 2-3 sentences:\n\n{text}"
            ))],
            tools: Vec::new(),
        };
        Ok(self.chat(request).await?.message.content)
    }

    async fn complete(&self, prompt: &str) -> Result<String, ModelError> {
        let request = ChatRequest {
            messages: vec![ChatMessage::user(prompt)],
            tools: Vec::new(),
        };
        Ok(self.chat(request).await?.message.content)
    }
}
