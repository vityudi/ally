//! Model Runtime: a single interface over interchangeable inference backends
//! (llama.cpp, Ollama, ONNX Runtime, MLX, OpenAI/Anthropic-compatible APIs, PALM).

use async_trait::async_trait;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ModelError {
    #[error("backend error: {0}")]
    Backend(String),
}

#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[async_trait]
pub trait ModelBackend: Send + Sync {
    async fn chat(&self, messages: &[ChatMessage]) -> Result<String, ModelError>;
    async fn embed(&self, text: &str) -> Result<Vec<f32>, ModelError>;
}
