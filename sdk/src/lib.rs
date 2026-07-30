//! Ally SDK: the only interface applications are meant to depend on.
//! Applications never talk to the model, memory or tools directly —
//! they talk to `Ally`, and the Runtime does the rest.

use ally_context::ContextEngine;
use ally_events::{Event, EventBus, EventHandler};
use ally_memory::{MemoryEngine, MemoryError, MemoryKind, MemoryRecord};
use ally_models::{ChatRequest, ChatResponse, ModelBackend, ModelError, OllamaBackend, ToolSpec};
use ally_planner::{Intent, Planner};
use ally_plugins::{Plugin, PluginManager};
use ally_security::{Permission, PermissionSet};
use ally_storage::{SqliteStorage, Storage, StorageError};
use ally_tools::{ToolError, ToolOrchestrator};
use serde_json::Value;
use std::sync::Arc;
use thiserror::Error;

/// Default local Ollama model used when an application doesn't configure
/// one explicitly. `0.5b` reports tool-calling support but did not
/// reliably emit tool calls in testing; `1.5b` is the smallest size that
/// did, so it's the default despite the larger download. Swap it with
/// `Ally::with_model` for anything else.
const DEFAULT_MODEL: &str = "qwen2.5:1.5b";

/// Caps how many tool-call round trips `Ally::chat` will make in a single
/// turn, so a model that keeps requesting tools can't loop forever.
const MAX_TOOL_ROUNDS: usize = 4;

#[derive(Debug, Error)]
pub enum ChatError {
    #[error(transparent)]
    Model(#[from] ModelError),
    #[error(transparent)]
    Tool(#[from] ToolError),
}

pub struct Ally {
    pub planner: Planner,
    pub memory: MemoryEngine,
    pub context: ContextEngine,
    pub tools: ToolOrchestrator,
    pub plugins: PluginManager,
    pub events: EventBus,
    pub permissions: PermissionSet,
    pub model: Arc<dyn ModelBackend>,
}

impl Default for Ally {
    fn default() -> Self {
        Self::new()
    }
}

impl Ally {
    /// Creates an `Ally` backed by a throwaway in-memory SQLite database.
    /// Handy for tests and demos; data does not survive past the process.
    /// Use [`Ally::open`] to persist memory across runs.
    pub fn new() -> Self {
        let storage = Arc::new(
            SqliteStorage::open_in_memory().expect("failed to open in-memory storage"),
        );
        Self::with_storage(storage)
    }

    /// Creates an `Ally` backed by a SQLite database file on disk.
    pub fn open(path: &str) -> Result<Self, StorageError> {
        let storage = Arc::new(SqliteStorage::open(path)?);
        Ok(Self::with_storage(storage))
    }

    /// Creates an `Ally` backed by any [`Storage`] implementation, using
    /// the default local Ollama backend as its Model Runtime.
    pub fn with_storage(storage: Arc<dyn Storage>) -> Self {
        Self {
            planner: Planner::new(),
            memory: MemoryEngine::new(storage),
            context: ContextEngine::new(),
            tools: ToolOrchestrator::new(),
            plugins: PluginManager::new(),
            events: EventBus::new(),
            permissions: PermissionSet::default(),
            model: Arc::new(OllamaBackend::new(DEFAULT_MODEL)),
        }
    }

    /// Swaps the Model Runtime backend (e.g. a different Ollama model, or
    /// any other `ModelBackend` implementation).
    pub fn with_model(&mut self, model: Arc<dyn ModelBackend>) {
        self.model = model;
    }

    /// Subscribes an additional listener to every Runtime event
    /// (`ConversationStarted`, `MemoryCreated`, `ToolExecuted`, ...).
    pub fn on_event(&mut self, handler: Box<dyn EventHandler>) {
        self.events.subscribe(handler);
    }

    /// Grants permissions to every subsequent `execute_tool` /
    /// `chat` call, replacing whatever was previously granted.
    pub fn grant_permissions(&mut self, permissions: Vec<Permission>) {
        self.permissions = PermissionSet::new(permissions);
    }

    /// Entry point for every user interaction: intent -> plan -> (future:
    /// memory retrieval, tool execution, context assembly, model call).
    pub fn handle_intent(&self, intent: Intent) -> ally_planner::Plan {
        self.planner.plan(&intent, &self.events)
    }

    /// Closes out a conversation started by `handle_intent`.
    pub fn end_conversation(&self, conversation_id: String) {
        self.events.publish(Event::ConversationEnded { conversation_id });
    }

    /// Persists a memory record and emits `MemoryCreated`.
    pub async fn remember(&self, record: MemoryRecord) -> Result<(), MemoryError> {
        self.memory.store(record, &self.events).await
    }

    /// Reads back memories of a given kind.
    pub async fn recall(&self, kind: MemoryKind) -> Result<Vec<MemoryRecord>, MemoryError> {
        self.memory.retrieve(kind).await
    }

    /// Installs a plugin: registers every tool it exposes with the Tool
    /// Orchestrator (so they become callable and visible to the model via
    /// `tool_specs`), then emits `PluginInstalled`.
    pub fn install_plugin(&mut self, plugin: Box<dyn Plugin>) {
        for tool in plugin.tools() {
            self.tools.register(tool);
        }
        self.plugins.install(plugin, &self.events);
    }

    /// Executes a registered tool under the currently granted permissions,
    /// emitting `ToolExecuted` on success.
    pub async fn execute_tool(&self, tool_name: &str, input: Value) -> Result<Value, ToolError> {
        self.tools
            .execute(tool_name, input, &self.permissions, &self.events)
            .await
    }

    /// `ToolSpec`s for every registered tool, ready to hand to a
    /// `ChatRequest` so the model knows what it can request.
    pub fn tool_specs(&self) -> Vec<ToolSpec> {
        self.tools
            .list()
            .map(|t| ToolSpec {
                name: t.name().to_string(),
                description: t.description().to_string(),
                parameters: t.parameters_schema(),
            })
            .collect()
    }

    /// Sends a chat turn to the configured Model Runtime. The model never
    /// executes a tool itself: when it requests one (`ChatResponse::tool_calls`),
    /// this dispatches the call through the permission-checked Tool
    /// Orchestrator, feeds the result back as a `tool` message, and asks
    /// the model again — up to `MAX_TOOL_ROUNDS` times — before returning
    /// the final response.
    pub async fn chat(&self, mut request: ChatRequest) -> Result<ChatResponse, ChatError> {
        let mut response = self.model.chat(request.clone()).await?;

        let mut rounds = 0;
        while !response.tool_calls.is_empty() && rounds < MAX_TOOL_ROUNDS {
            rounds += 1;
            for call in &response.tool_calls {
                let result = self.execute_tool(&call.name, call.arguments.clone()).await?;
                request.messages.push(ally_models::ChatMessage::tool(result.to_string()));
            }
            response = self.model.chat(request.clone()).await?;
        }

        Ok(response)
    }
}
