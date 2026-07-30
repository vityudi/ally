//! Ally SDK: the only interface applications are meant to depend on.
//! Applications never talk to the model, memory or tools directly —
//! they talk to `Ally`, and the Runtime does the rest.

use ally_context::ContextEngine;
use ally_events::{Event, EventBus, EventHandler};
use ally_memory::{MemoryEngine, MemoryError, MemoryKind, MemoryRecord};
use ally_planner::{Intent, Planner};
use ally_plugins::{Plugin, PluginManager};
use ally_security::PermissionSet;
use ally_storage::{SqliteStorage, Storage, StorageError};
use ally_tools::{ToolError, ToolOrchestrator};
use serde_json::Value;
use std::sync::Arc;

pub struct Ally {
    pub planner: Planner,
    pub memory: MemoryEngine,
    pub context: ContextEngine,
    pub tools: ToolOrchestrator,
    pub plugins: PluginManager,
    pub events: EventBus,
    pub permissions: PermissionSet,
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

    /// Creates an `Ally` backed by any [`Storage`] implementation.
    pub fn with_storage(storage: Arc<dyn Storage>) -> Self {
        Self {
            planner: Planner::new(),
            memory: MemoryEngine::new(storage),
            context: ContextEngine::new(),
            tools: ToolOrchestrator::new(),
            plugins: PluginManager::new(),
            events: EventBus::new(),
            permissions: PermissionSet::default(),
        }
    }

    /// Subscribes an additional listener to every Runtime event
    /// (`ConversationStarted`, `MemoryCreated`, `ToolExecuted`, ...).
    pub fn on_event(&mut self, handler: Box<dyn EventHandler>) {
        self.events.subscribe(handler);
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

    /// Installs a plugin and emits `PluginInstalled`.
    pub fn install_plugin(&mut self, plugin: Box<dyn Plugin>) {
        self.plugins.install(plugin, &self.events);
    }

    /// Executes a registered tool under the currently granted permissions,
    /// emitting `ToolExecuted` on success.
    pub async fn execute_tool(&self, tool_name: &str, input: Value) -> Result<Value, ToolError> {
        self.tools
            .execute(tool_name, input, &self.permissions, &self.events)
            .await
    }
}
