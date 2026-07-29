//! Ally SDK: the only interface applications are meant to depend on.
//! Applications never talk to the model, memory or tools directly —
//! they talk to `Ally`, and the Runtime does the rest.

use ally_context::ContextEngine;
use ally_events::EventBus;
use ally_memory::MemoryEngine;
use ally_planner::{Intent, Planner};
use ally_plugins::PluginManager;
use ally_security::PermissionSet;
use ally_tools::ToolOrchestrator;

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
        Self {
            planner: Planner::new(),
            memory: MemoryEngine::new(),
            context: ContextEngine::new(),
            tools: ToolOrchestrator::new(),
            plugins: PluginManager::new(),
            events: EventBus::new(),
            permissions: PermissionSet::default(),
        }
    }
}

impl Ally {
    pub fn new() -> Self {
        Self::default()
    }

    /// Entry point for every user interaction: intent -> plan -> (future:
    /// memory retrieval, tool execution, context assembly, model call).
    pub fn handle_intent(&self, intent: Intent) -> ally_planner::Plan {
        self.planner.plan(&intent, &self.events)
    }
}
