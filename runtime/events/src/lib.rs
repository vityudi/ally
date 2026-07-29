//! Event Bus: decoupled communication between Runtime modules.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Event {
    ConversationStarted { conversation_id: String },
    ConversationEnded { conversation_id: String },
    MemoryCreated { memory_id: String },
    ToolExecuted { tool_name: String },
    PluginInstalled { plugin_name: String },
    ReminderCreated { reminder_id: String },
}

pub trait EventHandler: Send + Sync {
    fn handle(&self, event: &Event);
}

#[derive(Default)]
pub struct EventBus {
    handlers: Vec<Box<dyn EventHandler>>,
}

impl EventBus {
    pub fn new() -> Self {
        Self { handlers: Vec::new() }
    }

    pub fn subscribe(&mut self, handler: Box<dyn EventHandler>) {
        self.handlers.push(handler);
    }

    pub fn publish(&self, event: Event) {
        for handler in &self.handlers {
            handler.handle(&event);
        }
    }
}
