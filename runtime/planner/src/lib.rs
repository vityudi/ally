//! Planner: transforms user intentions into deterministic, executable plans.
//! The Language Model participates only when a plan cannot be resolved
//! without language generation.

use ally_events::{Event, EventBus};

#[derive(Debug, Clone)]
pub struct Intent {
    pub name: String,
}

#[derive(Debug, Clone, Default)]
pub struct Plan {
    pub steps: Vec<String>,
}

#[derive(Default)]
pub struct Planner;

impl Planner {
    pub fn new() -> Self {
        Self
    }

    pub fn plan(&self, intent: &Intent, events: &EventBus) -> Plan {
        events.publish(Event::ConversationStarted {
            conversation_id: intent.name.clone(),
        });
        Plan::default()
    }
}
