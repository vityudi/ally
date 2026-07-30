//! Planner: transforms user intentions into deterministic, executable plans.
//! The Language Model participates only when a plan cannot be resolved
//! without language generation.

use ally_events::{Event, EventBus};

#[derive(Debug, Clone)]
pub struct Intent {
    pub name: String,
}

/// One deterministic action in a [`Plan`]. The Planner only ever produces
/// references to memory kinds and tool names — never free-form strings —
/// so a future executor can dispatch a step without re-parsing it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlanStep {
    RetrieveMemory { kind: &'static str },
    ExecuteTool { name: &'static str },
    /// Language generation is required for this step (e.g. producing a
    /// confirmation message). This is the only step type the Runtime
    /// cannot resolve deterministically.
    GenerateResponse,
}

#[derive(Debug, Clone, Default)]
pub struct Plan {
    pub steps: Vec<PlanStep>,
    /// Whether any step in this plan needs the Language Model. `false`
    /// means the Runtime can respond without ever calling a model.
    pub requires_model: bool,
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
        resolve(intent)
    }
}

/// Rule-based intent resolver. Deliberately simple pattern matching for
/// Phase 2 — a learned/ML resolver can replace this later without
/// changing the `Plan`/`PlanStep` contract callers depend on.
fn resolve(intent: &Intent) -> Plan {
    match intent.name.as_str() {
        // Mirrors the walkthrough in docs/ARCHITECTURE.md:
        // Retrieve invoices -> Retrieve calendar -> Create reminder -> Generate confirmation.
        "finance.schedule_payment" => Plan {
            steps: vec![
                PlanStep::RetrieveMemory { kind: "episodic" },
                PlanStep::ExecuteTool { name: "calendar.retrieve" },
                PlanStep::ExecuteTool { name: "calendar.create_reminder" },
                PlanStep::GenerateResponse,
            ],
            requires_model: true,
        },
        // Unknown intents fall back to a plain conversational response —
        // no tool or memory access, but still not resolvable without the
        // model.
        _ => Plan {
            steps: vec![PlanStep::GenerateResponse],
            requires_model: true,
        },
    }
}
