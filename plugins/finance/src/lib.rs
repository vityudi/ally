//! finance plugin: schedules payment reminders. First real capability
//! wired into the Runtime — everything else here stays a stub.

use ally_security::Permission;
use ally_tools::{Tool, ToolError};
use async_trait::async_trait;
use serde_json::{json, Value};

/// Deterministically "schedules" a reminder (in Phase 3 this just echoes
/// the request back; wiring it to `runtime/scheduler` is future work).
/// Mirrors the `finance.schedule_payment` walkthrough in
/// `docs/ARCHITECTURE.md`: the Planner already routes to this action, and
/// a Language Model can request it as a tool call via `Ally::chat`.
pub struct CreateReminderTool;

#[async_trait]
impl Tool for CreateReminderTool {
    fn name(&self) -> &str {
        "finance.create_reminder"
    }

    fn description(&self) -> &str {
        "Schedules a reminder for a financial task, such as an upcoming bill payment."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "note": { "type": "string", "description": "What the reminder is about" },
                "due": { "type": "string", "description": "When it is due, e.g. 'tomorrow' or an ISO date" }
            },
            "required": ["note", "due"]
        })
    }

    fn required_permissions(&self) -> Vec<Permission> {
        vec![Permission::Notifications]
    }

    async fn execute(&self, input: Value) -> Result<Value, ToolError> {
        let note = input
            .get("note")
            .and_then(Value::as_str)
            .ok_or_else(|| ToolError::Execution("missing 'note'".to_string()))?;
        let due = input
            .get("due")
            .and_then(Value::as_str)
            .ok_or_else(|| ToolError::Execution("missing 'due'".to_string()))?;

        Ok(json!({ "status": "scheduled", "note": note, "due": due }))
    }
}

pub struct FinancePlugin;

impl ally_plugins::Plugin for FinancePlugin {
    fn name(&self) -> &str {
        "finance"
    }

    fn permissions(&self) -> Vec<Permission> {
        vec![Permission::Notifications]
    }

    fn tools(&self) -> Vec<Box<dyn Tool>> {
        vec![Box::new(CreateReminderTool)]
    }
}
