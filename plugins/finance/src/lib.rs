//! finance plugin: schedules payment reminders. First real capability
//! wired into the Runtime — everything else here stays a stub.

use ally_scheduler::{ScheduledTask, Scheduler};
use ally_security::Permission;
use ally_tools::{Tool, ToolError};
use async_trait::async_trait;
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};

/// Schedules a reminder by adding a task to the shared `Scheduler` — the
/// wiring the crate's Phase 3 version deferred (it only echoed its input
/// back). Mirrors the `finance.schedule_payment` walkthrough in
/// `docs/ARCHITECTURE.md`: the Planner already routes to this action, and
/// a Language Model can request it as a tool call via `Ally::chat`.
pub struct CreateReminderTool {
    scheduler: Arc<Mutex<Scheduler>>,
}

impl CreateReminderTool {
    pub fn new(scheduler: Arc<Mutex<Scheduler>>) -> Self {
        Self { scheduler }
    }
}

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
        vec![Permission::Write]
    }

    async fn execute(&self, input: Value) -> Result<Value, ToolError> {
        let note = input
            .get("note")
            .and_then(Value::as_str)
            .filter(|s| !s.trim().is_empty())
            .ok_or_else(|| ToolError::Execution("missing or empty 'note'".to_string()))?;
        let due = input
            .get("due")
            .and_then(Value::as_str)
            .filter(|s| !s.trim().is_empty())
            .ok_or_else(|| ToolError::Execution("missing or empty 'due'".to_string()))?;

        self.scheduler
            .lock()
            .expect("scheduler mutex poisoned")
            .add_task(ScheduledTask {
                name: format!("{note} ({due})"),
            });

        Ok(json!({ "status": "scheduled", "note": note, "due": due }))
    }
}

pub struct FinancePlugin {
    scheduler: Arc<Mutex<Scheduler>>,
}

impl FinancePlugin {
    /// Builds the finance plugin around a scheduler shared with the host
    /// application, so it can later inspect or run due reminders (e.g.
    /// via `Scheduler::run_due`).
    pub fn new(scheduler: Arc<Mutex<Scheduler>>) -> Self {
        Self { scheduler }
    }
}

impl ally_plugins::Plugin for FinancePlugin {
    fn name(&self) -> &str {
        "finance"
    }

    fn permissions(&self) -> Vec<Permission> {
        vec![Permission::Write]
    }

    fn tools(&self) -> Vec<Box<dyn Tool>> {
        vec![Box::new(CreateReminderTool::new(self.scheduler.clone()))]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ally_events::EventBus;
    use ally_security::{PermissionSet, SecurityError};
    use ally_tools::ToolOrchestrator;

    fn tool_with_scheduler() -> (CreateReminderTool, Arc<Mutex<Scheduler>>) {
        let scheduler = Arc::new(Mutex::new(Scheduler::new()));
        (CreateReminderTool::new(scheduler.clone()), scheduler)
    }

    #[tokio::test]
    async fn execute_adds_a_task_to_the_scheduler() {
        let (tool, scheduler) = tool_with_scheduler();

        let result = tool
            .execute(json!({ "note": "credit card", "due": "tomorrow" }))
            .await
            .expect("execute should succeed");

        assert_eq!(result["status"], "scheduled");
        assert_eq!(scheduler.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn execute_rejects_empty_note() {
        let (tool, _scheduler) = tool_with_scheduler();

        let err = tool
            .execute(json!({ "note": "  ", "due": "tomorrow" }))
            .await
            .unwrap_err();

        assert!(matches!(err, ToolError::Execution(_)));
    }

    #[tokio::test]
    async fn execute_rejects_empty_due() {
        let (tool, _scheduler) = tool_with_scheduler();

        let err = tool
            .execute(json!({ "note": "credit card", "due": "" }))
            .await
            .unwrap_err();

        assert!(matches!(err, ToolError::Execution(_)));
    }

    #[tokio::test]
    async fn orchestrator_denies_execution_without_write_permission() {
        let scheduler = Arc::new(Mutex::new(Scheduler::new()));
        let mut orchestrator = ToolOrchestrator::new();
        orchestrator.register(Box::new(CreateReminderTool::new(scheduler)));

        let events = EventBus::new();
        let granted = PermissionSet::new(vec![]);

        let err = orchestrator
            .execute(
                "finance.create_reminder",
                json!({ "note": "credit card", "due": "tomorrow" }),
                &granted,
                &events,
            )
            .await
            .unwrap_err();

        assert!(matches!(
            err,
            ToolError::Security(SecurityError::Denied(Permission::Write))
        ));
    }
}
