//! finance plugin: personal finance tools backed by an in-memory ledger
//! (transactions, budgets, goals) plus the pre-existing reminder scheduler.
//! Mirrors the tool surface of the reference Kyvo application (see
//! `apps/api/src/lib/tools.ts` there) closely enough that a Language Model
//! can register spending, check a real balance instead of guessing one,
//! and track budgets/goals — the same "AI is never the source of truth
//! for a number" principle the framework's docs call for.

use ally_scheduler::{ScheduledTask, Scheduler};
use ally_security::Permission;
use ally_tools::{Tool, ToolError};
use async_trait::async_trait;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

fn required_str(input: &Value, field: &str) -> Result<String, ToolError> {
    input
        .get(field)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .ok_or_else(|| ToolError::Execution(format!("missing or empty '{field}'")))
}

fn optional_str(input: &Value, field: &str) -> Option<String> {
    input
        .get(field)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
}

fn positive_amount(input: &Value, field: &str) -> Result<f64, ToolError> {
    let amount = input
        .get(field)
        .and_then(Value::as_f64)
        .ok_or_else(|| ToolError::Execution(format!("missing or non-numeric '{field}'")))?;
    if amount <= 0.0 {
        return Err(ToolError::Execution(format!("'{field}' must be greater than zero")));
    }
    Ok(amount)
}

#[derive(Debug, Clone)]
struct Budget {
    limit: f64,
}

#[derive(Debug, Clone)]
struct Goal {
    target_amount: f64,
    current_amount: f64,
    deadline: Option<String>,
}

/// Shared state behind every tool in this plugin except the reminder
/// scheduler. Kept in-memory (like `Scheduler`) rather than in
/// `runtime/storage` for now — see `ARCHITECTURE.md` before promoting this
/// to a persisted ledger shared across plugins.
#[derive(Default)]
struct Ledger {
    balance: f64,
    budgets: HashMap<String, Budget>,
    goals: HashMap<String, Goal>,
}

/// Schedules a reminder by adding a task to the shared `Scheduler`. Mirrors
/// the `finance.schedule_payment` walkthrough in `docs/ARCHITECTURE.md`.
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
        let note = required_str(&input, "note")?;
        let due = required_str(&input, "due")?;

        self.scheduler
            .lock()
            .expect("scheduler mutex poisoned")
            .add_task(ScheduledTask { name: format!("{note} ({due})") });

        Ok(json!({ "status": "scheduled", "note": note, "due": due }))
    }
}

/// Registers an expense, subtracting it from the ledger's balance. The
/// model should call this only once it already knows amount and category
/// with confidence — an ambiguous request should be clarified with the
/// user first, not guessed (see `docs/PRINCIPLES.md`).
pub struct RegisterExpenseTool {
    ledger: Arc<Mutex<Ledger>>,
}

#[async_trait]
impl Tool for RegisterExpenseTool {
    fn name(&self) -> &str {
        "finance.register_expense"
    }

    fn description(&self) -> &str {
        "Records a money outflow (a purchase or bill paid) and subtracts it from the balance."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "amount": { "type": "number", "description": "How much was spent, in the user's currency" },
                "category": { "type": "string", "description": "Spending category, e.g. 'food' or 'transport'" },
                "note": { "type": "string", "description": "Optional free-text description" }
            },
            "required": ["amount", "category"]
        })
    }

    fn required_permissions(&self) -> Vec<Permission> {
        vec![Permission::Write]
    }

    async fn execute(&self, input: Value) -> Result<Value, ToolError> {
        let amount = positive_amount(&input, "amount")?;
        let category = required_str(&input, "category")?;
        let note = optional_str(&input, "note");

        let balance = {
            let mut ledger = self.ledger.lock().expect("ledger mutex poisoned");
            ledger.balance -= amount;
            ledger.balance
        };

        Ok(json!({
            "status": "registered", "kind": "expense",
            "amount": amount, "category": category, "note": note, "balance": balance
        }))
    }
}

/// Registers income, adding it to the ledger's balance.
pub struct RegisterIncomeTool {
    ledger: Arc<Mutex<Ledger>>,
}

#[async_trait]
impl Tool for RegisterIncomeTool {
    fn name(&self) -> &str {
        "finance.register_income"
    }

    fn description(&self) -> &str {
        "Records a money inflow (salary, a payment received) and adds it to the balance."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "amount": { "type": "number", "description": "How much was received, in the user's currency" },
                "note": { "type": "string", "description": "Optional free-text description" }
            },
            "required": ["amount"]
        })
    }

    fn required_permissions(&self) -> Vec<Permission> {
        vec![Permission::Write]
    }

    async fn execute(&self, input: Value) -> Result<Value, ToolError> {
        let amount = positive_amount(&input, "amount")?;
        let note = optional_str(&input, "note");

        let balance = {
            let mut ledger = self.ledger.lock().expect("ledger mutex poisoned");
            ledger.balance += amount;
            ledger.balance
        };

        Ok(json!({ "status": "registered", "kind": "income", "amount": amount, "note": note, "balance": balance }))
    }
}

/// The only source of truth for the current balance — a model must call
/// this instead of estimating a number from conversation memory.
pub struct GetBalanceTool {
    ledger: Arc<Mutex<Ledger>>,
}

#[async_trait]
impl Tool for GetBalanceTool {
    fn name(&self) -> &str {
        "finance.get_balance"
    }

    fn description(&self) -> &str {
        "Returns the current balance. Always call this instead of guessing a balance from memory."
    }

    fn parameters_schema(&self) -> Value {
        json!({ "type": "object", "properties": {} })
    }

    fn required_permissions(&self) -> Vec<Permission> {
        vec![Permission::Read]
    }

    async fn execute(&self, _input: Value) -> Result<Value, ToolError> {
        let balance = self.ledger.lock().expect("ledger mutex poisoned").balance;
        Ok(json!({ "balance": balance }))
    }
}

/// Creates or replaces a monthly spending limit for a category.
pub struct CreateBudgetTool {
    ledger: Arc<Mutex<Ledger>>,
}

#[async_trait]
impl Tool for CreateBudgetTool {
    fn name(&self) -> &str {
        "finance.create_budget"
    }

    fn description(&self) -> &str {
        "Sets a monthly spending limit for a category."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "category": { "type": "string", "description": "Spending category this budget applies to" },
                "limit": { "type": "number", "description": "Maximum spend per month, in the user's currency" }
            },
            "required": ["category", "limit"]
        })
    }

    fn required_permissions(&self) -> Vec<Permission> {
        vec![Permission::Write]
    }

    async fn execute(&self, input: Value) -> Result<Value, ToolError> {
        let category = required_str(&input, "category")?;
        let limit = positive_amount(&input, "limit")?;

        self.ledger
            .lock()
            .expect("ledger mutex poisoned")
            .budgets
            .insert(category.clone(), Budget { limit });

        Ok(json!({ "status": "created", "category": category, "limit": limit }))
    }
}

/// Creates a savings goal, tracked separately from the general balance.
pub struct CreateGoalTool {
    ledger: Arc<Mutex<Ledger>>,
}

#[async_trait]
impl Tool for CreateGoalTool {
    fn name(&self) -> &str {
        "finance.create_goal"
    }

    fn description(&self) -> &str {
        "Creates a savings goal with a target amount and an optional deadline."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "name": { "type": "string", "description": "Short name for the goal, e.g. 'trip to December'" },
                "target_amount": { "type": "number", "description": "How much needs to be saved in total" },
                "deadline": { "type": "string", "description": "Optional target date, e.g. an ISO date" }
            },
            "required": ["name", "target_amount"]
        })
    }

    fn required_permissions(&self) -> Vec<Permission> {
        vec![Permission::Write]
    }

    async fn execute(&self, input: Value) -> Result<Value, ToolError> {
        let name = required_str(&input, "name")?;
        let target_amount = positive_amount(&input, "target_amount")?;
        let deadline = optional_str(&input, "deadline");

        self.ledger.lock().expect("ledger mutex poisoned").goals.insert(
            name.clone(),
            Goal { target_amount, current_amount: 0.0, deadline: deadline.clone() },
        );

        Ok(json!({ "status": "created", "name": name, "target_amount": target_amount, "deadline": deadline }))
    }
}

/// Lists every budget created so far, so the model can answer "what are my
/// budgets" without guessing from conversation history.
pub struct ListBudgetsTool {
    ledger: Arc<Mutex<Ledger>>,
}

#[async_trait]
impl Tool for ListBudgetsTool {
    fn name(&self) -> &str {
        "finance.list_budgets"
    }

    fn description(&self) -> &str {
        "Lists every budget (category + monthly limit) created so far."
    }

    fn parameters_schema(&self) -> Value {
        json!({ "type": "object", "properties": {} })
    }

    fn required_permissions(&self) -> Vec<Permission> {
        vec![Permission::Read]
    }

    async fn execute(&self, _input: Value) -> Result<Value, ToolError> {
        let budgets: Vec<Value> = self
            .ledger
            .lock()
            .expect("ledger mutex poisoned")
            .budgets
            .iter()
            .map(|(category, budget)| json!({ "category": category, "limit": budget.limit }))
            .collect();

        Ok(json!({ "budgets": budgets }))
    }
}

/// Lists every savings goal created so far, including progress toward it.
pub struct ListGoalsTool {
    ledger: Arc<Mutex<Ledger>>,
}

#[async_trait]
impl Tool for ListGoalsTool {
    fn name(&self) -> &str {
        "finance.list_goals"
    }

    fn description(&self) -> &str {
        "Lists every savings goal created so far, with target and current amount."
    }

    fn parameters_schema(&self) -> Value {
        json!({ "type": "object", "properties": {} })
    }

    fn required_permissions(&self) -> Vec<Permission> {
        vec![Permission::Read]
    }

    async fn execute(&self, _input: Value) -> Result<Value, ToolError> {
        let goals: Vec<Value> = self
            .ledger
            .lock()
            .expect("ledger mutex poisoned")
            .goals
            .iter()
            .map(|(name, goal)| {
                json!({
                    "name": name,
                    "target_amount": goal.target_amount,
                    "current_amount": goal.current_amount,
                    "deadline": goal.deadline,
                })
            })
            .collect();

        Ok(json!({ "goals": goals }))
    }
}

pub struct FinancePlugin {
    scheduler: Arc<Mutex<Scheduler>>,
    ledger: Arc<Mutex<Ledger>>,
}

impl FinancePlugin {
    /// Builds the finance plugin around a scheduler shared with the host
    /// application, so it can later inspect or run due reminders (e.g.
    /// via `Scheduler::run_due`). The ledger (balance, budgets, goals) is
    /// private to this plugin instance.
    pub fn new(scheduler: Arc<Mutex<Scheduler>>) -> Self {
        Self { scheduler, ledger: Arc::new(Mutex::new(Ledger::default())) }
    }
}

impl ally_plugins::Plugin for FinancePlugin {
    fn name(&self) -> &str {
        "finance"
    }

    fn permissions(&self) -> Vec<Permission> {
        vec![Permission::Read, Permission::Write]
    }

    fn tools(&self) -> Vec<Box<dyn Tool>> {
        vec![
            Box::new(CreateReminderTool::new(self.scheduler.clone())),
            Box::new(RegisterExpenseTool { ledger: self.ledger.clone() }),
            Box::new(RegisterIncomeTool { ledger: self.ledger.clone() }),
            Box::new(GetBalanceTool { ledger: self.ledger.clone() }),
            Box::new(CreateBudgetTool { ledger: self.ledger.clone() }),
            Box::new(CreateGoalTool { ledger: self.ledger.clone() }),
            Box::new(ListBudgetsTool { ledger: self.ledger.clone() }),
            Box::new(ListGoalsTool { ledger: self.ledger.clone() }),
        ]
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

    fn empty_ledger() -> Arc<Mutex<Ledger>> {
        Arc::new(Mutex::new(Ledger::default()))
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

    #[tokio::test]
    async fn expense_then_income_settle_the_balance() {
        let ledger = empty_ledger();
        let expense = RegisterExpenseTool { ledger: ledger.clone() };
        let income = RegisterIncomeTool { ledger: ledger.clone() };
        let balance = GetBalanceTool { ledger: ledger.clone() };

        expense
            .execute(json!({ "amount": 47.0, "category": "food" }))
            .await
            .expect("expense should succeed");
        income
            .execute(json!({ "amount": 100.0 }))
            .await
            .expect("income should succeed");

        let result = balance.execute(json!({})).await.expect("get_balance should succeed");
        assert_eq!(result["balance"], 53.0);
    }

    #[tokio::test]
    async fn register_expense_rejects_non_positive_amount() {
        let expense = RegisterExpenseTool { ledger: empty_ledger() };

        let err = expense
            .execute(json!({ "amount": 0.0, "category": "food" }))
            .await
            .unwrap_err();

        assert!(matches!(err, ToolError::Execution(_)));
    }

    #[tokio::test]
    async fn create_budget_and_goal_round_trip() {
        let ledger = empty_ledger();
        let budget = CreateBudgetTool { ledger: ledger.clone() };
        let goal = CreateGoalTool { ledger: ledger.clone() };

        budget
            .execute(json!({ "category": "food", "limit": 500.0 }))
            .await
            .expect("create_budget should succeed");
        goal.execute(json!({ "name": "trip", "target_amount": 3000.0, "deadline": "2026-12-01" }))
            .await
            .expect("create_goal should succeed");

        let ledger = ledger.lock().unwrap();
        assert_eq!(ledger.budgets["food"].limit, 500.0);
        assert_eq!(ledger.goals["trip"].target_amount, 3000.0);
        assert_eq!(ledger.goals["trip"].deadline.as_deref(), Some("2026-12-01"));
    }
}
