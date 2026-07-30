//! finance plugin: personal finance tools backed by an in-memory ledger
//! (transactions, budgets, goals) plus the pre-existing reminder scheduler.
//! Mirrors the tool surface of the reference Kyvo application (see
//! `apps/api/src/lib/tools.ts` there) closely enough that a Language Model
//! can register spending, check a real balance instead of guessing one,
//! and track budgets/goals — the same "AI is never the source of truth
//! for a number" principle the framework's docs call for.

mod categories;
mod dates;
mod nlu;

use ally_scheduler::{ScheduledTask, Scheduler};
use ally_security::Permission;
use ally_tools::{ExtractOutcome, Tool, ToolError};
use async_trait::async_trait;
use chrono::{Datelike, NaiveDate};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Parses an optional `YYYY-MM-DD` field, defaulting to today (the local
/// system date) when absent. A small model asked to compute "today"
/// itself is unreliable, so the server — not the model — is the source of
/// truth for the current date; the model only needs to pass a date at all
/// when explicitly backdating (e.g. logging a receipt from yesterday).
fn optional_date(input: &Value, field: &str) -> Result<NaiveDate, ToolError> {
    match input.get(field).and_then(Value::as_str).map(str::trim) {
        None | Some("") => Ok(chrono::Local::now().date_naive()),
        // A model asked for an ISO date sometimes tacks on a time
        // component anyway (e.g. "2026-07-30T10:00"); only the date part
        // before 'T' is ever meaningful here, so take that instead of
        // rejecting an otherwise-valid date over a part we don't use.
        Some(raw) => NaiveDate::parse_from_str(raw.split('T').next().unwrap_or(raw), "%Y-%m-%d")
            .map_err(|_| ToolError::Execution(format!("'{field}' must be an ISO date (YYYY-MM-DD)"))),
    }
}

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TransactionKind {
    Expense,
    Income,
}

#[derive(Debug, Clone)]
struct Transaction {
    kind: TransactionKind,
    amount: f64,
    category: Option<String>,
    note: Option<String>,
    date: NaiveDate,
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
    transactions: Vec<Transaction>,
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
                "note": { "type": "string", "description": "Optional free-text description" },
                "date": { "type": "string", "description": "Optional ISO date (YYYY-MM-DD) this was spent on; omit to use today" }
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
        let category = categories::resolve_category(&category).map(str::to_string).unwrap_or(category);
        let note = optional_str(&input, "note");
        let date = optional_date(&input, "date")?;

        let balance = {
            let mut ledger = self.ledger.lock().expect("ledger mutex poisoned");
            ledger.balance -= amount;
            ledger.transactions.push(Transaction {
                kind: TransactionKind::Expense,
                amount,
                category: Some(category.clone()),
                note: note.clone(),
                date,
            });
            ledger.balance
        };

        Ok(json!({
            "status": "registered", "kind": "expense",
            "amount": amount, "category": category, "note": note, "date": date.to_string(), "balance": balance
        }))
    }

    fn trigger_phrases(&self) -> Vec<&str> {
        vec!["gastei", "paguei", "comprei"]
    }

    fn extract_args(&self, message: &str) -> ExtractOutcome {
        nlu::extract_expense_args(message, chrono::Local::now().date_naive())
    }

    fn describe_result(&self, result: &Value) -> Option<String> {
        // Deterministic on purpose: asking the model to phrase this result
        // reliably invented details it never returned (e.g. a restaurant
        // name) even when told to use only the tool result — see this
        // project's small-model-prompting notes. Every field used here is
        // already in `result` from `execute` above.
        let amount = result.get("amount")?.as_f64()?;
        let category = result.get("category").and_then(Value::as_str).unwrap_or("outros");
        let date = result.get("date").and_then(Value::as_str).unwrap_or("");
        let balance = result.get("balance").and_then(Value::as_f64)?;
        Some(format!(
            "Gasto de {amount:.2} em \"{category}\" registrado ({date}). Saldo atual: {balance:.2}."
        ))
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
                "note": { "type": "string", "description": "Optional free-text description" },
                "date": { "type": "string", "description": "Optional ISO date (YYYY-MM-DD) this was received on; omit to use today" }
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
        let date = optional_date(&input, "date")?;

        let balance = {
            let mut ledger = self.ledger.lock().expect("ledger mutex poisoned");
            ledger.balance += amount;
            ledger.transactions.push(Transaction {
                kind: TransactionKind::Income,
                amount,
                category: None,
                note: note.clone(),
                date,
            });
            ledger.balance
        };

        Ok(json!({
            "status": "registered", "kind": "income",
            "amount": amount, "note": note, "date": date.to_string(), "balance": balance
        }))
    }

    fn trigger_phrases(&self) -> Vec<&str> {
        vec!["recebi", "ganhei"]
    }

    fn extract_args(&self, message: &str) -> ExtractOutcome {
        nlu::extract_income_args(message, chrono::Local::now().date_naive())
    }

    fn describe_result(&self, result: &Value) -> Option<String> {
        // See RegisterExpenseTool::describe_result for why this is
        // deterministic instead of model-phrased.
        let amount = result.get("amount")?.as_f64()?;
        let date = result.get("date").and_then(Value::as_str).unwrap_or("");
        let balance = result.get("balance").and_then(Value::as_f64)?;
        Some(format!("Recebimento de {amount:.2} registrado ({date}). Saldo atual: {balance:.2}."))
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

    fn trigger_phrases(&self) -> Vec<&str> {
        vec!["meu saldo", "qual e o meu saldo", "qual o meu saldo", "what's my balance", "what is my balance"]
    }

    fn describe_result(&self, result: &Value) -> Option<String> {
        // Deterministic on purpose, same reasoning as
        // `RegisterExpenseTool::describe_result` — this also skips a whole
        // extra model generation for what's otherwise a one-number answer.
        let balance = result.get("balance")?.as_f64()?;
        Some(format!("Seu saldo atual e de {balance:.2}."))
    }
}

/// Answers "how much did I spend [in a period]" with the actual recorded
/// expense transactions, instead of the model estimating from
/// conversation history. `finance.get_balance` mixes income and expenses
/// into one running total and isn't scoped to a period, so this is the
/// tool for period-scoped spending questions specifically.
pub struct GetSpendingTool {
    ledger: Arc<Mutex<Ledger>>,
}

#[async_trait]
impl Tool for GetSpendingTool {
    fn name(&self) -> &str {
        "finance.get_spending"
    }

    fn description(&self) -> &str {
        "Returns total money spent (expenses only, not income) between two dates, inclusive, \
         plus the matching transactions. Defaults to today if no dates are given. Always call \
         this for 'how much did I spend' questions instead of guessing from memory."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "from": { "type": "string", "description": "Optional ISO start date (YYYY-MM-DD), inclusive; defaults to today" },
                "to": { "type": "string", "description": "Optional ISO end date (YYYY-MM-DD), inclusive; defaults to today" },
                "category": { "type": "string", "description": "Optional category to filter by, e.g. 'food' or 'transport'" }
            }
        })
    }

    fn required_permissions(&self) -> Vec<Permission> {
        vec![Permission::Read]
    }

    async fn execute(&self, input: Value) -> Result<Value, ToolError> {
        let from = optional_date(&input, "from")?;
        let to = optional_date(&input, "to")?;
        if from > to {
            return Err(ToolError::Execution("'from' must not be after 'to'".to_string()));
        }
        let category = optional_str(&input, "category")
            .map(|category| categories::resolve_category(&category).map(str::to_string).unwrap_or(category));

        let ledger = self.ledger.lock().expect("ledger mutex poisoned");
        let matching: Vec<&Transaction> = ledger
            .transactions
            .iter()
            .filter(|t| {
                t.kind == TransactionKind::Expense
                    && t.date >= from
                    && t.date <= to
                    && category.as_ref().is_none_or(|category| t.category.as_deref() == Some(category.as_str()))
            })
            .collect();

        let total: f64 = matching.iter().map(|t| t.amount).sum();
        let transactions: Vec<Value> = matching
            .iter()
            .map(|t| {
                json!({
                    "amount": t.amount,
                    "category": t.category,
                    "note": t.note,
                    "date": t.date.to_string(),
                })
            })
            .collect();

        Ok(json!({
            "from": from.to_string(), "to": to.to_string(),
            "total": total, "transactions": transactions
        }))
    }

    fn trigger_phrases(&self) -> Vec<&str> {
        vec![
            "quanto gastei",
            "quanto eu gastei",
            "quanto gastou",
            "how much did i spend",
            "how much have i spent",
        ]
    }

    fn extract_args(&self, message: &str) -> ExtractOutcome {
        nlu::extract_spending_args(message, chrono::Local::now().date_naive())
    }

    fn describe_result(&self, result: &Value) -> Option<String> {
        // Deterministic on purpose, same reasoning as
        // `RegisterExpenseTool::describe_result` — letting the model phrase
        // this produced a first-person "gastei" (as if the assistant spent
        // the money) and dropped the per-transaction detail, even though
        // `transactions` already has it. Second person ("voce gastou") and
        // listing what each transaction was for are both free once this
        // isn't left to the model to improvise.
        let total = result.get("total")?.as_f64()?;
        let transactions = result.get("transactions")?.as_array()?;

        if transactions.is_empty() {
            return Some("Voce nao teve nenhum gasto registrado nesse periodo.".to_string());
        }

        let items: Vec<String> = transactions
            .iter()
            .filter_map(|t| {
                let amount = t.get("amount")?.as_f64()?;
                let category = t.get("category").and_then(Value::as_str).unwrap_or("outros");
                Some(format!("{amount:.2} em \"{category}\""))
            })
            .collect();

        Some(format!("Voce gastou {total:.2} no total: {}.", items.join(", ")))
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
        let category = categories::resolve_category(&category).map(str::to_string).unwrap_or(category);
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

/// Answers "how much can I still spend on X" / "am I within budget on X" by
/// comparing a category's budget limit against what's actually been spent
/// in that category this month — the model must never estimate this from
/// memory since it requires both a stored limit and a live sum.
pub struct BudgetStatusTool {
    ledger: Arc<Mutex<Ledger>>,
}

#[async_trait]
impl Tool for BudgetStatusTool {
    fn name(&self) -> &str {
        "finance.get_budget_status"
    }

    fn description(&self) -> &str {
        "Returns a category's budget limit, amount spent this month, and remaining amount."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "category": { "type": "string", "description": "Budget category to check, e.g. 'food' or 'transport'" }
            },
            "required": ["category"]
        })
    }

    fn required_permissions(&self) -> Vec<Permission> {
        vec![Permission::Read]
    }

    async fn execute(&self, input: Value) -> Result<Value, ToolError> {
        let category = required_str(&input, "category")?;
        let category = categories::resolve_category(&category).map(str::to_string).unwrap_or(category);

        let ledger = self.ledger.lock().expect("ledger mutex poisoned");
        let limit = ledger
            .budgets
            .get(&category)
            .map(|budget| budget.limit)
            .ok_or_else(|| ToolError::Execution(format!("no budget set for category '{category}'")))?;

        let today = chrono::Local::now().date_naive();
        let month_start = today.with_day(1).expect("day 1 is always valid");
        let spent: f64 = ledger
            .transactions
            .iter()
            .filter(|t| {
                t.kind == TransactionKind::Expense
                    && t.category.as_deref() == Some(category.as_str())
                    && t.date >= month_start
                    && t.date <= today
            })
            .map(|t| t.amount)
            .sum();

        Ok(json!({
            "category": category, "limit": limit, "spent": spent,
            "remaining": limit - spent, "over_budget": spent > limit
        }))
    }

    fn trigger_phrases(&self) -> Vec<&str> {
        vec![
            "quanto ainda posso gastar",
            "estou dentro do orcamento",
            "estou dentro do orçamento",
            "quanto sobrou do orcamento",
            "quanto sobrou do orçamento",
        ]
    }

    fn extract_args(&self, message: &str) -> ExtractOutcome {
        nlu::extract_budget_status_args(message)
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

    fn trigger_phrases(&self) -> Vec<&str> {
        vec!["meus orcamentos", "meus orçamentos", "quais sao meus orcamentos", "my budgets", "what are my budgets"]
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

    fn trigger_phrases(&self) -> Vec<&str> {
        vec!["minha meta", "minhas metas", "my goals", "my savings goal", "how is my goal"]
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
            Box::new(GetSpendingTool { ledger: self.ledger.clone() }),
            Box::new(CreateBudgetTool { ledger: self.ledger.clone() }),
            Box::new(CreateGoalTool { ledger: self.ledger.clone() }),
            Box::new(BudgetStatusTool { ledger: self.ledger.clone() }),
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
    async fn get_spending_defaults_to_today_and_ignores_income() {
        let ledger = empty_ledger();
        let expense = RegisterExpenseTool { ledger: ledger.clone() };
        let income = RegisterIncomeTool { ledger: ledger.clone() };
        let spending = GetSpendingTool { ledger: ledger.clone() };

        expense
            .execute(json!({ "amount": 78.0, "category": "market" }))
            .await
            .expect("expense should succeed");
        income
            .execute(json!({ "amount": 100.0 }))
            .await
            .expect("income should succeed");

        let result = spending.execute(json!({})).await.expect("get_spending should succeed");
        assert_eq!(result["total"], 78.0);
        assert_eq!(result["transactions"].as_array().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn get_spending_filters_by_explicit_date_range() {
        let ledger = empty_ledger();
        let expense = RegisterExpenseTool { ledger: ledger.clone() };
        let spending = GetSpendingTool { ledger: ledger.clone() };

        expense
            .execute(json!({ "amount": 30.0, "category": "food", "date": "2026-01-01" }))
            .await
            .expect("expense should succeed");
        expense
            .execute(json!({ "amount": 20.0, "category": "food", "date": "2026-02-01" }))
            .await
            .expect("expense should succeed");

        let result = spending
            .execute(json!({ "from": "2026-01-01", "to": "2026-01-31" }))
            .await
            .expect("get_spending should succeed");
        assert_eq!(result["total"], 30.0);
    }

    #[tokio::test]
    async fn get_spending_rejects_from_after_to() {
        let spending = GetSpendingTool { ledger: empty_ledger() };

        let err = spending
            .execute(json!({ "from": "2026-02-01", "to": "2026-01-01" }))
            .await
            .unwrap_err();

        assert!(matches!(err, ToolError::Execution(_)));
    }

    #[tokio::test]
    async fn register_expense_rejects_invalid_date() {
        let expense = RegisterExpenseTool { ledger: empty_ledger() };

        let err = expense
            .execute(json!({ "amount": 10.0, "category": "food", "date": "not-a-date" }))
            .await
            .unwrap_err();

        assert!(matches!(err, ToolError::Execution(_)));
    }

    #[tokio::test]
    async fn register_expense_tolerates_a_time_component_on_the_date() {
        let expense = RegisterExpenseTool { ledger: empty_ledger() };

        let result = expense
            .execute(json!({ "amount": 10.0, "category": "food", "date": "2026-07-30T10:00:00" }))
            .await
            .expect("execute should succeed");

        assert_eq!(result["date"], "2026-07-30");
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
