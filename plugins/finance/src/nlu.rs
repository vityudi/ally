//! Ties trigger-phrase-matched messages to concrete tool arguments for the
//! finance plugin. Pure functions over `&str` (plus a `today` date for
//! testability) returning `ally_tools::ExtractOutcome` — the actual
//! `Tool::extract_args` overrides in `lib.rs` are thin wrappers around these
//! that also supply the real current date.

use crate::categories;
use crate::dates;
use ally_tools::ExtractOutcome;
use chrono::NaiveDate;
use serde_json::json;

/// Finds the first number in `message`, treating whichever of `,`/`.`
/// appears last in a numeric run as the decimal separator (so both
/// "50,00"/"1.500,00" pt-BR style and "50.00"/"1,500.00" en style parse
/// correctly) and dropping any earlier separator as a thousands grouping.
/// A lone separator in an otherwise plain integer (e.g. "1.500" with no
/// other digits after) is ambiguous and treated as decimal — a known,
/// accepted limitation for four-digit-or-larger amounts written that way.
fn extract_amount(message: &str) -> Option<f64> {
    let chars: Vec<char> = message.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if chars[i].is_ascii_digit() {
            let start = i;
            while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == ',' || chars[i] == '.') {
                i += 1;
            }
            let mut end = i;
            while end > start && matches!(chars[end - 1], ',' | '.') {
                end -= 1;
            }
            let token: String = chars[start..end].iter().collect();
            if let Some(amount) = parse_amount_token(&token) {
                return Some(amount);
            }
        } else {
            i += 1;
        }
    }
    None
}

fn parse_amount_token(token: &str) -> Option<f64> {
    if token.is_empty() {
        return None;
    }
    let last_comma = token.rfind(',');
    let last_dot = token.rfind('.');
    let decimal_pos = last_comma.into_iter().chain(last_dot).max();

    let mut normalized = String::new();
    for (idx, ch) in token.char_indices() {
        match ch {
            '0'..='9' => normalized.push(ch),
            ',' | '.' if Some(idx) == decimal_pos => normalized.push('.'),
            ',' | '.' => {} // thousands separator, dropped
            _ => {}
        }
    }
    normalized.parse::<f64>().ok()
}

/// A single date to use for a transaction: the resolved period collapsed to
/// one day when it's a single-day range, otherwise `today` (matching the
/// existing schema default for an omitted date).
fn resolve_transaction_date(message: &str, today: NaiveDate) -> NaiveDate {
    match dates::resolve_period(message, today) {
        Some((from, to)) if from == to => from,
        _ => today,
    }
}

/// `finance.get_spending`: every argument is optional, so this never fails
/// — it always returns `Extracted`, defaulting to today/no-category filter
/// exactly like the tool's own schema defaults when nothing more specific
/// is found in the message.
pub fn extract_spending_args(message: &str, today: NaiveDate) -> ExtractOutcome {
    let category = categories::category_matches_in(message);
    let (from, to) = dates::resolve_period(message, today).unwrap_or((today, today));

    let mut args = json!({ "from": from.to_string(), "to": to.to_string() });
    if let Some(category) = category {
        args["category"] = json!(category);
    }
    ExtractOutcome::Extracted(args)
}

/// `finance.register_expense`: requires an amount, so this returns `Failed`
/// when none is found rather than registering a bogus zero/missing-amount
/// expense — the caller must fall through to the model in that case.
pub fn extract_expense_args(message: &str, today: NaiveDate) -> ExtractOutcome {
    let Some(amount) = extract_amount(message) else {
        return ExtractOutcome::Failed;
    };
    let Some(category) = categories::category_matches_in(message) else {
        return ExtractOutcome::Failed;
    };
    let date = resolve_transaction_date(message, today);

    ExtractOutcome::Extracted(json!({ "amount": amount, "category": category, "date": date.to_string() }))
}

/// `finance.register_income`: requires an amount; no category.
pub fn extract_income_args(message: &str, today: NaiveDate) -> ExtractOutcome {
    let Some(amount) = extract_amount(message) else {
        return ExtractOutcome::Failed;
    };
    let date = resolve_transaction_date(message, today);

    ExtractOutcome::Extracted(json!({ "amount": amount, "date": date.to_string() }))
}

/// `finance.get_budget_status`: requires a category to look up.
pub fn extract_budget_status_args(message: &str) -> ExtractOutcome {
    match categories::category_matches_in(message) {
        Some(category) => ExtractOutcome::Extracted(json!({ "category": category })),
        None => ExtractOutcome::Failed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn today() -> NaiveDate {
        NaiveDate::from_ymd_opt(2026, 7, 30).unwrap()
    }

    #[test]
    fn extract_amount_handles_plain_integer() {
        assert_eq!(extract_amount("gastei 50 no mercado"), Some(50.0));
    }

    #[test]
    fn extract_amount_handles_pt_br_comma_decimal() {
        assert_eq!(extract_amount("gastei 49,90 no mercado"), Some(49.9));
    }

    #[test]
    fn extract_amount_handles_dot_decimal() {
        assert_eq!(extract_amount("spent 49.90 at the market"), Some(49.9));
    }

    #[test]
    fn extract_amount_handles_thousands_and_decimal_together() {
        assert_eq!(extract_amount("paguei 1.500,00 de aluguel"), Some(1500.0));
        assert_eq!(extract_amount("paid 1,500.00 for rent"), Some(1500.0));
    }

    #[test]
    fn extract_amount_returns_none_without_a_number() {
        assert_eq!(extract_amount("gastei um pouco no mercado"), None);
    }

    #[test]
    fn expense_args_extracts_amount_category_and_date() {
        let outcome = extract_expense_args("gastei 50 reais no mercado hoje", today());
        assert_eq!(
            outcome,
            ExtractOutcome::Extracted(json!({ "amount": 50.0, "category": "food", "date": "2026-07-30" }))
        );
    }

    #[test]
    fn expense_args_defaults_date_to_today_when_unspecified() {
        let outcome = extract_expense_args("paguei 30 de transporte", today());
        assert_eq!(
            outcome,
            ExtractOutcome::Extracted(json!({ "amount": 30.0, "category": "transport", "date": "2026-07-30" }))
        );
    }

    #[test]
    fn expense_args_resolves_relative_date() {
        let outcome = extract_expense_args("paguei 30 de transporte ontem", today());
        assert_eq!(
            outcome,
            ExtractOutcome::Extracted(json!({ "amount": 30.0, "category": "transport", "date": "2026-07-29" }))
        );
    }

    #[test]
    fn expense_args_fails_without_an_amount() {
        assert_eq!(extract_expense_args("paguei no mercado hoje", today()), ExtractOutcome::Failed);
    }

    #[test]
    fn expense_args_fails_without_a_category() {
        assert_eq!(extract_expense_args("gastei 50 hoje", today()), ExtractOutcome::Failed);
    }

    #[test]
    fn income_args_extracts_amount_and_date() {
        let outcome = extract_income_args("recebi 2000 de salario", today());
        assert_eq!(outcome, ExtractOutcome::Extracted(json!({ "amount": 2000.0, "date": "2026-07-30" })));
    }

    #[test]
    fn income_args_fails_without_an_amount() {
        assert_eq!(extract_income_args("recebi meu pagamento hoje", today()), ExtractOutcome::Failed);
    }

    #[test]
    fn spending_args_defaults_to_today_with_no_category() {
        let outcome = extract_spending_args("quanto gastei", today());
        assert_eq!(outcome, ExtractOutcome::Extracted(json!({ "from": "2026-07-30", "to": "2026-07-30" })));
    }

    #[test]
    fn spending_args_extracts_category_and_period() {
        let outcome = extract_spending_args("quanto gastei em transporte esse mes", today());
        assert_eq!(
            outcome,
            ExtractOutcome::Extracted(
                json!({ "from": "2026-07-01", "to": "2026-07-30", "category": "transport" })
            )
        );
    }

    #[test]
    fn budget_status_args_extracts_category() {
        assert_eq!(
            extract_budget_status_args("estou dentro do orcamento de transporte"),
            ExtractOutcome::Extracted(json!({ "category": "transport" }))
        );
    }

    #[test]
    fn budget_status_args_fails_without_a_category() {
        assert_eq!(extract_budget_status_args("estou dentro do orcamento"), ExtractOutcome::Failed);
    }
}
