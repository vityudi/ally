//! Deterministic pt-BR/en relative-date resolution ("hoje", "ontem", "esse
//! mês", "last month", "em julho", ...) into a concrete `(from, to)` date
//! range. Pure `&str -> Option<...>`: this module has no knowledge of tools
//! or arguments, so it can be tested in isolation and reused by any
//! extractor that needs a period.
//!
//! No `regex`/`unicode-normalization` dependency exists in this workspace,
//! so accents are folded with a small fixed table rather than a general
//! Unicode normalizer — the character set relevant to pt-BR date words is
//! small and known in advance.

use chrono::{Datelike, Duration, NaiveDate};

fn fold_accents(input: &str) -> String {
    input
        .chars()
        .map(|c| match c {
            'á' | 'â' | 'ã' | 'à' => 'a',
            'é' | 'ê' => 'e',
            'í' => 'i',
            'ó' | 'ô' | 'õ' => 'o',
            'ú' => 'u',
            'ç' => 'c',
            other => other,
        })
        .collect()
}

fn normalize(message: &str) -> String {
    fold_accents(&message.to_lowercase())
}

/// First day of the month containing `date`.
fn month_start(date: NaiveDate) -> NaiveDate {
    date.with_day(1).expect("day 1 is always valid")
}

/// Last day of the month containing `date`.
fn month_end(date: NaiveDate) -> NaiveDate {
    let next_month_start = if date.month() == 12 {
        NaiveDate::from_ymd_opt(date.year() + 1, 1, 1)
    } else {
        NaiveDate::from_ymd_opt(date.year(), date.month() + 1, 1)
    }
    .expect("next month's day 1 is always valid");
    next_month_start - Duration::days(1)
}

fn month_range(year: i32, month: u32) -> (NaiveDate, NaiveDate) {
    let start = NaiveDate::from_ymd_opt(year, month, 1).expect("valid year/month");
    (start, month_end(start))
}

/// Monday..=today of the week containing `today` (weeks start on Monday, the
/// ISO/pt-BR convention).
fn this_week(today: NaiveDate) -> (NaiveDate, NaiveDate) {
    let days_since_monday = today.weekday().num_days_from_monday() as i64;
    (today - Duration::days(days_since_monday), today)
}

fn last_week(today: NaiveDate) -> (NaiveDate, NaiveDate) {
    let (this_monday, _) = this_week(today);
    let last_sunday = this_monday - Duration::days(1);
    let last_monday = last_sunday - Duration::days(6);
    (last_monday, last_sunday)
}

const PT_MONTHS: [(&str, u32); 12] = [
    ("janeiro", 1),
    ("fevereiro", 2),
    ("marco", 3),
    ("abril", 4),
    ("maio", 5),
    ("junho", 6),
    ("julho", 7),
    ("agosto", 8),
    ("setembro", 9),
    ("outubro", 10),
    ("novembro", 11),
    ("dezembro", 12),
];

const EN_MONTHS: [(&str, u32); 12] = [
    ("january", 1),
    ("february", 2),
    ("march", 3),
    ("april", 4),
    ("may", 5),
    ("june", 6),
    ("july", 7),
    ("august", 8),
    ("september", 9),
    ("october", 10),
    ("november", 11),
    ("december", 12),
];

fn named_month(message: &str, today: NaiveDate) -> Option<(NaiveDate, NaiveDate)> {
    let month = PT_MONTHS
        .iter()
        .chain(EN_MONTHS.iter())
        .find(|(name, _)| message.contains(name))
        .map(|(_, month)| *month)?;

    // A bare month name ("julho") almost always means the most recent
    // occurrence of that month: this year if it hasn't passed yet this
    // year, otherwise last year (e.g. asking about "julho" in January
    // means last July, not five months from now).
    let year = if month <= today.month() { today.year() } else { today.year() - 1 };
    Some(month_range(year, month))
}

/// Resolves a pt-BR/en relative-date phrase inside `message` to a concrete
/// inclusive `(from, to)` range, checking longer/more specific phrases
/// before their shorter substrings (e.g. "semana passada" before "semana").
/// Returns `None` if no recognized date phrase is present.
pub fn resolve_period(message: &str, today: NaiveDate) -> Option<(NaiveDate, NaiveDate)> {
    let message = normalize(message);

    if message.contains("semana passada") || message.contains("last week") {
        return Some(last_week(today));
    }
    if message.contains("mes passado") || message.contains("last month") {
        let prev = if today.month() == 1 {
            NaiveDate::from_ymd_opt(today.year() - 1, 12, 1)
        } else {
            NaiveDate::from_ymd_opt(today.year(), today.month() - 1, 1)
        }
        .expect("valid previous month");
        return Some(month_range(prev.year(), prev.month()));
    }
    if message.contains("ano passado") || message.contains("last year") {
        return Some((
            NaiveDate::from_ymd_opt(today.year() - 1, 1, 1).expect("valid date"),
            NaiveDate::from_ymd_opt(today.year() - 1, 12, 31).expect("valid date"),
        ));
    }
    if message.contains("esta semana") || message.contains("essa semana") || message.contains("this week") {
        return Some(this_week(today));
    }
    if message.contains("este mes") || message.contains("esse mes") || message.contains("this month") {
        return Some((month_start(today), today));
    }
    if message.contains("ontem") || message.contains("yesterday") {
        let yesterday = today - Duration::days(1);
        return Some((yesterday, yesterday));
    }
    if message.contains("hoje") || message.contains("today") {
        return Some((today, today));
    }
    if let Some(range) = named_month(&message, today) {
        return Some(range);
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    // 2026-07-30 is a Thursday.
    fn today() -> NaiveDate {
        NaiveDate::from_ymd_opt(2026, 7, 30).unwrap()
    }

    #[test]
    fn resolves_hoje_and_today() {
        assert_eq!(resolve_period("quanto gastei hoje?", today()), Some((today(), today())));
        assert_eq!(resolve_period("how much did I spend today", today()), Some((today(), today())));
    }

    #[test]
    fn resolves_ontem_and_yesterday() {
        let y = today() - Duration::days(1);
        assert_eq!(resolve_period("gastei ontem", today()), Some((y, y)));
        assert_eq!(resolve_period("spent yesterday", today()), Some((y, y)));
    }

    #[test]
    fn resolves_this_week_monday_to_today() {
        let (from, to) = resolve_period("quanto gastei essa semana", today()).unwrap();
        assert_eq!(from, NaiveDate::from_ymd_opt(2026, 7, 27).unwrap()); // Monday
        assert_eq!(to, today());
    }

    #[test]
    fn resolves_last_week_full_range() {
        let (from, to) = resolve_period("quanto gastei semana passada", today()).unwrap();
        assert_eq!(from, NaiveDate::from_ymd_opt(2026, 7, 20).unwrap());
        assert_eq!(to, NaiveDate::from_ymd_opt(2026, 7, 26).unwrap());
    }

    #[test]
    fn resolves_this_month_start_to_today() {
        let (from, to) = resolve_period("quanto gastei esse mes", today()).unwrap();
        assert_eq!(from, NaiveDate::from_ymd_opt(2026, 7, 1).unwrap());
        assert_eq!(to, today());
    }

    #[test]
    fn resolves_last_month_full_range() {
        let (from, to) = resolve_period("quanto gastei mes passado", today()).unwrap();
        assert_eq!(from, NaiveDate::from_ymd_opt(2026, 6, 1).unwrap());
        assert_eq!(to, NaiveDate::from_ymd_opt(2026, 6, 30).unwrap());
    }

    #[test]
    fn resolves_last_month_across_year_boundary() {
        let jan = NaiveDate::from_ymd_opt(2026, 1, 15).unwrap();
        let (from, to) = resolve_period("last month", jan).unwrap();
        assert_eq!(from, NaiveDate::from_ymd_opt(2025, 12, 1).unwrap());
        assert_eq!(to, NaiveDate::from_ymd_opt(2025, 12, 31).unwrap());
    }

    #[test]
    fn resolves_last_year_full_range() {
        let (from, to) = resolve_period("ano passado", today()).unwrap();
        assert_eq!(from, NaiveDate::from_ymd_opt(2025, 1, 1).unwrap());
        assert_eq!(to, NaiveDate::from_ymd_opt(2025, 12, 31).unwrap());
    }

    #[test]
    fn resolves_named_month_this_year_when_not_yet_passed() {
        // "julho" (month 7) while today is also month 7 -> this year.
        let (from, to) = resolve_period("quanto gastei em julho", today()).unwrap();
        assert_eq!(from, NaiveDate::from_ymd_opt(2026, 7, 1).unwrap());
        assert_eq!(to, NaiveDate::from_ymd_opt(2026, 7, 31).unwrap());
    }

    #[test]
    fn resolves_named_month_previous_year_when_already_passed() {
        // "dezembro" (month 12) is after today's month (7) -> last year's December.
        let (from, to) = resolve_period("quanto gastei em dezembro", today()).unwrap();
        assert_eq!(from, NaiveDate::from_ymd_opt(2025, 12, 1).unwrap());
        assert_eq!(to, NaiveDate::from_ymd_opt(2025, 12, 31).unwrap());
    }

    #[test]
    fn accent_folding_matches_accented_and_unaccented_forms() {
        assert!(resolve_period("quanto gastei em julho", today()).is_some());
        assert!(resolve_period("qual foi meu gasto no m\u{ea}s passado", today()).is_some());
    }

    #[test]
    fn returns_none_when_no_recognized_phrase() {
        assert_eq!(resolve_period("quanto gastei em transporte", today()), None);
    }

    #[test]
    fn semana_passada_takes_priority_over_semana() {
        // Regression: must not match "esta semana"/"this week" logic for
        // "semana passada" due to substring overlap in phrase ordering.
        let (from, _) = resolve_period("gasto da semana passada", today()).unwrap();
        assert_eq!(from, NaiveDate::from_ymd_opt(2026, 7, 20).unwrap());
    }
}
