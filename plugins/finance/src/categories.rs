//! Canonical spending categories with pt-BR/en synonym resolution. Categories
//! are otherwise free text (see `Transaction::category` in `lib.rs`), which
//! means "transporte" and "transport" are unrelated strings as far as budget
//! matching or spending-by-category filtering is concerned. Resolving known
//! synonyms to one of `CATEGORIES` before storing/comparing collapses that
//! without rejecting a genuinely novel category the user names.

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

fn normalize(text: &str) -> String {
    fold_accents(&text.to_lowercase())
}

/// Canonical category keys. Stored on `Transaction`/`Budget` regardless of
/// which language the user spoke in — the model phrases the final answer
/// back in the user's language, so an English internal key is fine.
// Only exercised by tests today (a cross-check that every synonym maps to a
// declared category) — kept `pub` for tools that want to validate/enumerate
// categories, e.g. a future `finance.list_categories`.
#[allow(dead_code)]
pub const CATEGORIES: &[&str] =
    &["food", "transport", "housing", "health", "leisure", "education", "shopping", "bills", "salary", "other"];

const SYNONYMS: &[(&str, &[&str])] = &[
    ("food", &["mercado", "supermercado", "comida", "alimentacao", "restaurante", "food", "groceries"]),
    ("transport", &["transporte", "uber", "gasolina", "combustivel", "transport", "gas"]),
    ("housing", &["aluguel", "casa", "moradia", "rent", "housing"]),
    ("health", &["saude", "remedio", "farmacia", "health", "medicine"]),
    ("leisure", &["lazer", "entretenimento", "cinema", "leisure", "entertainment"]),
    ("education", &["educacao", "curso", "escola", "education", "course"]),
    ("shopping", &["roupa", "compras", "shopping", "clothes"]),
    ("bills", &["conta", "contas", "luz", "agua", "internet", "bills", "utilities"]),
    ("salary", &["salario", "pagamento", "salary"]),
];

/// Resolves a single category value (e.g. an explicit `category` argument)
/// to its canonical key via the synonym table. Returns `None` — never a
/// guessed default like "other" — when the text doesn't match any known
/// synonym, so callers can fall back to the original text unchanged.
pub fn resolve_category(text: &str) -> Option<&'static str> {
    let normalized = normalize(text.trim());
    SYNONYMS
        .iter()
        .find(|(_, synonyms)| synonyms.iter().any(|synonym| normalized == *synonym))
        .map(|(canonical, _)| *canonical)
}

/// Scans a full free-text message for the first known category synonym
/// appearing anywhere in it (word-boundary aware via whitespace/punctuation
/// splitting), used to find an optional category filter inside a longer
/// sentence like "quanto gastei em transporte esse mes".
pub fn category_matches_in(message: &str) -> Option<&'static str> {
    let normalized = normalize(message);
    let words: Vec<&str> = normalized.split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()).collect();

    SYNONYMS
        .iter()
        .find(|(_, synonyms)| synonyms.iter().any(|synonym| words.contains(synonym)))
        .map(|(canonical, _)| *canonical)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_synonym_canonical_key_is_a_declared_category() {
        for (canonical, _) in SYNONYMS {
            assert!(CATEGORIES.contains(canonical), "undeclared category: {canonical}");
        }
    }

    #[test]
    fn resolves_every_synonym_to_its_canonical_category() {
        for (canonical, synonyms) in SYNONYMS {
            for synonym in *synonyms {
                assert_eq!(resolve_category(synonym), Some(*canonical), "synonym: {synonym}");
            }
        }
    }

    #[test]
    fn resolve_category_is_case_and_accent_insensitive() {
        assert_eq!(resolve_category("TRANSPORTE"), Some("transport"));
        assert_eq!(resolve_category("Alimenta\u{e7}\u{e3}o"), Some("food"));
    }

    #[test]
    fn resolve_category_returns_none_for_unknown_text() {
        assert_eq!(resolve_category("viagem para marte"), None);
    }

    #[test]
    fn category_matches_in_finds_synonym_inside_a_sentence() {
        assert_eq!(category_matches_in("quanto gastei em transporte esse mes"), Some("transport"));
        assert_eq!(category_matches_in("gastei 50 reais no mercado hoje"), Some("food"));
        assert_eq!(category_matches_in("paguei o aluguel ontem"), Some("housing"));
    }

    #[test]
    fn category_matches_in_returns_none_when_nothing_matches() {
        assert_eq!(category_matches_in("quanto gastei hoje"), None);
    }
}
