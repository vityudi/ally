//! Context Engine: decides what information actually reaches the Language
//! Model, instead of forwarding the entire conversation history.

use ally_memory::MemoryRecord;

#[derive(Debug, Default, Clone)]
pub struct ContextPackage {
    pub conversation_summary: String,
    pub recent_messages: Vec<String>,
    pub relevant_memories: Vec<MemoryRecord>,
    pub tool_results: Vec<String>,
}

/// Default token budget for an assembled [`ContextPackage`]. Deliberately
/// conservative: this is a compact package handed to a small local model,
/// not a full conversation transcript.
const DEFAULT_MAX_TOKENS: usize = 2048;

pub struct ContextEngine {
    max_tokens: usize,
}

impl Default for ContextEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl ContextEngine {
    pub fn new() -> Self {
        Self::with_budget(DEFAULT_MAX_TOKENS)
    }

    pub fn with_budget(max_tokens: usize) -> Self {
        Self { max_tokens }
    }

    /// Assembles a context package, keeping the conversation summary in
    /// full and filling the remaining budget with recent messages first,
    /// then relevant memories — dropping whatever no longer fits.
    ///
    /// Callers should already order `recent_messages` and `memories` by
    /// priority (most important first): this only enforces the budget, it
    /// does not re-rank.
    pub fn assemble(
        &self,
        conversation_summary: String,
        recent_messages: Vec<String>,
        memories: &[MemoryRecord],
    ) -> ContextPackage {
        let mut budget = self
            .max_tokens
            .saturating_sub(approx_tokens(&conversation_summary));

        let recent_messages = take_within_budget(recent_messages, &mut budget, |m| approx_tokens(m));
        let relevant_memories =
            take_within_budget(memories.to_vec(), &mut budget, |m| approx_tokens(&m.content));

        ContextPackage {
            conversation_summary,
            recent_messages,
            relevant_memories,
            tool_results: Vec::new(),
        }
    }
}

/// Rough token estimate (~4 characters per token, a common approximation
/// for English text). Replace with a real tokenizer once a Model Runtime
/// backend exists (Phase 3) and can report exact counts per model.
fn approx_tokens(text: &str) -> usize {
    (text.len() / 4).max(1)
}

fn take_within_budget<T>(items: Vec<T>, budget: &mut usize, cost_of: impl Fn(&T) -> usize) -> Vec<T> {
    let mut kept = Vec::new();
    for item in items {
        let cost = cost_of(&item);
        if cost > *budget {
            break;
        }
        *budget -= cost;
        kept.push(item);
    }
    kept
}
