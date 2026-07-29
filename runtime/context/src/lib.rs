//! Context Engine: decides what information actually reaches the Language
//! Model, instead of forwarding the entire conversation history.

use ally_memory::MemoryRecord;

#[derive(Debug, Default, Clone)]
pub struct ContextPackage {
    pub conversation_summary: String,
    pub recent_messages: Vec<String>,
    pub relevant_memories: Vec<String>,
    pub tool_results: Vec<String>,
}

#[derive(Default)]
pub struct ContextEngine;

impl ContextEngine {
    pub fn new() -> Self {
        Self
    }

    pub fn assemble(
        &self,
        conversation_summary: String,
        recent_messages: Vec<String>,
        memories: &[MemoryRecord],
    ) -> ContextPackage {
        ContextPackage {
            conversation_summary,
            recent_messages,
            relevant_memories: memories.iter().map(|m| m.content.clone()).collect(),
            tool_results: Vec::new(),
        }
    }
}
