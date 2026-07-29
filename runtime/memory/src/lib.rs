//! Memory Engine: episodic, semantic and procedural memory as first-class
//! Runtime components, not prompt text.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryKind {
    /// Past events: meetings, conversations, purchases.
    Episodic,
    /// Persistent facts: preferences, favorite bank, dietary restrictions.
    Semantic,
    /// Learned behaviors: "when discussing finances, retrieve transactions first".
    Procedural,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRecord {
    pub id: String,
    pub kind: MemoryKind,
    pub content: String,
}

#[derive(Default)]
pub struct MemoryEngine {
    records: Vec<MemoryRecord>,
}

impl MemoryEngine {
    pub fn new() -> Self {
        Self { records: Vec::new() }
    }

    pub fn store(&mut self, record: MemoryRecord) {
        self.records.push(record);
    }

    pub fn retrieve(&self, kind: MemoryKind) -> Vec<&MemoryRecord> {
        self.records
            .iter()
            .filter(|r| matches_kind(&r.kind, &kind))
            .collect()
    }
}

fn matches_kind(a: &MemoryKind, b: &MemoryKind) -> bool {
    matches!(
        (a, b),
        (MemoryKind::Episodic, MemoryKind::Episodic)
            | (MemoryKind::Semantic, MemoryKind::Semantic)
            | (MemoryKind::Procedural, MemoryKind::Procedural)
    )
}
