//! Memory Engine: episodic, semantic and procedural memory as first-class
//! Runtime components, not prompt text.
//!
//! Records are persisted through the abstract [`Storage`] trait (SQLite by
//! default, see `runtime/storage`). Each [`MemoryKind`] keeps its own id
//! index in storage, so `retrieve` only loads the records that actually
//! match instead of scanning every stored memory.

use ally_events::{Event, EventBus};
use ally_storage::Storage;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryKind {
    /// Past events: meetings, conversations, purchases.
    Episodic,
    /// Persistent facts: preferences, favorite bank, dietary restrictions.
    Semantic,
    /// Learned behaviors: "when discussing finances, retrieve transactions first".
    Procedural,
}

impl MemoryKind {
    fn as_str(&self) -> &'static str {
        match self {
            MemoryKind::Episodic => "episodic",
            MemoryKind::Semantic => "semantic",
            MemoryKind::Procedural => "procedural",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRecord {
    pub id: String,
    pub kind: MemoryKind,
    pub content: String,
}

#[derive(Debug, Error)]
pub enum MemoryError {
    #[error(transparent)]
    Storage(#[from] ally_storage::StorageError),
    #[error("failed to (de)serialize memory record: {0}")]
    Serde(#[from] serde_json::Error),
}

pub struct MemoryEngine {
    storage: Arc<dyn Storage>,
}

impl MemoryEngine {
    pub fn new(storage: Arc<dyn Storage>) -> Self {
        Self { storage }
    }

    /// Persists a record and adds it to its kind's index. Emits
    /// `MemoryCreated` once the write succeeds.
    pub async fn store(&self, record: MemoryRecord, events: &EventBus) -> Result<(), MemoryError> {
        let bytes = serde_json::to_vec(&record)?;
        self.storage.set(&record_key(&record.id), bytes).await?;

        let index_key = index_key(record.kind);
        let mut ids = self.load_index(&index_key).await?;
        if !ids.contains(&record.id) {
            ids.push(record.id.clone());
            self.storage.set(&index_key, serde_json::to_vec(&ids)?).await?;
        }

        events.publish(Event::MemoryCreated {
            memory_id: record.id.clone(),
        });
        Ok(())
    }

    /// Loads every record of `kind` from storage via its index — never a
    /// scan over unrelated kinds.
    pub async fn retrieve(&self, kind: MemoryKind) -> Result<Vec<MemoryRecord>, MemoryError> {
        let ids = self.load_index(&index_key(kind)).await?;
        let mut records = Vec::with_capacity(ids.len());
        for id in ids {
            if let Some(bytes) = self.storage.get(&record_key(&id)).await? {
                records.push(serde_json::from_slice(&bytes)?);
            }
        }
        Ok(records)
    }

    async fn load_index(&self, key: &str) -> Result<Vec<String>, MemoryError> {
        match self.storage.get(key).await? {
            Some(bytes) => Ok(serde_json::from_slice(&bytes)?),
            None => Ok(Vec::new()),
        }
    }
}

fn record_key(id: &str) -> String {
    format!("memory:record:{id}")
}

fn index_key(kind: MemoryKind) -> String {
    format!("memory:index:{}", kind.as_str())
}
