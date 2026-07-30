//! Memory Engine: episodic, semantic and procedural memory as first-class
//! Runtime components, not prompt text.
//!
//! Records are persisted through the abstract [`Storage`] trait (SQLite by
//! default, see `runtime/storage`). Each [`MemoryKind`] keeps its own id
//! index in storage, so `retrieve` only loads the records that actually
//! match instead of scanning every stored memory. Similarity search itself
//! is delegated to a [`VectorIndex`] (see `vector` module) rather than done
//! inline, so the search backend can be swapped independently of record
//! storage.

mod retriever;
mod vector;

pub use retriever::{GroundedMemory, Retriever, SemanticEpisodicRetriever};
pub use vector::{BruteForceVectorIndex, VectorIndex};

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
    /// Vector embedding of `content`, used by `retrieve_relevant` for
    /// similarity search. `None` for records written before embeddings
    /// were wired up, or by callers that don't have a Model Runtime handy
    /// (e.g. tests) — such records are simply never matched by relevance
    /// search, only by `retrieve(kind)`.
    #[serde(default)]
    pub embedding: Option<Vec<f32>>,
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
    vector_index: Arc<dyn VectorIndex>,
}

impl MemoryEngine {
    /// Uses [`BruteForceVectorIndex`] (over the same `storage`) for
    /// similarity search. Use [`MemoryEngine::with_vector_index`] to plug
    /// in a different backend (e.g. a real vector database).
    pub fn new(storage: Arc<dyn Storage>) -> Self {
        let vector_index = Arc::new(BruteForceVectorIndex::new(storage.clone()));
        Self::with_vector_index(storage, vector_index)
    }

    pub fn with_vector_index(storage: Arc<dyn Storage>, vector_index: Arc<dyn VectorIndex>) -> Self {
        Self { storage, vector_index }
    }

    /// Persists a record and adds it to its kind's index. If the record
    /// carries an embedding, also upserts it into the vector index so
    /// `retrieve_relevant` can find it. Emits `MemoryCreated` once the
    /// write succeeds.
    pub async fn store(&self, record: MemoryRecord, events: &EventBus) -> Result<(), MemoryError> {
        let bytes = serde_json::to_vec(&record)?;
        self.storage.set(&record_key(&record.id), bytes).await?;

        let index_key = index_key(record.kind);
        let mut ids = self.load_index(&index_key).await?;
        if !ids.contains(&record.id) {
            ids.push(record.id.clone());
            self.storage.set(&index_key, serde_json::to_vec(&ids)?).await?;
        }

        if let Some(embedding) = record.embedding.clone() {
            self.vector_index.upsert(&record.id, record.kind, embedding).await?;
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

    async fn load_record(&self, id: &str) -> Result<Option<MemoryRecord>, MemoryError> {
        match self.storage.get(&record_key(id)).await? {
            Some(bytes) => Ok(Some(serde_json::from_slice(&bytes)?)),
            None => Ok(None),
        }
    }

    /// Ranks records of `kinds` by similarity to `query_embedding` via the
    /// configured [`VectorIndex`], dropping anything below `min_similarity`
    /// and returning at most `top_k`, highest similarity first.
    ///
    /// `min_similarity` is the anti-hallucination guard: a weak match is
    /// worse than no match at all, since handing the model a barely
    /// related memory invites it to answer as if that memory were the
    /// fact being asked about. Records without an embedding are never
    /// matched here — only by `retrieve(kind)`.
    pub async fn retrieve_relevant(
        &self,
        kinds: &[MemoryKind],
        query_embedding: &[f32],
        top_k: usize,
        min_similarity: f32,
    ) -> Result<Vec<(MemoryRecord, f32)>, MemoryError> {
        let hits = self
            .vector_index
            .search(kinds, query_embedding, top_k, min_similarity)
            .await?;

        let mut scored = Vec::with_capacity(hits.len());
        for (id, similarity) in hits {
            if let Some(record) = self.load_record(&id).await? {
                scored.push((record, similarity));
            }
        }
        Ok(scored)
    }
}

fn record_key(id: &str) -> String {
    format!("memory:record:{id}")
}

fn index_key(kind: MemoryKind) -> String {
    format!("memory:index:{}", kind.as_str())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ally_storage::SqliteStorage;

    fn record(id: &str, embedding: Vec<f32>) -> MemoryRecord {
        MemoryRecord {
            id: id.to_string(),
            kind: MemoryKind::Semantic,
            content: format!("content for {id}"),
            embedding: Some(embedding),
        }
    }

    #[tokio::test]
    async fn retrieve_relevant_ranks_by_similarity_and_applies_threshold() {
        let storage = Arc::new(SqliteStorage::open_in_memory().unwrap());
        let engine = MemoryEngine::new(storage);
        let events = EventBus::new();

        // Close to the query vector [1.0, 0.0, 0.0].
        engine
            .store(record("close", vec![0.9, 0.1, 0.0]), &events)
            .await
            .unwrap();
        // Orthogonal to the query vector — should be filtered out by the threshold.
        engine
            .store(record("unrelated", vec![0.0, 1.0, 0.0]), &events)
            .await
            .unwrap();

        let results = engine
            .retrieve_relevant(&[MemoryKind::Semantic], &[1.0, 0.0, 0.0], 5, 0.5)
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0.id, "close");
        assert!(results[0].1 > 0.5);
    }
}
