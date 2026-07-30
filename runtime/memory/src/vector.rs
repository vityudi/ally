//! Similarity search, decoupled from record persistence.
//!
//! [`MemoryEngine`](crate::MemoryEngine) owns *what* a memory is (id, kind,
//! content); [`VectorIndex`] owns *finding* memories by embedding
//! similarity. Splitting them means swapping in a real vector database
//! (sqlite-vec, Qdrant, pgvector...) later is a new `VectorIndex` impl, not
//! a rewrite of `MemoryEngine`'s storage/indexing logic.

use crate::{MemoryError, MemoryKind};
use ally_storage::Storage;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Pluggable similarity search over memory embeddings.
#[async_trait]
pub trait VectorIndex: Send + Sync {
    /// Registers or updates the embedding for `id`, so it becomes a
    /// candidate for future [`search`](VectorIndex::search) calls scoped to
    /// `kind`.
    async fn upsert(&self, id: &str, kind: MemoryKind, embedding: Vec<f32>) -> Result<(), MemoryError>;

    /// Returns up to `top_k` ids among `kinds` whose embedding clears
    /// `min_similarity` against `query_embedding`, highest similarity
    /// first.
    async fn search(
        &self,
        kinds: &[MemoryKind],
        query_embedding: &[f32],
        top_k: usize,
        min_similarity: f32,
    ) -> Result<Vec<(String, f32)>, MemoryError>;
}

#[derive(Serialize, Deserialize)]
struct Entry {
    id: String,
    embedding: Vec<f32>,
}

/// Default [`VectorIndex`]: an in-storage per-kind list of `(id, embedding)`
/// pairs, scored by brute-force cosine similarity. O(n) in the number of
/// embedded memories per kind — fine for a personal assistant's memory
/// store, not meant to scale to a shared corpus.
pub struct BruteForceVectorIndex {
    storage: Arc<dyn Storage>,
}

impl BruteForceVectorIndex {
    pub fn new(storage: Arc<dyn Storage>) -> Self {
        Self { storage }
    }

    async fn load(&self, kind: MemoryKind) -> Result<Vec<Entry>, MemoryError> {
        match self.storage.get(&vector_index_key(kind)).await? {
            Some(bytes) => Ok(serde_json::from_slice(&bytes)?),
            None => Ok(Vec::new()),
        }
    }
}

#[async_trait]
impl VectorIndex for BruteForceVectorIndex {
    async fn upsert(&self, id: &str, kind: MemoryKind, embedding: Vec<f32>) -> Result<(), MemoryError> {
        let mut entries = self.load(kind).await?;
        match entries.iter_mut().find(|e| e.id == id) {
            Some(entry) => entry.embedding = embedding,
            None => entries.push(Entry { id: id.to_string(), embedding }),
        }
        self.storage
            .set(&vector_index_key(kind), serde_json::to_vec(&entries)?)
            .await?;
        Ok(())
    }

    async fn search(
        &self,
        kinds: &[MemoryKind],
        query_embedding: &[f32],
        top_k: usize,
        min_similarity: f32,
    ) -> Result<Vec<(String, f32)>, MemoryError> {
        let mut scored = Vec::new();
        for &kind in kinds {
            for entry in self.load(kind).await? {
                let similarity = cosine_similarity(query_embedding, &entry.embedding);
                if similarity >= min_similarity {
                    scored.push((entry.id, similarity));
                }
            }
        }

        scored.sort_by(|(_, a), (_, b)| b.total_cmp(a));
        scored.truncate(top_k);
        Ok(scored)
    }
}

/// Cosine similarity between two vectors. Returns `0.0` for mismatched or
/// zero-length inputs rather than panicking — a malformed/empty embedding
/// should just never match, not crash retrieval.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

fn vector_index_key(kind: MemoryKind) -> String {
    format!("memory:vector_index:{}", kind.as_str())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ally_storage::SqliteStorage;

    #[tokio::test]
    async fn search_ranks_by_similarity_and_applies_threshold() {
        let index = BruteForceVectorIndex::new(Arc::new(SqliteStorage::open_in_memory().unwrap()));

        index
            .upsert("close", MemoryKind::Semantic, vec![0.9, 0.1, 0.0])
            .await
            .unwrap();
        index
            .upsert("unrelated", MemoryKind::Semantic, vec![0.0, 1.0, 0.0])
            .await
            .unwrap();

        let results = index
            .search(&[MemoryKind::Semantic], &[1.0, 0.0, 0.0], 5, 0.5)
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "close");
        assert!(results[0].1 > 0.5);
    }

    #[tokio::test]
    async fn upsert_overwrites_existing_embedding_for_same_id() {
        let index = BruteForceVectorIndex::new(Arc::new(SqliteStorage::open_in_memory().unwrap()));

        index
            .upsert("a", MemoryKind::Semantic, vec![0.0, 1.0, 0.0])
            .await
            .unwrap();
        index
            .upsert("a", MemoryKind::Semantic, vec![1.0, 0.0, 0.0])
            .await
            .unwrap();

        let results = index
            .search(&[MemoryKind::Semantic], &[1.0, 0.0, 0.0], 5, 0.9)
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "a");
    }
}
