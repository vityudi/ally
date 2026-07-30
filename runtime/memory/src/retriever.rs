//! Retrieval policy for grounding a chat turn in memory.
//!
//! [`MemoryEngine`] exposes similarity search as a general-purpose API
//! (any kinds, any top_k, any threshold — see `recall_relevant`). A
//! [`Retriever`] is a fixed *policy* on top of it: which kinds to search
//! and what top_k/threshold to use for a specific purpose. The Ally SDK
//! calls one at the start of every `chat` turn instead of hardcoding that
//! policy inline, so applications can swap it (e.g. to also search
//! `Procedural` memories) without editing `ally-sdk`.

use crate::{MemoryEngine, MemoryError, MemoryKind};
use async_trait::async_trait;
use std::sync::Arc;

/// A memory judged relevant enough to ground a model's answer.
pub struct GroundedMemory {
    pub id: String,
    pub kind: MemoryKind,
    pub content: String,
    pub similarity: f32,
}

#[async_trait]
pub trait Retriever: Send + Sync {
    /// Returns the memories relevant to `query_embedding`, most similar
    /// first.
    async fn retrieve(&self, query_embedding: &[f32]) -> Result<Vec<GroundedMemory>, MemoryError>;
}

/// Default [`Retriever`]: searches `Semantic` and `Episodic` memories.
/// `Procedural` memories are excluded — they describe *how* to behave
/// (e.g. "retrieve transactions first when discussing finances"), not
/// facts to ground an answer in, so surfacing them here would confuse
/// grounding with instruction-following.
pub struct SemanticEpisodicRetriever {
    memory: Arc<MemoryEngine>,
    top_k: usize,
    min_similarity: f32,
}

impl SemanticEpisodicRetriever {
    /// `min_similarity` is the anti-hallucination guard: a weak match is
    /// worse than no match at all, since handing the model a barely
    /// related memory invites it to answer as if that memory were the
    /// fact being asked about.
    pub fn new(memory: Arc<MemoryEngine>, top_k: usize, min_similarity: f32) -> Self {
        Self { memory, top_k, min_similarity }
    }
}

#[async_trait]
impl Retriever for SemanticEpisodicRetriever {
    async fn retrieve(&self, query_embedding: &[f32]) -> Result<Vec<GroundedMemory>, MemoryError> {
        let scored = self
            .memory
            .retrieve_relevant(
                &[MemoryKind::Semantic, MemoryKind::Episodic],
                query_embedding,
                self.top_k,
                self.min_similarity,
            )
            .await?;

        Ok(scored
            .into_iter()
            .map(|(record, similarity)| GroundedMemory {
                id: record.id,
                kind: record.kind,
                content: record.content,
                similarity,
            })
            .collect())
    }
}
