//! Ally SDK: the only interface applications are meant to depend on.
//! Applications never talk to the model, memory or tools directly —
//! they talk to `Ally`, and the Runtime does the rest.
#![deny(missing_docs)]

use ally_context::ContextEngine;
use ally_events::EventBus;
use ally_memory::{MemoryEngine, MemoryError, SemanticEpisodicRetriever};
use ally_models::{LlamaCppBackend, ModelBackend, ModelError};
use ally_planner::Planner;
use ally_plugins::PluginManager;
use ally_security::PermissionSet;
use ally_storage::{SqliteStorage, Storage, StorageError};
use ally_tools::{ToolError, ToolOrchestrator};
use serde_json::Value;
use std::sync::Arc;
use thiserror::Error;

// Re-exported so applications only ever need `use ally_sdk::{...}` and
// never depend on a `runtime/*` crate directly — see the module doc above.
pub use ally_context::ContextPackage;
pub use ally_events::{Event, EventHandler, LoggingHandler};
pub use ally_memory::{MemoryKind, MemoryRecord, Retriever};
pub use ally_models::{ChatMessage, ChatRequest, ChatResponse, ToolSpec};
pub use ally_planner::{Intent, Plan};
pub use ally_plugins::{Plugin, PluginError};
pub use ally_security::Permission;

/// Where `Ally::with_storage` points `LlamaCppBackend::lazy_default` to
/// look for (and download into, on first use) the pinned default GGUF
/// weights — see `ally_models::llama_cpp`'s module doc. Relative to the
/// process's current working directory, matching the existing `/models`
/// convention (`models/README.md`) and how `Ally::open`'s SQLite path is
/// already resolved the same way. Not an OS-appropriate user data
/// directory yet — a reasonable follow-up, not required for dropping the
/// external Ollama dependency.
const DEFAULT_MODELS_DIR: &str = "models";

/// Caps how many tool-call round trips `Ally::chat` will make in a single
/// turn, so a model that keeps requesting tools can't loop forever.
const MAX_TOOL_ROUNDS: usize = 4;

/// How many memories `Ally::chat` pulls into the grounding message per turn.
const MEMORIES_PER_TURN: usize = 5;

/// Minimum cosine similarity a memory must clear to be handed to the model
/// as grounding for an answer. A weak match is worse than no match: it
/// invites the model to answer as if a barely-related memory were the
/// fact being asked about, instead of admitting it doesn't know.
const MIN_MEMORY_SIMILARITY: f32 = 0.5;

/// Merged into the leading system message on every `chat` turn, whether or
/// not any memory was retrieved. This is a Runtime-level guarantee (see
/// the grounding principle: answer from retrieved data, never guess), not
/// an application prompt choice — an app's own system prompt (e.g.
/// "You are Ally, a personal assistant") never mentions that the person
/// talking to it is a separate individual it starts out knowing nothing
/// about, and a small model left to fill that gap on its own has been
/// observed conflating "you" (the user) with itself. Verified live against
/// qwen2.5:1.5b that adding this one extra sentence does not break
/// trigger-matched tool routing (which returns before this runs) or
/// model-decided tool calls (which still fire normally with it present).
const IDENTITY_GUARDRAIL: &str = "You are the assistant; the person you're \
    talking to is a separate individual, not you. Never assume their name, \
    identity or other facts about them unless stated as a fact below or \
    earlier in this conversation — if you don't know, say so instead of \
    guessing.";

/// Failure from [`Ally::chat`]: either the Model Runtime backend failed,
/// or a tool it requested failed (including a denied permission).
#[derive(Debug, Error)]
pub enum ChatError {
    /// The Model Runtime backend returned an error.
    #[error(transparent)]
    Model(#[from] ModelError),
    /// A tool call the model requested failed to execute.
    #[error(transparent)]
    Tool(#[from] ToolError),
    /// Memory retrieval failed while assembling grounding for this turn.
    #[error(transparent)]
    Memory(#[from] MemoryError),
}

/// Failure from [`Ally::remember`] or [`Ally::recall_relevant`]: embedding
/// text requires the Model Runtime backend, so these can now fail for the
/// same reasons a chat call can, in addition to storage failures.
#[derive(Debug, Error)]
pub enum RememberError {
    /// The Model Runtime backend failed to produce an embedding.
    #[error(transparent)]
    Model(#[from] ModelError),
    /// The Memory Engine failed to persist or read back a record.
    #[error(transparent)]
    Memory(#[from] MemoryError),
}

/// The single entry point applications use to talk to the Ally Runtime.
/// Every Runtime module (Planner, Memory, Context, Tools, Plugins,
/// Events, Permissions, Model) is reachable only through `Ally`'s methods
/// — its fields are private so the Runtime's internals can change shape
/// without breaking application code built against this crate.
pub struct Ally {
    planner: Planner,
    memory: Arc<MemoryEngine>,
    retriever: Arc<dyn Retriever>,
    context: ContextEngine,
    tools: ToolOrchestrator,
    plugins: PluginManager,
    events: EventBus,
    permissions: PermissionSet,
    model: Arc<dyn ModelBackend>,
}

impl Default for Ally {
    fn default() -> Self {
        Self::new()
    }
}

impl Ally {
    /// Creates an `Ally` backed by a throwaway in-memory SQLite database.
    /// Handy for tests and demos; data does not survive past the process.
    /// Use [`Ally::open`] to persist memory across runs.
    pub fn new() -> Self {
        let storage = Arc::new(
            SqliteStorage::open_in_memory().expect("failed to open in-memory storage"),
        );
        Self::with_storage(storage)
    }

    /// Creates an `Ally` backed by a SQLite database file on disk.
    pub fn open(path: &str) -> Result<Self, StorageError> {
        let storage = Arc::new(SqliteStorage::open(path)?);
        Ok(Self::with_storage(storage))
    }

    /// Creates an `Ally` backed by any [`Storage`] implementation, using
    /// the in-process `LlamaCppBackend` as its Model Runtime — no external
    /// daemon required. Weights are downloaded into [`DEFAULT_MODELS_DIR`]
    /// lazily, on the first call that actually needs the model (`chat`,
    /// `chat_stream`, `embed`, and therefore `remember`/`recall_relevant`
    /// too), not here — see `ally_models::LlamaCppBackend::lazy_default`.
    pub fn with_storage(storage: Arc<dyn Storage>) -> Self {
        let memory = Arc::new(MemoryEngine::new(storage));
        let retriever = Arc::new(SemanticEpisodicRetriever::new(
            memory.clone(),
            MEMORIES_PER_TURN,
            MIN_MEMORY_SIMILARITY,
        ));
        Self {
            planner: Planner::new(),
            memory,
            retriever,
            context: ContextEngine::new(),
            tools: ToolOrchestrator::new(),
            plugins: PluginManager::new(),
            events: EventBus::new(),
            permissions: PermissionSet::default(),
            model: Arc::new(LlamaCppBackend::lazy_default(DEFAULT_MODELS_DIR)),
        }
    }

    /// Swaps the Model Runtime backend — e.g. `ally_models::OllamaBackend`
    /// to talk to an existing `ollama serve` instead, or any other
    /// `ModelBackend` implementation.
    pub fn with_model(&mut self, model: Arc<dyn ModelBackend>) {
        self.model = model;
    }

    /// Swaps the grounding retrieval policy used by [`Ally::chat`] — e.g.
    /// to also search `Procedural` memories, or to change how many
    /// memories are pulled in per turn. Defaults to
    /// [`SemanticEpisodicRetriever`].
    pub fn with_retriever(&mut self, retriever: Arc<dyn Retriever>) {
        self.retriever = retriever;
    }

    /// Which specific model the current backend talks to (e.g. a GGUF
    /// filename or an Ollama model tag) — handy for a startup banner so
    /// the user knows what they're chatting with before the first message
    /// triggers the (possibly slow) load/download.
    pub fn model_id(&self) -> String {
        self.model.model_id()
    }

    /// Subscribes an additional listener to every Runtime event
    /// (`ConversationStarted`, `MemoryCreated`, `ToolExecuted`, ...).
    pub fn on_event(&mut self, handler: Box<dyn EventHandler>) {
        self.events.subscribe(handler);
    }

    /// Grants permissions to every subsequent `execute_tool` /
    /// `chat` call, replacing whatever was previously granted.
    pub fn grant_permissions(&mut self, permissions: Vec<Permission>) {
        self.permissions = PermissionSet::new(permissions);
    }

    /// Entry point for every user interaction: intent -> plan -> (future:
    /// memory retrieval, tool execution, context assembly, model call).
    pub fn handle_intent(&self, intent: Intent) -> Plan {
        self.planner.plan(&intent, &self.events)
    }

    /// Closes out a conversation started by `handle_intent`.
    pub fn end_conversation(&self, conversation_id: String) {
        self.events.publish(Event::ConversationEnded { conversation_id });
    }

    /// Persists a memory record and emits `MemoryCreated`. If `record`
    /// doesn't already carry an embedding, one is computed from its
    /// `content` via the configured Model Runtime backend first, so later
    /// `recall_relevant` calls can find it.
    pub async fn remember(&self, mut record: MemoryRecord) -> Result<(), RememberError> {
        if record.embedding.is_none() {
            record.embedding = Some(self.model.embed(&record.content).await?);
        }
        self.memory.store(record, &self.events).await?;
        Ok(())
    }

    /// Reads back memories of a given kind.
    pub async fn recall(&self, kind: MemoryKind) -> Result<Vec<MemoryRecord>, MemoryError> {
        self.memory.retrieve(kind).await
    }

    /// Embeds `query` and returns the memories of `kinds` most relevant to
    /// it, most similar first, each paired with its similarity score.
    /// Memories below [`MIN_MEMORY_SIMILARITY`] are never returned.
    pub async fn recall_relevant(
        &self,
        kinds: &[MemoryKind],
        query: &str,
        top_k: usize,
    ) -> Result<Vec<(MemoryRecord, f32)>, RememberError> {
        let query_embedding = self.model.embed(query).await?;
        Ok(self
            .memory
            .retrieve_relevant(kinds, &query_embedding, top_k, MIN_MEMORY_SIMILARITY)
            .await?)
    }

    /// Installs a plugin: registers every tool it exposes with the Tool
    /// Orchestrator (so they become callable and visible to the model via
    /// `tool_specs`), then emits `PluginInstalled`. Fails if any tool the
    /// plugin exposes requires a permission the plugin itself didn't
    /// declare via `Plugin::permissions()`.
    pub fn install_plugin(&mut self, plugin: Box<dyn Plugin>) -> Result<(), PluginError> {
        for tool in plugin.tools() {
            self.tools.register(tool);
        }
        self.plugins.install(plugin, &self.events)
    }

    /// Names of every plugin installed so far, in installation order.
    pub fn installed_plugins(&self) -> impl Iterator<Item = &str> {
        self.plugins.installed()
    }

    /// Assembles a token-budgeted [`ContextPackage`] from a summary, the
    /// recent conversation messages and previously recalled memories.
    pub fn assemble_context(
        &self,
        summary: String,
        recent_messages: Vec<String>,
        memories: &[MemoryRecord],
    ) -> ContextPackage {
        self.context.assemble(summary, recent_messages, memories)
    }

    /// Executes a registered tool under the currently granted permissions,
    /// emitting `ToolExecuted` on success.
    pub async fn execute_tool(&self, tool_name: &str, input: Value) -> Result<Value, ToolError> {
        self.tools
            .execute(tool_name, input, &self.permissions, &self.events)
            .await
    }

    /// `ToolSpec`s for every registered tool, ready to hand to a
    /// `ChatRequest` so the model knows what it can request.
    pub fn tool_specs(&self) -> Vec<ToolSpec> {
        self.tools
            .list()
            .map(|t| ToolSpec {
                name: t.name().to_string(),
                description: t.description().to_string(),
                parameters: t.parameters_schema(),
            })
            .collect()
    }

    /// Sends a chat turn to the configured Model Runtime.
    ///
    /// First, the latest user message is checked against every registered
    /// tool's `trigger_phrases`, with arguments deterministically resolved
    /// via `ToolOrchestrator::match_with_args`. A small local model's
    /// decision of *whether* to call a tool gets markedly less reliable as
    /// a conversation grows multi-turn — even though the tool itself works
    /// fine once actually called — so a tool matched this way (e.g. "how
    /// much did I spend on transport this month") is routed to directly
    /// instead of leaving that decision to the model every time. The model
    /// is still used, but only for one thing: phrasing the tool's result
    /// naturally, never for deciding whether/which tool to call or with
    /// what arguments. If no trigger phrase matches — or one matches but
    /// required arguments couldn't be extracted — this falls through to
    /// the regular model-driven flow below.
    ///
    /// Otherwise, this embeds the latest user message and asks the
    /// configured [`Retriever`] for relevant memories, then merges
    /// [`IDENTITY_GUARDRAIL`] — plus, if any memories matched, a grounding
    /// block listing them — into the leading system message. The guardrail
    /// is merged on every turn, not just when memories are found: without
    /// it, a small model reminded only that "you are Ally" has been
    /// observed conflating the user asking a question with Ally itself
    /// instead of admitting it doesn't know them yet. A `MemoryRetrieved`
    /// event is published with the memory ids actually used, so a wrong
    /// answer can be traced back to what was (or wasn't) retrieved for it.
    ///
    /// The model never executes a tool itself: when it requests one
    /// (`ChatResponse::tool_calls`), this dispatches the call through the
    /// permission-checked Tool Orchestrator, feeds the result back as a
    /// `tool` message, and asks the model again — up to `MAX_TOOL_ROUNDS`
    /// times — before returning the final response.
    pub async fn chat(&self, mut request: ChatRequest) -> Result<ChatResponse, ChatError> {
        if let Some(query) = request.messages.iter().rev().find(|m| m.role == "user") {
            if let Some((tool, args)) = self.tools.match_with_args(&query.content) {
                let tool_name = tool.name().to_string();
                let result = self.execute_tool(&tool_name, args).await?;

                if let Some(message) = tool.describe_result(&result) {
                    return Ok(ChatResponse {
                        message: ChatMessage::assistant(message),
                        tool_calls: Vec::new(),
                    });
                }

                let phrasing_request = ChatRequest {
                    messages: vec![
                        ChatMessage::system(
                            "Reply naturally, in the same language as the user's message, \
                             using ONLY the data in the tool result below. Do not invent or \
                             add any other numbers.",
                        ),
                        query.clone(),
                        ChatMessage::tool(result.to_string()),
                    ],
                    tools: Vec::new(),
                };
                let response = self.model.chat(phrasing_request).await?;
                return Ok(ChatResponse { message: response.message, tool_calls: Vec::new() });
            }
        }

        if let Some(query) = request.messages.iter().rev().find(|m| m.role == "user") {
            let query = query.content.clone();
            let query_embedding = self.model.embed(&query).await?;
            let grounded = self.retriever.retrieve(&query_embedding).await?;

            let mut grounding = String::from(IDENTITY_GUARDRAIL);
            let mut memory_ids = Vec::new();

            if !grounded.is_empty() {
                let candidates: Vec<MemoryRecord> = grounded
                    .into_iter()
                    .map(|g| MemoryRecord {
                        id: g.id,
                        kind: g.kind,
                        content: g.content,
                        embedding: None,
                    })
                    .collect();
                let package = self.context.assemble(String::new(), Vec::new(), &candidates);

                if !package.relevant_memories.is_empty() {
                    memory_ids = package.relevant_memories.iter().map(|m| m.id.clone()).collect();

                    grounding.push_str(
                        "\n\nUse ONLY the facts below to answer questions about the user. \
                         If the answer isn't in these facts, say you don't have that \
                         information — never guess.\n\n",
                    );
                    for memory in &package.relevant_memories {
                        grounding.push_str(&format!("[mem:{}] {}\n", memory.id, memory.content));
                    }
                }
            }

            // Merged into the caller's existing leading system message
            // rather than inserted as a second one: verified live that
            // with two separate system messages, qwen2.5:1.5b (via Ollama)
            // silently ignored this grounding entirely and answered from
            // the *other* system message's instructions instead — the
            // retrieval and event above fired correctly, but the final
            // answer acted as if it hadn't. A single combined system
            // message fixes it. Always merged (even with no memories
            // found) since the identity guardrail applies to every turn.
            match request.messages.first_mut() {
                Some(first) if first.role == "system" => {
                    first.content = format!("{grounding}\n{}", first.content);
                }
                _ => request.messages.insert(0, ChatMessage::system(grounding)),
            }
            if !memory_ids.is_empty() {
                self.events.publish(Event::MemoryRetrieved { memory_ids, query });
            }
        }

        let mut response = self.model.chat(request.clone()).await?;

        let mut rounds = 0;
        while !response.tool_calls.is_empty() && rounds < MAX_TOOL_ROUNDS {
            rounds += 1;
            for call in &response.tool_calls {
                // A tool failure (bad/missing arguments, denied permission)
                // is fed back to the model as a tool result rather than
                // aborting the turn, so it gets a chance to recover — e.g.
                // ask the user for the missing field — instead of the whole
                // `chat` call failing as if the backend were unreachable.
                let message = match self.execute_tool(&call.name, call.arguments.clone()).await {
                    Ok(result) => ChatMessage::tool(result.to_string()),
                    Err(err) => ChatMessage::tool(format!("error: {err}")),
                };
                request.messages.push(message);
            }
            response = self.model.chat(request.clone()).await?;
        }

        if let Some(query) = request.messages.iter().rev().find(|m| m.role == "user") {
            if let Some(fact) = self.extract_personal_fact(&query.content).await {
                let id = format!(
                    "fact-{}",
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .expect("system clock is before UNIX epoch")
                        .as_nanos()
                );
                // Best-effort: a failure to persist a remembered fact
                // shouldn't fail the whole turn — the user still gets
                // `response` back either way.
                let _ = self
                    .remember(MemoryRecord { id, kind: MemoryKind::Semantic, content: fact, embedding: None })
                    .await;
            }
        }

        Ok(response)
    }

    /// Best-effort: asks the model whether the user's latest message stated
    /// a durable personal fact (name, preference, where they live, a
    /// restriction, ...) worth remembering long-term, via a short, isolated
    /// prompt — deliberately NOT appended to the rest of the conversation's
    /// history. Longer, multi-rule prompts made qwen2.5:1.5b fail outright
    /// in earlier testing (see the project's small-model-prompting notes),
    /// so this asks exactly one thing and nothing else. Returns `None` on
    /// any backend failure or an explicit "none" reply — never blocks
    /// `chat` from returning `response` to the user, since missing a fact
    /// to remember is a much smaller problem than failing to reply.
    async fn extract_personal_fact(&self, message: &str) -> Option<String> {
        // Deterministic guard before trusting the model's judgment, same
        // pattern as the trigger-phrase tool routing above: a trailing "?"
        // is a free, reliable signal that this is a question, not a stated
        // fact. Skips both a misclassification (seen live: a small local
        // model replied with a "fact" extracted from "voce sabe algo sobre
        // mim?" instead of "none") and an unnecessary model call.
        if message.trim().ends_with('?') {
            return None;
        }

        let extraction_request = ChatRequest {
            messages: vec![
                ChatMessage::system(
                    "The user said this to a personal assistant. If it STATES a durable \
                     personal fact about them (name, preference, where they live, a \
                     restriction, etc.), reply with that fact as one short third-person \
                     sentence, in the SAME language the user wrote in. If it's a question, \
                     a command, or doesn't state a new fact, reply with exactly: none",
                ),
                ChatMessage::user(message),
            ],
            tools: Vec::new(),
        };
        let response = self.model.chat(extraction_request).await.ok()?;
        let fact = response.message.content.trim();
        if fact.is_empty() || fact.eq_ignore_ascii_case("none") {
            return None;
        }
        Some(fact.to_string())
    }
}
