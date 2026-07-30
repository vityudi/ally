//! In-process inference via `llama.cpp` (through the `llama-cpp-2`
//! bindings) — the default Model Runtime backend (see
//! `ally_sdk::Ally::with_storage`), so Ally runs as a single executable
//! with no external daemon to install or keep running.
//! [`crate::OllamaBackend`] is still available and still useful — point
//! `Ally::with_model` at it if you'd rather talk to a bigger model already
//! served by a local/remote `ollama serve` — but it's no longer what a
//! plain `Ally::new()` gives you.
//!
//! Weights are not bundled into the binary (a Q4_K_M GGUF for a 1.5B-class
//! model is already ~1 GB, which would wreck incremental builds and
//! distribution). Instead [`LlamaCppBackend::lazy_default`] defers both
//! the download (`download::ensure_downloaded`, into the git-ignored
//! `/models` directory — see `models/README.md`) and the model load until
//! the first `chat`/`chat_stream`/`embed` call actually needs it, via
//! `tokio::sync::OnceCell`. This keeps `Ally::new()`/`with_storage()`
//! synchronous and infallible exactly like before — nothing pays the
//! multi-second load (or first-run multi-minute download) until it
//! actually talks to the model. For a first run with no network access,
//! `ALLY_CHAT_MODEL_PATH`/`ALLY_EMBEDDING_MODEL_PATH` point at local GGUFs
//! instead — see [`LlamaCppBackend::lazy_default`].
//!
//! Tool calling is the one place this backend can't just defer to
//! llama.cpp: Ollama's `/api/chat` parses tool calls out of the model's
//! raw output server-side, but `llama_chat_apply_template` (what
//! `LlamaModel::apply_chat_template` wraps) takes no `tools` argument, so
//! there is no way to feed Qwen2.5's Jinja template its `tools` variable
//! through this binding. Instead `tool_system_preamble` pre-renders
//! exactly the text that branch of the template would have produced and
//! folds it into the system message by hand, and `parse_tool_calls`
//! extracts the `<tool_call>{...}</tool_call>` blocks Qwen2.5 is trained
//! to emit back out of the generated text. Every other part of the
//! template (plain turns, `<tool_response>` wrapping for tool-role
//! messages) is independent of the `tools` variable, so it's left to the
//! model's own embedded template. If `apply_chat_template` fails for any
//! reason (e.g. a future model ships a template llama.cpp's built-in
//! interpreter can't evaluate), `manual_chatml_prompt` is a plain ChatML
//! fallback so the backend still produces a usable prompt.

mod download;

use crate::{ChatMessage, ChatRequest, ChatResponse, ModelBackend, ModelError, ToolCall, ToolSpec};
use async_trait::async_trait;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaChatMessage, LlamaChatTemplate, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use ouroboros::self_referencing;
use serde_json::Value;
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use tokio::sync::OnceCell;

/// Context window size for chat contexts. Generous enough for a handful of
/// tool-augmented turns on a 1.5B-class model without needing per-request
/// tuning for this POC.
const DEFAULT_CONTEXT_SIZE: u32 = 4096;

/// Hard cap on generated tokens per `chat`/`chat_stream` call, so a model
/// that never emits EOS can't hang the caller forever.
const MAX_NEW_TOKENS: usize = 1024;

fn backend() -> &'static LlamaBackend {
    static BACKEND: OnceLock<LlamaBackend> = OnceLock::new();
    BACKEND.get_or_init(|| {
        // Without this, llama.cpp/ggml print their own raw model-load /
        // tensor / sampler / timing dump straight to stderr on every first
        // call — dozens of lines a chat REPL user has no use for. There's
        // no tracing subscriber installed, so routing them "to tracing"
        // would just drop them; `with_logs_enabled(false)` suppresses the
        // native callback outright instead.
        llama_cpp_2::send_logs_to_tracing(llama_cpp_2::LogOptions::default().with_logs_enabled(false));
        LlamaBackend::init().expect("llama.cpp backend failed to initialize")
    })
}

#[self_referencing]
struct LoadedModel {
    model: LlamaModel,
    #[borrows(model)]
    #[covariant]
    context: LlamaContext<'this>,
}

/// `LlamaContext` holds a raw `NonNull` pointer and so is not `Send`/`Sync`
/// by default, but nothing in llama.cpp ties a context to the thread that
/// created it — it only requires that a given context is never decoded
/// from two threads *concurrently*. `LlamaCppBackend` only ever touches a
/// `LoadedModel` from inside a `std::sync::Mutex` lock, which is exactly
/// that serialization guarantee, so it's sound to assert both here.
struct SendSyncLoadedModel(LoadedModel);
unsafe impl Send for SendSyncLoadedModel {}
unsafe impl Sync for SendSyncLoadedModel {}

fn load(path: &Path, embeddings: bool) -> Result<SendSyncLoadedModel, ModelError> {
    let model_params = LlamaModelParams::default();
    let model = LlamaModel::load_from_file(backend(), path, &model_params)
        .map_err(|e| ModelError::Backend(format!("failed to load {}: {e}", path.display())))?;

    // `LlamaContextParams::default()` (== upstream's `llama_context_default_params()`)
    // hardcodes `n_threads`/`n_threads_batch` to `GGML_DEFAULT_N_THREADS`
    // (4), not the machine's actual core count — Ollama's server detects
    // and uses all available cores, so leaving this at the llama.cpp
    // default made generation visibly slower here on anything with more
    // than 4 cores. Match Ollama's behavior explicitly instead.
    let n_threads = std::thread::available_parallelism().map(|n| n.get() as i32).unwrap_or(4);

    let loaded = LoadedModelTryBuilder {
        model,
        context_builder: |model| {
            let ctx_params = LlamaContextParams::default()
                .with_n_ctx(NonZeroU32::new(DEFAULT_CONTEXT_SIZE))
                .with_embeddings(embeddings)
                .with_n_threads(n_threads)
                .with_n_threads_batch(n_threads);
            model.new_context(backend(), ctx_params).map_err(|e| {
                ModelError::Backend(format!("failed to create llama.cpp context: {e}"))
            })
        },
    }
    .try_build()?;

    Ok(SendSyncLoadedModel(loaded))
}

/// Where a GGUF for a given role (chat/embedding) comes from: either an
/// explicit local path (eager constructors), or a pinned Hugging Face file
/// downloaded into `models_dir` on first use ([`LlamaCppBackend::lazy_default`]).
enum GgufSource {
    Path(PathBuf),
    Download { pinned: &'static download::PinnedGguf, models_dir: PathBuf },
}

impl GgufSource {
    /// Human-readable identifier for startup messages — the GGUF filename
    /// either way, without needing to resolve (and thus possibly download)
    /// the file first.
    fn label(&self) -> String {
        match self {
            GgufSource::Path(path) => path.display().to_string(),
            GgufSource::Download { pinned, .. } => pinned.file.to_string(),
        }
    }

    async fn resolve(&self) -> Result<PathBuf, ModelError> {
        match self {
            GgufSource::Path(path) => Ok(path.clone()),
            GgufSource::Download { pinned, models_dir } => {
                download::ensure_downloaded(pinned, models_dir).await
            }
        }
    }
}

pub struct LlamaCppBackend {
    chat_source: GgufSource,
    embedding_source: Option<GgufSource>,
    chat: OnceCell<Mutex<SendSyncLoadedModel>>,
    chat_template: OnceCell<LlamaChatTemplate>,
    embedding: OnceCell<Mutex<SendSyncLoadedModel>>,
}

impl LlamaCppBackend {
    /// Points at a chat model already on disk. `embed()` will error unless
    /// [`LlamaCppBackend::with_embedding_model`] is used instead. Nothing
    /// is loaded until the first `chat`/`chat_stream` call.
    pub fn new(gguf_path: impl AsRef<Path>) -> Self {
        Self::from_sources(GgufSource::Path(gguf_path.as_ref().to_path_buf()), None)
    }

    /// Same as [`LlamaCppBackend::new`], but also points at a second,
    /// embedding-dedicated GGUF for `embed()` — a qwen2.5-class chat model
    /// isn't usable for embeddings any more than it is via Ollama, which
    /// has the same split (see `ollama::DEFAULT_EMBEDDING_MODEL`).
    pub fn with_embedding_model(
        gguf_path: impl AsRef<Path>,
        embedding_gguf_path: impl AsRef<Path>,
    ) -> Self {
        Self::from_sources(
            GgufSource::Path(gguf_path.as_ref().to_path_buf()),
            Some(GgufSource::Path(embedding_gguf_path.as_ref().to_path_buf())),
        )
    }

    /// The Ally SDK's default: no path required up front, nothing touches
    /// disk or network until the first `chat`/`chat_stream`/`embed` call,
    /// at which point the pinned default GGUFs (`download::DEFAULT_CHAT_MODEL`
    /// / `DEFAULT_EMBEDDING_MODEL`) are downloaded into `models_dir` (if
    /// not already cached there) and loaded.
    ///
    /// `ALLY_CHAT_MODEL_PATH` / `ALLY_EMBEDDING_MODEL_PATH`, if set, each
    /// override the corresponding pinned download with a local GGUF path
    /// instead — no network access at all for that model, and no filename
    /// constraint (a file already sitting in `models_dir` under the pinned
    /// name is picked up automatically without either variable, but
    /// requires matching that exact name; these variables work with any
    /// path/filename). Meant for offline first runs or swapping in a
    /// different model without recompiling.
    pub fn lazy_default(models_dir: impl Into<PathBuf>) -> Self {
        let models_dir = models_dir.into();

        let chat_source = match std::env::var_os("ALLY_CHAT_MODEL_PATH") {
            Some(path) => GgufSource::Path(PathBuf::from(path)),
            None => {
                GgufSource::Download { pinned: &download::DEFAULT_CHAT_MODEL, models_dir: models_dir.clone() }
            }
        };
        let embedding_source = match std::env::var_os("ALLY_EMBEDDING_MODEL_PATH") {
            Some(path) => GgufSource::Path(PathBuf::from(path)),
            None => GgufSource::Download { pinned: &download::DEFAULT_EMBEDDING_MODEL, models_dir },
        };

        Self::from_sources(chat_source, Some(embedding_source))
    }

    fn from_sources(chat_source: GgufSource, embedding_source: Option<GgufSource>) -> Self {
        Self {
            chat_source,
            embedding_source,
            chat: OnceCell::new(),
            chat_template: OnceCell::new(),
            embedding: OnceCell::new(),
        }
    }

    async fn ensure_chat(&self) -> Result<&Mutex<SendSyncLoadedModel>, ModelError> {
        self.chat
            .get_or_try_init(|| async {
                let path = self.chat_source.resolve().await?;
                let loaded = tokio::task::block_in_place(|| load(&path, false))?;
                Ok(Mutex::new(loaded))
            })
            .await
    }

    async fn ensure_chat_template(&self) -> Result<&LlamaChatTemplate, ModelError> {
        let chat_mutex = self.ensure_chat().await?;
        self.chat_template
            .get_or_try_init(|| async {
                let chat = chat_mutex.lock().expect("llama.cpp chat mutex poisoned");
                chat.0
                    .borrow_model()
                    .chat_template(None)
                    .map_err(|e| ModelError::Backend(format!("model has no chat template: {e}")))
            })
            .await
    }

    async fn ensure_embedding(&self) -> Result<&Mutex<SendSyncLoadedModel>, ModelError> {
        let source = self
            .embedding_source
            .as_ref()
            .ok_or_else(|| ModelError::Backend("no embedding model configured".to_string()))?;
        self.embedding
            .get_or_try_init(|| async {
                let path = source.resolve().await?;
                let loaded = tokio::task::block_in_place(|| load(&path, true))?;
                Ok(Mutex::new(loaded))
            })
            .await
    }
}

/// Renders exactly the text Qwen2.5's own chat template would produce for
/// its `{%- if tools %}` branch, so it can be folded into the system
/// message by hand — see the module doc for why this can't be done via
/// the template engine itself through this binding.
fn tool_system_preamble(tools: &[ToolSpec]) -> String {
    let mut s = String::from(
        "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n\
         You are provided with function signatures within <tools></tools> XML tags:\n<tools>",
    );
    for tool in tools {
        let spec = serde_json::json!({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
        });
        s.push('\n');
        s.push_str(&spec.to_string());
    }
    s.push_str(
        "\n</tools>\n\nFor each function call, return a json object with function name and \
         arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n\
         {\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>",
    );
    s
}

/// Merges the tool preamble into `request.messages`, matching the shape
/// Qwen2.5's template produces (see module doc for the specifics).
fn messages_with_tools(request: &ChatRequest) -> Vec<ChatMessage> {
    let mut messages = request.messages.clone();

    if !request.tools.is_empty() {
        let preamble = tool_system_preamble(&request.tools);
        if let Some(first) = messages.first_mut().filter(|m| m.role == "system") {
            first.content.push_str(&preamble);
        } else {
            let base = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";
            messages.insert(0, ChatMessage::system(format!("{base}{preamble}")));
        }
    }

    messages
}

fn manual_chatml_prompt(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    for m in messages {
        if m.role == "tool" {
            prompt.push_str("<|im_start|>user\n<tool_response>\n");
            prompt.push_str(&m.content);
            prompt.push_str("\n</tool_response><|im_end|>\n");
        } else {
            prompt.push_str("<|im_start|>");
            prompt.push_str(&m.role);
            prompt.push('\n');
            prompt.push_str(&m.content);
            prompt.push_str("<|im_end|>\n");
        }
    }
    prompt.push_str("<|im_start|>assistant\n");
    prompt
}

fn build_prompt(
    model: &LlamaModel,
    chat_template: &LlamaChatTemplate,
    request: &ChatRequest,
) -> Result<String, ModelError> {
    let messages = messages_with_tools(request);

    let llama_messages: Result<Vec<LlamaChatMessage>, _> = messages
        .iter()
        .map(|m| LlamaChatMessage::new(m.role.clone(), m.content.clone()))
        .collect();
    let llama_messages =
        llama_messages.map_err(|e| ModelError::Backend(format!("invalid chat message: {e}")))?;

    match model.apply_chat_template(chat_template, &llama_messages, true) {
        Ok(prompt) => Ok(prompt),
        Err(_) => Ok(manual_chatml_prompt(&messages)),
    }
}

/// Extracts `<tool_call>{"name": ..., "arguments": ...}</tool_call>`
/// blocks Qwen2.5-family models emit inline in their output, returning the
/// remaining plain-text content alongside the parsed [`ToolCall`]s. A
/// block that isn't valid JSON is left in the returned text verbatim
/// rather than silently dropped.
fn parse_tool_calls(text: &str) -> (String, Vec<ToolCall>) {
    const OPEN: &str = "<tool_call>";
    const CLOSE: &str = "</tool_call>";

    let mut cleaned = String::new();
    let mut tool_calls = Vec::new();
    let mut rest = text;

    while let Some(start) = rest.find(OPEN) {
        cleaned.push_str(&rest[..start]);
        let after_open = &rest[start + OPEN.len()..];

        let Some(end) = after_open.find(CLOSE) else {
            cleaned.push_str(&rest[start..]);
            rest = "";
            break;
        };

        let body = after_open[..end].trim();
        match serde_json::from_str::<Value>(body) {
            Ok(value) => {
                let name = value
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                let arguments = value.get("arguments").cloned().unwrap_or(Value::Null);
                tool_calls.push(ToolCall { name, arguments });
            }
            Err(_) => {
                cleaned.push_str(OPEN);
                cleaned.push_str(body);
                cleaned.push_str(CLOSE);
            }
        }

        rest = &after_open[end + CLOSE.len()..];
    }
    cleaned.push_str(rest);

    (cleaned.trim().to_string(), tool_calls)
}

/// Runs the greedy decode loop shared by `chat` and `chat_stream`. Greedy
/// (temperature 0) sampling mirrors `OllamaOptions`'s default in
/// `ollama.rs` — same reasoning: with sampling noise removed, tool-call
/// emission is far more deterministic on small models.
fn generate(
    loaded: &mut LoadedModel,
    chat_template: &LlamaChatTemplate,
    request: &ChatRequest,
    mut on_token: Option<Box<dyn for<'a> FnMut(&'a str) + Send>>,
) -> Result<ChatResponse, ModelError> {
    let prompt = build_prompt(loaded.borrow_model(), chat_template, request)?;

    loaded.with_mut(|fields| {
        let model: &LlamaModel = fields.model;
        let ctx: &mut LlamaContext = fields.context;

        // Ally resends the full message history on every `chat` call
        // (there's no incremental/cached decoding across turns), so the
        // previous turn's KV cache is stale and must be dropped before
        // decoding this turn's prompt from position 0 — otherwise
        // llama.cpp rejects the batch with a "sequence positions must
        // remain consecutive" error the moment position 0 follows
        // whatever position the last turn ended on.
        ctx.clear_kv_cache();

        let tokens = model
            .str_to_token(&prompt, AddBos::Always)
            .map_err(|e| ModelError::Backend(format!("tokenize failed: {e}")))?;

        let mut batch = LlamaBatch::new(tokens.len().max(512), 1);
        let last_index = tokens.len() as i32 - 1;
        for (i, token) in (0_i32..).zip(tokens.into_iter()) {
            batch
                .add(token, i, &[0], i == last_index)
                .map_err(|e| ModelError::Backend(format!("batch add failed: {e}")))?;
        }
        ctx.decode(&mut batch)
            .map_err(|e| ModelError::Backend(format!("decode failed: {e}")))?;

        let mut sampler = LlamaSampler::greedy();
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut n_cur = batch.n_tokens();
        let mut content = String::new();

        for _ in 0..MAX_NEW_TOKENS {
            let token = sampler.sample(ctx, batch.n_tokens() - 1);
            sampler.accept(token);

            if model.is_eog_token(token) {
                break;
            }

            let piece = model
                .token_to_piece(token, &mut decoder, true, None)
                .map_err(|e| ModelError::Backend(format!("detokenize failed: {e}")))?;

            if let Some(cb) = on_token.as_mut() {
                cb(&piece);
            }
            content.push_str(&piece);

            batch.clear();
            batch
                .add(token, n_cur, &[0], true)
                .map_err(|e| ModelError::Backend(format!("batch add failed: {e}")))?;
            n_cur += 1;

            ctx.decode(&mut batch)
                .map_err(|e| ModelError::Backend(format!("decode failed: {e}")))?;
        }

        let (cleaned, tool_calls) = parse_tool_calls(&content);
        Ok(ChatResponse {
            message: ChatMessage::assistant(cleaned),
            tool_calls,
        })
    })
}

#[async_trait]
impl ModelBackend for LlamaCppBackend {
    fn name(&self) -> &str {
        "llama-cpp"
    }

    fn model_id(&self) -> String {
        self.chat_source.label()
    }

    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse, ModelError> {
        let chat_mutex = self.ensure_chat().await?;
        let chat_template = self.ensure_chat_template().await?;
        tokio::task::block_in_place(|| {
            let mut chat = chat_mutex.lock().expect("llama.cpp chat mutex poisoned");
            generate(&mut chat.0, chat_template, &request, None)
        })
    }

    async fn chat_stream(
        &self,
        request: ChatRequest,
        on_token: Box<dyn for<'a> FnMut(&'a str) + Send>,
    ) -> Result<ChatResponse, ModelError> {
        let chat_mutex = self.ensure_chat().await?;
        let chat_template = self.ensure_chat_template().await?;
        tokio::task::block_in_place(|| {
            let mut chat = chat_mutex.lock().expect("llama.cpp chat mutex poisoned");
            generate(&mut chat.0, chat_template, &request, Some(on_token))
        })
    }

    async fn embed(&self, text: &str) -> Result<Vec<f32>, ModelError> {
        let embedding_mutex = self.ensure_embedding().await?;

        tokio::task::block_in_place(|| {
            let mut loaded = embedding_mutex.lock().expect("llama.cpp embedding mutex poisoned");
            loaded.0.with_mut(|fields| {
                let model: &LlamaModel = fields.model;
                let ctx: &mut LlamaContext = fields.context;

                // Same reasoning as `generate`'s clear_kv_cache: every
                // `embed` call is a fresh, standalone input, not a
                // continuation of the previous one.
                ctx.clear_kv_cache();

                let tokens = model
                    .str_to_token(text, AddBos::Always)
                    .map_err(|e| ModelError::Backend(format!("tokenize failed: {e}")))?;

                let mut batch = LlamaBatch::new(tokens.len().max(512), 1);
                // Mean pooling needs every token's hidden state, not just
                // the last one (unlike causal `generate`, which only
                // needs logits for the final position) — so every token
                // is marked as an output.
                for (i, token) in (0_i32..).zip(tokens.into_iter()) {
                    batch
                        .add(token, i, &[0], true)
                        .map_err(|e| ModelError::Backend(format!("batch add failed: {e}")))?;
                }
                // This context was created with `.with_embeddings(true)`,
                // which makes it non-causal — llama.cpp requires such
                // contexts to be run through `encode`, not `decode`.
                ctx.encode(&mut batch)
                    .map_err(|e| ModelError::Backend(format!("encode failed: {e}")))?;

                let embedding = ctx
                    .embeddings_seq_ith(0)
                    .map_err(|e| ModelError::Backend(format!("embeddings failed: {e}")))?;
                Ok(embedding.to_vec())
            })
        })
    }

    /// Overrides the trait's `len()/4` approximation with an exact count
    /// from the chat model's own tokenizer, when it's already loaded —
    /// the first backend able to do this at all, per the invitation in
    /// `ModelBackend::count_tokens`'s doc. `count_tokens` isn't async, so
    /// it can't trigger the lazy load itself; falls back to the same
    /// approximation as every other backend until something else (a
    /// `chat`/`embed` call) has loaded the model. Nothing in this repo
    /// currently calls `count_tokens`, so this is a documented edge case,
    /// not an observed regression.
    fn count_tokens(&self, text: &str) -> usize {
        match self.chat.get() {
            Some(chat) => {
                let chat = chat.lock().expect("llama.cpp chat mutex poisoned");
                chat.0
                    .borrow_model()
                    .str_to_token(text, AddBos::Never)
                    .map(|tokens| tokens.len())
                    .unwrap_or_else(|_| (text.len() / 4).max(1))
            }
            None => (text.len() / 4).max(1),
        }
    }
}
