# Models

Local model weights and inference artifacts live here (e.g. GGUF files for
llama.cpp, ONNX exports, future PALM checkpoints). This directory holds data,
not code — see `runtime/models` for the Model Runtime abstraction that loads
and swaps these backends.

Populated automatically: `Ally::new()`/`Ally::open()` default to
`LlamaCppBackend::lazy_default`, which downloads the pinned default chat +
embedding GGUFs here on the first `chat`/`embed` call, verifies their
SHA-256, and reuses the cached files on every run after that — see
`runtime/models/src/llama_cpp/download.rs`. Nothing here needs to be
fetched by hand.

## Offline first run / using a different model

No network access on first run, or want to point at a GGUF you already
have? Two options:

- Drop the file in here under the exact pinned filename
  (`qwen2.5-1.5b-instruct-q4_k_m.gguf` for chat,
  `all-MiniLM-L6-v2-Q4_K_M.gguf` for embeddings — see the constants in
  `download.rs`) and it's used as-is, no download attempted. Note the
  SHA-256 check only runs on files this crate downloads itself, not on
  ones already present, so this doesn't verify integrity.
- Or set `ALLY_CHAT_MODEL_PATH` / `ALLY_EMBEDDING_MODEL_PATH` to any path,
  any filename — no download is attempted for whichever one is set, and
  no renaming is required.

Contents of this directory are git-ignored except for this file.
