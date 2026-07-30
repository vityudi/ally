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

Contents of this directory are git-ignored except for this file.
