<div align="center">

# Ally Framework

**An Open Personal Intelligence Framework — local-first, model-agnostic, privacy by default.**

[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-stable%20%7C%20edition%202021-orange?logo=rust&logoColor=white)](https://www.rust-lang.org)
[![Build](https://img.shields.io/badge/build-cargo%20workspace-informational?logo=rust&logoColor=white)](#building)
[![Tokio](https://img.shields.io/badge/async-tokio-blueviolet?logo=rust&logoColor=white)](https://tokio.rs)
[![Serde](https://img.shields.io/badge/serialization-serde-9cf?logo=rust&logoColor=white)](https://serde.rs)
[![Status](https://img.shields.io/badge/status-phase%205%20of%206-blue)](#status)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

</div>

---

Ally is not another chatbot or LLM wrapper. It is a **Personal Intelligence
Runtime**: the Language Model sits at the *end* of the pipeline, generating
language, while deterministic Runtime modules handle planning, memory,
context assembly, tool execution, permissions and storage.

## 📖 Read first

Read the project's constitution before contributing:

| Document | Purpose |
|---|---|
| [`docs/MANIFESTO.md`](docs/MANIFESTO.md) | What we believe and why |
| [`docs/FOUNDATION.md`](docs/FOUNDATION.md) | Vision and core principles |
| [`docs/PRINCIPLES.md`](docs/PRINCIPLES.md) | The 25 non-negotiable engineering principles |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Technical architecture spec |

## 🗂️ Repository layout

```
ally-framework/
  runtime/       # Core engine: planner, memory, context, tools, models,
                 # plugins (manager), scheduler, events, storage, security, api
  sdk/           # The only interface applications should depend on
  cli/           # Local CLI entry point to the Runtime
  plugins/       # Installable capability plugins (finance, calendar, ...)
  examples/      # Example applications built on the SDK
  docs/          # Vision, principles and architecture documents
  benchmarks/    # Performance and model evaluation benchmarks
  models/        # Local model weights (git-ignored, data only)
  tests/         # Workspace-level integration tests
  tools/         # Developer scripts for working on this repo
```

## 🚧 Status

Phases 1-5 of the roadmap in [`docs/FOUNDATION.md`](docs/FOUNDATION.md) have
working implementations: Runtime foundation and plugin system, Memory/Planner/
Context engines, native local inference via `llama.cpp` (default) and Ollama,
the SDK and finance plugin, and a training/eval pipeline
(`benchmarks/eval-harness`, `tools/dataset-gen`). Still early and evolving —
see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the target shape and
Phase 6 (PALM, a custom small model) for what's still ahead.

## 🛠️ Building

```sh
cargo build --workspace
```

Requires a recent stable Rust toolchain (edition 2021). The default Model
Runtime backend runs inference in-process via `llama.cpp`
(`runtime/models/src/llama_cpp.rs`), so the build also needs `cmake` and a
C/C++ toolchain (MSVC Build Tools on Windows, gcc/clang on Linux/macOS) —
`llama-cpp-2` compiles `llama.cpp` itself as part of `cargo build`. To
build `ally-models` without that step (e.g. to only use
`ally_models::OllamaBackend`, talking to an existing `ollama serve`),
disable its default feature: `cargo build -p ally-models --no-default-features`.

## 🧪 Try it: chat REPL

```sh
cargo run -p ally-chat
```

No external services needed — `Ally::new()` defaults to the in-process
`LlamaCppBackend`. The first message you send triggers a one-time download
(~1.1 GB) of the pinned default GGUF weights into `models/` (see
`models/README.md`), then loads them; that first call is slow, but every
run after that reuses the cached weights and just pays the (much shorter)
load time.

- Type in Portuguese or English; type `sair` / `exit` / `quit` to leave.
- The REPL installs the finance plugin, so it's a good way to poke at
  tool-calling — try things like `gastei 15 reais no almoco hoje` or
  `qual e o meu saldo`.
- To talk to a different/bigger model via an existing `ollama serve`
  instead of the local default, call
  `ally.with_model(Arc::new(ally_models::OllamaBackend::new("model-name")))`
  before the REPL loop starts, in `examples/chat/src/main.rs`.
- `cargo run -p ally-cli` and `cargo run -p kyvo-example` are smaller,
  non-interactive demos of the same SDK (see `examples/` and `cli/`).

## 📜 License

Licensed under the [Apache License 2.0](LICENSE).
