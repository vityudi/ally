<div align="center">

# Ally Framework

**An Open Personal Intelligence Framework — local-first, model-agnostic, privacy by default.**

[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-stable%20%7C%20edition%202021-orange?logo=rust&logoColor=white)](https://www.rust-lang.org)
[![Build](https://img.shields.io/badge/build-cargo%20workspace-informational?logo=rust&logoColor=white)](#building)
[![Tokio](https://img.shields.io/badge/async-tokio-blueviolet?logo=rust&logoColor=white)](https://tokio.rs)
[![Serde](https://img.shields.io/badge/serialization-serde-9cf?logo=rust&logoColor=white)](https://serde.rs)
[![Status](https://img.shields.io/badge/status-early%20scaffold-yellow)](#status)
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

Early scaffold (**Phase 1: Runtime foundation**). Every crate currently
exposes minimal stubs matching the responsibilities described in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — no real planning, memory or
inference logic is implemented yet.

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

## 📜 License

Licensed under the [Apache License 2.0](LICENSE).
