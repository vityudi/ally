# Ally Framework

> An Open Personal Intelligence Framework — local-first, model-agnostic,
> privacy by default.

Ally is not another chatbot or LLM wrapper. It is a **Personal Intelligence
Runtime**: the Language Model sits at the *end* of the pipeline, generating
language, while deterministic Runtime modules handle planning, memory,
context assembly, tool execution, permissions and storage.

Read the project's constitution before contributing:

- [`docs/MANIFESTO.md`](docs/MANIFESTO.md) — what we believe and why
- [`docs/FOUNDATION.md`](docs/FOUNDATION.md) — vision and core principles
- [`docs/PRINCIPLES.md`](docs/PRINCIPLES.md) — the 25 non-negotiable engineering principles
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — technical architecture spec

## Repository layout

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

## Status

Early scaffold (Phase 1: Runtime foundation). Every crate currently exposes
minimal stubs matching the responsibilities described in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — no real planning, memory or
inference logic is implemented yet.

## Building

```sh
cargo build --workspace
```

Requires a recent stable Rust toolchain (edition 2021).
