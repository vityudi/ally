# Tools

Developer-facing scripts and utilities for working on the Ally Framework
itself (codegen, release scripts, dataset prep for PALM, etc.) — not to be
confused with `runtime/tools`, which is the Runtime's Tool Orchestrator.

## `dataset-gen`

Generates the synthetic planning/intent dataset consumed by
[`benchmarks/eval-harness`](../benchmarks/README.md): template-based
(prompt → expected tool call) examples, no RNG dependency, deterministic
and idempotent to regenerate.

```sh
cargo run -p dataset-gen
```

Writes `datasets/planning-intent/finance.jsonl`.
