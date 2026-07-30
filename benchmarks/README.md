# Benchmarks

Performance and quality benchmarks for the Ally Runtime: latency of the
Planner/Context/Memory pipeline, tool execution overhead, and (later)
evaluation of PALM against other small models on personal-assistant tasks.

## `model-latency`

Measures chat latency and process memory for the configured Model Runtime
backend (Ollama by default). Requires a local `ollama serve` with the
target model pulled — this is a manual dev tool, not part of `cargo test`.

```sh
ollama pull qwen2.5:0.5b
cargo run -p model-latency-benchmark -- qwen2.5:0.5b 5
```

Future work: extend to other `ModelBackend` implementations as they land
(Phase 3+), and add planner/memory/context pipeline benchmarks that don't
require a model at all.
