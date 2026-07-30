//! Measures chat latency and process memory for the configured Model
//! Runtime backend. Requires a local `ollama serve` with the target model
//! pulled — this is a manual dev tool, not part of `cargo test`.
//!
//! Usage:
//!   cargo run -p model-latency-benchmark -- [model] [rounds]

use ally_models::{ChatMessage, ChatRequest, ModelBackend, OllamaBackend};
use std::time::Instant;
use sysinfo::{get_current_pid, System};

const DEFAULT_MODEL: &str = "qwen2.5:0.5b";
const DEFAULT_ROUNDS: usize = 5;
const PROMPT: &str = "In one short sentence, what is a personal intelligence runtime?";

#[tokio::main]
async fn main() {
    let mut args = std::env::args().skip(1);
    let model_name = args.next().unwrap_or_else(|| DEFAULT_MODEL.to_string());
    let rounds: usize = args
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_ROUNDS);

    let backend = OllamaBackend::new(&model_name);
    println!("model: {model_name}  rounds: {rounds}");

    let mut latencies_ms = Vec::with_capacity(rounds);
    for round in 1..=rounds {
        let request = ChatRequest {
            messages: vec![ChatMessage::user(PROMPT)],
            tools: Vec::new(),
        };

        let started = Instant::now();
        match backend.chat(request).await {
            Ok(_) => {
                let elapsed = started.elapsed();
                println!("round {round}: {:.0} ms", elapsed.as_secs_f64() * 1000.0);
                latencies_ms.push(elapsed.as_secs_f64() * 1000.0);
            }
            Err(err) => {
                eprintln!(
                    "round {round}: failed ({err}). Is `ollama serve` running with `{model_name}` pulled?"
                );
                return;
            }
        }
    }

    if !latencies_ms.is_empty() {
        let min = latencies_ms.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = latencies_ms.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let avg = latencies_ms.iter().sum::<f64>() / latencies_ms.len() as f64;
        println!("latency (ms): min={min:.0} avg={avg:.0} max={max:.0}");
    }

    report_process_memory();
}

fn report_process_memory() {
    let mut system = System::new_all();
    system.refresh_all();

    match get_current_pid().ok().and_then(|pid| system.process(pid)) {
        Some(process) => {
            let mb = process.memory() as f64 / (1024.0 * 1024.0);
            println!("process memory (RSS): {mb:.1} MB");
        }
        None => eprintln!("could not read process memory on this platform"),
    }
}
