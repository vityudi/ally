//! Runs the synthetic planning/intent dataset (see `tools/dataset-gen`)
//! through `Ally::chat` and compares tool-call accuracy and latency
//! across one or more Ollama models — a repeatable version of the manual
//! observation from Phase 3 (`examples/kyvo`): small models report tool
//! support but don't reliably use it. Requires a local `ollama serve`
//! with the target models pulled — this is a manual dev tool, not part
//! of `cargo test`.
//!
//! Usage:
//!   cargo run -p eval-harness -- [dataset_path] [model1,model2,...]

use ally_models::OllamaBackend;
use ally_plugin_finance::FinancePlugin;
use ally_scheduler::Scheduler;
use ally_sdk::{Ally, ChatError, ChatMessage, ChatRequest, Event, EventHandler, Permission};
use serde::Deserialize;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use sysinfo::{get_current_pid, System};

const DEFAULT_MODELS: &str = "qwen2.5:0.5b,qwen2.5:1.5b";
const SYSTEM_PROMPT: &str = "You are a personal assistant. If the user needs a reminder or \
    scheduled task, call the appropriate tool immediately with your best-guess arguments. \
    Otherwise just respond normally.";

#[derive(Debug, Deserialize)]
struct DatasetExample {
    #[allow(dead_code)]
    id: String,
    #[allow(dead_code)]
    intent: String,
    prompt: String,
    expected_tool: Option<String>,
}

/// Records every `Event::ToolExecuted` tool name so a single eval example
/// can be checked against its expected tool without changing `Ally::chat`
/// itself.
struct ToolCallRecorder(Arc<Mutex<Vec<String>>>);

impl EventHandler for ToolCallRecorder {
    fn handle(&self, event: &Event) {
        if let Event::ToolExecuted { tool_name } = event {
            self.0.lock().expect("recorder mutex poisoned").push(tool_name.clone());
        }
    }
}

struct ModelReport {
    model: String,
    total: usize,
    correct: usize,
    positive_total: usize,
    positive_correct: usize,
    negative_total: usize,
    negative_correct: usize,
    latencies_ms: Vec<f64>,
    rss_mb: Option<f64>,
}

#[tokio::main]
async fn main() {
    let mut args = std::env::args().skip(1);
    let dataset_path = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| workspace_root().join("datasets/planning-intent/finance.jsonl"));
    let models: Vec<String> = args
        .next()
        .unwrap_or_else(|| DEFAULT_MODELS.to_string())
        .split(',')
        .map(|s| s.trim().to_string())
        .collect();

    let examples = load_dataset(&dataset_path);
    println!("loaded {} examples from {}", examples.len(), dataset_path.display());

    let mut reports = Vec::new();
    for model in &models {
        println!("\n=== model: {model} ===");
        match run_model(model, &examples).await {
            Some(report) => reports.push(report),
            None => println!(
                "skipped {model} (could not reach the Model Runtime — is `ollama serve` \
                 running with `{model}` pulled?)"
            ),
        }
    }

    print_comparison_table(&reports);
}

async fn run_model(model: &str, examples: &[DatasetExample]) -> Option<ModelReport> {
    let scheduler = Arc::new(Mutex::new(Scheduler::new()));
    let mut ally = Ally::new();
    ally.grant_permissions(vec![Permission::Write]);
    ally.install_plugin(Box::new(FinancePlugin::new(scheduler)))
        .expect("finance plugin should install cleanly");
    ally.with_model(Arc::new(OllamaBackend::new(model)));

    let calls = Arc::new(Mutex::new(Vec::new()));
    ally.on_event(Box::new(ToolCallRecorder(calls.clone())));

    let tools = ally.tool_specs();

    let mut report = ModelReport {
        model: model.to_string(),
        total: 0,
        correct: 0,
        positive_total: 0,
        positive_correct: 0,
        negative_total: 0,
        negative_correct: 0,
        latencies_ms: Vec::with_capacity(examples.len()),
        rss_mb: None,
    };

    for example in examples {
        calls.lock().expect("recorder mutex poisoned").clear();

        let request = ChatRequest {
            messages: vec![
                ChatMessage::system(SYSTEM_PROMPT),
                ChatMessage::user(example.prompt.clone()),
            ],
            tools: tools.clone(),
        };

        let started = Instant::now();
        let outcome = ally.chat(request).await;
        let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;

        // A `Tool` error means the model reached out and asked for a tool
        // but the call itself was malformed (e.g. missing arguments) —
        // that's a legitimate (incorrect) eval outcome for this example,
        // not a reason to give up on the whole model. A `Model` error
        // means the backend itself is unreachable, which is fatal for
        // every remaining example.
        match &outcome {
            Ok(_) => {}
            Err(ChatError::Tool(err)) => {
                eprintln!("  example {} called a tool that failed: {err}", example.id);
            }
            Err(ChatError::Model(err)) => {
                eprintln!("  example {} failed: {err}", example.id);
                return None;
            }
        }

        report.latencies_ms.push(elapsed_ms);

        let called = calls.lock().expect("recorder mutex poisoned").clone();
        let correct = match &example.expected_tool {
            Some(expected) => called.iter().any(|c| c == expected),
            None => called.is_empty(),
        };

        report.total += 1;
        if correct {
            report.correct += 1;
        }
        match &example.expected_tool {
            Some(_) => {
                report.positive_total += 1;
                if correct {
                    report.positive_correct += 1;
                }
            }
            None => {
                report.negative_total += 1;
                if correct {
                    report.negative_correct += 1;
                }
            }
        }
    }

    report.rss_mb = process_memory_mb();
    Some(report)
}

fn load_dataset(path: &PathBuf) -> Vec<DatasetExample> {
    let contents = std::fs::read_to_string(path)
        .unwrap_or_else(|err| panic!("failed to read dataset {}: {err}", path.display()));
    contents
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).expect("dataset line should be valid JSON"))
        .collect()
}

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("benchmarks/eval-harness should be two levels below the workspace root")
        .to_path_buf()
}

fn process_memory_mb() -> Option<f64> {
    let mut system = System::new_all();
    system.refresh_all();
    get_current_pid()
        .ok()
        .and_then(|pid| system.process(pid))
        .map(|process| process.memory() as f64 / (1024.0 * 1024.0))
}

fn print_comparison_table(reports: &[ModelReport]) {
    if reports.is_empty() {
        println!("\nno models produced results");
        return;
    }

    println!("\n=== comparison ===");
    println!(
        "{:<16} {:>8} {:>10} {:>10} {:>9} {:>9} {:>9} {:>9}",
        "model", "accuracy", "pos.recall", "neg.prec.", "min ms", "avg ms", "max ms", "rss MB"
    );
    for report in reports {
        let accuracy = pct(report.correct, report.total);
        let pos_recall = pct(report.positive_correct, report.positive_total);
        let neg_precision = pct(report.negative_correct, report.negative_total);
        let (min, avg, max) = latency_stats(&report.latencies_ms);
        let rss = report
            .rss_mb
            .map(|mb| format!("{mb:.1}"))
            .unwrap_or_else(|| "n/a".to_string());

        println!(
            "{:<16} {:>7.1}% {:>9.1}% {:>9.1}% {:>9.0} {:>9.0} {:>9.0} {:>9}",
            report.model, accuracy, pos_recall, neg_precision, min, avg, max, rss
        );
    }
}

fn pct(correct: usize, total: usize) -> f64 {
    if total == 0 {
        return 0.0;
    }
    (correct as f64 / total as f64) * 100.0
}

fn latency_stats(latencies_ms: &[f64]) -> (f64, f64, f64) {
    if latencies_ms.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let min = latencies_ms.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = latencies_ms.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let avg = latencies_ms.iter().sum::<f64>() / latencies_ms.len() as f64;
    (min, avg, max)
}
