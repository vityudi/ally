//! Generates the synthetic planning/intent dataset consumed by
//! `benchmarks/eval-harness`. Template-based and fully deterministic (no
//! RNG dependency) — regenerating the file is idempotent.
//!
//! Usage:
//!   cargo run -p dataset-gen

use serde::Serialize;
use std::fs;
use std::path::PathBuf;

#[derive(Serialize)]
struct DatasetExample {
    id: String,
    intent: String,
    prompt: String,
    /// `None` means no tool call is expected — the prompt is off-topic
    /// and a good model should just respond conversationally.
    expected_tool: Option<String>,
}

const BILLS: &[&str] = &[
    "credit card",
    "electricity bill",
    "rent",
    "phone bill",
    "internet bill",
    "water bill",
];

const DUE_PHRASES: &[&str] = &["today", "tomorrow", "next Friday", "in 3 days", "on the 5th"];

const TEMPLATES: &[&str] = &[
    "I need to pay my {bill} {due}.",
    "Remind me to pay the {bill} {due}.",
    "Can you set up a reminder for my {bill} due {due}?",
    "Don't let me forget the {bill}, it's due {due}.",
    "Please schedule a payment reminder for {bill} {due}.",
];

const OFF_TOPIC_PROMPTS: &[&str] = &[
    "What's the weather like?",
    "Tell me a joke.",
    "What's 12 times 8?",
    "Who wrote Romeo and Juliet?",
    "What's the capital of Japan?",
    "How do I boil an egg?",
    "Recommend me a good book.",
    "What time zone is Berlin in?",
    "Translate 'good morning' to Spanish.",
    "What's the tallest mountain in the world?",
];

fn main() {
    let mut examples = Vec::new();

    let mut positive_count = 0;
    for bill in BILLS {
        for due in DUE_PHRASES {
            for template in TEMPLATES {
                positive_count += 1;
                let prompt = template.replace("{bill}", bill).replace("{due}", due);
                examples.push(DatasetExample {
                    id: format!("finance.schedule_payment-{positive_count:03}"),
                    intent: "finance.schedule_payment".to_string(),
                    prompt,
                    expected_tool: Some("finance.create_reminder".to_string()),
                });
            }
        }
    }

    for (i, prompt) in OFF_TOPIC_PROMPTS.iter().enumerate() {
        examples.push(DatasetExample {
            id: format!("none-{:03}", i + 1),
            intent: "none".to_string(),
            prompt: prompt.to_string(),
            expected_tool: None,
        });
    }

    let negative_count = OFF_TOPIC_PROMPTS.len();

    let out_dir = workspace_root().join("datasets").join("planning-intent");
    fs::create_dir_all(&out_dir).expect("failed to create datasets/planning-intent");
    let out_path = out_dir.join("finance.jsonl");

    let mut body = String::new();
    for example in &examples {
        body.push_str(&serde_json::to_string(example).expect("example should serialize"));
        body.push('\n');
    }
    fs::write(&out_path, body).expect("failed to write dataset file");

    println!(
        "wrote {} examples ({positive_count} positive, {negative_count} negative) to {}",
        examples.len(),
        out_path.display()
    );
}

/// `tools/dataset-gen` -> workspace root is two levels up from the crate manifest.
fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("tools/dataset-gen should be two levels below the workspace root")
        .to_path_buf()
}
