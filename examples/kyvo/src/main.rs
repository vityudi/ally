//! Example application ("Kyvo") showing how a host app talks only to the
//! Ally SDK — never to a model, memory store or tool directly.
//!
//! Mirrors the finance walkthrough in `docs/ARCHITECTURE.md`: the user
//! asks to pay a credit card, the model decides it needs the
//! `finance.create_reminder` tool, the SDK executes it through the
//! permission-checked Tool Orchestrator (the model never touches it
//! directly), and the model turns the tool result into a confirmation.
//!
//! Requires a local Ollama daemon (`ollama serve`) with a small
//! tool-calling-capable model pulled, e.g.:
//!   ollama pull qwen2.5:1.5b
//!
//! Tested against qwen2.5:0.5b and qwen2.5:1.5b: the 0.5b model reports
//! "tools" capability but never actually emits a tool call for this
//! prompt, even with a directive system message — it's simply too small.
//! 1.5b calls the tool reliably. `ally_sdk::DEFAULT_MODEL` matches this
//! finding; swap it via `Ally::with_model` for anything else.

use ally_scheduler::Scheduler;
use ally_sdk::{Ally, ChatMessage, ChatRequest, LoggingHandler, Permission};
use std::sync::{Arc, Mutex};

#[tokio::main]
async fn main() {
    let mut ally = Ally::new();
    ally.on_event(Box::new(LoggingHandler));
    ally.grant_permissions(vec![Permission::Read, Permission::Write]);

    let scheduler = Arc::new(Mutex::new(Scheduler::new()));
    ally.install_plugin(Box::new(ally_plugin_finance::FinancePlugin::new(scheduler)))
        .expect("finance plugin should install cleanly");

    println!(
        "kyvo-example running on Ally SDK (installed plugins: {})",
        ally.installed_plugins().count()
    );

    let request = ChatRequest {
        messages: vec![
            ChatMessage::system(
                "You are Kyvo, a personal finance assistant. When the user mentions a \
                 bill or payment due, you MUST call the finance.create_reminder \
                 function immediately with your best guess for note and due date. Do \
                 not ask clarifying questions first.",
            ),
            ChatMessage::user("I need to pay my credit card tomorrow."),
        ],
        tools: ally.tool_specs(),
    };

    match ally.chat(request).await {
        Ok(response) => println!("kyvo: {}", response.message.content),
        Err(err) => {
            eprintln!(
                "kyvo: could not reach the Model Runtime ({err}). \
                 Is `ollama serve` running with a model pulled?"
            );
        }
    }
}
