//! Interactive REPL to actually converse with whatever model backend the
//! Ally SDK is configured with (default: `LlamaCppBackend`, fully
//! in-process — no external daemon).
//!
//! Unlike `examples/kyvo` (a single hardcoded prompt), this keeps the full
//! message history across turns, so you can have a real back-and-forth and
//! see how the model behaves with follow-ups, ambiguous input, or requests
//! that should/shouldn't trigger the finance plugin's tools.
//!
//! No setup required: the first message triggers a one-time download
//! (~1.1 GB) of the pinned default GGUF weights into `models/`, then loads
//! them — see `ally_sdk::DEFAULT_MODELS_DIR` and
//! `ally_models::LlamaCppBackend::lazy_default`. That first call is slow;
//! subsequent runs reuse the cached weights and only pay the (much
//! shorter) model-load time. To use a bigger model via an existing
//! `ollama serve` instead, call
//! `ally.with_model(Arc::new(ally_models::OllamaBackend::new("...")))`
//! before the REPL loop starts.
//!
//! Run with: `cargo run -p ally-chat`
//! Type `sair` / `exit` / `quit` to leave.
//!
//! Memory persists across runs by default, in `ally-chat.sqlite3` in the
//! current directory. Pass `--reset` (or delete the file yourself) to start
//! from a throwaway in-memory store instead, wiping everything remembered
//! so far for that run only — the file on disk is left untouched.

use ally_scheduler::Scheduler;
use ally_sdk::{Ally, ChatMessage, ChatRequest, LoggingHandler, Permission};
use std::io::{self, Write};
use std::sync::{Arc, Mutex};

const STORAGE_PATH: &str = "ally-chat.sqlite3";

#[tokio::main]
async fn main() {
    let reset = std::env::args().any(|a| a == "--reset");
    let mut ally = if reset {
        Ally::new()
    } else {
        Ally::open(STORAGE_PATH).expect("failed to open local storage")
    };
    ally.on_event(Box::new(LoggingHandler));
    ally.grant_permissions(vec![Permission::Read, Permission::Write]);

    let scheduler = Arc::new(Mutex::new(Scheduler::new()));
    ally
        .install_plugin(Box::new(ally_plugin_finance::FinancePlugin::new(scheduler)))
        .expect("finance plugin should install cleanly");

    println!("ally-chat — installed plugins: {}", ally.installed_plugins().count());
    println!(
        "memoria: {}",
        if reset { "temporaria (--reset)".to_string() } else { format!("persistente ({STORAGE_PATH})") }
    );
    println!("modelo: {}", ally.model_id());
    println!("Digite sua mensagem e aperte Enter. Digite 'sair' para encerrar.\n");

    // Deliberately short: a stacked, multi-clause rulebook reliably makes
    // qwen2.5:1.5b return a completely empty response (no text, no tool
    // call) instead of partially following the rules — confirmed by
    // bisecting the previous, longer prompt sentence by sentence. Small
    // local models need a handful of essential rules, not a policy
    // document; add more only after re-verifying tool calls still fire.
    let mut messages: Vec<ChatMessage> = vec![ChatMessage::system(
        "You are Ally, a personal assistant. Reply in the user's language \
         (Portuguese or English). If a request already gives you everything \
         a tool needs, call it now instead of asking to confirm. Never state \
         a balance or amount from memory — always call the matching tool.",
    )];

    loop {
        print!("voce> ");
        io::stdout().flush().ok();

        let mut input = String::new();
        if io::stdin().read_line(&mut input).unwrap_or(0) == 0 {
            break; // EOF (e.g. piped input ran out)
        }
        let input = input.trim();
        if input.is_empty() {
            continue;
        }
        if matches!(input, "sair" | "exit" | "quit") {
            break;
        }

        messages.push(ChatMessage::user(input));

        let request = ChatRequest { messages: messages.clone(), tools: ally.tool_specs() };

        match ally.chat(request).await {
            Ok(response) => {
                if response.message.content.trim().is_empty() {
                    // The model kept requesting tools without ever settling
                    // on a final text reply, hit Ally::chat's internal tool
                    // round cap, and returned empty content — a real model
                    // limitation (small models juggling several tools), not
                    // a network error, so it gets its own message.
                    println!(
                        "ally> (o modelo nao chegou a uma resposta final depois de tentar \
                         usar ferramentas repetidas vezes — tente reformular de forma mais \
                         direta)\n"
                    );
                } else {
                    println!("ally> {}\n", response.message.content);
                }
                messages.push(response.message);
            }
            Err(err) => {
                eprintln!("ally> (erro ao falar com o backend: {err})\n");
                messages.pop(); // don't keep a turn that never got a reply
            }
        }
    }

    println!("ate mais!");
}
