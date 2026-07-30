//! Interactive REPL to actually converse with whatever model backend the
//! Ally SDK is configured with (default: Ollama, `ally_sdk::DEFAULT_MODEL`).
//!
//! Unlike `examples/kyvo` (a single hardcoded prompt), this keeps the full
//! message history across turns, so you can have a real back-and-forth and
//! see how the model behaves with follow-ups, ambiguous input, or requests
//! that should/shouldn't trigger the finance plugin's tools.
//!
//! Requires a local Ollama daemon (`ollama serve`) with a tool-calling
//! model pulled, e.g. `ollama pull qwen2.5:1.5b`.
//!
//! Run with: `cargo run -p ally-chat`
//! Type `sair` / `exit` / `quit` to leave.

use ally_scheduler::Scheduler;
use ally_sdk::{Ally, ChatMessage, ChatRequest, LoggingHandler, Permission};
use std::io::{self, Write};
use std::sync::{Arc, Mutex};

#[tokio::main]
async fn main() {
    let mut ally = Ally::new();
    ally.on_event(Box::new(LoggingHandler));
    ally.grant_permissions(vec![Permission::Read, Permission::Write]);

    let scheduler = Arc::new(Mutex::new(Scheduler::new()));
    ally
        .install_plugin(Box::new(ally_plugin_finance::FinancePlugin::new(scheduler)))
        .expect("finance plugin should install cleanly");

    println!("ally-chat — installed plugins: {}", ally.installed_plugins().count());
    println!("Digite sua mensagem e aperte Enter. Digite 'sair' para encerrar.\n");

    let mut messages: Vec<ChatMessage> = vec![ChatMessage::system(
        "You are Ally, a helpful personal assistant. Always reply in the same \
         language the user's last message was written in (Portuguese or \
         English) — never mix languages in one reply. Use a tool only when \
         the user's request clearly maps to one of the tools you were given; \
         otherwise just answer directly. If a tool call fails because a \
         required field is missing, ask the user for that specific field \
         instead of giving up. Never state a balance, budget or goal amount \
         from memory — always call finance.get_balance or the relevant tool \
         first. Currently your tools only cover personal finance (expenses, \
         income, balance, budgets, goals, reminders) — for anything else, \
         say plainly that you don't have a tool for that yet instead of \
         pretending you handled it.",
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
                eprintln!(
                    "ally> (erro ao falar com o backend: {err}. \
                     `ollama serve` esta rodando com um modelo pull-ado?)\n"
                );
                messages.pop(); // don't keep a turn that never got a reply
            }
        }
    }

    println!("ate mais!");
}
