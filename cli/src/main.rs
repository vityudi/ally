//! Ally CLI: minimal local entry point to the Ally Runtime.

use ally_sdk::{Ally, Intent, LoggingHandler, MemoryKind, MemoryRecord};

#[tokio::main]
async fn main() {
    let mut ally = Ally::open("ally-cli.sqlite3").expect("failed to open local storage");
    ally.on_event(Box::new(LoggingHandler));

    let plan = ally.handle_intent(Intent {
        name: "finance.schedule_payment".to_string(),
    });
    println!("plan: {plan:?}");

    ally.remember(MemoryRecord {
        id: "note-1".to_string(),
        kind: MemoryKind::Semantic,
        content: "user prefers concise answers".to_string(),
        embedding: None,
    })
    .await
    .expect("failed to store memory");

    let recalled = ally
        .recall(MemoryKind::Semantic)
        .await
        .expect("failed to recall memory");
    println!("recalled semantic memories: {recalled:?}");

    let context = ally.assemble_context(
        "User asked to schedule a credit card payment.".to_string(),
        vec!["user: I need to pay my credit card tomorrow.".to_string()],
        &recalled,
    );
    println!("context: {context:?}");

    ally.end_conversation("finance.schedule_payment".to_string());
}
