//! Example application ("Kyvo") showing how a host app talks only to the
//! Ally SDK — never to a model, memory store or tool directly.

use ally_sdk::Ally;

fn main() {
    let ally = Ally::new();
    println!(
        "kyvo-example running on Ally SDK (installed plugins: {})",
        ally.plugins.installed().count()
    );
}
