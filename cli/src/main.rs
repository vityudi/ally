//! Ally CLI: minimal local entry point to the Ally Runtime.

use ally_sdk::Ally;

fn main() {
    let _ally = Ally::new();
    println!("ally-cli: runtime initialized (scaffold — no commands wired yet)");
}
