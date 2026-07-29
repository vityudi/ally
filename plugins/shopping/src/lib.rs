//! shopping plugin: capability stub, installable independently of core.

use ally_security::Permission;
use ally_tools::Tool;

pub struct ShoppingPlugin;

impl ally_plugins::Plugin for ShoppingPlugin {
    fn name(&self) -> &str {
        "shopping"
    }

    fn permissions(&self) -> Vec<Permission> {
        Vec::new()
    }

    fn tools(&self) -> Vec<Box<dyn Tool>> {
        Vec::new()
    }
}
