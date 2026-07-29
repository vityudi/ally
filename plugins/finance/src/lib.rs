//! finance plugin: capability stub, installable independently of core.

use ally_security::Permission;
use ally_tools::Tool;

pub struct FinancePlugin;

impl ally_plugins::Plugin for FinancePlugin {
    fn name(&self) -> &str {
        "finance"
    }

    fn permissions(&self) -> Vec<Permission> {
        Vec::new()
    }

    fn tools(&self) -> Vec<Box<dyn Tool>> {
        Vec::new()
    }
}
