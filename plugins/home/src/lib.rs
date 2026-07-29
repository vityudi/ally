//! home plugin: capability stub, installable independently of core.

use ally_security::Permission;
use ally_tools::Tool;

pub struct HomePlugin;

impl ally_plugins::Plugin for HomePlugin {
    fn name(&self) -> &str {
        "home"
    }

    fn permissions(&self) -> Vec<Permission> {
        Vec::new()
    }

    fn tools(&self) -> Vec<Box<dyn Tool>> {
        Vec::new()
    }
}
