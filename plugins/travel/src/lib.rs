//! travel plugin: capability stub, installable independently of core.

use ally_security::Permission;
use ally_tools::Tool;

pub struct TravelPlugin;

impl ally_plugins::Plugin for TravelPlugin {
    fn name(&self) -> &str {
        "travel"
    }

    fn permissions(&self) -> Vec<Permission> {
        Vec::new()
    }

    fn tools(&self) -> Vec<Box<dyn Tool>> {
        Vec::new()
    }
}
