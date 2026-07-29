//! health plugin: capability stub, installable independently of core.

use ally_security::Permission;
use ally_tools::Tool;

pub struct HealthPlugin;

impl ally_plugins::Plugin for HealthPlugin {
    fn name(&self) -> &str {
        "health"
    }

    fn permissions(&self) -> Vec<Permission> {
        Vec::new()
    }

    fn tools(&self) -> Vec<Box<dyn Tool>> {
        Vec::new()
    }
}
