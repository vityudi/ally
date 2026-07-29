//! knowledge plugin: capability stub, installable independently of core.

use ally_security::Permission;
use ally_tools::Tool;

pub struct KnowledgePlugin;

impl ally_plugins::Plugin for KnowledgePlugin {
    fn name(&self) -> &str {
        "knowledge"
    }

    fn permissions(&self) -> Vec<Permission> {
        Vec::new()
    }

    fn tools(&self) -> Vec<Box<dyn Tool>> {
        Vec::new()
    }
}
