//! calendar plugin: capability stub, installable independently of core.

use ally_security::Permission;
use ally_tools::Tool;

pub struct CalendarPlugin;

impl ally_plugins::Plugin for CalendarPlugin {
    fn name(&self) -> &str {
        "calendar"
    }

    fn permissions(&self) -> Vec<Permission> {
        Vec::new()
    }

    fn tools(&self) -> Vec<Box<dyn Tool>> {
        Vec::new()
    }
}
