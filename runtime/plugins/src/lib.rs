//! Plugin Manager: discovery, permission declaration and lifecycle of
//! installable capabilities (finance, calendar, health, ...).

use ally_security::Permission;
use ally_tools::Tool;

pub trait Plugin: Send + Sync {
    fn name(&self) -> &str;
    fn permissions(&self) -> Vec<Permission>;
    fn tools(&self) -> Vec<Box<dyn Tool>>;
}

#[derive(Default)]
pub struct PluginManager {
    plugins: Vec<Box<dyn Plugin>>,
}

impl PluginManager {
    pub fn new() -> Self {
        Self { plugins: Vec::new() }
    }

    pub fn install(&mut self, plugin: Box<dyn Plugin>) {
        self.plugins.push(plugin);
    }

    pub fn installed(&self) -> impl Iterator<Item = &str> {
        self.plugins.iter().map(|p| p.name())
    }
}
