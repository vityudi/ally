//! Plugin Manager: discovery, permission declaration and lifecycle of
//! installable capabilities (finance, calendar, health, ...).

use ally_events::{Event, EventBus};
use ally_security::Permission;
use ally_tools::Tool;
use thiserror::Error;

pub trait Plugin: Send + Sync {
    fn name(&self) -> &str;
    fn permissions(&self) -> Vec<Permission>;
    fn tools(&self) -> Vec<Box<dyn Tool>>;
}

/// Failure installing a `Plugin` into a `PluginManager`.
#[derive(Debug, Error)]
pub enum PluginError {
    /// A tool declared `required_permissions()` that its owning plugin
    /// never declared via `Plugin::permissions()`. A plugin's declared
    /// capability list must be a superset of what its tools actually
    /// request, so this is rejected at install time rather than only
    /// caught (or silently allowed) at tool-execution time.
    #[error(
        "plugin '{plugin}' tool '{tool}' requires undeclared permission {permission:?}"
    )]
    UndeclaredPermission {
        plugin: String,
        tool: String,
        permission: Permission,
    },
}

#[derive(Default)]
pub struct PluginManager {
    plugins: Vec<Box<dyn Plugin>>,
}

impl PluginManager {
    pub fn new() -> Self {
        Self { plugins: Vec::new() }
    }

    /// Installs a plugin after checking that every tool it exposes only
    /// requires permissions the plugin itself declares.
    pub fn install(
        &mut self,
        plugin: Box<dyn Plugin>,
        events: &EventBus,
    ) -> Result<(), PluginError> {
        let declared = plugin.permissions();
        for tool in plugin.tools() {
            for permission in tool.required_permissions() {
                if !declared.contains(&permission) {
                    return Err(PluginError::UndeclaredPermission {
                        plugin: plugin.name().to_string(),
                        tool: tool.name().to_string(),
                        permission,
                    });
                }
            }
        }

        events.publish(Event::PluginInstalled {
            plugin_name: plugin.name().to_string(),
        });
        self.plugins.push(plugin);
        Ok(())
    }

    pub fn installed(&self) -> impl Iterator<Item = &str> {
        self.plugins.iter().map(|p| p.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use serde_json::{json, Value};

    struct FakeTool {
        permissions: Vec<Permission>,
    }

    #[async_trait]
    impl Tool for FakeTool {
        fn name(&self) -> &str {
            "fake.tool"
        }

        fn description(&self) -> &str {
            "test tool"
        }

        fn parameters_schema(&self) -> Value {
            json!({})
        }

        fn required_permissions(&self) -> Vec<Permission> {
            self.permissions.clone()
        }

        async fn execute(&self, _input: Value) -> Result<Value, ally_tools::ToolError> {
            Ok(json!({}))
        }
    }

    struct FakePlugin {
        permissions: Vec<Permission>,
        tool_permissions: Vec<Permission>,
    }

    impl Plugin for FakePlugin {
        fn name(&self) -> &str {
            "fake"
        }

        fn permissions(&self) -> Vec<Permission> {
            self.permissions.clone()
        }

        fn tools(&self) -> Vec<Box<dyn Tool>> {
            vec![Box::new(FakeTool {
                permissions: self.tool_permissions.clone(),
            })]
        }
    }

    #[test]
    fn install_accepts_declared_permissions() {
        let mut manager = PluginManager::new();
        let events = EventBus::new();
        let plugin = FakePlugin {
            permissions: vec![Permission::Write],
            tool_permissions: vec![Permission::Write],
        };

        assert!(manager.install(Box::new(plugin), &events).is_ok());
        assert_eq!(manager.installed().collect::<Vec<_>>(), vec!["fake"]);
    }

    #[test]
    fn install_rejects_undeclared_permissions() {
        let mut manager = PluginManager::new();
        let events = EventBus::new();
        let plugin = FakePlugin {
            permissions: vec![Permission::Write],
            tool_permissions: vec![Permission::Network],
        };

        let err = manager.install(Box::new(plugin), &events).unwrap_err();
        assert!(matches!(
            err,
            PluginError::UndeclaredPermission {
                permission: Permission::Network,
                ..
            }
        ));
        assert_eq!(manager.installed().count(), 0);
    }
}
