//! Tool Orchestrator: permission-aware execution of actions in the outside world.
//! The Language Model never accesses these resources directly.

use ally_security::{Permission, PermissionSet};
use async_trait::async_trait;
use serde_json::Value;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ToolError {
    #[error(transparent)]
    Security(#[from] ally_security::SecurityError),
    #[error("execution failed: {0}")]
    Execution(String),
}

#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn required_permissions(&self) -> Vec<Permission>;
    async fn execute(&self, input: Value) -> Result<Value, ToolError>;
}

#[derive(Default)]
pub struct ToolOrchestrator {
    tools: Vec<Box<dyn Tool>>,
}

impl ToolOrchestrator {
    pub fn new() -> Self {
        Self { tools: Vec::new() }
    }

    pub fn register(&mut self, tool: Box<dyn Tool>) {
        self.tools.push(tool);
    }

    pub async fn execute(
        &self,
        tool_name: &str,
        input: Value,
        granted: &PermissionSet,
    ) -> Result<Value, ToolError> {
        let tool = self
            .tools
            .iter()
            .find(|t| t.name() == tool_name)
            .ok_or_else(|| ToolError::Execution(format!("unknown tool: {tool_name}")))?;

        for permission in tool.required_permissions() {
            granted.require(permission)?;
        }

        tool.execute(input).await
    }
}
