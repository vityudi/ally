//! Tool Orchestrator: permission-aware execution of actions in the outside world.
//! The Language Model never accesses these resources directly.

use ally_events::{Event, EventBus};
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
    /// Human-readable description handed to a Language Model so it can
    /// decide when to request this tool (see `ally_models::ToolSpec`).
    fn description(&self) -> &str;
    /// JSON Schema for this tool's `execute` input.
    fn parameters_schema(&self) -> Value;
    fn required_permissions(&self) -> Vec<Permission>;
    async fn execute(&self, input: Value) -> Result<Value, ToolError>;

    /// Case-insensitive substring phrases that mean this tool should be
    /// called directly, bypassing the Language Model's decision of
    /// whether/which tool to call. Reserved for tools whose
    /// `parameters_schema` requires no arguments (read-only queries): a
    /// small local model's decision of *whether* to call a tool gets
    /// markedly less reliable as a conversation grows multi-turn, even
    /// though the tool itself works fine once called — see
    /// `docs/PRINCIPLES.md` on preferring determinism. A tool needing no
    /// argument extraction can be triggered by plain keyword matching
    /// instead, with no natural-language-understanding risk. Tools that
    /// need arguments (amounts, categories, dates) should leave this
    /// empty — those still go through the model.
    fn trigger_phrases(&self) -> Vec<&str> {
        Vec::new()
    }
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

    /// Every registered tool, in registration order — used to build the
    /// `ToolSpec` list handed to a Model Runtime backend.
    pub fn list(&self) -> impl Iterator<Item = &dyn Tool> {
        self.tools.iter().map(|t| t.as_ref())
    }

    pub async fn execute(
        &self,
        tool_name: &str,
        input: Value,
        granted: &PermissionSet,
        events: &EventBus,
    ) -> Result<Value, ToolError> {
        let tool = self
            .tools
            .iter()
            .find(|t| t.name() == tool_name)
            .ok_or_else(|| ToolError::Execution(format!("unknown tool: {tool_name}")))?;

        for permission in tool.required_permissions() {
            granted.require(permission)?;
        }

        let result = tool.execute(input).await?;
        events.publish(Event::ToolExecuted {
            tool_name: tool_name.to_string(),
        });
        Ok(result)
    }

    /// Finds the first registered tool whose `trigger_phrases()` matches
    /// `message` (case-insensitive substring), if any. See
    /// `Tool::trigger_phrases` for why this exists.
    pub fn match_trigger_phrase(&self, message: &str) -> Option<&dyn Tool> {
        let message = message.to_lowercase();
        self.tools.iter().map(|t| t.as_ref()).find(|tool| {
            tool.trigger_phrases()
                .iter()
                .any(|phrase| message.contains(&phrase.to_lowercase()))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FixedTool;

    #[async_trait]
    impl Tool for FixedTool {
        fn name(&self) -> &str {
            "test.fixed"
        }
        fn description(&self) -> &str {
            "test tool"
        }
        fn parameters_schema(&self) -> Value {
            serde_json::json!({ "type": "object", "properties": {} })
        }
        fn required_permissions(&self) -> Vec<Permission> {
            Vec::new()
        }
        async fn execute(&self, _input: Value) -> Result<Value, ToolError> {
            Ok(serde_json::json!({ "ok": true }))
        }
        fn trigger_phrases(&self) -> Vec<&str> {
            vec!["quanto gastei", "how much did i spend"]
        }
    }

    #[test]
    fn match_trigger_phrase_is_case_insensitive_and_substring() {
        let mut orchestrator = ToolOrchestrator::new();
        orchestrator.register(Box::new(FixedTool));

        assert!(orchestrator.match_trigger_phrase("Quanto Gastei hoje?").is_some());
        assert!(orchestrator.match_trigger_phrase("well, HOW MUCH DID I SPEND today").is_some());
        assert!(orchestrator.match_trigger_phrase("what's my balance").is_none());
    }

    #[test]
    fn tools_without_trigger_phrases_never_match() {
        struct Untriggered;

        #[async_trait]
        impl Tool for Untriggered {
            fn name(&self) -> &str {
                "test.untriggered"
            }
            fn description(&self) -> &str {
                "test tool"
            }
            fn parameters_schema(&self) -> Value {
                serde_json::json!({ "type": "object", "properties": {} })
            }
            fn required_permissions(&self) -> Vec<Permission> {
                Vec::new()
            }
            async fn execute(&self, _input: Value) -> Result<Value, ToolError> {
                Ok(serde_json::json!({}))
            }
        }

        let mut orchestrator = ToolOrchestrator::new();
        orchestrator.register(Box::new(Untriggered));

        assert!(orchestrator.match_trigger_phrase("anything at all").is_none());
    }
}
