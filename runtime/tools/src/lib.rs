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

/// Result of attempting to deterministically pull `execute` arguments out of
/// a matched trigger phrase's raw message. See `Tool::extract_args`.
#[derive(Debug, Clone, PartialEq)]
pub enum ExtractOutcome {
    /// The tool has no extractor; the orchestrator should call `execute`
    /// with an empty object, exactly as it always has for zero-argument
    /// tools like a plain balance check.
    NotAttempted,
    /// The extractor found valid arguments to call `execute` with.
    Extracted(Value),
    /// The phrase matched, but a *required* argument (e.g. an amount)
    /// could not be found in the message. The match must be abandoned
    /// entirely rather than calling `execute` with missing/guessed data —
    /// this matters most for write tools, where an empty amount would
    /// otherwise register a bogus transaction.
    Failed,
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
    /// whether/which tool to call. A small local model's decision of
    /// *whether* to call a tool gets markedly less reliable as a
    /// conversation grows multi-turn, even though the tool itself works
    /// fine once called — see `docs/PRINCIPLES.md` on preferring
    /// determinism. Tools whose `parameters_schema` requires no arguments
    /// can rely on the default `extract_args` (always `NotAttempted`,
    /// resolved to `{}`); tools that need arguments (amounts, categories,
    /// dates) should also override `extract_args` to pull them
    /// deterministically out of the message, or a match will still be
    /// abandoned in favor of the model (see `ExtractOutcome`).
    fn trigger_phrases(&self) -> Vec<&str> {
        Vec::new()
    }

    /// Deterministically extracts `execute` arguments from the raw message
    /// that matched one of this tool's `trigger_phrases`. Defaults to
    /// `NotAttempted`, meaning "call with `{}`" — the same behavior as
    /// before this hook existed. Override for tools whose schema has
    /// arguments that can be reliably parsed out of the message (a period,
    /// a category, an amount); return `Failed` when a *required* argument
    /// isn't found so the orchestrator falls through to the model instead
    /// of calling `execute` with invalid input.
    fn extract_args(&self, _message: &str) -> ExtractOutcome {
        ExtractOutcome::NotAttempted
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

    /// Like `match_trigger_phrase`, but also resolves the arguments to call
    /// the matched tool with, via `Tool::extract_args`. Returns `None` both
    /// when no phrase matches and when a phrase matches but extraction
    /// `Failed` (required arguments missing) — in either case the caller
    /// should fall through to the model instead of executing the tool.
    pub fn match_with_args(&self, message: &str) -> Option<(&dyn Tool, Value)> {
        let tool = self.match_trigger_phrase(message)?;
        match tool.extract_args(message) {
            ExtractOutcome::NotAttempted => Some((tool, serde_json::json!({}))),
            ExtractOutcome::Extracted(args) => Some((tool, args)),
            ExtractOutcome::Failed => None,
        }
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

    struct ExtractingTool(ExtractOutcome);

    #[async_trait]
    impl Tool for ExtractingTool {
        fn name(&self) -> &str {
            "test.extracting"
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
            vec!["quanto gastei"]
        }
        fn extract_args(&self, _message: &str) -> ExtractOutcome {
            self.0.clone()
        }
    }

    #[test]
    fn match_with_args_defaults_to_empty_object_when_not_attempted() {
        let mut orchestrator = ToolOrchestrator::new();
        orchestrator.register(Box::new(ExtractingTool(ExtractOutcome::NotAttempted)));

        let (tool, args) = orchestrator.match_with_args("quanto gastei hoje?").unwrap();
        assert_eq!(tool.name(), "test.extracting");
        assert_eq!(args, serde_json::json!({}));
    }

    #[test]
    fn match_with_args_uses_extracted_arguments() {
        let mut orchestrator = ToolOrchestrator::new();
        orchestrator.register(Box::new(ExtractingTool(ExtractOutcome::Extracted(
            serde_json::json!({ "category": "transport" }),
        ))));

        let (_, args) = orchestrator.match_with_args("quanto gastei em transporte").unwrap();
        assert_eq!(args, serde_json::json!({ "category": "transport" }));
    }

    #[test]
    fn match_with_args_abandons_match_on_extraction_failure() {
        let mut orchestrator = ToolOrchestrator::new();
        orchestrator.register(Box::new(ExtractingTool(ExtractOutcome::Failed)));

        assert!(orchestrator.match_with_args("quanto gastei em algo confuso").is_none());
    }
}
