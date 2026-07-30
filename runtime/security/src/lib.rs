//! Security Layer: capability-based permission declaration and enforcement.

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// A capability a `Tool` may require and an application may grant, per the
/// Security Layer contract in `docs/ARCHITECTURE.md`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Permission {
    Read,
    Write,
    Network,
    Filesystem,
    Microphone,
    Camera,
}

#[derive(Debug, Error)]
pub enum SecurityError {
    #[error("permission denied: {0:?}")]
    Denied(Permission),
}

/// The permissions an application has granted to the Runtime for the
/// current session. Checked by `ally_tools::ToolOrchestrator::execute`
/// before every tool call.
#[derive(Default, Clone)]
pub struct PermissionSet {
    granted: Vec<Permission>,
}

impl PermissionSet {
    pub fn new(granted: Vec<Permission>) -> Self {
        Self { granted }
    }

    pub fn require(&self, permission: Permission) -> Result<(), SecurityError> {
        if self.granted.contains(&permission) {
            Ok(())
        } else {
            Err(SecurityError::Denied(permission))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn require_succeeds_when_granted() {
        let set = PermissionSet::new(vec![Permission::Write]);
        assert!(set.require(Permission::Write).is_ok());
    }

    #[test]
    fn require_fails_when_not_granted() {
        let set = PermissionSet::new(vec![Permission::Write]);
        let err = set.require(Permission::Network).unwrap_err();
        assert!(matches!(err, SecurityError::Denied(Permission::Network)));
    }
}
