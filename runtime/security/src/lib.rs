//! Security Layer: capability-based permission declaration and enforcement.

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Permission {
    Filesystem,
    Network,
    Camera,
    Microphone,
    Location,
    Contacts,
    Notifications,
}

#[derive(Debug, Error)]
pub enum SecurityError {
    #[error("permission denied: {0:?}")]
    Denied(Permission),
}

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
