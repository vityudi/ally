//! Local API surface exposed by the Runtime (HTTP, WebSocket, Unix socket,
//! named pipe or FFI, depending on what the host application needs).

use ally_sdk::Ally;

pub struct ApiServer {
    #[allow(dead_code)]
    ally: Ally,
}

impl ApiServer {
    pub fn new(ally: Ally) -> Self {
        Self { ally }
    }
}
