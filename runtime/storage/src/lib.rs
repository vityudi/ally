//! Storage Layer: pluggable persistence, independent of any specific database.
//!
//! Every Runtime module that needs to persist data (Memory, Scheduler, ...)
//! depends on the [`Storage`] trait, never on a concrete database. SQLite is
//! the Phase 2 default because it is embedded and local-first (no server
//! process, works on desktop, mobile and embedded targets); other backends
//! (Postgres, RocksDB, cloud sync) can implement the same trait later
//! without touching callers.

use async_trait::async_trait;
use rusqlite::{params, Connection, OptionalExtension};
use std::sync::Mutex;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum StorageError {
    #[error("key not found: {0}")]
    NotFound(String),
    #[error("backend error: {0}")]
    Backend(String),
}

#[async_trait]
pub trait Storage: Send + Sync {
    async fn get(&self, key: &str) -> Result<Option<Vec<u8>>, StorageError>;
    async fn set(&self, key: &str, value: Vec<u8>) -> Result<(), StorageError>;
    async fn delete(&self, key: &str) -> Result<(), StorageError>;
}

/// SQLite-backed key/value [`Storage`]. A single `kv` table holds every
/// record; higher-level modules (e.g. the Memory Engine) are responsible
/// for namespacing their own keys and for any indexing they need.
pub struct SqliteStorage {
    conn: Mutex<Connection>,
}

impl SqliteStorage {
    /// Opens (or creates) a SQLite database file at `path`.
    pub fn open(path: &str) -> Result<Self, StorageError> {
        let conn = Connection::open(path).map_err(|e| StorageError::Backend(e.to_string()))?;
        Self::from_connection(conn)
    }

    /// Opens a throwaway in-memory database. Data does not survive past
    /// the process — useful for tests and short-lived demos.
    pub fn open_in_memory() -> Result<Self, StorageError> {
        let conn =
            Connection::open_in_memory().map_err(|e| StorageError::Backend(e.to_string()))?;
        Self::from_connection(conn)
    }

    fn from_connection(conn: Connection) -> Result<Self, StorageError> {
        conn.execute(
            "CREATE TABLE IF NOT EXISTS kv (
                key   TEXT PRIMARY KEY,
                value BLOB NOT NULL
            )",
            [],
        )
        .map_err(|e| StorageError::Backend(e.to_string()))?;
        Ok(Self { conn: Mutex::new(conn) })
    }
}

#[async_trait]
impl Storage for SqliteStorage {
    async fn get(&self, key: &str) -> Result<Option<Vec<u8>>, StorageError> {
        let conn = self.conn.lock().expect("sqlite connection poisoned");
        conn.query_row("SELECT value FROM kv WHERE key = ?1", [key], |row| row.get(0))
            .optional()
            .map_err(|e| StorageError::Backend(e.to_string()))
    }

    async fn set(&self, key: &str, value: Vec<u8>) -> Result<(), StorageError> {
        let conn = self.conn.lock().expect("sqlite connection poisoned");
        conn.execute(
            "INSERT INTO kv (key, value) VALUES (?1, ?2)
             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            params![key, value],
        )
        .map_err(|e| StorageError::Backend(e.to_string()))?;
        Ok(())
    }

    async fn delete(&self, key: &str) -> Result<(), StorageError> {
        let conn = self.conn.lock().expect("sqlite connection poisoned");
        conn.execute("DELETE FROM kv WHERE key = ?1", [key])
            .map_err(|e| StorageError::Backend(e.to_string()))?;
        Ok(())
    }
}
