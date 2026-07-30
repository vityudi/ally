//! Minimal, auditable downloader for the GGUF weights `LlamaCppBackend`
//! needs to run in-process. There is no download mechanism anywhere else
//! in this repo — Ollama's own `ollama pull` was the only precedent, and
//! it lives outside this codebase. Kept deliberately simple: no resume,
//! no dynamic "latest" resolution. A specific Hugging Face repo, file and
//! revision are pinned as constants below, downloaded into the git-ignored
//! `/models` directory (see `models/README.md`), and verified against a
//! hardcoded SHA-256 before being accepted — reproducibility and integrity
//! matter more than convenience for weights that get executed.

use sha2::{Digest, Sha256};
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::ModelError;

/// A single GGUF file pinned to an exact Hugging Face revision, so "the
/// default model" means the same bytes on every machine instead of
/// whatever happens to be at the tip of `main` on download day.
pub struct PinnedGguf {
    pub repo: &'static str,
    pub file: &'static str,
    pub revision: &'static str,
    pub sha256: &'static str,
    pub size_bytes: u64,
}

/// Chat model: matches the `qwen2.5:1.5b` Ollama previously pulled by
/// default (see `sdk::DEFAULT_MODEL`), quantized to Q4_K_M — the standard
/// "good enough, small enough" choice for 1.5B-class models. Revision and
/// checksum captured from the repo's `main` branch / file metadata.
pub const DEFAULT_CHAT_MODEL: PinnedGguf = PinnedGguf {
    repo: "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
    file: "qwen2.5-1.5b-instruct-q4_k_m.gguf",
    revision: "91cad51170dc346986eccefdc2dd33a9da36ead9",
    sha256: "6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e",
    size_bytes: 1_117_320_736,
};

/// Embedding model: mirrors `ollama::DEFAULT_EMBEDDING_MODEL` ("all-minilm"),
/// same underlying `all-MiniLM-L6-v2` weights in GGUF form.
pub const DEFAULT_EMBEDDING_MODEL: PinnedGguf = PinnedGguf {
    repo: "second-state/All-MiniLM-L6-v2-Embedding-GGUF",
    file: "all-MiniLM-L6-v2-Q4_K_M.gguf",
    revision: "544f204f2eaa2d71361ffc74d6df7170285b286a",
    sha256: "2ec4cee28a27a9c973d5f5230930d6ef6e52694bd2bc71be26a9bef5b1d755e6",
    size_bytes: 20_999_104,
};

/// Resolves a [`PinnedGguf`] to a local path inside `models_dir`,
/// downloading and checksum-verifying it first if it isn't already there.
/// Reuses the cached file on subsequent calls without re-downloading.
pub async fn ensure_downloaded(
    gguf: &PinnedGguf,
    models_dir: &Path,
) -> Result<PathBuf, ModelError> {
    let final_path = models_dir.join(gguf.file);
    if final_path.is_file() {
        return Ok(final_path);
    }

    std::fs::create_dir_all(models_dir)
        .map_err(|e| ModelError::Backend(format!("failed to create {}: {e}", models_dir.display())))?;

    let url = format!(
        "https://huggingface.co/{}/resolve/{}/{}",
        gguf.repo, gguf.revision, gguf.file
    );
    let part_path = models_dir.join(format!("{}.part", gguf.file));

    let result = download_to(&url, &part_path, gguf).await;
    if result.is_err() {
        let _ = std::fs::remove_file(&part_path);
    }
    result?;

    std::fs::rename(&part_path, &final_path).map_err(|e| {
        ModelError::Backend(format!("failed to finalize {}: {e}", final_path.display()))
    })?;
    Ok(final_path)
}

async fn download_to(url: &str, part_path: &Path, gguf: &PinnedGguf) -> Result<(), ModelError> {
    let response = reqwest::get(url)
        .await
        .map_err(|e| ModelError::Backend(format!("failed to fetch {url}: {e}")))?
        .error_for_status()
        .map_err(|e| ModelError::Backend(format!("failed to fetch {url}: {e}")))?;

    let mut file = std::fs::File::create(part_path)
        .map_err(|e| ModelError::Backend(format!("failed to create {}: {e}", part_path.display())))?;

    let mut hasher = Sha256::new();
    let mut downloaded: u64 = 0;
    let mut stream = response.bytes_stream();
    use futures_util::StreamExt;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| ModelError::Backend(format!("download interrupted: {e}")))?;
        hasher.update(&chunk);
        downloaded += chunk.len() as u64;
        file.write_all(&chunk)
            .map_err(|e| ModelError::Backend(format!("failed writing {}: {e}", part_path.display())))?;
    }

    if downloaded != gguf.size_bytes {
        return Err(ModelError::Backend(format!(
            "size mismatch for {}: expected {} bytes, got {downloaded}",
            gguf.file, gguf.size_bytes
        )));
    }

    let digest = hex::encode(hasher.finalize());
    if digest != gguf.sha256 {
        return Err(ModelError::Backend(format!(
            "checksum mismatch for {}: expected {}, got {digest}",
            gguf.file, gguf.sha256
        )));
    }
    Ok(())
}
