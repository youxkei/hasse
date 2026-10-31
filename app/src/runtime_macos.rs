//! The bundled ONNX Runtime dylib for macOS.
//!
//! The official macOS release ships only a dylib, so it is embedded in the
//! executable and written to Application Support on first start.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

static ONNXRUNTIME: &[u8] =
    include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/../runtime/libonnxruntime.dylib"));

/// Return `preferred` if it exists, otherwise the embedded dylib's path.
pub fn ensure(preferred: Option<&Path>) -> Result<PathBuf> {
    if let Some(path) = preferred {
        if path.is_file() {
            return Ok(path.to_path_buf());
        }
    }

    let dir = support_dir()?.join("runtime");
    let path = dir.join("libonnxruntime.1.22.0.dylib");
    if std::fs::metadata(&path).map(|m| m.len() == ONNXRUNTIME.len() as u64).unwrap_or(false) {
        return Ok(path);
    }
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    let tmp = dir.join(format!("libonnxruntime.{}.tmp", std::process::id()));
    std::fs::write(&tmp, ONNXRUNTIME).with_context(|| format!("writing {}", tmp.display()))?;
    std::fs::rename(&tmp, &path).with_context(|| format!("writing {}", path.display()))?;
    Ok(path)
}

/// `~/Library/Application Support/hasse`, where models and the runtime live.
pub fn support_dir() -> Result<PathBuf> {
    let home = std::env::var_os("HOME").context("HOME is not set")?;
    Ok(PathBuf::from(home).join("Library/Application Support/hasse"))
}
