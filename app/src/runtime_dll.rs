//! Ship ONNX Runtime inside the executable.
//!
//! Microsoft publishes no static ONNX Runtime for Windows, so the DLL is
//! embedded and written to `%LOCALAPPDATA%\hasse\runtime` on first start.
//! The copy is reused when its size matches the embedded one.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

const VERSION: &str = "1.22.0";
static ONNXRUNTIME: &[u8] =
    include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/../runtime/onnxruntime.dll"));

/// Return `preferred` if it exists, otherwise the embedded DLL's path.
pub fn ensure(preferred: Option<&Path>) -> Result<PathBuf> {
    if let Some(path) = preferred {
        if path.is_file() {
            return Ok(path.to_path_buf());
        }
    }

    let dir = base_dir().join("hasse").join("runtime");
    let path = dir.join(format!("onnxruntime-{VERSION}.dll"));
    if std::fs::metadata(&path).map(|m| m.len() == ONNXRUNTIME.len() as u64).unwrap_or(false) {
        return Ok(path);
    }
    if let Err(e) = write_atomic(&dir, &path) {
        eprintln!("runtime: {e:#}; falling back to a temporary directory");
        let fallback = std::env::temp_dir().join(format!("hasse-{}", std::process::id()));
        let path = fallback.join("onnxruntime.dll");
        write_atomic(&fallback, &path).context("extracting onnxruntime.dll")?;
        return Ok(path);
    }
    Ok(path)
}

fn base_dir() -> PathBuf {
    std::env::var_os("LOCALAPPDATA")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir)
}

fn write_atomic(dir: &Path, path: &Path) -> Result<()> {
    std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    let tmp = dir.join(format!("onnxruntime-{VERSION}.{}.tmp", std::process::id()));
    std::fs::write(&tmp, ONNXRUNTIME).with_context(|| format!("writing {}", tmp.display()))?;
    let _ = std::fs::remove_file(path);
    std::fs::rename(&tmp, path).or_else(|_| {
        std::fs::copy(&tmp, path)?;
        let _ = std::fs::remove_file(&tmp);
        Ok::<(), std::io::Error>(())
    })?;
    Ok(())
}
