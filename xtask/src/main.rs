//! Build helper: fetch the ONNX Runtime library for this OS, then build the app.
//!
//! The app embeds the runtime with `include_bytes!` (`runtime/onnxruntime.dll`
//! on Windows, `runtime/libonnxruntime.dylib` on macOS), so the file must exist
//! before `cargo build`. This task downloads the official release once,
//! verifies its SHA-256, and writes it in place.
//!
//! Build prerequisites are a `default.yaml` at the repository root (gitignored,
//! machine-specific, embedded by the app) and the exported model artifacts in
//! `engine/models/`. The result is `target/release/hasse.exe` on Windows and
//! `target/release/hasse` on macOS.
//!
//!     cargo xtask runtime   # just fetch/verify the runtime
//!     cargo xtask build     # fetch + cargo build -p hasse --release
//!     cargo xtask build --debug

use std::io::{Cursor, Read};
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{bail, Context, Result};
use sha2::{Digest, Sha256};

struct Artifact {
    url: &'static str,
    entry: &'static str,
    sha256: &'static str,
    size: u64,
    output: &'static str,
    format: Format,
}

#[derive(Clone, Copy)]
enum Format {
    Zip,
    TarGz,
}

fn artifact() -> Result<Artifact> {
    if cfg!(target_os = "windows") {
        Ok(Artifact {
            url: "https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-win-x64-1.22.0.zip",
            entry: "onnxruntime-win-x64-1.22.0/lib/onnxruntime.dll",
            sha256: "579b636403983254346a5c1d80bd28f1519cd1e284cd204f8d4ff41f8d711559",
            size: 12_418_080,
            output: "runtime/onnxruntime.dll",
            format: Format::Zip,
        })
    } else if cfg!(target_os = "macos") {
        if !cfg!(target_arch = "aarch64") {
            bail!(
                "this xtask only knows the macOS arm64 runtime; use an Apple Silicon Mac, or add the universal2 artifact"
            );
        }
        Ok(Artifact {
            url: "https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-osx-arm64-1.22.0.tgz",
            entry: "onnxruntime-osx-arm64-1.22.0/lib/libonnxruntime.1.22.0.dylib",
            sha256: "2b885992d3d6fa4130d39ec84a80d7504ff52750027c547bb22c86165f19406a",
            size: 33_481_272,
            output: "runtime/libonnxruntime.dylib",
            format: Format::TarGz,
        })
    } else {
        bail!("unsupported operating system")
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("runtime") => {
            let path = ensure_runtime()?;
            println!("{}", path.display());
            Ok(())
        }
        Some("build") => {
            ensure_runtime()?;
            build(&args[1..])
        }
        Some("help") | Some("-h") | Some("--help") | None => {
            usage();
            Ok(())
        }
        Some(other) => {
            usage();
            bail!("unknown command: {other}");
        }
    }
}

fn usage() {
    eprintln!(
        "usage:\n  cargo xtask runtime        fetch/verify the bundled ONNX Runtime\n  cargo xtask build [--debug]  fetch it, then build the hasse app"
    );
}

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask lives in the workspace root")
        .to_path_buf()
}

fn ensure_runtime() -> Result<PathBuf> {
    let artifact = artifact()?;
    let path = root().join(artifact.output);
    if valid(&path, &artifact) {
        eprintln!("runtime: using {}", path.display());
        return Ok(path);
    }

    eprintln!("runtime: downloading {}", artifact.url);
    let mut response = ureq::get(artifact.url)
        .call()
        .context("download failed")?;
    let bytes = response
        .body_mut()
        .with_config()
        .limit(200 * 1024 * 1024)
        .read_to_vec()
        .context("reading the download")?;
    eprintln!("runtime: downloaded {} MiB", bytes.len() / (1024 * 1024));

    let library = match artifact.format {
        Format::Zip => extract_zip(&bytes, artifact.entry)?,
        Format::TarGz => extract_tar_gz(&bytes, artifact.entry)?,
    };
    if library.len() as u64 != artifact.size || hex(&Sha256::digest(&library)) != artifact.sha256 {
        bail!(
            "{} failed verification (size {}, sha256 {})",
            artifact.entry,
            library.len(),
            hex(&Sha256::digest(&library))
        );
    }

    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, &library).with_context(|| format!("writing {}", tmp.display()))?;
    std::fs::rename(&tmp, &path).with_context(|| format!("writing {}", path.display()))?;
    eprintln!("runtime: wrote {}", path.display());
    Ok(path)
}

fn valid(path: &Path, artifact: &Artifact) -> bool {
    let Ok(data) = std::fs::read(path) else {
        return false;
    };
    data.len() as u64 == artifact.size && hex(&Sha256::digest(&data)) == artifact.sha256
}

fn extract_zip(bytes: &[u8], entry_name: &str) -> Result<Vec<u8>> {
    let mut archive = zip::ZipArchive::new(Cursor::new(bytes)).context("opening the zip")?;
    let mut entry = archive
        .by_name(entry_name)
        .with_context(|| format!("{entry_name} is not in the archive"))?;
    let mut library = Vec::new();
    entry
        .read_to_end(&mut library)
        .context("extracting the runtime")?;
    Ok(library)
}

fn extract_tar_gz(bytes: &[u8], entry_name: &str) -> Result<Vec<u8>> {
    let decoder = flate2::read::GzDecoder::new(Cursor::new(bytes));
    let mut archive = tar::Archive::new(decoder);
    for entry in archive.entries().context("reading the tarball")? {
        let mut entry = entry.context("reading a tarball entry")?;
        if entry.path()?.to_string_lossy() == entry_name {
            let mut library = Vec::new();
            entry
                .read_to_end(&mut library)
                .context("extracting the runtime")?;
            return Ok(library);
        }
    }
    bail!("{entry_name} is not in the archive")
}

fn hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push_str(&format!("{b:02x}"));
    }
    out
}

fn build(flags: &[String]) -> Result<()> {
    let debug = flags.iter().any(|f| f == "--debug");
    let mut cmd = Command::new("cargo");
    cmd.current_dir(root()).args(["build", "-p", "hasse"]);
    if !debug {
        cmd.arg("--release");
    }
    eprintln!("xtask: cargo {}", if debug { "build (debug)" } else { "build --release" });
    let status = cmd.status().context("running cargo build")?;
    if !status.success() {
        bail!("cargo build failed");
    }
    let profile = if debug { "debug" } else { "release" };
    let binary = if cfg!(target_os = "windows") { "hasse.exe" } else { "hasse" };
    println!("{}", root().join("target").join(profile).join(binary).display());
    Ok(())
}
