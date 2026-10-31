use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::Deserialize;

/// Defaults embedded at build time. `default.yaml` is machine-specific (it
/// names the microphone) and gitignored; it must exist when the app is built.
const DEFAULT_YAML: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../default.yaml"));

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub hotkey: HotkeyConfig,
    pub audio: AudioConfig,
    pub models: ModelsConfig,
    pub paste: PasteConfig,
    /// Directory used to resolve relative paths. With no config file this is
    /// the executable's directory, so the app runs from anywhere.
    #[serde(skip)]
    pub base_dir: PathBuf,
    #[serde(skip)]
    pub config_file: Option<PathBuf>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct HotkeyConfig {
    /// Push-to-talk key. Held while speaking.
    pub key: String,
    /// Swallow the key so the focused app never sees it.
    pub swallow: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AudioConfig {
    /// Empty selects the system default input device.
    pub device: String,
    /// Use only the first channel instead of averaging every channel.
    pub first_channel_only: bool,
    /// Recordings longer than this are cut at quiet points.
    pub max_seconds: f32,
    /// Reject taps shorter than this.
    pub min_seconds: f32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelsConfig {
    /// Directory holding frontend.onnx, decoder.safetensors and friends.
    /// `None` searches next to the executable, then the usual dev layout.
    pub dir: Option<PathBuf>,
    /// ONNX Runtime DLL. `None` uses the copy embedded in the executable.
    pub onnxruntime: Option<PathBuf>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PasteConfig {
    /// Time between Ctrl+V and restoring the clipboard.
    pub delay_ms: u64,
    /// Put the previous clipboard contents back.
    pub restore_clipboard: bool,
}

impl Default for Config {
    fn default() -> Self {
        let mut config: Config = serde_yaml::from_str(DEFAULT_YAML)
            .expect("default.yaml is valid YAML");
        config.base_dir = exe_dir();
        config.config_file = None;
        config
    }
}

impl Config {
    /// Load `path`, or the nearest `hasse.yaml` walking up from the executable
    /// and the working directory, or the user config (`~/.hasse.yaml`,
    /// `~/.config/hasse/hasse.yaml`). No config file is fine: the embedded
    /// `default.yaml` applies. A found file is merged over the defaults, so it
    /// only needs the keys it changes.
    pub fn load(explicit: Option<&Path>) -> Result<Self> {
        let path = match explicit {
            Some(p) => Some(p.to_path_buf()),
            None => find_config(),
        };
        let Some(path) = path else {
            return Ok(Config::default());
        };
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?;
        let mut base: serde_yaml::Value =
            serde_yaml::from_str(DEFAULT_YAML).expect("default.yaml is valid YAML");
        let overlay: serde_yaml::Value = serde_yaml::from_str(&text)
            .with_context(|| format!("parsing {}", path.display()))?;
        merge(&mut base, overlay);
        let mut config: Config = serde_yaml::from_value(base)
            .with_context(|| format!("applying {}", path.display()))?;
        config.base_dir = path.parent().unwrap_or(Path::new(".")).to_path_buf();
        config.config_file = Some(path);
        Ok(config)
    }

    /// Find the model directory. An explicit `[models] dir` wins; otherwise
    /// look beside the executable and up the usual development layout.
    pub fn models_dir(&self) -> Result<PathBuf> {
        if let Some(dir) = &self.models.dir {
            let path = if dir.is_absolute() { dir.clone() } else { self.base_dir.join(dir) };
            if path.join("frontend.onnx").is_file() {
                return Ok(path);
            }
            bail!("frontend.onnx not found in {}", path.display());
        }

        #[cfg(target_os = "macos")]
        if let Some(home) = std::env::var_os("HOME") {
            let candidate =
                PathBuf::from(home).join("Library/Application Support/hasse/models");
            if candidate.join("frontend.onnx").is_file() {
                return Ok(candidate);
            }
        }

        let mut roots = vec![self.base_dir.clone()];
        // Always include the executable's directory: a config file elsewhere
        // must not hide the models installed next to the binary.
        roots.push(exe_dir());
        if let Ok(cwd) = std::env::current_dir() {
            roots.push(cwd);
        }
        for root in roots {
            let mut dir = Some(root);
            for _ in 0..5 {
                let Some(current) = dir else { break };
                for rel in ["models", "engine/models"] {
                    let candidate = current.join(rel);
                    if candidate.join("frontend.onnx").is_file() {
                        return Ok(candidate);
                    }
                }
                dir = current.parent().map(Path::to_path_buf);
            }
        }
        bail!(
            "could not find frontend.onnx. Put the model files in a `models` directory next to the executable, or set models.dir in hasse.yaml"
        )
    }

    pub fn onnxruntime_path(&self) -> Option<PathBuf> {
        self.models.onnxruntime.as_ref().map(|path| {
            if path.is_absolute() {
                path.clone()
            } else {
                self.base_dir.join(path)
            }
        })
    }
}

pub fn exe_dir() -> PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(Path::to_path_buf))
        .unwrap_or_else(|| PathBuf::from("."))
}

/// Merge `overlay` into `base`: mappings merge key by key, everything else is
/// replaced. This keeps the embedded defaults under a partial user file.
fn merge(base: &mut serde_yaml::Value, overlay: serde_yaml::Value) {
    match (base, overlay) {
        (serde_yaml::Value::Mapping(base_map), serde_yaml::Value::Mapping(overlay_map)) => {
            for (key, value) in overlay_map {
                match base_map.get_mut(&key) {
                    Some(existing) => merge(existing, value),
                    None => {
                        base_map.insert(key, value);
                    }
                }
            }
        }
        (base, overlay) => *base = overlay,
    }
}

/// The configuration file. `hasse.yaml` next to the executable or in an
/// ancestor (a checkout's own settings) wins; otherwise the user locations:
/// `~/.hasse.yaml`, `$XDG_CONFIG_HOME/hasse/hasse.yaml`,
/// `~/.config/hasse/hasse.yaml`, and `%APPDATA%\hasse\hasse.yaml` on Windows
/// or `~/Library/Application Support/hasse/hasse.yaml` on macOS.
fn find_config() -> Option<PathBuf> {
    let mut roots = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            roots.push(parent.to_path_buf());
        }
    }
    if let Ok(cwd) = std::env::current_dir() {
        roots.push(cwd);
    }
    for root in roots {
        let mut dir = Some(root.as_path());
        while let Some(d) = dir {
            let candidate = d.join("hasse.yaml");
            if candidate.is_file() {
                return Some(candidate);
            }
            dir = d.parent();
        }
    }
    user_config()
}

fn user_config() -> Option<PathBuf> {
    let home = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE"))?;
    let home = PathBuf::from(home);
    let mut candidates = vec![home.join(".hasse.yaml")];
    if let Some(xdg) = std::env::var_os("XDG_CONFIG_HOME") {
        candidates.push(PathBuf::from(xdg).join("hasse").join("hasse.yaml"));
    }
    candidates.push(home.join(".config").join("hasse").join("hasse.yaml"));
    #[cfg(windows)]
    {
        if let Some(appdata) = std::env::var_os("APPDATA") {
            candidates.push(PathBuf::from(appdata).join("hasse").join("hasse.yaml"));
        }
    }
    #[cfg(target_os = "macos")]
    {
        candidates.push(
            home.join("Library")
                .join("Application Support")
                .join("hasse")
                .join("hasse.yaml"),
        );
    }
    candidates.into_iter().find(|path| path.is_file())
}
