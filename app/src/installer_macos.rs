//! `hasse install` / `hasse uninstall` on macOS.
//!
//! Install copies the binary into `~/Applications/hasse.app` (an `LSUIElement`
//! bundle), the models into Application Support, and registers a LaunchAgent
//! that starts the app at login. There are no options.

use std::os::unix::fs::PermissionsExt;
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};

const LABEL: &str = "com.hasse.ime";
const RUNTIME_FILES: &[&str] = &[
    "frontend.onnx",
    "frontend.onnx.data",
    "frontend.safetensors",
    "decoder.safetensors",
    "vocab.json",
    "meta.json",
];

pub fn install(models: &Path) -> Result<()> {
    let home = home()?;
    let app = home.join("Applications").join("hasse.app");
    let macos = app.join("Contents").join("MacOS");
    let bin = macos.join("hasse");
    let plist = launch_agent_path(&home)?;
    let models_dst = crate::runtime_dll::support_dir()?.join("models");

    if stop_running()? {
        println!("  stopped the running instance");
    }

    std::fs::create_dir_all(&macos).with_context(|| format!("creating {}", macos.display()))?;
    let exe_src = std::env::current_exe().context("locating the running executable")?;
    if exe_src != bin {
        std::fs::copy(&exe_src, &bin)
            .with_context(|| format!("copying {} to {}", exe_src.display(), bin.display()))?;
    }
    std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o755))?;
    println!("  app: {}", app.display());

    std::fs::write(app.join("Contents").join("Info.plist"), info_plist())
        .context("writing Info.plist")?;

    std::fs::create_dir_all(&models_dst)
        .with_context(|| format!("creating {}", models_dst.display()))?;
    for name in RUNTIME_FILES {
        let src = models.join(name);
        let dst = models_dst.join(name);
        if !src.is_file() {
            bail!("missing model file: {}", src.display());
        }
        if src != dst {
            std::fs::copy(&src, &dst)
                .with_context(|| format!("copying {}", src.display()))?;
        }
    }
    println!("  models: {}", models_dst.display());

    // Ad-hoc signing keeps the bundle self-consistent for TCC.
    let _ = Command::new("/usr/bin/codesign")
        .args(["--force", "--deep", "--sign", "-"])
        .arg(&app)
        .status();

    std::fs::create_dir_all(plist.parent().unwrap_or(Path::new("/")))
        .context("creating ~/Library/LaunchAgents")?;
    std::fs::write(&plist, launch_agent(&bin, &home))
        .with_context(|| format!("writing {}", plist.display()))?;
    register(&plist)?;
    println!("  autostart: LaunchAgent {LABEL}");

    println!("installed to {}", app.display());
    Ok(())
}

pub fn uninstall() -> Result<()> {
    let home = home()?;
    let _ = stop_running();
    let _ = Command::new("/bin/launchctl")
        .args(["bootout", &format!("gui/{}/{}", uid(), LABEL)])
        .status();
    let plist = launch_agent_path(&home)?;
    if plist.is_file() {
        std::fs::remove_file(&plist)
            .with_context(|| format!("removing {}", plist.display()))?;
    }
    println!("removed the LaunchAgent {LABEL}");
    println!(
        "the app and models are still in {} and {}; delete them to finish",
        home.join("Applications/hasse.app").display(),
        crate::runtime_dll::support_dir()?.display()
    );
    Ok(())
}

fn register(plist: &Path) -> Result<()> {
    let _ = Command::new("/bin/launchctl")
        .args(["bootout", &format!("gui/{}/{}", uid(), LABEL)])
        .status();
    let domain = format!("gui/{}", uid());
    let status = Command::new("/bin/launchctl")
        .args(["bootstrap", &domain])
        .arg(plist)
        .status();
    if status.map(|s| s.success()).unwrap_or(false) {
        return Ok(());
    }
    let status = Command::new("/bin/launchctl")
        .args(["load", "-w"])
        .arg(plist)
        .status();
    if status.map(|s| s.success()).unwrap_or(false) {
        return Ok(());
    }
    // Starting it directly is still enough for this session; the plist remains
    // for the next login.
    let bin = plist_bin(plist);
    if let Some(bin) = bin {
        let _ = Command::new(bin).spawn();
    }
    Ok(())
}

fn plist_bin(_plist: &Path) -> Option<PathBuf> {
    let home = home().ok()?;
    Some(home.join("Applications/hasse.app/Contents/MacOS/hasse"))
}

fn stop_running() -> Result<bool> {
    let dir = crate::runtime_dll::support_dir()?;
    let pid_path = dir.join("hasse.pid");
    let Ok(text) = std::fs::read_to_string(&pid_path) else {
        return Ok(false);
    };
    let Ok(pid) = text.trim().parse::<libc::pid_t>() else {
        return Ok(false);
    };
    unsafe {
        libc::kill(pid, libc::SIGTERM);
    }
    // The instance holds an exclusive flock on hasse.lock; it becomes
    // available when the process is gone.
    let lock = dir.join("hasse.lock");
    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline {
        let available = std::fs::OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .open(&lock)
            .ok()
            .map(|file| unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0)
            .unwrap_or(false);
        if available {
            return Ok(true);
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    Ok(true)
}

fn home() -> Result<PathBuf> {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .context("HOME is not set")
}

fn uid() -> u32 {
    unsafe { libc::getuid() }
}

fn launch_agent_path(home: &Path) -> Result<PathBuf> {
    Ok(home
        .join("Library")
        .join("LaunchAgents")
        .join(format!("{LABEL}.plist")))
}

fn info_plist() -> String {
    r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleIdentifier</key><string>com.hasse.ime</string>
    <key>CFBundleName</key><string>hasse</string>
    <key>CFBundleDisplayName</key><string>hasse</string>
    <key>CFBundleExecutable</key><string>hasse</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleVersion</key><string>0.1.0</string>
    <key>CFBundleShortVersionString</key><string>0.1.0</string>
    <key>LSUIElement</key><true/>
    <key>NSHighResolutionCapable</key><true/>
</dict>
</plist>
"#
    .to_string()
}

fn launch_agent(bin: &Path, home: &Path) -> String {
    let log = home.join("Library/Logs/hasse.log");
    format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>{LABEL}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{bin}</string>
    </array>
    <key>RunAtLoad</key><true/>
    <key>KeepAlive</key><false/>
    <key>ProcessType</key><string>Interactive</string>
    <key>StandardOutPath</key><string>{log}</string>
    <key>StandardErrorPath</key><string>{log}</string>
</dict>
</plist>
"#,
        bin = xml_escape(&bin.display().to_string()),
        log = xml_escape(&log.display().to_string()),
    )
}

fn xml_escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}
