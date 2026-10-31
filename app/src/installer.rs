//! `hasse install` / `hasse uninstall`: put the executable and model files in
//! `%LOCALAPPDATA%\hasse`, register a logon task in Task Scheduler, and start it.
//!
//! There are no options: install always autostarts and always starts now, and
//! it creates no shortcut — the app lives in the tray and starts at logon.
//!
//! Autostart uses Task Scheduler and never the registry. The task is handed
//! over as an XML definition and registered with `schtasks /create /xml` from
//! this ordinary process, so it needs no elevation.
//!
//! After installation the app needs no config file: [`crate::config`] looks for
//! `models\` next to the executable first.

use std::os::windows::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::{bail, Context, Result};

const RUNTIME_FILES: &[&str] = &[
    "frontend.onnx",
    "frontend.onnx.data",
    "frontend.safetensors",
    "decoder.safetensors",
    "vocab.json",
    "meta.json",
];

const TASK_NAME: &str = "hasse";
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

pub fn install(models: &Path) -> Result<()> {
    let dir = install_dir()?;
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    if stop_running() {
        println!("  stopped the running instance");
    }

    let exe_src = std::env::current_exe().context("locating the running executable")?;
    let exe_dst = dir.join("hasse.exe");
    if exe_src != exe_dst {
        std::fs::copy(&exe_src, &exe_dst).with_context(|| {
            format!(
                "copying {} to {} (close a running hasse.exe first)",
                exe_src.display(),
                exe_dst.display()
            )
        })?;
    }
    println!("  exe: {}", exe_dst.display());

    let models_dst = dir.join("models");
    std::fs::create_dir_all(&models_dst)
        .with_context(|| format!("creating {}", models_dst.display()))?;
    for name in RUNTIME_FILES {
        let src = models.join(name);
        let dst = models_dst.join(name);
        if !src.is_file() {
            bail!("missing model file: {}", src.display());
        }
        if src == dst {
            continue;
        }
        let bytes = std::fs::copy(&src, &dst)
            .with_context(|| format!("copying {}", src.display()))?;
        println!("  {name} ({} MiB)", bytes / 1024 / 1024);
    }
    println!("  models: {}", models_dst.display());

    register_task(&exe_dst, &dir)?;
    println!("  autostart: Task Scheduler task \"{TASK_NAME}\" at logon");

    if schtasks(&["/run", "/tn", TASK_NAME], &dir).is_err() {
        start_direct(&exe_dst, &dir)?;
    } else {
        println!("  started: {TASK_NAME}");
    }

    println!("installed to {}", dir.display());
    Ok(())
}

pub fn uninstall() -> Result<()> {
    let dir = install_dir()?;
    println!(
        "autostart task: {}",
        if delete_task(&dir) { "removed" } else { "not present" }
    );
    println!(
        "the program and models are still in {}; delete that folder to finish",
        dir.display()
    );
    Ok(())
}

fn install_dir() -> Result<PathBuf> {
    let base = std::env::var_os("LOCALAPPDATA")
        .map(PathBuf::from)
        .context("LOCALAPPDATA is not set")?;
    Ok(base.join("hasse"))
}

/// Register the logon task from an ordinary process. The definition is kept
/// beside the installed exe when Task Scheduler refuses it.
fn register_task(exe: &Path, dir: &Path) -> Result<()> {
    let xml = task_xml(exe, dir, &user());
    let path = dir.join("hasse-task.xml");
    std::fs::write(&path, utf16le(&xml))
        .with_context(|| format!("writing {}", path.display()))?;
    let path_arg = path.display().to_string();
    match schtasks(&["/create", "/xml", &path_arg, "/tn", TASK_NAME, "/f"], dir) {
        Ok(()) => {
            let _ = std::fs::remove_file(&path);
            Ok(())
        }
        Err(e) => Err(anyhow::anyhow!(
            "Task Scheduler refused the task ({e}); the definition is {}",
            path.display()
        )),
    }
}

/// Ask a running instance to quit through its stop event, and wait briefly for
/// it to go. `false` when nothing was running (or it was a build without the
/// event).
fn stop_running() -> bool {
    unsafe {
        let Ok(event) = windows::Win32::System::Threading::OpenEventW(
            windows::Win32::System::Threading::EVENT_MODIFY_STATE,
            false,
            windows::core::w!("Local\\hasse-stop"),
        ) else {
            return false;
        };
        let _ = windows::Win32::System::Threading::SetEvent(event);
        let _ = windows::Win32::Foundation::CloseHandle(event);
        for _ in 0..100 {
            if windows::Win32::System::Threading::OpenEventW(
                windows::Win32::System::Threading::EVENT_MODIFY_STATE,
                false,
                windows::core::w!("Local\\hasse-stop"),
            )
            .is_err()
            {
                return true;
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        true
    }
}

/// Delete the task if it exists; `false` when there was none.
fn delete_task(dir: &Path) -> bool {
    if schtasks(&["/query", "/tn", TASK_NAME], dir).is_err() {
        return false;
    }
    schtasks(&["/delete", "/tn", TASK_NAME, "/f"], dir).is_ok()
}

/// Start the installed copy directly, detached from this console.
fn start_direct(exe: &Path, dir: &Path) -> Result<()> {
    let mut child = Command::new(exe)
        .current_dir(dir)
        .creation_flags(CREATE_NO_WINDOW)
        .spawn()
        .with_context(|| format!("starting {}", exe.display()))?;
    std::thread::sleep(std::time::Duration::from_millis(500));
    if let Ok(Some(status)) = child.try_wait() {
        bail!("{} exited immediately ({status})", exe.display());
    }
    println!("  started: {}", exe.display());
    Ok(())
}

/// Run `schtasks`, reporting its own output only when it fails.
fn schtasks(args: &[&str], cwd: &Path) -> Result<(), String> {
    let mut command = Command::new("schtasks.exe");
    command
        .args(args)
        .stdin(Stdio::null())
        .creation_flags(CREATE_NO_WINDOW);
    if cwd.is_dir() {
        command.current_dir(cwd);
    }
    let out = command
        .output()
        .map_err(|e| format!("cannot run schtasks.exe: {e}"))?;
    if out.status.success() {
        Ok(())
    } else {
        Err(format!(
            "schtasks {} exited with {}: {}",
            args.first().copied().unwrap_or(""),
            out.status,
            String::from_utf8_lossy(&out.stderr).trim()
        ))
    }
}

/// The task definition.
///
/// XML rather than `schtasks /create`'s flags: the flags cannot say these, and
/// the defaults of three of them are wrong for an IME.
///
/// - `ExecutionTimeLimit` `PT0S`: by default a task is stopped after 72 hours.
/// - `DisallowStartIfOnBatteries` / `StopIfGoingOnBatteries` false: by default a
///   task does not start on battery and is stopped going onto it.
/// - `Priority` 4 is the normal priority class; the default 7 is below normal,
///   and this process carries a low-level keyboard hook.
/// - `MultipleInstancesPolicy` `IgnoreNew`: a second logon start does not
///   launch a second IME.
fn task_xml(exe: &Path, dir: &Path, user: &str) -> String {
    format!(
        r#"<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>hasse: Japanese voice input IME, started at logon</Description>
  </RegistrationInfo>
  <Triggers>
    <LogonTrigger>
      <Enabled>true</Enabled>
      <UserId>{user}</UserId>
    </LogonTrigger>
  </Triggers>
  <Principals>
    <Principal id="Author">
      <UserId>{user}</UserId>
      <LogonType>InteractiveToken</LogonType>
      <RunLevel>LeastPrivilege</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <AllowHardTerminate>true</AllowHardTerminate>
    <StartWhenAvailable>false</StartWhenAvailable>
    <RunOnlyIfNetworkAvailable>false</RunOnlyIfNetworkAvailable>
    <IdleSettings>
      <StopOnIdleEnd>false</StopOnIdleEnd>
      <RestartOnIdle>false</RestartOnIdle>
    </IdleSettings>
    <AllowStartOnDemand>true</AllowStartOnDemand>
    <Enabled>true</Enabled>
    <Hidden>false</Hidden>
    <RunOnlyIfIdle>false</RunOnlyIfIdle>
    <RestartOnFailure>
      <Interval>PT1M</Interval>
      <Count>3</Count>
    </RestartOnFailure>
    <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>
    <Priority>4</Priority>
  </Settings>
  <Actions Context="Author">
    <Exec>
      <Command>{command}</Command>
      <WorkingDirectory>{working}</WorkingDirectory>
    </Exec>
  </Actions>
</Task>
"#,
        user = xml_escape(user),
        command = xml_escape(&exe.display().to_string()),
        working = xml_escape(&dir.display().to_string()),
    )
}

fn xml_escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// The user the task runs as and is triggered by.
fn user() -> String {
    let name = std::env::var("USERNAME").unwrap_or_default();
    match std::env::var("USERDOMAIN") {
        Ok(domain) if !domain.is_empty() => format!("{domain}\\{name}"),
        _ => name,
    }
}

/// UTF-16 with a byte order mark, which is what the declaration says and what
/// `schtasks /xml` reads.
fn utf16le(text: &str) -> Vec<u8> {
    let mut bytes = vec![0xFF, 0xFE];
    for unit in text.encode_utf16() {
        bytes.extend_from_slice(&unit.to_le_bytes());
    }
    bytes
}
