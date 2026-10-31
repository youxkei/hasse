//! hasse: a Japanese voice-input IME for Windows and macOS.
//!
//! The pipeline is: capture the microphone (`audio`), keep the first channel while
//! the push-to-talk key is held (`hotkey`), resample to 16 kHz, run the mel
//! frontend (`mel`) and the FastConformer encoder plus the TDT/KDA decoder
//! (`asr`), then insert the text through the clipboard (`clipboard`). The overlay
//! and tray show the state; `controller` below owns the state machine.
//!
//! Platform backends are chosen with `cfg`: the Windows modules use Win32 APIs,
//! and the macOS modules use rdev, arboard, Core Graphics and eframe. Build and
//! model-export notes are in the `xtask` and `tools` sources.

mod asr;
mod audio;
mod config;
mod key_event;
mod mel;
mod status;
mod tone;
mod tray;

#[cfg(windows)]
mod clipboard;
#[cfg(target_os = "macos")]
#[path = "clipboard_macos.rs"]
mod clipboard;

#[cfg(windows)]
mod hotkey;
#[cfg(target_os = "macos")]
#[path = "hotkey_macos.rs"]
mod hotkey;

#[cfg(windows)]
mod installer;
#[cfg(target_os = "macos")]
#[path = "installer_macos.rs"]
mod installer;

#[cfg(windows)]
mod overlay;
#[cfg(target_os = "macos")]
#[path = "overlay_macos.rs"]
mod overlay;

#[cfg(windows)]
mod runtime_dll;
#[cfg(target_os = "macos")]
#[path = "runtime_macos.rs"]
mod runtime_dll;

#[cfg(windows)]
mod sound;
#[cfg(target_os = "macos")]
#[path = "sound_macos.rs"]
mod sound;

use std::path::PathBuf;
use std::sync::atomic::AtomicU32;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use crossbeam_channel::{Receiver, Sender};

use asr::Asr;
use audio::{AudioCmd, Recorded};
use config::Config;
use hotkey::KeyEvent;
use overlay::Status;

fn main() -> Result<()> {
    #[cfg(windows)]
    unsafe {
        let _ = windows::Win32::System::Console::SetConsoleOutputCP(65001);
    }

    let raw: Vec<String> = std::env::args().skip(1).collect();

    if raw.first().map(String::as_str) == Some("install") {
        let config_path = flag_value(&raw, "--config").map(PathBuf::from);
        let config = Config::load(config_path.as_deref())?;
        let models = config.models_dir()?;
        return installer::install(&models);
    }
    if raw.first().map(String::as_str) == Some("uninstall") {
        return installer::uninstall();
    }

    let mut config_path: Option<PathBuf> = None;
    let mut wav: Option<PathBuf> = None;
    let mut list_devices = false;
    let mut no_paste = false;
    let mut args = raw.iter();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--config" => config_path = args.next().map(PathBuf::from),
            "--wav" => wav = args.next().map(PathBuf::from),
            "--devices" => list_devices = true,
            "--no-paste" => no_paste = true,
            "--se-test" => {
                sound::start();
                std::thread::sleep(Duration::from_millis(400));
                sound::stop();
                std::thread::sleep(Duration::from_millis(400));
                sound::error();
                std::thread::sleep(Duration::from_millis(500));
                return Ok(());
            }
            "--clipboard-test" => {
                let (before, after) = clipboard::roundtrip("hasse clipboard test")?;
                println!("before: {before:?}");
                println!("after:  {after:?}");
                println!("{}", if before == after { "clipboard preserved" } else { "CLIPBOARD MISMATCH" });
                return Ok(());
            }
            "-h" | "--help" => {
                println!(
                    "hasse — Japanese voice input IME\n\n  hasse [--config FILE]         run the IME (default)\n  hasse install                 install, register a logon task, and start\n  hasse uninstall               remove the logon task\n  hasse --wav FILE              transcribe a wav and print it\n  hasse --devices               list input devices and the selected one\n  hasse --no-paste              print the recognized text instead of typing it\n  hasse --se-test               play the start, stop and error sounds\n  hasse --clipboard-test        check that the clipboard is saved and restored\n\n  Quit from the tray icon's menu."
                );
                return Ok(());
            }
            other => eprintln!("unknown argument {other:?} (try --help)"),
        }
    }

    let config = Config::load(config_path.as_deref())?;

    if list_devices {
        return list_input_devices(&config);
    }
    if let Some(path) = wav {
        return transcribe_file(&config, &path);
    }
    run(config, no_paste)
}

fn transcribe_file(config: &Config, path: &std::path::Path) -> Result<()> {
    let (mono, rate) = read_wav(path)?;
    let audio = audio::to_model_rate(&mono, rate);
    eprintln!("loaded {} ({:.2}s @ {rate} Hz)", path.display(), audio.len() as f32 / 16000.0);
    let mut asr = Asr::load(&config.models_dir()?, config.onnxruntime_path().as_deref())?;
    let started = Instant::now();
    let text = asr.transcribe(&audio)?;
    println!("{text}");
    eprintln!("elapsed {:.2}s", started.elapsed().as_secs_f32());
    Ok(())
}

fn run(config: Config, no_paste: bool) -> Result<()> {
    let Some(_instance) = acquire_instance() else {
        eprintln!("hasse: already running");
        return Ok(());
    };
    hide_own_console();
    overlay::create()?;
    listen_for_stop();
    let level = Arc::new(AtomicU32::new(0));
    overlay::set_level_source(Arc::clone(&level));

    let (audio_tx, audio_rx) = crossbeam_channel::unbounded();
    let (recorded_tx, recorded_rx) = crossbeam_channel::unbounded();
    audio::spawn(
        config.audio.device.clone(),
        config.audio.first_channel_only,
        config.audio.max_seconds,
        audio_rx,
        recorded_tx,
        level,
    );

    let hotkeys = hotkey::spawn(&config.hotkey.key, config.hotkey.swallow)?;
    let controller_config = config.clone();
    std::thread::Builder::new()
        .name("hasse-control".into())
        .spawn(move || controller(controller_config, no_paste, hotkeys, audio_tx, recorded_rx))?;

    // On Windows the tray is created here; on macOS the eframe app creates it.
    #[cfg(windows)]
    let _tray = match tray::create() {
        Ok(tray) => Some(tray),
        Err(e) => {
            eprintln!("tray: {e:#}");
            None
        }
    };

    overlay::run_loop()?;
    Ok(())
}

fn flag_value(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

/// One IME per logon session. The handle lives until the process exits.
#[cfg(windows)]
fn acquire_instance() -> Option<windows::Win32::Foundation::HANDLE> {
    unsafe {
        let handle = windows::Win32::System::Threading::CreateMutexW(
            None,
            true,
            windows::core::w!("Local\\hasse-voice-ime"),
        )
        .ok()?;
        if windows::Win32::Foundation::GetLastError()
            == windows::Win32::Foundation::ERROR_ALREADY_EXISTS
        {
            return None;
        }
        Some(handle)
    }
}

/// The installer asks a running instance to quit through this event.
#[cfg(windows)]
fn listen_for_stop() {
    unsafe {
        let Ok(event) = windows::Win32::System::Threading::CreateEventW(
            None,
            true,
            false,
            windows::core::w!("Local\\hasse-stop"),
        ) else {
            return;
        };
        let raw = event.0 as isize;
        let _ = std::thread::Builder::new()
            .name("hasse-stop".into())
            .spawn(move || {
                windows::Win32::System::Threading::WaitForSingleObject(
                    windows::Win32::Foundation::HANDLE(raw as *mut core::ffi::c_void),
                    windows::Win32::System::Threading::INFINITE,
                );
                overlay::quit();
            });
    }
}

/// Hide the console when it belongs to this process alone (started by Task
/// Scheduler, the installer, or Explorer). A console shared with a shell stays.
#[cfg(windows)]
fn hide_own_console() {
    unsafe {
        let mut pids = [0u32; 2];
        let count = windows::Win32::System::Console::GetConsoleProcessList(&mut pids);
        if count == 1 {
            let window = windows::Win32::System::Console::GetConsoleWindow();
            if !window.is_invalid() {
                let _ = windows::Win32::UI::WindowsAndMessaging::ShowWindow(
                    window,
                    windows::Win32::UI::WindowsAndMessaging::SW_HIDE,
                );
            }
        }
    }
}

/// A macOS lock file held for the process lifetime; the lock is released when
/// the process exits.
#[cfg(target_os = "macos")]
struct InstanceGuard {
    _lock: std::fs::File,
    pid_path: std::path::PathBuf,
}

#[cfg(target_os = "macos")]
impl Drop for InstanceGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.pid_path);
    }
}

#[cfg(target_os = "macos")]
fn acquire_instance() -> Option<InstanceGuard> {
    use std::os::unix::io::AsRawFd;
    let dir = runtime_dll::support_dir().ok()?;
    std::fs::create_dir_all(&dir).ok()?;
    let lock = std::fs::OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .open(dir.join("hasse.lock"))
        .ok()?;
    if unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
        return None;
    }
    let pid_path = dir.join("hasse.pid");
    let _ = std::fs::write(&pid_path, std::process::id().to_string());
    Some(InstanceGuard { _lock: lock, pid_path })
}

#[cfg(target_os = "macos")]
extern "C" fn on_sigterm(_: libc::c_int) {
    overlay::quit();
}

#[cfg(target_os = "macos")]
fn listen_for_stop() {
    unsafe {
        libc::signal(
            libc::SIGTERM,
            on_sigterm as extern "C" fn(libc::c_int) as libc::sighandler_t,
        );
    }
}

#[cfg(target_os = "macos")]
fn hide_own_console() {}

fn controller(
    config: Config,
    no_paste: bool,
    hotkeys: Receiver<KeyEvent>,
    audio_tx: Sender<AudioCmd>,
    recorded_rx: Receiver<Recorded>,
) {
    overlay::set(Status::Starting);
    let mut asr = match config
        .models_dir()
        .and_then(|dir| Asr::load(&dir, config.onnxruntime_path().as_deref()))
    {
        Ok(asr) => Some(asr),
        Err(e) => {
            overlay::set(Status::Error(format!("モデルを読み込めません: {e:#}")));
            eprintln!("asr: {e:#}");
            None
        }
    };
    if asr.is_some() {
        overlay::hide();
    }

    loop {
        match hotkeys.recv() {
            Ok(KeyEvent::Pressed) => {
                if asr.is_some() {
                    let _ = audio_tx.send(AudioCmd::Start);
                    overlay::set(Status::Recording);
                    sound::start();
                } else {
                    sound::error();
                }
            }
            Ok(KeyEvent::Released) => {
                if asr.is_none() {
                    continue;
                }
                let _ = audio_tx.send(AudioCmd::Stop);
                sound::stop();
                let Ok(recorded) = recorded_rx.recv_timeout(Duration::from_secs(3)) else {
                    overlay::hide();
                    continue;
                };
                let samples = audio::to_model_rate(&recorded.samples, recorded.sample_rate);
                if (samples.len() as f32) < config.audio.min_seconds * 16000.0 {
                    overlay::hide();
                    continue;
                }
                overlay::set(Status::Transcribing);
                match asr.as_mut().unwrap().transcribe(&samples) {
                    Ok(text) if !text.trim().is_empty() => {
                        let text = text.trim();
                        if no_paste {
                            eprintln!("asr: {text}");
                            overlay::set(Status::Typed);
                        } else {
                            match clipboard::paste(text, config.paste.delay_ms, config.paste.restore_clipboard) {
                                Ok(()) => overlay::set(Status::Typed),
                                Err(e) => overlay::set(Status::Error(format!("貼り付けに失敗: {e}"))),
                            }
                        }
                    }
                    Ok(_) => {
                        if no_paste {
                            eprintln!("asr: (empty)");
                        }
                        overlay::hide();
                    }
                    Err(e) => {
                        eprintln!("asr: {e:#}");
                        overlay::set(Status::Error(format!("認識に失敗: {e:#}")));
                    }
                }
            }
            Err(_) => break,
        }
    }
    overlay::quit();
}

fn read_wav(path: &std::path::Path) -> Result<(Vec<f32>, u32)> {
    let mut reader = hound::WavReader::open(path)
        .with_context(|| format!("opening {}", path.display()))?;
    let spec = reader.spec();
    let mut samples: Vec<f32> = Vec::new();
    match spec.sample_format {
        hound::SampleFormat::Float => {
            for s in reader.samples::<f32>() {
                samples.push(s?);
            }
        }
        hound::SampleFormat::Int => {
            if spec.bits_per_sample <= 16 {
                for s in reader.samples::<i16>() {
                    samples.push(s? as f32 / 32768.0);
                }
            } else {
                for s in reader.samples::<i32>() {
                    samples.push(s? as f32 / 2147483648.0);
                }
            }
        }
    }
    let channels = spec.channels as usize;
    let mono = if channels > 1 {
        samples
            .chunks(channels)
            .map(|c| c.iter().sum::<f32>() / channels as f32)
            .collect()
    } else {
        samples
    };
    Ok((mono, spec.sample_rate))
}

fn list_input_devices(config: &Config) -> Result<()> {
    use cpal::traits::{DeviceTrait, HostTrait};
    let host = cpal::default_host();
    for device in host.input_devices()? {
        let rate = device
            .default_input_config()
            .map(|c| format!("{} Hz, {} ch", c.sample_rate(), c.channels()))
            .unwrap_or_else(|_| "<no default config>".into());
        println!("{device}  ({rate})");
    }
    let selected = if config.audio.device.is_empty() {
        "system default".to_string()
    } else {
        config.audio.device.clone()
    };
    println!("selected: {selected}");
    Ok(())
}
