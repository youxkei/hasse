//! hasse: a Japanese voice-input IME for Windows and macOS.
//!
//! The pipeline is: capture the microphone (`audio`), keep the first channel while
//! the push-to-talk key is held (`hotkey`), resample to 16 kHz, run the mel
//! frontend (`mel`) and the FastConformer encoder plus the TDT/KDA decoder
//! (`asr`), then show the transcript in the overlay for review and correction,
//! and insert it through the clipboard (`clipboard`). A correction trains the
//! token-level phrase bias used by the decoder (`bias`). The overlay and tray
//! show the state; `controller` below owns the state machine.
//!
//! Platform backends are chosen with `cfg`: the Windows modules use Win32 APIs,
//! and the macOS modules use rdev, arboard, Core Graphics and eframe. Build and
//! model-export notes are in the `xtask` and `tools` sources.

mod asr;
mod audio;
mod bias;
mod config;
mod focus;
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
use bias::BiasFile;
use config::Config;
use hotkey::KeyEvent;
use overlay::Status;

fn main() -> Result<()> {
    #[cfg(windows)]
    unsafe {
        use windows::Win32::UI::HiDpi::{
            SetProcessDpiAwarenessContext, DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2,
        };
        // The overlay draws itself with GDI; without this Windows bitmap-stretches
        // its surface and the text turns blurry on a scaled monitor.
        let _ = SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
        let _ = windows::Win32::System::Console::SetConsoleOutputCP(65001);
    }

    let raw: Vec<String> = std::env::args().skip(1).collect();

    let config_path = flag_value(&raw, "--config").map(PathBuf::from);
    match subcommand(&raw) {
        Some("bias") => {
            let config = Config::load(config_path.as_deref())?;
            let args = strip_config(&raw);
            return bias_command(&config, &args[1..]);
        }
        Some("install") => {
            let config = Config::load(config_path.as_deref())?;
            let models = config.models_dir()?;
            return installer::install(&models);
        }
        Some("uninstall") => return installer::uninstall(),
        _ => {}
    }

    let mut wav: Option<PathBuf> = None;
    let mut list_devices = false;
    let mut no_paste = false;
    let mut tokenize: Option<String> = None;
    let mut args = raw.iter();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--config" => {
                let _ = args.next();
            }
            "--wav" => wav = args.next().map(PathBuf::from),
            "--devices" => list_devices = true,
            "--no-paste" => no_paste = true,
            "--tokenize" => tokenize = args.next().cloned(),
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
                    "hasse — Japanese voice input IME\n\n  hasse [--config FILE]         run the IME (default)\n  hasse install                 install, register a logon task, and start\n  hasse uninstall               remove the logon task\n  hasse bias                    list the phrases learned from corrections\n  hasse bias add TEXT           add a rule (--from, --weight, --before/--after)\n  hasse bias remove TEXT        drop the rules for TEXT\n  hasse bias rebuild            rebuild bias.yaml from corrections.jsonl\n  hasse bias rewrite TEXT       show the learned replacements applied to TEXT\n  hasse --wav FILE              transcribe a wav and print it\n  hasse --devices               list input devices and the selected one\n  hasse --no-paste              print the recognized text instead of typing it\n  hasse --tokenize TEXT         show the model's token ids\n  hasse --se-test               play the start, stop and error sounds\n  hasse --clipboard-test        check that the clipboard is saved and restored\n\n  Quit from the tray icon's menu."
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
    if let Some(text) = tokenize {
        return tokenize_text(&config, &text);
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
    configure_bias(config, &mut asr)?;
    asr.set_beam(config.decode.beam);
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

    // On Windows the overlay owns the message loop, so the tray icon is created
    // here and kept alive until the loop returns; the macOS overlay creates it
    // once eframe's loop is running.
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

/// The first positional argument, skipping `--config` and its value. This is
/// how `hasse --config FILE bias` and `hasse bias --config FILE` agree.
fn subcommand(args: &[String]) -> Option<&str> {
    let mut i = 0;
    while i < args.len() {
        let arg = args[i].as_str();
        if arg == "--config" {
            i += 2;
            continue;
        }
        if arg.starts_with('-') {
            i += 1;
            continue;
        }
        return Some(arg);
    }
    None
}

fn strip_config(args: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--config" {
            i += 2;
            continue;
        }
        out.push(args[i].clone());
        i += 1;
    }
    out
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
        Ok(mut asr) => {
            if let Err(e) = configure_bias(&config, &mut asr) {
                eprintln!("bias: {e:#}");
            }
            asr.set_beam(config.decode.beam);
            Some(asr)
        }
        Err(e) => {
            overlay::set(Status::Error(format!("モデルを読み込めません: {e:#}")));
            eprintln!("asr: {e:#}");
            None
        }
    };
    if asr.is_some() {
        overlay::hide();
    }

    // The window that had the keyboard when recording began, so the review
    // overlay can hand it back after the user confirms.
    let mut focus = None;

    loop {
        match hotkeys.recv() {
            Ok(KeyEvent::Pressed) => {
                if asr.is_some() {
                    focus = focus::capture();
                    overlay::set_anchor(focus.as_ref().and_then(focus::monitor_rect));
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
                let recognized = match asr.as_mut().unwrap().transcribe(&samples) {
                    Ok(text) if !text.trim().is_empty() => text.trim().to_string(),
                    Ok(_) => {
                        if no_paste {
                            eprintln!("asr: (empty)");
                        }
                        overlay::hide();
                        continue;
                    }
                    Err(e) => {
                        eprintln!("asr: {e:#}");
                        overlay::set(Status::Error(format!("認識に失敗: {e:#}")));
                        continue;
                    }
                };

                if no_paste {
                    eprintln!("asr: {recognized}");
                    overlay::hide();
                    continue;
                }

                let inserted = if config.review.enabled {
                    match overlay::review(&recognized) {
                        Some(edited) => {
                            // Keys pressed while reviewing are not new
                            // utterances; drop them before the next loop.
                            drain_hotkeys(&hotkeys);
                            let edited = edited.trim().to_string();
                            if edited.is_empty() {
                                restore_focus(&focus);
                                overlay::hide();
                                continue;
                            }
                            if edited != recognized {
                                learn(&config, asr.as_mut().unwrap(), &recognized, &edited);
                            }
                            restore_focus(&focus);
                            // Let the app that had the focus take it back
                            // before Ctrl+V arrives.
                            std::thread::sleep(Duration::from_millis(120));
                            edited
                        }
                        None => {
                            drain_hotkeys(&hotkeys);
                            restore_focus(&focus);
                            overlay::hide();
                            continue;
                        }
                    }
                } else {
                    recognized
                };

                match clipboard::paste(&inserted, config.paste.delay_ms, config.paste.restore_clipboard) {
                    Ok(()) => overlay::hide(),
                    Err(e) => overlay::set(Status::Error(format!("貼り付けに失敗: {e}"))),
                }
            }
            Err(_) => break,
        }
    }
    overlay::quit();
}

/// Load `bias.yaml` into a decoder. Phrases that fail to parse leave the model
/// unmodified rather than stopping recognition.
fn configure_bias(config: &Config, asr: &mut Asr) -> Result<()> {
    if !config.bias.enabled {
        return Ok(());
    }
    let file = BiasFile::load(&config.bias_path())?;
    asr.set_phrases(&file.phrases, config.bias.weight);
    Ok(())
}

/// Fold a correction into `bias.yaml` and apply it to the running decoder.
fn learn(config: &Config, asr: &mut Asr, recognized: &str, corrected: &str) {
    match bias::learn(config, recognized, corrected) {
        Ok(phrases) => {
            if config.bias.enabled {
                asr.set_phrases(&phrases, config.bias.weight);
            }
        }
        Err(e) => eprintln!("bias: {e:#}"),
    }
}

fn drain_hotkeys(hotkeys: &Receiver<KeyEvent>) {
    while hotkeys.try_recv().is_ok() {}
}

fn restore_focus(focus: &Option<focus::Focus>) {
    if let Some(focus) = focus.as_ref() {
        focus::restore(focus);
    }
}

fn bias_command(config: &Config, args: &[String]) -> Result<()> {
    let path = config.bias_path();
    let mut file = BiasFile::load(&path)?;
    match args.first().map(String::as_str) {
        None => {
            if file.phrases.is_empty() {
                println!("no learned phrases ({})", path.display());
            } else {
                for phrase in &file.phrases {
                    let replacement = match &phrase.from {
                        Some(from) => format!("{from} → {}", phrase.to),
                        None => phrase.to.clone(),
                    };
                    match (&phrase.before, &phrase.after) {
                        (None, None) => println!("{:+.2}  {replacement}", phrase.weight),
                        (Some(before), None) => {
                            println!("{:+.2}  {before} | {replacement}", phrase.weight)
                        }
                        (None, Some(after)) => {
                            println!("{:+.2}  {replacement} | {after}", phrase.weight)
                        }
                        (Some(before), Some(after)) => println!(
                            "{:+.2}  {before} | {replacement} | {after}",
                            phrase.weight
                        ),
                    }
                }
                println!("{}", path.display());
            }
        }
        Some("add") => {
            let Some(text) = args.get(1) else {
                anyhow::bail!(
                    "usage: hasse bias add TEXT [--from TEXT] [--weight W] \
                     [--before CONTEXT] [--after CONTEXT]"
                );
            };
            let weight = flag_value(args, "--weight")
                .and_then(|w| w.parse::<f32>().ok())
                .unwrap_or(1.0);
            let from = flag_value(args, "--from");
            let before = flag_value(args, "--before");
            let after = flag_value(args, "--after");
            file.add(text, weight, from.as_deref(), before.as_deref(), after.as_deref());
            file.save(&path)?;
        }
        Some("rewrite") => {
            let Some(text) = args.get(1) else {
                anyhow::bail!("usage: hasse bias rewrite TEXT");
            };
            let tokenizer = bias::load_tokenizer(config)?;
            let compiled =
                bias::CompiledBias::compile(&file.phrases, &tokenizer, config.bias.weight);
            println!("{}", compiled.rewrite(text));
        }
        Some("rebuild") => {
            let tokenizer = bias::load_tokenizer(config)?;
            let log = Config::data_dir().join("corrections.jsonl");
            let (corrections, rebuilt) = bias::rebuild_from_log(&log, &tokenizer)?;
            rebuilt.save(&path)?;
            println!(
                "rebuilt {} rules from {corrections} corrections ({})",
                rebuilt.phrases.len(),
                path.display()
            );
        }
        Some("remove") => {
            let Some(text) = args.get(1) else {
                anyhow::bail!("usage: hasse bias remove TEXT");
            };
            if file.remove(text) {
                file.save(&path)?;
            } else {
                println!("not found: {text}");
            }
        }
        Some(other) => anyhow::bail!("unknown bias command {other:?}"),
    }
    Ok(())
}

/// Print the token ids the model vocabulary gives `text`. Kept for checking the
/// Rust SentencePiece encoder against the Python reference after a model
/// change.
fn tokenize_text(config: &Config, text: &str) -> Result<()> {
    let dir = config.models_dir()?;
    let vocab_path = dir.join("vocab.json");
    let vocab: Vec<String> = serde_json::from_str(
        &std::fs::read_to_string(&vocab_path)
            .with_context(|| format!("reading {}", vocab_path.display()))?,
    )
    .context("parsing vocab.json")?;
    let tokenizer = bias::Tokenizer::new(&vocab);
    for id in tokenizer.encode(text) {
        let piece = vocab.get(id).map(String::as_str).unwrap_or("");
        println!("{id:5}  {piece}");
    }
    Ok(())
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
