//! macOS push-to-talk: rdev's global keyboard listener.
//!
//! A global listener on macOS needs the Accessibility/Input Monitoring
//! permissions; `request_permissions` asks for them with the system prompt.
//! `swallow` has no equivalent here: the key also reaches the focused app.

use anyhow::{anyhow, Result};
use crossbeam_channel::Receiver;
use rdev::{listen, Event, EventType, Key};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

pub use crate::key_event::KeyEvent;

pub fn spawn(key: &str, swallow: bool) -> Result<Receiver<KeyEvent>> {
    let target = parse_key(key)?;
    if swallow {
        eprintln!("hotkey: macOS cannot swallow a key; it also reaches the focused app");
    }
    request_permissions();

    let (tx, rx) = crossbeam_channel::unbounded();
    let held = Arc::new(AtomicBool::new(false));
    std::thread::Builder::new()
        .name("hasse-hotkey".into())
        .spawn(move || {
            let callback = move |event: Event| match event.event_type {
                EventType::KeyPress(k) if k == target => {
                    if !held.swap(true, Ordering::SeqCst) {
                        let _ = tx.send(KeyEvent::Pressed);
                    }
                }
                EventType::KeyRelease(k) if k == target => {
                    if held.swap(false, Ordering::SeqCst) {
                        let _ = tx.send(KeyEvent::Released);
                    }
                }
                _ => {}
            };
            if let Err(e) = listen(callback) {
                eprintln!("hotkey: {e:?}; grant Accessibility and Input Monitoring to hasse");
            }
        })?;
    Ok(rx)
}

/// Ask for the two TCC permissions this app needs: listening to keys
/// (Input Monitoring) and posting Cmd+V (Accessibility).
pub fn request_permissions() {
    #[link(name = "CoreGraphics", kind = "framework")]
    unsafe extern "C" {
        fn CGPreflightListenEventAccess() -> bool;
        fn CGRequestListenEventAccess() -> bool;
        fn CGPreflightPostEventAccess() -> bool;
        fn CGRequestPostEventAccess() -> bool;
    }
    unsafe {
        if !CGPreflightListenEventAccess() {
            let _ = CGRequestListenEventAccess();
        }
        if !CGPreflightPostEventAccess() {
            let _ = CGRequestPostEventAccess();
        }
    }
}

pub fn parse_key(name: &str) -> Result<Key> {
    let n = name.trim();
    let lower = n.to_ascii_lowercase();
    let key = match lower.as_str() {
        "rightshift" => Key::ShiftRight,
        "leftshift" => Key::ShiftLeft,
        "rightctrl" => Key::ControlRight,
        "leftctrl" => Key::ControlLeft,
        "rightalt" => Key::AltGr,
        "leftalt" => Key::Alt,
        "rightmeta" | "rightcmd" => Key::MetaRight,
        "leftmeta" | "leftcmd" => Key::MetaLeft,
        "capslock" => Key::CapsLock,
        "space" => Key::Space,
        "scrolllock" => Key::ScrollLock,
        "pause" => Key::Pause,
        "tab" => Key::Tab,
        "insert" => Key::Insert,
        "delete" => Key::Delete,
        "home" => Key::Home,
        "end" => Key::End,
        "pageup" => Key::PageUp,
        "pagedown" => Key::PageDown,
        _ => {
            if let Some(rest) = lower.strip_prefix('f') {
                match rest.parse::<u32>() {
                    Ok(1) => Key::F1,
                    Ok(2) => Key::F2,
                    Ok(3) => Key::F3,
                    Ok(4) => Key::F4,
                    Ok(5) => Key::F5,
                    Ok(6) => Key::F6,
                    Ok(7) => Key::F7,
                    Ok(8) => Key::F8,
                    Ok(9) => Key::F9,
                    Ok(10) => Key::F10,
                    Ok(11) => Key::F11,
                    Ok(12) => Key::F12,
                    _ => return Err(anyhow!("unknown key {name:?}")),
                }
            } else if n.chars().count() == 1 {
                let c = n.chars().next().unwrap().to_ascii_uppercase();
                match c {
                    'A' => Key::KeyA,
                    'B' => Key::KeyB,
                    'C' => Key::KeyC,
                    'D' => Key::KeyD,
                    'E' => Key::KeyE,
                    'F' => Key::KeyF,
                    'G' => Key::KeyG,
                    'H' => Key::KeyH,
                    'I' => Key::KeyI,
                    'J' => Key::KeyJ,
                    'K' => Key::KeyK,
                    'L' => Key::KeyL,
                    'M' => Key::KeyM,
                    'N' => Key::KeyN,
                    'O' => Key::KeyO,
                    'P' => Key::KeyP,
                    'Q' => Key::KeyQ,
                    'R' => Key::KeyR,
                    'S' => Key::KeyS,
                    'T' => Key::KeyT,
                    'U' => Key::KeyU,
                    'V' => Key::KeyV,
                    'W' => Key::KeyW,
                    'X' => Key::KeyX,
                    'Y' => Key::KeyY,
                    'Z' => Key::KeyZ,
                    '0' => Key::Num0,
                    '1' => Key::Num1,
                    '2' => Key::Num2,
                    '3' => Key::Num3,
                    '4' => Key::Num4,
                    '5' => Key::Num5,
                    '6' => Key::Num6,
                    '7' => Key::Num7,
                    '8' => Key::Num8,
                    '9' => Key::Num9,
                    _ => return Err(anyhow!("unknown key {name:?}")),
                }
            } else {
                return Err(anyhow!("unknown key {name:?}"));
            }
        }
    };
    Ok(key)
}
