//! Global push-to-talk key via a low-level keyboard hook.
//!
//! `RegisterHotKey` only reports presses, and cannot report a bare modifier's
//! release, so push-to-talk needs `WH_KEYBOARD_LL`. The hook runs on its own
//! thread with a message loop, as Windows requires.
//!
//! Quit is the tray menu's 終了, not a keyboard shortcut: a second global
//! shortcut would claim another key from every application.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;

use anyhow::{anyhow, Result};
use crossbeam_channel::{Receiver, Sender};
use windows::Win32::Foundation::{LPARAM, LRESULT, WPARAM};
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::UI::WindowsAndMessaging::{
    CallNextHookEx, DispatchMessageW, GetMessageW, SetWindowsHookExW, TranslateMessage,
    KBDLLHOOKSTRUCT, MSG, WH_KEYBOARD_LL, WM_KEYDOWN, WM_KEYUP, WM_SYSKEYDOWN, WM_SYSKEYUP,
};

pub use crate::key_event::KeyEvent;

struct HookConfig {
    vk: u32,
    swallow: bool,
    tx: Sender<KeyEvent>,
    held: AtomicBool,
}

static CFG: OnceLock<HookConfig> = OnceLock::new();

pub fn spawn(key: &str, swallow: bool) -> Result<Receiver<KeyEvent>> {
    let vk = parse_key(key)?;
    let (tx, rx) = crossbeam_channel::unbounded();
    CFG.set(HookConfig { vk, swallow, tx, held: AtomicBool::new(false) })
        .map_err(|_| anyhow!("hook already running"))?;

    std::thread::Builder::new()
        .name("hasse-hotkey".into())
        .spawn(|| {
            let result = unsafe {
                let module = GetModuleHandleW(None).ok().map(Into::into);
                SetWindowsHookExW(WH_KEYBOARD_LL, Some(hook_proc), module, 0)
            };
            let hook = match result {
                Ok(h) => h,
                Err(e) => {
                    eprintln!("hotkey: failed to install hook: {e}");
                    return;
                }
            };
            let mut msg = MSG::default();
            unsafe {
                while GetMessageW(&mut msg, None, 0, 0).0 > 0 {
                    let _ = TranslateMessage(&msg);
                    DispatchMessageW(&msg);
                }
            }
            let _ = hook;
        })?;
    Ok(rx)
}

unsafe extern "system" fn hook_proc(code: i32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
    if code < 0 {
        return unsafe { CallNextHookEx(None, code, wparam, lparam) };
    }
    let Some(cfg) = CFG.get() else {
        return unsafe { CallNextHookEx(None, code, wparam, lparam) };
    };
    let kb = unsafe { &*(lparam.0 as *const KBDLLHOOKSTRUCT) };
    let msg = wparam.0 as u32;
    let down = msg == WM_KEYDOWN || msg == WM_SYSKEYDOWN;
    let up = msg == WM_KEYUP || msg == WM_SYSKEYUP;

    if kb.vkCode == cfg.vk {
        if down {
            if !cfg.held.swap(true, Ordering::SeqCst) {
                let _ = cfg.tx.send(KeyEvent::Pressed);
            }
            if cfg.swallow {
                return LRESULT(1);
            }
        } else if up && cfg.held.swap(false, Ordering::SeqCst) {
            let _ = cfg.tx.send(KeyEvent::Released);
            if cfg.swallow {
                return LRESULT(1);
            }
        }
    }
    unsafe { CallNextHookEx(None, code, wparam, lparam) }
}

pub fn parse_key(name: &str) -> Result<u32> {
    let n = name.trim();
    let lower = n.to_ascii_lowercase();
    let vk = match lower.as_str() {
        "rightctrl" => 0xA3,
        "leftctrl" => 0xA2,
        "rightalt" => 0xA5,
        "leftalt" => 0xA4,
        "rightshift" => 0xA1,
        "leftshift" => 0xA0,
        "capslock" => 0x14,
        "space" => 0x20,
        "scrolllock" => 0x91,
        "pause" => 0x13,
        "tab" => 0x09,
        "insert" => 0x2D,
        "delete" => 0x2E,
        "home" => 0x24,
        "end" => 0x23,
        "pageup" => 0x21,
        "pagedown" => 0x22,
        _ => {
            if let Some(rest) = lower.strip_prefix('f') {
                if let Ok(n) = rest.parse::<u32>() {
                    if (1..=24).contains(&n) {
                        0x70 + n - 1
                    } else {
                        return Err(anyhow!("unknown key {name:?}"));
                    }
                } else {
                    return Err(anyhow!("unknown key {name:?}"));
                }
            } else if n.chars().count() == 1 {
                let c = n.chars().next().unwrap().to_ascii_uppercase();
                if c.is_ascii_alphanumeric() {
                    c as u32
                } else {
                    return Err(anyhow!("unknown key {name:?}"));
                }
            } else {
                return Err(anyhow!("unknown key {name:?}"));
            }
        }
    };
    Ok(vk)
}
