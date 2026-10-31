//! Insert text by clipboard + Ctrl+V, restoring what was on the clipboard.
//!
//! The user asked for Handy's approach: keep the clipboard contents. We
//! enumerate every format the clipboard currently offers and duplicate each
//! handle with `OleDuplicateData`, so text, images, files and custom formats
//! all come back.

use std::sync::Once;
use std::time::Duration;

use anyhow::{anyhow, Result};
use windows::Win32::Foundation::{GlobalFree, HANDLE, HGLOBAL};
use windows::Win32::System::DataExchange::{
    CloseClipboard, EmptyClipboard, EnumClipboardFormats, GetClipboardData, OpenClipboard,
    SetClipboardData,
};
use windows::Win32::System::Memory::{
    GlobalAlloc, GlobalLock, GlobalUnlock, GLOBAL_ALLOC_FLAGS, GMEM_MOVEABLE,
};
use windows::Win32::System::Ole::{OleDuplicateData, OleInitialize, CF_UNICODETEXT, CLIPBOARD_FORMAT};
use windows::Win32::UI::Input::KeyboardAndMouse::{
    SendInput, INPUT, INPUT_KEYBOARD, KEYBDINPUT, KEYBD_EVENT_FLAGS, KEYEVENTF_KEYUP, VIRTUAL_KEY,
    VK_CONTROL,
};

static OLE: Once = Once::new();

struct Snapshot {
    items: Vec<(u16, HANDLE)>,
}

impl Snapshot {
    fn capture() -> Option<Self> {
        OLE.call_once(|| unsafe {
            let _ = OleInitialize(None);
        });
        if !open_clipboard() {
            return None;
        }
        let mut items = Vec::new();
        let mut format = 0u32;
        loop {
            format = unsafe { EnumClipboardFormats(format) };
            if format == 0 {
                break;
            }
            let Ok(handle) = (unsafe { GetClipboardData(format) }) else {
                continue;
            };
            if handle.is_invalid() {
                continue;
            }
            let dup = unsafe {
                OleDuplicateData(handle, CLIPBOARD_FORMAT(format as u16), GLOBAL_ALLOC_FLAGS(0))
            };
            if !dup.is_invalid() {
                items.push((format as u16, dup));
            }
        }
        let _ = unsafe { CloseClipboard() };
        Some(Self { items })
    }

    fn restore(self) {
        if !open_clipboard() {
            return;
        }
        unsafe {
            let _ = EmptyClipboard();
            for (format, handle) in self.items {
                if SetClipboardData(format as u32, Some(handle)).is_err() {
                    let _ = GlobalFree(Some(HGLOBAL(handle.0)));
                }
            }
            let _ = CloseClipboard();
        }
    }
}

/// Put `text` on the clipboard and press Ctrl+V into the focused window.
pub fn paste(text: &str, delay_ms: u64, restore: bool) -> Result<()> {
    let snapshot = if restore { Snapshot::capture() } else { None };
    if !set_text(text) {
        return Err(anyhow!("could not write the clipboard"));
    }
    std::thread::sleep(Duration::from_millis(30));
    send_ctrl_v();
    std::thread::sleep(Duration::from_millis(delay_ms));
    if let Some(snapshot) = snapshot {
        snapshot.restore();
    }
    Ok(())
}

/// Read the clipboard's Unicode text, for diagnostics.
pub fn get_text() -> String {
    if !open_clipboard() {
        return String::new();
    }
    let mut out = String::new();
    unsafe {
        if let Ok(handle) = GetClipboardData(CF_UNICODETEXT.0 as u32) {
            let hglobal = HGLOBAL(handle.0);
            let ptr = GlobalLock(hglobal);
            if !ptr.is_null() {
                let p = ptr as *const u16;
                let mut len = 0usize;
                while *p.add(len) != 0 {
                    len += 1;
                }
                out = String::from_utf16_lossy(std::slice::from_raw_parts(p, len));
                let _ = GlobalUnlock(hglobal);
            }
        }
        let _ = CloseClipboard();
    }
    out
}

/// Capture, replace and restore the clipboard; returns (before, after).
pub fn roundtrip(text: &str) -> Result<(String, String)> {
    let before = get_text();
    let snapshot = Snapshot::capture();
    if !set_text(text) {
        return Err(anyhow!("could not write the clipboard"));
    }
    if let Some(snapshot) = snapshot {
        snapshot.restore();
    }
    Ok((before, get_text()))
}

fn open_clipboard() -> bool {
    for _ in 0..10 {
        if unsafe { OpenClipboard(None) }.is_ok() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    false
}

fn set_text(text: &str) -> bool {
    let mut utf16: Vec<u16> = text.encode_utf16().collect();
    utf16.push(0);
    let bytes = utf16.len() * 2;
    unsafe {
        let Ok(handle) = GlobalAlloc(GMEM_MOVEABLE, bytes) else {
            return false;
        };
        let ptr = GlobalLock(handle);
        if ptr.is_null() {
            let _ = GlobalFree(Some(handle));
            return false;
        }
        std::ptr::copy_nonoverlapping(utf16.as_ptr() as *const u8, ptr as *mut u8, bytes);
        let _ = GlobalUnlock(handle);
        if !open_clipboard() {
            let _ = GlobalFree(Some(handle));
            return false;
        }
        let _ = EmptyClipboard();
        let ok = SetClipboardData(CF_UNICODETEXT.0 as u32, Some(HANDLE(handle.0))).is_ok();
        let _ = CloseClipboard();
        if !ok {
            let _ = GlobalFree(Some(handle));
        }
        ok
    }
}

fn send_ctrl_v() {
    let key = |vk: VIRTUAL_KEY, up: bool| {
        let mut input = INPUT::default();
        input.r#type = INPUT_KEYBOARD;
        input.Anonymous.ki = KEYBDINPUT {
            wVk: vk,
            wScan: 0,
            dwFlags: if up { KEYEVENTF_KEYUP } else { KEYBD_EVENT_FLAGS(0) },
            time: 0,
            dwExtraInfo: 0,
        };
        input
    };
    let inputs = [
        key(VK_CONTROL, false),
        key(VIRTUAL_KEY(0x56), false), // V
        key(VIRTUAL_KEY(0x56), true),
        key(VK_CONTROL, true),
    ];
    unsafe {
        SendInput(&inputs, std::mem::size_of::<INPUT>() as i32);
    }
}
