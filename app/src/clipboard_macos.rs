//! macOS text insertion: save the clipboard text, replace it, press Cmd+V, and
//! restore the text.
//!
//! Only text is preserved on macOS (the user's choice); images and other
//! pasteboard types are not snapshotted.

use anyhow::{anyhow, Result};
use arboard::Clipboard;
use core_graphics::event::{CGEvent, CGEventFlags, CGEventTapLocation};
use core_graphics::event_source::{CGEventSource, CGEventSourceStateID};
use std::thread::sleep;
use std::time::Duration;

/// Put `text` on the clipboard and press Cmd+V into the focused window.
pub fn paste(text: &str, delay_ms: u64, restore: bool) -> Result<()> {
    let mut clipboard = Clipboard::new().map_err(|e| anyhow!("clipboard: {e}"))?;
    let previous = if restore { clipboard.get_text().ok() } else { None };
    clipboard.set_text(text).map_err(|e| anyhow!("clipboard: {e}"))?;
    sleep(Duration::from_millis(30));
    send_cmd_v()?;
    sleep(Duration::from_millis(delay_ms));
    if let Some(previous) = previous {
        let _ = clipboard.set_text(previous);
    }
    Ok(())
}

pub fn get_text() -> String {
    Clipboard::new()
        .ok()
        .and_then(|mut clipboard| clipboard.get_text().ok())
        .unwrap_or_default()
}

/// Replacing and restoring the clipboard text, for `--clipboard-test`.
pub fn roundtrip(text: &str) -> Result<(String, String)> {
    let before = get_text();
    let mut clipboard = Clipboard::new().map_err(|e| anyhow!("clipboard: {e}"))?;
    clipboard.set_text(text).map_err(|e| anyhow!("clipboard: {e}"))?;
    clipboard.set_text(before.clone()).map_err(|e| anyhow!("clipboard: {e}"))?;
    Ok((before, get_text()))
}

fn send_cmd_v() -> Result<()> {
    const V: u16 = 9; // kVK_ANSI_V
    let source = CGEventSource::new(CGEventSourceStateID::HIDSystemState)
        .map_err(|_| anyhow!("creating a CGEventSource"))?;
    let down = CGEvent::new_keyboard_event(source.clone(), V, true)
        .map_err(|_| anyhow!("creating the Cmd+V key-down event"))?;
    down.set_flags(CGEventFlags::CGEventFlagCommand);
    down.post(CGEventTapLocation::HID);

    let up = CGEvent::new_keyboard_event(source, V, false)
        .map_err(|_| anyhow!("creating the Cmd+V key-up event"))?;
    up.set_flags(CGEventFlags::CGEventFlagCommand);
    up.post(CGEventTapLocation::HID);
    Ok(())
}
