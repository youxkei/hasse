//! System tray icon and its menu.

use anyhow::{Context, Result};
use tray_icon::menu::{Menu, MenuEvent, MenuItem};
use tray_icon::{Icon, TrayIcon, TrayIconBuilder};

/// Create the tray icon. Must be called on the thread that runs the message loop.
pub fn create() -> Result<TrayIcon> {
    let menu = Menu::new();
    let quit = MenuItem::with_id("quit", "終了", true, None);
    menu.append(&quit).context("building the tray menu")?;
    let icon = Icon::from_rgba(draw_icon(), 32, 32).context("building the tray icon")?;
    let tray = TrayIconBuilder::new()
        .with_menu(Box::new(menu))
        .with_tooltip("hasse — 右Shiftで音声入力（終了はこのメニュー）")
        .with_icon(icon)
        .with_menu_on_left_click(false)
        .build()
        .context("creating the tray icon")?;
    Ok(tray)
}

/// True when the user chose 終了 from the tray menu.
pub fn quit_requested() -> bool {
    while let Ok(event) = MenuEvent::receiver().try_recv() {
        if event.id.0 == "quit" {
            return true;
        }
    }
    false
}

/// A 32x32 microphone on a rounded dark square.
fn draw_icon() -> Vec<u8> {
    const S: usize = 32;
    let mut px = vec![0u8; S * S * 4];
    let mut put = |x: usize, y: usize, c: [u8; 4]| {
        let i = (y * S + x) * 4;
        px[i..i + 4].copy_from_slice(&c);
    };
    let bg = [38u8, 43, 54, 255];
    let fg = [243u8, 246, 250, 255];

    for y in 0..S as i32 {
        for x in 0..S as i32 {
            let dx = if x < 6 { 6 - x } else if x > S as i32 - 7 { x - (S as i32 - 7) } else { 0 };
            let dy = if y < 6 { 6 - y } else if y > S as i32 - 7 { y - (S as i32 - 7) } else { 0 };
            if dx * dx + dy * dy <= 36 {
                put(x as usize, y as usize, bg);
            }
        }
    }

    let rect = |put: &mut dyn FnMut(usize, usize, [u8; 4]), x0: usize, x1: usize, y0: usize, y1: usize| {
        for y in y0..y1 {
            for x in x0..x1 {
                put(x, y, fg);
            }
        }
    };
    // Capsule.
    rect(&mut put, 13, 19, 7, 8);
    rect(&mut put, 12, 20, 8, 19);
    rect(&mut put, 13, 19, 19, 20);
    // U-shaped cradle.
    rect(&mut put, 9, 11, 15, 23);
    rect(&mut put, 21, 23, 15, 23);
    rect(&mut put, 10, 22, 22, 24);
    // Stand and base.
    rect(&mut put, 15, 17, 24, 27);
    rect(&mut put, 12, 20, 27, 28);

    px
}
