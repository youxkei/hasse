//! A tiny always-on-top, click-through indicator drawn with GDI.

use std::sync::atomic::{AtomicIsize, AtomicU32, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};
use windows::core::w;
use windows::Win32::Foundation::{COLORREF, HWND, LPARAM, LRESULT, RECT, WPARAM};
use windows::Win32::Graphics::Gdi::{
    BeginPaint, BitBlt, CreateCompatibleBitmap, CreateCompatibleDC, CreateFontW, CreateRoundRectRgn,
    CreateSolidBrush, DeleteDC, DeleteObject, DrawTextW, EndPaint, FillRect, FillRgn, HBRUSH,
    HGDIOBJ, HFONT, InvalidateRect, PAINTSTRUCT, SelectObject, SetBkMode, SetTextColor, SRCCOPY,
    CLEARTYPE_QUALITY, CLIP_DEFAULT_PRECIS, DEFAULT_CHARSET, DEFAULT_PITCH, DT_CENTER,
    DT_SINGLELINE, DT_VCENTER, FF_DONTCARE, FW_SEMIBOLD, TRANSPARENT,
};
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::UI::WindowsAndMessaging::{
    CreateWindowExW, DefWindowProcW, DispatchMessageW, GetClientRect, GetMessageW, LWA_ALPHA,
    LWA_COLORKEY, PostMessageW, RegisterClassW, SetLayeredWindowAttributes, SetTimer, ShowWindow,
    SystemParametersInfoW, TranslateMessage, MSG, SW_HIDE, SW_SHOWNOACTIVATE, SPI_GETWORKAREA,
    SYSTEM_PARAMETERS_INFO_UPDATE_FLAGS, WM_APP, WM_CLOSE, WM_DESTROY, WM_PAINT, WM_TIMER,
    WNDCLASSW, WS_EX_LAYERED, WS_EX_NOACTIVATE, WS_EX_TOOLWINDOW, WS_EX_TOPMOST, WS_EX_TRANSPARENT,
    WS_POPUP,
};

const WM_UPDATE: u32 = WM_APP + 1;
const TIMER_ID: usize = 1;
const WIDTH: i32 = 300;
const HEIGHT: i32 = 68;
/// The color-key pixels outside the rounded panel: magenta, absent from the palette.
const KEY_COLOR: COLORREF = COLORREF(0x00FF_00FF);
/// Opacity of the panel itself.
const OPACITY: u8 = 190;

pub use crate::status::Status;

struct UiState {
    text: String,
    accent: COLORREF,
    visible: bool,
    recording: bool,
    hide_at: Option<Instant>,
}

static UI: OnceLock<Mutex<UiState>> = OnceLock::new();
static LEVEL: OnceLock<Arc<AtomicU32>> = OnceLock::new();
static WINDOW: AtomicIsize = AtomicIsize::new(0);
static FONT: AtomicIsize = AtomicIsize::new(0);

fn ui() -> &'static Mutex<UiState> {
    UI.get_or_init(|| {
        Mutex::new(UiState {
            text: String::new(),
            accent: color(120, 170, 255),
            visible: false,
            recording: false,
            hide_at: None,
        })
    })
}

fn color(r: u8, g: u8, b: u8) -> COLORREF {
    COLORREF(r as u32 | ((g as u32) << 8) | ((b as u32) << 16))
}

fn hwnd() -> Option<HWND> {
    match WINDOW.load(Ordering::SeqCst) {
        0 => None,
        v => Some(HWND(v as *mut core::ffi::c_void)),
    }
}

pub fn set_level_source(level: Arc<AtomicU32>) {
    let _ = LEVEL.set(level);
}

/// Create the window on the calling thread; then call [`run_loop`].
pub fn create() -> Result<()> {
    unsafe {
        let instance = GetModuleHandleW(None)?;
        let class = WNDCLASSW {
            style: Default::default(),
            lpfnWndProc: Some(wndproc),
            cbClsExtra: 0,
            cbWndExtra: 0,
            hInstance: instance.into(),
            hIcon: Default::default(),
            hCursor: Default::default(),
            hbrBackground: HBRUSH(std::ptr::null_mut()),
            lpszMenuName: windows::core::PCWSTR::null(),
            lpszClassName: w!("HasseOverlay"),
        };
        if RegisterClassW(&class) == 0 {
            return Err(anyhow!("RegisterClassW failed"));
        }

        let mut work = RECT::default();
        let _ = SystemParametersInfoW(
            SPI_GETWORKAREA,
            0,
            Some(&mut work as *mut RECT as *mut core::ffi::c_void),
            SYSTEM_PARAMETERS_INFO_UPDATE_FLAGS(0),
        );
        let x = work.left + ((work.right - work.left) - WIDTH) / 2;
        let y = work.bottom - HEIGHT - 40;

        let ex = WS_EX_LAYERED
            | WS_EX_TRANSPARENT
            | WS_EX_TOPMOST
            | WS_EX_NOACTIVATE
            | WS_EX_TOOLWINDOW;
        let window = CreateWindowExW(
            ex,
            w!("HasseOverlay"),
            w!("hasse"),
            WS_POPUP,
            x,
            y,
            WIDTH,
            HEIGHT,
            None,
            None,
            Some(instance.into()),
            None,
        )?;
        WINDOW.store(window.0 as isize, Ordering::SeqCst);
        let _ = SetLayeredWindowAttributes(window, KEY_COLOR, OPACITY, LWA_COLORKEY | LWA_ALPHA);
        // One timer drives the level animation, transient statuses, and the
        // tray menu, whether or not the overlay is visible.
        let _ = SetTimer(Some(window), TIMER_ID, 50, None);
        let font = CreateFontW(
            -18,
            0,
            0,
            0,
            FW_SEMIBOLD.0 as i32,
            0,
            0,
            0,
            DEFAULT_CHARSET,
            windows::Win32::Graphics::Gdi::OUT_DEFAULT_PRECIS,
            CLIP_DEFAULT_PRECIS,
            CLEARTYPE_QUALITY,
            (DEFAULT_PITCH.0 | FF_DONTCARE.0) as u32,
            w!("Yu Gothic UI"),
        );
        FONT.store(font.0 as isize, Ordering::SeqCst);
        Ok(())
    }
}

pub fn run_loop() -> Result<()> {
    let mut msg = MSG::default();
    unsafe {
        while GetMessageW(&mut msg, None, 0, 0).0 > 0 {
            let _ = TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
    }
    Ok(())
}

pub fn set(status: Status) {
    {
        let mut state = ui().lock().unwrap();
        state.recording = false;
        state.hide_at = None;
        match status {
            Status::Starting => {
                state.text = "モデルを読み込み中…".into();
                state.accent = color(120, 170, 255);
                state.visible = true;
            }
            Status::Recording => {
                state.text = "録音中 — 離すと認識します".into();
                state.accent = color(255, 90, 90);
                state.visible = true;
                state.recording = true;
            }
            Status::Transcribing => {
                state.text = "認識中…".into();
                state.accent = color(255, 190, 90);
                state.visible = true;
            }
            Status::Typed => {
                state.text = "入力しました".into();
                state.accent = color(110, 210, 140);
                state.visible = true;
                state.hide_at = Some(Instant::now() + Duration::from_millis(900));
            }
            Status::Error(message) => {
                state.text = message;
                state.accent = color(255, 110, 110);
                state.visible = true;
                state.hide_at = Some(Instant::now() + Duration::from_millis(3500));
            }
        }
    }
    post_update();
}

pub fn hide() {
    if let Ok(mut state) = ui().lock() {
        state.visible = false;
        state.hide_at = None;
    }
    post_update();
}

pub fn quit() {
    if let Some(window) = hwnd() {
        unsafe {
            let _ = PostMessageW(Some(window), WM_CLOSE, WPARAM(0), LPARAM(0));
        }
    }
}

fn post_update() {
    if let Some(window) = hwnd() {
        unsafe {
            let _ = PostMessageW(Some(window), WM_UPDATE, WPARAM(0), LPARAM(0));
        }
    }
}

unsafe extern "system" fn wndproc(window: HWND, msg: u32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
    match msg {
        WM_UPDATE => {
            let show = ui().lock().map(|s| s.visible).unwrap_or(false);
            unsafe {
                let _ = ShowWindow(window, if show { SW_SHOWNOACTIVATE } else { SW_HIDE });
                if show {
                    let _ = InvalidateRect(Some(window), None, false);
                }
            }
            LRESULT(0)
        }
        WM_TIMER => {
            if crate::tray::quit_requested() {
                unsafe {
                    let _ = PostMessageW(Some(window), WM_CLOSE, WPARAM(0), LPARAM(0));
                }
                return LRESULT(0);
            }
            let (visible, expired) = {
                let mut state = ui().lock().unwrap();
                let expired = state.hide_at.map(|t| Instant::now() >= t).unwrap_or(false);
                if expired {
                    state.visible = false;
                    state.hide_at = None;
                }
                (state.visible, expired)
            };
            unsafe {
                if expired {
                    let _ = ShowWindow(window, SW_HIDE);
                } else if visible {
                    let _ = InvalidateRect(Some(window), None, false);
                }
            }
            LRESULT(0)
        }
        WM_PAINT => {
            paint(window);
            LRESULT(0)
        }
        WM_DESTROY => {
            unsafe {
                windows::Win32::UI::WindowsAndMessaging::PostQuitMessage(0);
            }
            LRESULT(0)
        }
        _ => unsafe { DefWindowProcW(window, msg, wparam, lparam) },
    }
}

fn paint(window: HWND) {
    unsafe {
        let mut ps = PAINTSTRUCT::default();
        let hdc = BeginPaint(window, &mut ps);
        let mut rect = RECT::default();
        let _ = GetClientRect(window, &mut rect);
        let w = rect.right - rect.left;
        let h = rect.bottom - rect.top;

        let mem = CreateCompatibleDC(Some(hdc));
        let bmp = CreateCompatibleBitmap(hdc, w, h);
        let old = SelectObject(mem, HGDIOBJ(bmp.0));

        // Everything outside the rounded panel becomes the transparent color key.
        let key = CreateSolidBrush(KEY_COLOR);
        let full = RECT { left: 0, top: 0, right: w, bottom: h };
        let _ = FillRect(mem, &full, key);
        let _ = DeleteObject(HGDIOBJ(key.0));

        let bg = CreateSolidBrush(color(30, 30, 36));
        let region = CreateRoundRectRgn(0, 0, w + 1, h + 1, 18, 18);
        let _ = FillRgn(mem, region, bg);
        let _ = DeleteObject(HGDIOBJ(region.0));
        let _ = DeleteObject(HGDIOBJ(bg.0));

        let state = ui().lock().unwrap();
        let font = HFONT(FONT.load(Ordering::SeqCst) as *mut core::ffi::c_void);
        let old_font = SelectObject(mem, HGDIOBJ(font.0));
        let _ = SetBkMode(mem, TRANSPARENT);
        let _ = SetTextColor(mem, color(245, 245, 248));

        let mut text: Vec<u16> = state.text.encode_utf16().collect();
        let mut text_rect = RECT { left: 16, top: 10, right: w - 16, bottom: h - 18 };
        let _ = DrawTextW(mem, &mut text, &mut text_rect, DT_CENTER | DT_VCENTER | DT_SINGLELINE);

        let level = if state.recording {
            LEVEL
                .get()
                .map(|l| f32::from_bits(l.load(Ordering::Relaxed)).clamp(0.0, 1.0))
                .unwrap_or(0.0)
        } else {
            1.0
        };
        let bar_w = ((w - 32) as f32 * level.max(0.04)) as i32;
        let accent = CreateSolidBrush(state.accent);
        let bar = CreateRoundRectRgn(16, h - 12, 16 + bar_w, h - 7, 6, 6);
        let _ = FillRgn(mem, bar, accent);
        let _ = DeleteObject(HGDIOBJ(bar.0));
        let _ = DeleteObject(HGDIOBJ(accent.0));

        let _ = SelectObject(mem, old_font);
        let _ = BitBlt(hdc, 0, 0, w, h, Some(mem), 0, 0, SRCCOPY);
        let _ = SelectObject(mem, old);
        let _ = DeleteObject(HGDIOBJ(bmp.0));
        let _ = DeleteDC(mem);
        let _ = EndPaint(window, &ps);
    }
}
