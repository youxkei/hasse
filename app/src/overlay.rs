//! Bottom-center overlay and the main message loop on Windows.
//!
//! The window is created once, with `CreateWindowExW`, as a borderless tool
//! dialog: `WS_EX_TOOLWINDOW | WS_EX_DLGMODALFRAME` are in the extended style
//! before the window is first shown. A tiling window manager reads those bits
//! when it first sees the window and decides from them whether to manage it, so
//! setting them after the fact is too late.
//!
//! While idle and recording the window is click-through (`WS_EX_TRANSPARENT |
//! WS_EX_NOACTIVATE`) and layered, with the color key making the area outside
//! the rounded panel transparent. A finished transcript is handed to [`review`],
//! which turns those two bits off, gives the window a rounded region, puts a
//! native multi-line edit over the panel, and waits for Ctrl+Enter (or Esc).
//! Both states keep the layer for the translucent panel; the review takes its
//! shape from the region so that the edit is not clipped by keyed-out pixels.
//! The edit control is the Win32 one, so the Japanese IME behaves exactly as
//! it does in any other application.
//!
//! Enter is left alone because it confirms an IME conversion; Ctrl+Enter is the
//! only key that cannot collide with Japanese input.

use std::sync::atomic::{AtomicIsize, AtomicU32, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};
use windows::core::{w, PCWSTR};
use windows::Win32::Foundation::{COLORREF, HWND, LPARAM, LRESULT, POINT, RECT, WPARAM};
use windows::Win32::Graphics::Gdi::{
    BeginPaint, BitBlt, CreateCompatibleBitmap, CreateCompatibleDC, CreateFontW,
    CreateRoundRectRgn, CreateSolidBrush, DeleteDC, DeleteObject, DrawTextW, EndPaint, FillRect,
    FillRgn, HBRUSH, HDC, HGDIOBJ, HFONT, InvalidateRect, MonitorFromPoint, PAINTSTRUCT,
    SelectObject, SetBkColor, SetBkMode, SetTextColor, SetWindowRgn, MONITOR_DEFAULTTONEAREST,
    SRCCOPY,
    CLEARTYPE_QUALITY, CLIP_DEFAULT_PRECIS, DEFAULT_CHARSET, DEFAULT_PITCH, DT_CENTER,
    DT_SINGLELINE, DT_VCENTER, FF_DONTCARE, FW_SEMIBOLD, TRANSPARENT,
};
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::System::Threading::{AttachThreadInput, GetCurrentThreadId};
use windows::Win32::UI::HiDpi::{GetDpiForMonitor, GetDpiForWindow, MDT_EFFECTIVE_DPI};
use windows::Win32::UI::Input::KeyboardAndMouse::{
    GetKeyState, SetActiveWindow, SetFocus, VK_CONTROL, VK_ESCAPE, VK_RETURN,
};
use windows::Win32::UI::WindowsAndMessaging::{
    BringWindowToTop, CallWindowProcW, CreateWindowExW, DefWindowProcW, DispatchMessageW,
    GetClientRect, GetForegroundWindow, GetMessageW, GetWindowTextLengthW, GetWindowTextW,
    GetWindowThreadProcessId, MoveWindow,
    PostMessageW, PostQuitMessage, RegisterClassW, SendMessageW, SetForegroundWindow,
    SetLayeredWindowAttributes, SetTimer, SetWindowLongPtrW, SetWindowPos, SetWindowTextW,
    ShowWindow, SwitchToThisWindow, SystemParametersInfoW, TranslateMessage, ES_AUTOVSCROLL,
    ES_MULTILINE, ES_WANTRETURN, GWLP_WNDPROC, GWL_EXSTYLE, LWA_ALPHA, LWA_COLORKEY, MSG,
    SPI_GETWORKAREA, SWP_FRAMECHANGED, SWP_NOACTIVATE, SWP_NOMOVE, SWP_NOSIZE, SWP_NOZORDER,
    SW_HIDE, SW_SHOW, SW_SHOWNOACTIVATE, SYSTEM_PARAMETERS_INFO_UPDATE_FLAGS, WM_APP, WM_CLOSE,
    WM_CTLCOLOREDIT, WM_DESTROY, WM_DPICHANGED, WM_KEYDOWN, WM_NCCALCSIZE, WM_PAINT, WM_SETFONT,
    WM_TIMER, WNDCLASSW, WNDPROC,
    WS_CHILD, WS_EX_DLGMODALFRAME, WS_EX_LAYERED, WS_EX_NOACTIVATE, WS_EX_TOOLWINDOW,
    WS_EX_TOPMOST, WS_EX_TRANSPARENT, WS_POPUP, WS_VISIBLE, WINDOW_STYLE,
};

const WM_UPDATE: u32 = WM_APP + 1;
const WM_REVIEW_DONE: u32 = WM_APP + 2;
const TIMER_ID: usize = 1;
/// Logical sizes, scaled by the DPI of the monitor the overlay lands on.
const STATUS_SIZE: (i32, i32) = (300, 68);
const REVIEW_SIZE: (i32, i32) = (720, 156);
/// Distance from the bottom of the work area.
const BOTTOM_MARGIN: i32 = 40;
/// Height of the text in the status panel, in logical pixels.
const FONT_HEIGHT: i32 = 18;
/// The color-key pixels outside the rounded panel: magenta, absent from the palette.
const KEY_COLOR: COLORREF = COLORREF(0x00FF_00FF);
/// Opacity of the panel itself.
const OPACITY: u8 = 190;

/// The styles the window is created with and keeps for its whole life. The two
/// that make a tiling window manager leave it alone are there from the start.
const DIALOG_EX: u32 =
    WS_EX_TOOLWINDOW.0 | WS_EX_DLGMODALFRAME.0 | WS_EX_TOPMOST.0 | WS_EX_LAYERED.0;
/// Click-through and never activated, for the idle and recording states.
const PASSIVE_EX: u32 = DIALOG_EX | WS_EX_TRANSPARENT.0 | WS_EX_NOACTIVATE.0;

pub use crate::status::Status;

struct UiState {
    text: String,
    accent: COLORREF,
    visible: bool,
    recording: bool,
    hide_at: Option<Instant>,
    /// A transcript waiting to be edited, set by the controller thread.
    pending_review: Option<String>,
    reviewing: bool,
}

static UI: OnceLock<Mutex<UiState>> = OnceLock::new();
static LEVEL: OnceLock<Arc<AtomicU32>> = OnceLock::new();
/// Work area of the monitor to place the overlay on, in pixels as this process
/// sees them; `None` for the primary monitor.
static ANCHOR: Mutex<Option<[f32; 4]>> = Mutex::new(None);
static CONFIRM: OnceLock<(Sender<Option<String>>, Mutex<Receiver<Option<String>>>)> =
    OnceLock::new();
static WINDOW: AtomicIsize = AtomicIsize::new(0);
static EDIT: AtomicIsize = AtomicIsize::new(0);
static EDIT_OLD_PROC: AtomicIsize = AtomicIsize::new(0);
static FONT: AtomicIsize = AtomicIsize::new(0);
/// The scale `FONT` was created for, as `f32` bits.
static FONT_SCALE: AtomicU32 = AtomicU32::new(0);
static EDIT_BRUSH: AtomicIsize = AtomicIsize::new(0);
/// The input level the panel currently shows, as `f32` bits, so a repaint
/// happens only when the bar actually moves.
static LEVEL_SHOWN: AtomicU32 = AtomicU32::new(u32::MAX);

fn ui() -> &'static Mutex<UiState> {
    UI.get_or_init(|| {
        Mutex::new(UiState {
            text: String::new(),
            accent: color(120, 170, 255),
            visible: false,
            recording: false,
            hide_at: None,
            pending_review: None,
            reviewing: false,
        })
    })
}

fn confirm_channel() -> &'static (Sender<Option<String>>, Mutex<Receiver<Option<String>>>) {
    CONFIRM.get_or_init(|| {
        let (tx, rx) = std::sync::mpsc::channel();
        (tx, Mutex::new(rx))
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

fn edit() -> Option<HWND> {
    match EDIT.load(Ordering::SeqCst) {
        0 => None,
        v => Some(HWND(v as *mut core::ffi::c_void)),
    }
}

pub fn set_level_source(level: Arc<AtomicU32>) {
    let _ = LEVEL.set(level);
}

/// Choose the monitor for the overlay: the work area of the monitor that held
/// the focused window when recording started.
pub fn set_anchor(bounds: Option<[f32; 4]>) {
    if let Ok(mut anchor) = ANCHOR.lock() {
        *anchor = bounds;
    }
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
    if let Some((tx, _)) = CONFIRM.get() {
        let _ = tx.send(None);
    }
}

/// Show `text` for editing and block until the user presses Ctrl+Enter
/// (`Some`) or Esc (`None`). The caller inserts and learns from the returned
/// text.
pub fn review(text: &str) -> Option<String> {
    {
        let mut state = ui().lock().unwrap();
        state.pending_review = Some(text.to_string());
        state.visible = true;
        state.recording = false;
        state.hide_at = None;
        state.text.clear();
    }
    post_update();
    confirm_channel().1.lock().unwrap().recv().ok().flatten()
}

pub fn run_loop() -> Result<()> {
    create()?;
    pump();
    Ok(())
}

/// The message loop. `TranslateMessage` is what turns a keydown into the
/// `WM_CHAR` the edit inserts, so the confirm gesture is kept away from it: the
/// edit posts the confirmation from `WM_KEYDOWN`, and the matching `WM_CHAR`
/// for Return would otherwise land in the text first.
fn pump() {
    let mut msg = MSG::default();
    unsafe {
        while GetMessageW(&mut msg, None, 0, 0).0 > 0 {
            if !is_confirm_key(&msg) {
                let _ = TranslateMessage(&msg);
            }
            DispatchMessageW(&msg);
        }
    }
}

/// Whether this message is Ctrl+Enter delivered to the review panel while it is
/// open. A keydown aimed anywhere else is left to the application that owns it.
fn is_confirm_key(msg: &MSG) -> bool {
    if msg.message != WM_KEYDOWN || msg.wParam.0 as u32 != VK_RETURN.0 as u32 {
        return false;
    }
    if !ui().lock().map(|s| s.reviewing).unwrap_or(false) {
        return false;
    }
    let Some(edit) = edit() else {
        return false;
    };
    if msg.hwnd != edit && Some(msg.hwnd) != hwnd() {
        return false;
    }
    unsafe { GetKeyState(VK_CONTROL.0 as i32) < 0 }
}

fn create() -> Result<()> {
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
            lpszMenuName: PCWSTR::null(),
            lpszClassName: w!("HasseOverlay"),
        };
        if RegisterClassW(&class) == 0 {
            return Err(anyhow!("RegisterClassW failed"));
        }

        let area = work_area();
        let (width, height) = STATUS_SIZE;
        let window = CreateWindowExW(
            // Tool window and dialog frame from creation: the window manager
            // decides from these bits on the first show, and never revisits.
            windows::Win32::UI::WindowsAndMessaging::WINDOW_EX_STYLE(PASSIVE_EX),
            w!("HasseOverlay"),
            w!("hasse"),
            WS_POPUP,
            area.left + ((area.right - area.left) - width) / 2,
            area.bottom - height - BOTTOM_MARGIN,
            width,
            height,
            None,
            None,
            Some(instance.into()),
            None,
        )?;
        WINDOW.store(window.0 as isize, Ordering::SeqCst);
        let _ = SetLayeredWindowAttributes(window, KEY_COLOR, OPACITY, LWA_COLORKEY | LWA_ALPHA);
        // DWM draws a one-pixel border and rounds the corners of a top-level
        // window by itself; the panel is shaped by the color key instead, so
        // both would show as a frame outside it.
        let no_border = windows::Win32::Graphics::Dwm::DWMWA_COLOR_NONE;
        let _ = windows::Win32::Graphics::Dwm::DwmSetWindowAttribute(
            window,
            windows::Win32::Graphics::Dwm::DWMWA_BORDER_COLOR,
            &no_border as *const u32 as *const core::ffi::c_void,
            std::mem::size_of::<u32>() as u32,
        );
        let donotround = windows::Win32::Graphics::Dwm::DWMWCP_DONOTROUND;
        let _ = windows::Win32::Graphics::Dwm::DwmSetWindowAttribute(
            window,
            windows::Win32::Graphics::Dwm::DWMWA_WINDOW_CORNER_PREFERENCE,
            &donotround as *const _ as *const core::ffi::c_void,
            std::mem::size_of::<i32>() as u32,
        );
        // One timer drives the level animation, transient statuses, and the
        // tray menu, whether or not the overlay is visible.
        let _ = SetTimer(Some(window), TIMER_ID, 50, None);
        place(window, STATUS_SIZE);
        EDIT_BRUSH.store(
            CreateSolidBrush(color(18, 18, 22)).0 as isize,
            Ordering::SeqCst,
        );
        // A review can be requested before the window exists.
        post_update();
        Ok(())
    }
}

fn post_update() {
    if let Some(window) = hwnd() {
        unsafe {
            let _ = PostMessageW(Some(window), WM_UPDATE, WPARAM(0), LPARAM(0));
        }
    }
}

fn work_area() -> RECT {
    let mut rect = RECT::default();
    unsafe {
        let _ = SystemParametersInfoW(
            SPI_GETWORKAREA,
            0,
            Some(&mut rect as *mut RECT as *mut core::ffi::c_void),
            SYSTEM_PARAMETERS_INFO_UPDATE_FLAGS(0),
        );
    }
    rect
}

/// The work area chosen by [`set_anchor`], or the primary monitor's.
fn target_area() -> RECT {
    let anchor = ANCHOR.lock().map(|a| *a).unwrap_or(None);
    match anchor {
        Some([left, top, width, height]) => RECT {
            left: left as i32,
            top: top as i32,
            right: (left + width) as i32,
            bottom: (top + height) as i32,
        },
        None => work_area(),
    }
}

/// The DPI scale of the monitor that covers most of `area`.
fn area_scale(area: RECT) -> f32 {
    unsafe {
        let center = POINT {
            x: (area.left + area.right) / 2,
            y: (area.top + area.bottom) / 2,
        };
        let monitor = MonitorFromPoint(center, MONITOR_DEFAULTTONEAREST);
        let mut dpi_x = 96u32;
        let mut dpi_y = 96u32;
        let _ = GetDpiForMonitor(monitor, MDT_EFFECTIVE_DPI, &mut dpi_x, &mut dpi_y);
        dpi_x as f32 / 96.0
    }
}

/// The DPI scale of the monitor a window is already on.
fn window_scale(window: HWND) -> f32 {
    unsafe {
        let dpi = GetDpiForWindow(window);
        if dpi == 0 {
            1.0
        } else {
            dpi as f32 / 96.0
        }
    }
}

/// Size `logical` for `scale`.
fn scaled(size: (i32, i32), scale: f32) -> (i32, i32) {
    (
        (size.0 as f32 * scale).round() as i32,
        (size.1 as f32 * scale).round() as i32,
    )
}

/// Place the window bottom-center on the target monitor and return the scale
/// its monitor uses.
fn place(window: HWND, size: (i32, i32)) -> f32 {
    let area = target_area();
    let scale = area_scale(area);
    let (width, height) = scaled(size, scale);
    let margin = (BOTTOM_MARGIN as f32 * scale).round() as i32;
    let x = area.left + ((area.right - area.left) - width) / 2;
    let y = area.bottom - height - margin;
    unsafe {
        let _ = MoveWindow(window, x, y, width, height, true);
    }
    scale
}

/// The font is recreated when the overlay moves to a monitor with another DPI.
fn ensure_font(scale: f32) -> HFONT {
    if (f32::from_bits(FONT_SCALE.load(Ordering::SeqCst)) - scale).abs() < 0.01 {
        return HFONT(FONT.load(Ordering::SeqCst) as *mut core::ffi::c_void);
    }
    unsafe {
        let font = CreateFontW(
            -((FONT_HEIGHT as f32 * scale).round() as i32),
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
        let old = FONT.swap(font.0 as isize, Ordering::SeqCst);
        if old != 0 {
            let _ = DeleteObject(HGDIOBJ(old as *mut core::ffi::c_void));
        }
        FONT_SCALE.store(scale.to_bits(), Ordering::SeqCst);
        font
    }
}

/// Write the extended style and let Windows re-evaluate the window. Only the
/// bits that change are written; the dialog and tool bits stay as they were
/// set at creation.
fn set_ex(window: HWND, ex: u32) {
    unsafe {
        SetWindowLongPtrW(window, GWL_EXSTYLE, ex as isize);
        let _ = SetWindowPos(
            window,
            None,
            0,
            0,
            0,
            0,
            SWP_FRAMECHANGED | SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE,
        );
    }
}

/// The color key makes the pixels outside the rounded panel transparent. The
/// idle and recording panel keeps it: it has no child window, so the whole
/// window is redrawn by [`paint`] and there is no other source of pixels.
fn use_color_key(window: HWND) {
    unsafe {
        let _ = SetLayeredWindowAttributes(window, KEY_COLOR, OPACITY, LWA_COLORKEY | LWA_ALPHA);
    }
}

/// The review panel is transparent as a whole and takes its rounded shape from
/// a window region, so no pixel needs to be keyed out.
fn use_alpha(window: HWND) {
    unsafe {
        let _ = SetLayeredWindowAttributes(window, COLORREF(0), OPACITY, LWA_ALPHA);
    }
}

/// Clip the window to the rounded panel, for the states without a layer.
fn round_window(window: HWND, scale: f32) {
    unsafe {
        let mut rect = RECT::default();
        let _ = GetClientRect(window, &mut rect);
        let radius = (18.0 * scale) as i32;
        let region = CreateRoundRectRgn(0, 0, rect.right + 1, rect.bottom + 1, radius, radius);
        let _ = SetWindowRgn(window, Some(region), true);
    }
}

/// The layered states take their shape from the color key.
fn clear_round(window: HWND) {
    unsafe {
        let _ = SetWindowRgn(window, None, true);
    }
}

/// The edit control is created once and hidden between reviews.
fn ensure_edit(window: HWND) -> Option<HWND> {
    if let Some(edit) = edit() {
        return Some(edit);
    }
    unsafe {
        let instance = GetModuleHandleW(None).ok()?;
        let style = WINDOW_STYLE((ES_MULTILINE | ES_AUTOVSCROLL | ES_WANTRETURN) as u32)
            | WS_CHILD
            | WS_VISIBLE;
        let edit = CreateWindowExW(
            Default::default(),
            w!("EDIT"),
            PCWSTR::null(),
            style,
            0,
            0,
            0,
            0,
            Some(window),
            None,
            Some(instance.into()),
            None,
        )
        .ok()?;
        EDIT.store(edit.0 as isize, Ordering::SeqCst);
        let old = SetWindowLongPtrW(edit, GWLP_WNDPROC, edit_proc as *const () as usize as isize);
        EDIT_OLD_PROC.store(old, Ordering::SeqCst);
        Some(edit)
    }
}

fn begin_review(window: HWND) {
    let Some(edit) = ensure_edit(window) else {
        return;
    };
    let text = {
        let mut state = ui().lock().unwrap();
        state.reviewing = true;
        state.visible = true;
        state.pending_review.take()
    };
    let Some(text) = text else {
        ui().lock().unwrap().reviewing = false;
        return;
    };
    let scale = place(window, REVIEW_SIZE);
    round_window(window, scale);
    set_ex(window, DIALOG_EX);
    use_alpha(window);
    let font = ensure_font(scale);
    let wide: Vec<u16> = text.encode_utf16().chain([0]).collect();
    unsafe {
        let _ = SetWindowTextW(edit, PCWSTR(wide.as_ptr()));
        let _ = SendMessageW(
            edit,
            WM_SETFONT,
            Some(WPARAM(font.0 as usize)),
            Some(LPARAM(1)),
        );
        layout_edit(window, edit, scale);
        let _ = ShowWindow(edit, SW_SHOW);
        let _ = ShowWindow(window, SW_SHOW);
        take_foreground(window, edit);
        let _ = InvalidateRect(Some(window), None, false);
    }
}

/// Bring the panel to the foreground and give the edit the keyboard.
///
/// Windows refuses `SetForegroundWindow` for a process that did not receive the
/// last input event, which a push-to-talk IME usually has not: the keys went to
/// the application that kept the focus while the transcript was produced.
/// Sharing the foreground thread's input state for the length of the call lifts
/// that refusal; `SwitchToThisWindow` is the fallback for when it does not.
fn take_foreground(window: HWND, edit: HWND) {
    unsafe {
        let foreground = GetForegroundWindow();
        let ours = GetCurrentThreadId();
        let theirs = GetWindowThreadProcessId(foreground, None);
        let attached = theirs != 0
            && theirs != ours
            && AttachThreadInput(theirs, ours, true).as_bool();
        let _ = BringWindowToTop(window);
        let _ = SetForegroundWindow(window);
        let _ = SetActiveWindow(window);
        let _ = SetFocus(Some(edit));
        if attached {
            let _ = AttachThreadInput(theirs, ours, false);
        }
        if GetForegroundWindow() != window {
            SwitchToThisWindow(window, true);
            let _ = SetFocus(Some(edit));
        }
    }
}

/// Fill the panel below the hint line with the edit control.
fn layout_edit(window: HWND, edit: HWND, scale: f32) {
    let mut rect = RECT::default();
    unsafe {
        let _ = GetClientRect(window, &mut rect);
        let (width, height) = (rect.right - rect.left, rect.bottom - rect.top);
        let _ = MoveWindow(
            edit,
            (16.0 * scale) as i32,
            (38.0 * scale) as i32,
            width - (32.0 * scale) as i32,
            height - (50.0 * scale) as i32,
            true,
        );
    }
}

/// Read the edit, hand the result to [`review`], and go back to click-through.
fn end_review(window: HWND, confirmed: bool) {
    let text = if confirmed {
        edit().and_then(read_edit_text).unwrap_or_default()
    } else {
        String::new()
    };
    {
        let mut state = ui().lock().unwrap();
        state.reviewing = false;
        state.visible = false;
        state.text.clear();
        state.pending_review = None;
    }
    if let Some(edit) = edit() {
        unsafe {
            let _ = ShowWindow(edit, SW_HIDE);
        }
    }
    unsafe {
        let _ = ShowWindow(window, SW_HIDE);
    }
    set_ex(window, PASSIVE_EX);
    use_color_key(window);
    clear_round(window);
    if let Some((tx, _)) = CONFIRM.get() {
        let _ = tx.send(if confirmed { Some(text) } else { None });
    }
}

fn read_edit_text(edit: HWND) -> Option<String> {
    unsafe {
        let length = GetWindowTextLengthW(edit);
        let mut buffer = vec![0u16; length as usize + 1];
        let written = GetWindowTextW(edit, &mut buffer);
        Some(String::from_utf16_lossy(&buffer[..written as usize]))
    }
}

fn post_review_done(confirmed: bool) {
    if let Some(window) = hwnd() {
        unsafe {
            let _ = PostMessageW(
                Some(window),
                WM_REVIEW_DONE,
                WPARAM(confirmed as usize),
                LPARAM(0),
            );
        }
    }
}

/// Ctrl+Enter confirms and Esc cancels, both before the edit sees the key.
/// Everything else goes to the edit, including the IME composition messages.
unsafe extern "system" fn edit_proc(
    window: HWND,
    msg: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    if msg == WM_KEYDOWN {
        let key = wparam.0 as u32;
        if key == VK_ESCAPE.0 as u32 {
            post_review_done(false);
            return LRESULT(0);
        }
        if key == VK_RETURN.0 as u32
            && unsafe { GetKeyState(VK_CONTROL.0 as i32) } < 0
        {
            post_review_done(true);
            return LRESULT(0);
        }
    }
    let old: WNDPROC = unsafe { std::mem::transmute(EDIT_OLD_PROC.load(Ordering::SeqCst)) };
    unsafe { CallWindowProcW(old, window, msg, wparam, lparam) }
}

unsafe extern "system" fn wndproc(
    window: HWND,
    msg: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    match msg {
        WM_UPDATE => {
            let (visible, reviewing) = {
                let state = ui().lock().unwrap();
                (state.visible, state.reviewing)
            };
            if !reviewing {
                if visible {
                    // The anchor follows the focused window, so the panel has
                    // to move with it; it is not only placed when created.
                    place(window, STATUS_SIZE);
                }
                unsafe {
                    let _ = ShowWindow(
                        window,
                        if visible { SW_SHOWNOACTIVATE } else { SW_HIDE },
                    );
                    if visible {
                        let _ = InvalidateRect(Some(window), None, false);
                    }
                }
            }
            let pending = ui().lock().unwrap().pending_review.is_some();
            if pending {
                begin_review(window);
            }
            LRESULT(0)
        }
        WM_REVIEW_DONE => {
            end_review(window, wparam.0 != 0);
            LRESULT(0)
        }
        WM_TIMER => {
            if crate::tray::quit_requested() {
                unsafe {
                    let _ = PostMessageW(Some(window), WM_CLOSE, WPARAM(0), LPARAM(0));
                }
                return LRESULT(0);
            }
            let (visible, recording, expired) = {
                let mut state = ui().lock().unwrap();
                let expired = state.hide_at.map(|t| Instant::now() >= t).unwrap_or(false);
                if expired {
                    state.visible = false;
                    state.hide_at = None;
                }
                (state.visible, state.recording, expired)
            };
            if expired {
                unsafe {
                    let _ = ShowWindow(window, SW_HIDE);
                }
                return LRESULT(0);
            }
            // Everything but the level bar is redrawn when its state changes.
            // The bar follows the microphone, so the timer only asks for a
            // repaint when the level differs from what is on screen, and never
            // over the review panel, whose caret the edit control draws itself.
            if !visible || !recording {
                return LRESULT(0);
            }
            let level = LEVEL
                .get()
                .map(|l| l.load(Ordering::Relaxed))
                .unwrap_or(0);
            if LEVEL_SHOWN.swap(level, Ordering::Relaxed) != level {
                unsafe {
                    let _ = InvalidateRect(Some(window), None, false);
                }
            }
            LRESULT(0)
        }
        // `WS_EX_DLGMODALFRAME` asks the window manager to leave the window
        // alone, and Windows draws a double border for it in the non-client
        // area. The border belongs to the non-client area, so giving the window
        // no non-client area removes it while the style bit stays set.
        WM_NCCALCSIZE => {
            if wparam.0 == 0 {
                unsafe { DefWindowProcW(window, msg, wparam, lparam) }
            } else {
                LRESULT(0)
            }
        }
        WM_DPICHANGED => {
            let reviewing = ui().lock().map(|s| s.reviewing).unwrap_or(false);
            let scale = place(window, if reviewing { REVIEW_SIZE } else { STATUS_SIZE });
            if reviewing {
                round_window(window, scale);
            }
            if let Some(edit) = edit() {
                layout_edit(window, edit, scale);
            }
            unsafe {
                let _ = InvalidateRect(Some(window), None, false);
            }
            LRESULT(0)
        }
        WM_PAINT => {
            paint(window);
            LRESULT(0)
        }
        WM_CTLCOLOREDIT => {
            let hdc = HDC(wparam.0 as *mut core::ffi::c_void);
            unsafe {
                let _ = SetBkColor(hdc, color(18, 18, 22));
                let _ = SetTextColor(hdc, color(245, 245, 248));
            }
            LRESULT(EDIT_BRUSH.load(Ordering::SeqCst))
        }
        WM_DESTROY => {
            unsafe {
                PostQuitMessage(0);
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
        let scale = window_scale(window);

        let mem = CreateCompatibleDC(Some(hdc));
        let bmp = CreateCompatibleBitmap(hdc, w, h);
        let old = SelectObject(mem, HGDIOBJ(bmp.0));

        // Everything outside the rounded panel becomes the transparent color key.
        let key = CreateSolidBrush(KEY_COLOR);
        let full = RECT {
            left: 0,
            top: 0,
            right: w,
            bottom: h,
        };
        let _ = FillRect(mem, &full, key);
        let _ = DeleteObject(HGDIOBJ(key.0));

        let bg = CreateSolidBrush(color(30, 30, 36));
        let radius = (18.0 * scale) as i32;
        let region = CreateRoundRectRgn(0, 0, w + 1, h + 1, radius, radius);
        let _ = FillRgn(mem, region, bg);
        let _ = DeleteObject(HGDIOBJ(region.0));
        let _ = DeleteObject(HGDIOBJ(bg.0));

        let state = ui().lock().unwrap();
        let font = ensure_font(scale);
        let old_font = SelectObject(mem, HGDIOBJ(font.0));
        let _ = SetBkMode(mem, TRANSPARENT);

        if state.reviewing {
            // Paint the edit's area in the edit's own background: the child
            // draws over it a moment later, and until it does, the key color
            // would leave a hole right through the panel.
            let background = CreateSolidBrush(color(18, 18, 22));
            let edit_area = RECT {
                left: (16.0 * scale) as i32,
                top: (38.0 * scale) as i32,
                right: w - (16.0 * scale) as i32,
                bottom: h - (12.0 * scale) as i32,
            };
            let _ = FillRect(mem, &edit_area, background);
            let _ = DeleteObject(HGDIOBJ(background.0));

            let _ = SetTextColor(mem, color(160, 170, 190));
            let mut label: Vec<u16> = "Ctrl+Enter で確定、Esc で取り消し"
                .encode_utf16()
                .collect();
            let mut label_rect = RECT {
                left: (18.0 * scale) as i32,
                top: (8.0 * scale) as i32,
                right: w - (18.0 * scale) as i32,
                bottom: (34.0 * scale) as i32,
            };
            let _ = DrawTextW(
                mem,
                &mut label,
                &mut label_rect,
                windows::Win32::Graphics::Gdi::DT_LEFT | DT_VCENTER | DT_SINGLELINE,
            );
        } else {
            let _ = SetTextColor(mem, color(245, 245, 248));
            // DrawTextW reads an empty slice as "the string is null-terminated"
            // and walks off the end of the buffer, so empty text is skipped.
            let mut text: Vec<u16> = state.text.encode_utf16().collect();
            if !text.is_empty() {
                let mut text_rect = RECT {
                    left: (16.0 * scale) as i32,
                    top: (10.0 * scale) as i32,
                    right: w - (16.0 * scale) as i32,
                    bottom: h - (18.0 * scale) as i32,
                };
                let _ = DrawTextW(
                    mem,
                    &mut text,
                    &mut text_rect,
                    DT_CENTER | DT_VCENTER | DT_SINGLELINE,
                );
            }

            let level = if state.recording {
                let bits = LEVEL
                    .get()
                    .map(|l| l.load(Ordering::Relaxed))
                    .unwrap_or(0);
                LEVEL_SHOWN.store(bits, Ordering::Relaxed);
                f32::from_bits(bits).clamp(0.0, 1.0)
            } else {
                1.0
            };
            let inset = (16.0 * scale) as i32;
            let bar_w = ((w - inset * 2) as f32 * level.max(0.04)) as i32;
            let accent = CreateSolidBrush(state.accent);
            let bar = CreateRoundRectRgn(
                inset,
                h - (12.0 * scale) as i32,
                inset + bar_w,
                h - (7.0 * scale) as i32,
                (6.0 * scale) as i32,
                (6.0 * scale) as i32,
            );
            let _ = FillRgn(mem, bar, accent);
            let _ = DeleteObject(HGDIOBJ(bar.0));
            let _ = DeleteObject(HGDIOBJ(accent.0));
        }

        let _ = SelectObject(mem, old_font);
        let _ = BitBlt(hdc, 0, 0, w, h, Some(mem), 0, 0, SRCCOPY);
        let _ = SelectObject(mem, old);
        let _ = DeleteObject(HGDIOBJ(bmp.0));
        let _ = DeleteDC(mem);
        let _ = EndPaint(window, &ps);
    }
}
