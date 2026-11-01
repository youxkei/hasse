//! Bottom-center overlay and the main event loop on macOS.
//!
//! eframe owns the UI thread, so the tray icon is created once the loop is
//! running. While idle and recording the window is click-through. A finished
//! transcript is handed to [`review`], which makes the window focusable, shows
//! an editable text box, and waits for Ctrl+Enter (or Esc). The controller then
//! learns from the difference and inserts the text.
//!
//! Enter is left alone because it confirms an IME conversion; Ctrl+Enter is the
//! only key that cannot collide with Japanese input.
//!
//! AppKit has no tiling window manager to keep out of, so the window needs none
//! of the Win32 tool-dialog styling the Windows overlay sets at creation.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};
use eframe::egui;

pub use crate::status::Status;

/// One window size for every state. Resizing the window recreates the
/// transparent surface; the status text is a small panel inside this window
/// instead.
const WINDOW_SIZE: [f32; 2] = [720.0, 156.0];
const PANEL_WIDTH: f32 = 300.0;
const PANEL_HEIGHT: f32 = 68.0;
const BOTTOM_MARGIN: f32 = 100.0;

struct UiState {
    text: String,
    accent: egui::Color32,
    visible: bool,
    recording: bool,
    hide_at: Option<Instant>,
    pending_review: Option<String>,
}

static UI: OnceLock<Mutex<UiState>> = OnceLock::new();
static LEVEL: OnceLock<Arc<AtomicU32>> = OnceLock::new();
static ANCHOR: Mutex<Option<[f32; 4]>> = Mutex::new(None);
static QUIT: AtomicBool = AtomicBool::new(false);
static CONFIRM: OnceLock<(Sender<Option<String>>, Mutex<Receiver<Option<String>>>)> =
    OnceLock::new();
/// Kept so that state changes from other threads can wake the UI even while
/// the window is hidden.
static CONTEXT: OnceLock<egui::Context> = OnceLock::new();

fn request_repaint() {
    if let Some(ctx) = CONTEXT.get() {
        ctx.request_repaint();
    }
}

fn ui_state() -> &'static Mutex<UiState> {
    UI.get_or_init(|| {
        Mutex::new(UiState {
            text: String::new(),
            accent: color(120, 170, 255),
            visible: false,
            recording: false,
            hide_at: None,
            pending_review: None,
        })
    })
}

/// The app sends the confirmed text (or `None` for a cancel); [`review`] waits
/// on the other end.
fn confirm_channel() -> &'static (Sender<Option<String>>, Mutex<Receiver<Option<String>>>) {
    CONFIRM.get_or_init(|| {
        let (tx, rx) = std::sync::mpsc::channel();
        (tx, Mutex::new(rx))
    })
}

fn color(r: u8, g: u8, b: u8) -> egui::Color32 {
    egui::Color32::from_rgb(r, g, b)
}

pub fn set_level_source(level: Arc<AtomicU32>) {
    let _ = LEVEL.set(level);
}

/// macOS keeps the overlay on the screen it is already on.
pub fn set_anchor(_bounds: Option<[f32; 4]>) {}

pub fn set(status: Status) {
    {
        let mut state = ui_state().lock().unwrap();
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
    request_repaint();
}

pub fn hide() {
    if let Ok(mut state) = ui_state().lock() {
        state.visible = false;
        state.hide_at = None;
    }
    request_repaint();
}

pub fn quit() {
    QUIT.store(true, Ordering::SeqCst);
    if let Some((tx, _)) = CONFIRM.get() {
        let _ = tx.send(None);
    }
    request_repaint();
}

/// Show `text` for editing and block until the user presses Ctrl+Enter
/// (`Some`) or Esc (`None`). The caller inserts and learns from the returned
/// text.
pub fn review(text: &str) -> Option<String> {
    {
        let mut state = ui_state().lock().unwrap();
        state.pending_review = Some(text.to_string());
        state.visible = true;
        state.recording = false;
        state.hide_at = None;
        state.text.clear();
    }
    request_repaint();
    confirm_channel().1.lock().unwrap().recv().ok().flatten()
}

pub fn run_loop() -> Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("hasse")
            .with_inner_size(WINDOW_SIZE)
            .with_decorations(false)
            .with_transparent(true)
            .with_always_on_top()
            .with_mouse_passthrough(true)
            .with_resizable(false)
            .with_taskbar(false)
            .with_active(false),
        ..Default::default()
    };
    eframe::run_native(
        "hasse",
        options,
        Box::new(|cc| {
            let _ = CONTEXT.set(cc.egui_ctx.clone());
            install_fonts(&cc.egui_ctx);
            Ok(Box::new(OverlayApp::new()))
        }),
    )
    .map_err(|e| anyhow!("eframe: {e}"))
}

struct OverlayApp {
    tray: Option<tray_icon::TrayIcon>,
    tray_tried: bool,
    mode: Mode,
    buffer: String,
    focus_editor: bool,
    positioned: Option<Option<[f32; 4]>>,
}

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Status,
    Review,
}

impl OverlayApp {
    fn new() -> Self {
        Self {
            tray: None,
            tray_tried: false,
            mode: Mode::Status,
            buffer: String::new(),
            focus_editor: false,
            positioned: None,
        }
    }

    /// Keep the window at the bottom center whenever its size or monitor
    /// changes.
    fn place(&mut self, ctx: &egui::Context) {
        let anchor = ANCHOR.lock().map(|a| *a).unwrap_or(None);
        if self.positioned == Some(anchor) {
            return;
        }
        let monitor = ctx.input(|i| i.viewport().monitor_size);
        let (left, top, width, height) = match anchor {
            Some([left, top, width, height]) => (left, top, width, height),
            None => match monitor {
                Some(monitor) => (0.0, 0.0, monitor.x, monitor.y),
                None => return,
            },
        };
        let pos = egui::pos2(
            left + (width - WINDOW_SIZE[0]) / 2.0,
            top + height - WINDOW_SIZE[1] - BOTTOM_MARGIN,
        );
        ctx.send_viewport_cmd(egui::ViewportCommand::InnerSize(egui::vec2(
            WINDOW_SIZE[0],
            WINDOW_SIZE[1],
        )));
        ctx.send_viewport_cmd(egui::ViewportCommand::OuterPosition(pos));
        self.positioned = Some(anchor);
    }

    fn finish_review(&mut self, ctx: &egui::Context) {
        self.mode = Mode::Status;
        self.buffer.clear();
        self.positioned = None;
        {
            let mut state = ui_state().lock().unwrap();
            state.visible = false;
            state.text.clear();
        }
        ctx.send_viewport_cmd(egui::ViewportCommand::MousePassthrough(true));
    }

    fn review_ui(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ctx.request_repaint_after(Duration::from_millis(16));

        // Read the keys before the editor runs, so Enter stays free for the
        // IME and Ctrl+Enter cannot be swallowed as a newline.
        let (confirm, cancel) = ui.input_mut(|i| {
            let m = i.modifiers;
            let confirm = (m.ctrl || m.command) && i.consume_key(m, egui::Key::Enter);
            let cancel = m == egui::Modifiers::NONE
                && i.consume_key(egui::Modifiers::NONE, egui::Key::Escape);
            (confirm, cancel)
        });
        if confirm || cancel {
            if let Some((tx, _)) = CONFIRM.get() {
                let _ = tx.send(if confirm { Some(self.buffer.clone()) } else { None });
            }
            self.finish_review(ctx);
            return;
        }

        let rect = ui.max_rect();
        ui.painter().rect_filled(
            rect,
            egui::CornerRadius::same(18),
            egui::Color32::from_rgba_unmultiplied(30, 30, 36, 235),
        );
        ui.painter().text(
            egui::pos2(rect.left() + 18.0, rect.top() + 20.0),
            egui::Align2::LEFT_CENTER,
            "Ctrl+Enter で確定、Esc で取り消し",
            egui::FontId::proportional(13.0),
            egui::Color32::from_rgb(160, 170, 190),
        );

        let mut visuals = ui.style().visuals.clone();
        visuals.extreme_bg_color = egui::Color32::from_rgba_unmultiplied(18, 18, 22, 220);
        visuals.widgets.inactive.bg_stroke =
            egui::Stroke::new(1.0, egui::Color32::from_rgb(70, 76, 92));
        visuals.selection.bg_fill = egui::Color32::from_rgb(60, 110, 200);
        ui.style_mut().visuals = visuals;

        let edit_rect = egui::Rect::from_min_max(
            egui::pos2(rect.left() + 16.0, rect.top() + 38.0),
            egui::pos2(rect.right() - 16.0, rect.bottom() - 12.0),
        );
        let response = ui.put(
            edit_rect,
            egui::TextEdit::multiline(&mut self.buffer)
                .desired_width(f32::INFINITY)
                .desired_rows(3)
                .font(egui::FontId::proportional(17.0))
                .text_color(egui::Color32::from_rgb(245, 245, 248)),
        );
        if self.focus_editor {
            response.request_focus();
            self.focus_editor = false;
        }
    }

    fn status_ui(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ctx.request_repaint_after(Duration::from_millis(50));

        let now = Instant::now();
        let (visible, text, accent, recording) = {
            let mut state = ui_state().lock().unwrap();
            if state.hide_at.map(|t| now >= t).unwrap_or(false) {
                state.visible = false;
                state.hide_at = None;
            }
            (state.visible, state.text.clone(), state.accent, state.recording)
        };
        if !visible {
            return;
        }

        let full = ui.max_rect();
        let panel = egui::Rect::from_min_size(
            egui::pos2(
                full.center().x - PANEL_WIDTH / 2.0,
                full.bottom() - PANEL_HEIGHT,
            ),
            egui::vec2(PANEL_WIDTH, PANEL_HEIGHT),
        );
        let painter = ui.painter();
        painter.rect_filled(
            panel,
            egui::CornerRadius::same(18),
            egui::Color32::from_rgba_unmultiplied(30, 30, 36, 190),
        );
        painter.text(
            egui::pos2(panel.center().x, panel.top() + 26.0),
            egui::Align2::CENTER_CENTER,
            &text,
            egui::FontId::proportional(16.0),
            egui::Color32::from_rgb(245, 245, 248),
        );

        let level = if recording {
            LEVEL
                .get()
                .map(|l| f32::from_bits(l.load(Ordering::Relaxed)).clamp(0.0, 1.0))
                .unwrap_or(0.0)
        } else {
            1.0
        };
        let bar = egui::Rect::from_min_size(
            egui::pos2(panel.left() + 16.0, panel.bottom() - 12.0),
            egui::vec2((PANEL_WIDTH - 32.0) * level.max(0.04), 5.0),
        );
        painter.rect_filled(bar, egui::CornerRadius::same(3), accent);
    }
}

impl eframe::App for OverlayApp {
    fn clear_color(&self, _visuals: &egui::Visuals) -> [f32; 4] {
        [0.0, 0.0, 0.0, 0.0]
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();

        // The tray must be created once the event loop is running.
        if !self.tray_tried {
            self.tray_tried = true;
            self.tray = crate::tray::create().ok();
        }

        if QUIT.load(Ordering::SeqCst) || crate::tray::quit_requested() {
            if let Some((tx, _)) = CONFIRM.get() {
                let _ = tx.send(None);
            }
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            return;
        }

        let pending = ui_state().lock().unwrap().pending_review.take();
        if let Some(text) = pending {
            self.mode = Mode::Review;
            self.buffer = text;
            self.focus_editor = true;
            self.positioned = None;
            ctx.send_viewport_cmd(egui::ViewportCommand::MousePassthrough(false));
            ctx.send_viewport_cmd(egui::ViewportCommand::Focus);
        }

        self.place(&ctx);

        match self.mode {
            Mode::Review => self.review_ui(ui, &ctx),
            Mode::Status => self.status_ui(ui, &ctx),
        }
    }
}

/// egui's bundled font has no Japanese. Load a system face; the first one that
/// exists wins.
fn install_fonts(ctx: &egui::Context) {
    let mut candidates: Vec<PathBuf> = Vec::new();
    candidates.push(PathBuf::from("/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"));
    candidates.push(PathBuf::from("/System/Library/Fonts/Hiragino Sans GB.ttc"));
    candidates.push(PathBuf::from(
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ));
    for path in candidates {
        let Ok(bytes) = std::fs::read(&path) else {
            continue;
        };
        let mut fonts = egui::text::FontDefinitions::default();
        fonts.font_data.insert(
            "jp".to_owned(),
            std::sync::Arc::new(egui::text::FontData::from_owned(bytes)),
        );
        fonts
            .families
            .entry(egui::text::FontFamily::Proportional)
            .or_default()
            .insert(0, "jp".to_owned());
        fonts
            .families
            .entry(egui::text::FontFamily::Monospace)
            .or_default()
            .push("jp".to_owned());
        ctx.set_fonts(fonts);
        return;
    }
}
