//! macOS overlay: a transparent, click-through, always-on-top eframe window at
//! the bottom center, and the main event loop that hosts the tray icon.

use anyhow::{anyhow, Result};
use eframe::egui;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

pub use crate::status::Status;

struct UiState {
    text: String,
    accent: egui::Color32,
    visible: bool,
    recording: bool,
    hide_at: Option<Instant>,
}

static UI: OnceLock<Mutex<UiState>> = OnceLock::new();
static LEVEL: OnceLock<Arc<AtomicU32>> = OnceLock::new();
static QUIT: AtomicBool = AtomicBool::new(false);

fn ui_state() -> &'static Mutex<UiState> {
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

fn color(r: u8, g: u8, b: u8) -> egui::Color32 {
    egui::Color32::from_rgb(r, g, b)
}

pub fn set_level_source(level: Arc<AtomicU32>) {
    let _ = LEVEL.set(level);
}

/// The window is created by [`run_loop`]; this exists for the shared startup.
pub fn create() -> Result<()> {
    Ok(())
}

pub fn set(status: Status) {
    {
        let mut state = ui_state().lock().unwrap();
        state.recording = false;
        state.hide_at = None;
        match status {
            Status::Starting => {
                state.text = "Loading model…".into();
                state.accent = color(120, 170, 255);
                state.visible = true;
            }
            Status::Recording => {
                state.text = "Recording — release to transcribe".into();
                state.accent = color(255, 90, 90);
                state.visible = true;
                state.recording = true;
            }
            Status::Transcribing => {
                state.text = "Transcribing…".into();
                state.accent = color(255, 190, 90);
                state.visible = true;
            }
            Status::Typed => {
                state.text = "Inserted".into();
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
    // The UI loop repaints every 50 ms, so the state is picked up next frame.
}

pub fn hide() {
    if let Ok(mut state) = ui_state().lock() {
        state.visible = false;
        state.hide_at = None;
    }
}

pub fn quit() {
    QUIT.store(true, Ordering::SeqCst);
}

pub fn run_loop() -> Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([300.0, 68.0])
            .with_decorations(false)
            .with_transparent(true)
            .with_always_on_top()
            .with_mouse_passthrough(true)
            .with_resizable(false)
            .with_active(false),
        ..Default::default()
    };
    eframe::run_native(
        "hasse",
        options,
        Box::new(|cc| {
            install_japanese_font(&cc.egui_ctx);
            Ok(Box::new(OverlayApp::new()))
        }),
    )
    .map_err(|e| anyhow!("eframe: {e}"))
}

struct OverlayApp {
    tray: Option<tray_icon::TrayIcon>,
    tray_tried: bool,
    positioned: bool,
}

impl OverlayApp {
    fn new() -> Self {
        Self { tray: None, tray_tried: false, positioned: false }
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
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            return;
        }

        if !self.positioned {
            if let Some(size) = ctx.input(|i| i.viewport().monitor_size) {
                let pos = egui::pos2((size.x - 300.0) / 2.0, size.y - 68.0 - 100.0);
                ctx.send_viewport_cmd(egui::ViewportCommand::OuterPosition(pos));
                self.positioned = true;
            }
        }

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

        let rect = ui.max_rect();
        let painter = ui.painter();
        painter.rect_filled(
            rect,
            egui::CornerRadius::same(18),
            egui::Color32::from_rgba_unmultiplied(30, 30, 36, 190),
        );
        painter.text(
            egui::pos2(rect.center().x, rect.top() + 26.0),
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
            egui::pos2(rect.left() + 16.0, rect.bottom() - 12.0),
            egui::vec2((rect.width() - 32.0) * level.max(0.04), 5.0),
        );
        painter.rect_filled(bar, egui::CornerRadius::same(3), accent);
    }
}

/// egui's bundled font has no Japanese; load a system face for the error text
/// and any Japanese labels.
fn install_japanese_font(ctx: &egui::Context) {
    const CANDIDATES: &[&str] = &[
        "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ];
    for path in CANDIDATES {
        if let Ok(bytes) = std::fs::read(path) {
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
            ctx.set_fonts(fonts);
            return;
        }
    }
}
