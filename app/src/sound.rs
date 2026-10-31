//! Short sound effects for push-to-talk start, stop, and errors.
//!
//! The WAV images are generated once and kept for the process lifetime, as
//! `PlaySoundW` with `SND_MEMORY | SND_ASYNC` reads them in the background.

use std::sync::OnceLock;

use windows::core::PCWSTR;
use windows::Win32::Media::Audio::{PlaySoundW, SND_ASYNC, SND_FLAGS, SND_MEMORY, SND_NODEFAULT};

pub fn start() {
    play(Side::Start);
}

pub fn stop() {
    play(Side::Stop);
}

pub fn error() {
    play(Side::Error);
}

#[derive(Clone, Copy)]
enum Side {
    Start,
    Stop,
    Error,
}

struct Sounds {
    start: Vec<u8>,
    stop: Vec<u8>,
    error: Vec<u8>,
}

fn sounds() -> &'static Sounds {
    static SOUNDS: OnceLock<Sounds> = OnceLock::new();
    SOUNDS.get_or_init(|| Sounds {
        // Rising for start, falling for stop, a low double tone for errors.
        start: crate::tone::start_wav(),
        stop: crate::tone::stop_wav(),
        error: crate::tone::error_wav(),
    })
}

fn play(side: Side) {
    let set = sounds();
    let bytes = match side {
        Side::Start => &set.start,
        Side::Stop => &set.stop,
        Side::Error => &set.error,
    };
    let flags = SND_FLAGS(SND_MEMORY.0 | SND_ASYNC.0 | SND_NODEFAULT.0);
    unsafe {
        // The WAV bytes stay alive in `SOUNDS` for the whole process.
        let _ = PlaySoundW(PCWSTR(bytes.as_ptr() as *const u16), None, flags);
    }
}
