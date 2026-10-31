//! macOS sound effects, played with `/usr/bin/afplay` on generated WAV files.

use std::path::PathBuf;
use std::process::Command;
use std::sync::OnceLock;

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

fn play(side: Side) {
    static FILES: OnceLock<[PathBuf; 3]> = OnceLock::new();
    let files = FILES.get_or_init(|| {
        let dir = std::env::var_os("HOME")
            .map(PathBuf::from)
            .unwrap_or_else(std::env::temp_dir)
            .join("Library/Caches/hasse");
        let _ = std::fs::create_dir_all(&dir);
        let files = [
            dir.join("start.wav"),
            dir.join("stop.wav"),
            dir.join("error.wav"),
        ];
        let _ = std::fs::write(&files[0], crate::tone::start_wav());
        let _ = std::fs::write(&files[1], crate::tone::stop_wav());
        let _ = std::fs::write(&files[2], crate::tone::error_wav());
        files
    });
    let index = match side {
        Side::Start => 0,
        Side::Stop => 1,
        Side::Error => 2,
    };
    let _ = Command::new("/usr/bin/afplay").arg(&files[index]).spawn();
}
