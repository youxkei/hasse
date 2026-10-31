//! Microphone capture and conversion to the model's 16 kHz mono format.
//!
//! Only the first channel of the selected input is used by default: on a stereo
//! input pair, the microphone is usually the first channel.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::SampleFormat;
use crossbeam_channel::{Receiver, Sender};

use crate::mel::SAMPLE_RATE;

pub enum AudioCmd {
    Start,
    Stop,
}

pub struct Recorded {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

/// Open the input device on its own thread (cpal streams are not `Send` on all
/// platforms) and serve [`AudioCmd`]. Recorded audio goes back through `recorded_tx`.
pub fn spawn(
    device_name: String,
    first_channel_only: bool,
    max_seconds: f32,
    commands: Receiver<AudioCmd>,
    recorded_tx: Sender<Recorded>,
    level: Arc<AtomicU32>,
) {
    std::thread::Builder::new()
        .name("hasse-audio".into())
        .spawn(move || {
            if let Err(e) =
                run(device_name, first_channel_only, max_seconds, commands, recorded_tx, level)
            {
                eprintln!("audio: {e:#}");
            }
        })
        .expect("spawn audio thread");
}

fn run(
    device_name: String,
    first_channel_only: bool,
    max_seconds: f32,
    commands: Receiver<AudioCmd>,
    recorded_tx: Sender<Recorded>,
    level: Arc<AtomicU32>,
) -> Result<()> {
    let host = cpal::default_host();
    let device = if device_name.is_empty() {
        host.default_input_device()
            .ok_or_else(|| anyhow!("no default input device"))?
    } else {
        let needle = device_name.to_lowercase();
        host.input_devices()?
            .find(|d| {
                let name = d.to_string();
                name == device_name || name.to_lowercase().contains(&needle)
            })
            .ok_or_else(|| anyhow!("input device {:?} not found", device_name))?
    };
    let supported = device.default_input_config()?;
    let sample_rate = supported.sample_rate();
    let channels = supported.channels() as usize;
    let format = supported.sample_format();
    let config = supported.config();
    eprintln!("audio: {device} @ {sample_rate} Hz, {channels} ch, {format:?}");

    let buffer = Arc::new(Mutex::new(Vec::<f32>::new()));
    let recording = Arc::new(AtomicBool::new(false));
    let cap = (max_seconds.max(1.0) * sample_rate as f32) as usize;

    let buf_cb = Arc::clone(&buffer);
    let rec_cb = Arc::clone(&recording);
    let level_cb = Arc::clone(&level);
    let data_callback = move |data: &[f32], _: &cpal::InputCallbackInfo| {
        if !rec_cb.load(Ordering::Relaxed) {
            return;
        }
        let mut sum_sq = 0.0f32;
        let mut count = 0usize;
        if let Ok(mut buf) = buf_cb.lock() {
            for frame in data.chunks(channels.max(1)) {
                // Channel 1 only by default; the second channel of an input
                // pair is ignored. `first_channel_only: false` averages them.
                let mono = if first_channel_only {
                    frame.first().copied().unwrap_or(0.0)
                } else {
                    frame.iter().sum::<f32>() / frame.len().max(1) as f32
                };
                if buf.len() < cap {
                    buf.push(mono);
                }
                sum_sq += mono * mono;
                count += 1;
            }
        }
        let rms = if count > 0 { (sum_sq / count as f32).sqrt() } else { 0.0 };
        level_cb.store(rms.to_bits(), Ordering::Relaxed);
    };

    let stream = match format {
        SampleFormat::F32 => device.build_input_stream(config, data_callback, |e| eprintln!("audio: {e}"), None)?,
        other => return Err(anyhow!("unsupported sample format {other:?}")),
    };
    stream.play()?;

    loop {
        match commands.recv() {
            Ok(AudioCmd::Start) => {
                if let Ok(mut buf) = buffer.lock() {
                    buf.clear();
                    // Reserve now, so the audio callback never reallocates.
                    let missing = cap.saturating_sub(buf.capacity());
                    buf.reserve(missing);
                }
                level.store(0.0f32.to_bits(), Ordering::Relaxed);
                recording.store(true, Ordering::SeqCst);
            }
            Ok(AudioCmd::Stop) => {
                recording.store(false, Ordering::SeqCst);
                let samples = buffer.lock().map(|mut b| std::mem::take(&mut *b)).unwrap_or_default();
                if recorded_tx.send(Recorded { samples, sample_rate }).is_err() {
                    break;
                }
            }
            Err(_) => break,
        }
    }
    drop(stream);
    Ok(())
}

/// Windowed-sinc resampling to the model's 16 kHz (cutoff 7.2 kHz).
pub fn to_model_rate(input: &[f32], in_rate: u32) -> Vec<f32> {
    if in_rate as usize == SAMPLE_RATE {
        return input.to_vec();
    }
    let ratio = in_rate as f64 / SAMPLE_RATE as f64;
    let out_len = (input.len() as f64 / ratio) as usize;
    // Cutoff in cycles per input sample.
    let fc = 0.45 * SAMPLE_RATE as f64 / in_rate as f64;
    const HALF: i32 = 32;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let pos = i as f64 * ratio;
        let center = pos.floor() as i64;
        let mut acc = 0.0f64;
        let mut wsum = 0.0f64;
        for j in -HALF..=HALF {
            let idx = center + j as i64;
            if idx < 0 || idx as usize >= input.len() {
                continue;
            }
            let d = pos - idx as f64;
            let x = 2.0 * fc * d;
            let sinc = if x.abs() < 1e-12 {
                1.0
            } else {
                (std::f64::consts::PI * x).sin() / (std::f64::consts::PI * x)
            };
            let t = d / (HALF as f64 + 1.0);
            if t.abs() >= 1.0 {
                continue;
            }
            let window = 0.42
                + 0.5 * (std::f64::consts::PI * t).cos()
                + 0.08 * (2.0 * std::f64::consts::PI * t).cos();
            let weight = sinc * window;
            acc += input[idx as usize] as f64 * weight;
            wsum += weight;
        }
        out.push(if wsum.abs() > 1e-9 { (acc / wsum) as f32 } else { 0.0 });
    }
    out
}
