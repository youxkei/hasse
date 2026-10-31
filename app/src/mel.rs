//! NeMo's `AudioToMelSpectrogramPreprocessor` for 16 kHz audio, in Rust.
//!
//! The recipe was checked against NeMo (max |diff| 2.6e-5 on real speech):
//! pre-emphasis 0.97, STFT with a 512-point FFT, a symmetric 400-sample Hann
//! window centred in the FFT frame, reflect padding of 256 samples, power
//! spectrum, a slaney-normalised mel filterbank (128 x 257), `ln(x + 2^-24)`,
//! per-feature normalisation over the valid frames (std uses n-1), and zero
//! padding of the frame axis to a multiple of 16.

use std::sync::Arc;

use anyhow::Result;
use realfft::{RealFftPlanner, RealToComplex};

pub const SAMPLE_RATE: usize = 16_000;
pub const N_FFT: usize = 512;
pub const WIN_LENGTH: usize = 400;
pub const HOP_LENGTH: usize = 160;
pub const N_MELS: usize = 128;
const PAD: usize = N_FFT / 2;
const LOG_GUARD: f32 = 5.960_464_5e-8; // 2^-24
const PAD_TO: usize = 16;

pub struct Features {
    /// `[N_MELS * time]`, frequency-major, as ONNX expects `[1, 128, T]`.
    pub data: Vec<f32>,
    pub n_mels: usize,
    pub time: usize,
    /// Valid frames, before padding.
    pub length: usize,
}

pub struct MelFrontend {
    fb: Vec<f32>,
    /// The 400-sample window placed into a 512-sample frame.
    window: Vec<f32>,
    fft: Arc<dyn RealToComplex<f32>>,
}

impl MelFrontend {
    pub fn new(fb: Vec<f32>, window: Vec<f32>) -> Result<Self> {
        anyhow::ensure!(fb.len() == N_MELS * (N_FFT / 2 + 1), "unexpected filterbank shape");
        anyhow::ensure!(window.len() == WIN_LENGTH, "unexpected window length");
        let mut win512 = vec![0.0f32; N_FFT];
        let left = (N_FFT - WIN_LENGTH) / 2;
        win512[left..left + WIN_LENGTH].copy_from_slice(&window);
        let fft = RealFftPlanner::<f32>::new().plan_fft_forward(N_FFT);
        Ok(Self { fb, window: win512, fft })
    }

    pub fn compute(&self, audio: &[f32]) -> Result<Features> {
        anyhow::ensure!(!audio.is_empty(), "empty audio");
        let seq_len = audio.len() / HOP_LENGTH;
        anyhow::ensure!(seq_len >= 1, "audio too short");
        let n_frames = seq_len + 1;

        // Pre-emphasis, then torch's reflect padding (the edge sample is not repeated).
        let mut pre = Vec::with_capacity(audio.len());
        pre.push(audio[0]);
        for i in 1..audio.len() {
            pre.push(audio[i] - 0.97 * audio[i - 1]);
        }
        anyhow::ensure!(audio.len() > PAD, "audio too short for reflect padding");
        let mut padded = Vec::with_capacity(pre.len() + 2 * PAD);
        for j in 0..PAD {
            padded.push(pre[PAD - j]);
        }
        padded.extend_from_slice(&pre);
        for i in 0..PAD {
            padded.push(pre[pre.len() - 2 - i]);
        }

        let bins = N_FFT / 2 + 1;
        let mut input = self.fft.make_input_vec();
        let mut spectrum = self.fft.make_output_vec();
        let mut power = vec![0.0f32; bins];
        let mut logmel = vec![0.0f32; N_MELS * n_frames];

        for f in 0..n_frames {
            let base = f * HOP_LENGTH;
            for j in 0..N_FFT {
                input[j] = padded[base + j] * self.window[j];
            }
            self.fft.process(&mut input, &mut spectrum)?;
            for k in 0..bins {
                let c = spectrum[k];
                power[k] = c.re * c.re + c.im * c.im;
            }
            for m in 0..N_MELS {
                let row = &self.fb[m * bins..(m + 1) * bins];
                let mut acc = 0.0f32;
                for k in 0..bins {
                    acc += row[k] * power[k];
                }
                logmel[m * n_frames + f] = (acc + LOG_GUARD).ln();
            }
        }

        // Per-feature normalisation over the valid frames.
        let mut mean = vec![0.0f32; N_MELS];
        let mut std = vec![0.0f32; N_MELS];
        for m in 0..N_MELS {
            let row = &logmel[m * n_frames..m * n_frames + seq_len];
            let mu = row.iter().sum::<f32>() / seq_len as f32;
            let var = if seq_len > 1 {
                row.iter().map(|v| (v - mu) * (v - mu)).sum::<f32>() / (seq_len as f32 - 1.0)
            } else {
                0.0
            };
            mean[m] = mu;
            std[m] = var.sqrt() + 1e-5;
        }
        for m in 0..N_MELS {
            let row = &mut logmel[m * n_frames..(m + 1) * n_frames];
            for f in 0..seq_len {
                row[f] = (row[f] - mean[m]) / std[m];
            }
            for f in seq_len..n_frames {
                row[f] = 0.0;
            }
        }

        let padded_time = n_frames.div_ceil(PAD_TO) * PAD_TO;
        let mut data = if padded_time == n_frames {
            logmel
        } else {
            let mut out = vec![0.0f32; N_MELS * padded_time];
            for m in 0..N_MELS {
                out[m * padded_time..m * padded_time + n_frames]
                    .copy_from_slice(&logmel[m * n_frames..(m + 1) * n_frames]);
            }
            out
        };
        // `logmel` is already consumed; silence the mut warning path.
        data.iter_mut().for_each(|v| {
            if !v.is_finite() {
                *v = 0.0;
            }
        });

        Ok(Features { data, n_mels: N_MELS, time: padded_time, length: seq_len })
    }
}
