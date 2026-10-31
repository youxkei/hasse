//! The short WAV tones used for start, stop and error feedback.

/// Build a 16-bit mono 22.05 kHz WAV from `(frequency, milliseconds)` tones.
pub fn wav(tones: &[(f32, u32)]) -> Vec<u8> {
    const RATE: u32 = 22_050;
    const AMPLITUDE: f32 = 0.22;
    let mut samples: Vec<i16> = Vec::new();
    for &(freq, ms) in tones {
        let count = (RATE as u64 * ms as u64 / 1000) as usize;
        let attack = RATE as f32 * 0.006;
        let release = RATE as f32 * 0.012;
        for i in 0..count {
            let t = i as f32 / RATE as f32;
            let env = (i as f32 / attack).min(1.0).min((count - i) as f32 / release);
            let v = (2.0 * std::f32::consts::PI * freq * t).sin() * env * AMPLITUDE;
            samples.push((v * i16::MAX as f32) as i16);
        }
        samples.extend(std::iter::repeat_n(0i16, (RATE as u64 * 10 / 1000) as usize));
    }

    let data_len = (samples.len() * 2) as u32;
    let mut out = Vec::with_capacity(44 + data_len as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&1u16.to_le_bytes()); // mono
    out.extend_from_slice(&RATE.to_le_bytes());
    out.extend_from_slice(&(RATE * 2).to_le_bytes()); // bytes per second
    out.extend_from_slice(&2u16.to_le_bytes()); // block align
    out.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    for sample in samples {
        out.extend_from_slice(&sample.to_le_bytes());
    }
    out
}

/// Rising two tones.
pub fn start_wav() -> Vec<u8> {
    wav(&[(880.0, 55), (1320.0, 85)])
}

/// Falling two tones.
pub fn stop_wav() -> Vec<u8> {
    wav(&[(1320.0, 55), (880.0, 95)])
}

/// A low double tone.
pub fn error_wav() -> Vec<u8> {
    wav(&[(330.0, 110), (262.0, 150)])
}
