//! Haas-ja-v1.0 inference: mel frontend, ONNX encoder, TDT greedy decoder.
//!
//! The decoder is a direct port of NeMo's `GreedyBatchedTDTLabelLoopingComputer`
//! (batch size 1) with the KDA prediction network written out in plain Rust.
//! It was validated to produce the same token ids as NeMo on real speech, and
//! the Rust output is checked against the Python reference by `--wav`.
//!
//! The encoder runs in ONNX Runtime on the CPU.

use std::path::Path;

use anyhow::{Context, Result};
use ort::session::Session;
use ort::value::Tensor;
use safetensors::SafeTensors;
use serde::Deserialize;

use crate::mel::{MelFrontend, SAMPLE_RATE};

const VOCAB_PIECES: usize = 3072;
const N_MELS: usize = 128;

#[derive(Debug, Clone, Deserialize)]
struct Meta {
    sample_rate: usize,
    vocab_size: usize,
    blank_id: usize,
    num_classes_with_blank: usize,
    durations: Vec<usize>,
    max_symbols: usize,
    hidden: usize,
    num_layers: usize,
    num_heads: usize,
    head_dim: usize,
    head_v_dim: usize,
    conv_size: usize,
}

struct Layer {
    q_proj: Vec<f32>,
    k_proj: Vec<f32>,
    v_proj: Vec<f32>,
    q_conv: Vec<f32>,
    k_conv: Vec<f32>,
    v_conv: Vec<f32>,
    f_proj_0: Vec<f32>,
    f_proj_1: Vec<f32>,
    b_proj: Vec<f32>,
    g_proj_0: Vec<f32>,
    g_proj_1: Vec<f32>,
    g_proj_1_bias: Vec<f32>,
    o_norm: Vec<f32>,
    o_proj: Vec<f32>,
    a_log: Vec<f32>,
    dt_bias: Vec<f32>,
    norm_w: Vec<f32>,
    norm_b: Vec<f32>,
}

struct Weights {
    embed: Vec<f32>,
    layers: Vec<Layer>,
    final_norm_w: Vec<f32>,
    final_norm_b: Vec<f32>,
    joint_pred_w: Vec<f32>,
    joint_pred_b: Vec<f32>,
    joint_out_w: Vec<f32>,
    joint_out_b: Vec<f32>,
}

struct State {
    /// Per layer, `[num_heads * head_dim * head_v_dim]` in (head, k, v) order.
    h: Vec<Vec<f32>>,
    /// Per layer, the q/k/v short-convolution caches, each `[dim * conv_size]`.
    conv: Vec<[Vec<f32>; 3]>,
}

pub struct Asr {
    session: Session,
    _env: ort::environment::Environment,
    meta: Meta,
    vocab: Vec<String>,
    weights: Weights,
    mel: MelFrontend,
}

impl Asr {
    pub fn load(models_dir: &Path, onnxruntime: Option<&Path>) -> Result<Self> {
        let dir = models_dir;
        let onnxruntime = crate::runtime_dll::ensure(onnxruntime)?;
        let meta_text = std::fs::read_to_string(dir.join("meta.json"))
            .with_context(|| format!("reading {}", dir.join("meta.json").display()))?;
        let meta: Meta = serde_json::from_str(&meta_text).context("parsing meta.json")?;
        anyhow::ensure!(meta.sample_rate == SAMPLE_RATE, "unexpected sample rate");
        anyhow::ensure!(meta.vocab_size == VOCAB_PIECES, "unexpected vocab size");
        anyhow::ensure!(meta.head_dim == meta.head_v_dim, "unequal k/v head dimensions");

        let vocab: Vec<String> = serde_json::from_str(
            &std::fs::read_to_string(dir.join("vocab.json")).context("reading vocab.json")?,
        )
        .context("parsing vocab.json")?;
        anyhow::ensure!(vocab.len() == VOCAB_PIECES, "unexpected vocab length");

        let frontend_bytes =
            std::fs::read(dir.join("frontend.safetensors")).context("reading frontend.safetensors")?;
        let frontend = SafeTensors::deserialize(&frontend_bytes).context("parsing frontend.safetensors")?;
        let fb = tensor_f32(&frontend, "fb")?;
        let window = tensor_f32(&frontend, "window")?;

        let dec_bytes =
            std::fs::read(dir.join("decoder.safetensors")).context("reading decoder.safetensors")?;
        let dec = SafeTensors::deserialize(&dec_bytes).context("parsing decoder.safetensors")?;
        let weights = Weights::load(&dec, meta.num_layers)?;

        let env = ort::init_from(&onnxruntime)
            .map_err(|e| anyhow::anyhow!("loading {}: {e}", onnxruntime.display()))?
            .with_name("hasse")
            .build()
            .map_err(|e| anyhow::anyhow!("ONNX Runtime: {e}"))
            .context("creating ONNX Runtime")?;
        let mut builder = Session::builder(&env)
            .map_err(|e| anyhow::anyhow!("ONNX Runtime: {e}"))?;
        let session = builder
            .commit_from_file(dir.join("frontend.onnx"))
            .map_err(|e| anyhow::anyhow!("loading frontend.onnx: {e}"))?;

        Ok(Self {
            session,
            _env: env,
            meta,
            vocab,
            weights,
            mel: MelFrontend::new(fb, window)?,
        })
    }

    /// Transcribe mono 16 kHz audio. Long audio is cut at quiet points like
    /// `haas_asr.transcribe_longform` does.
    pub fn transcribe(&mut self, audio: &[f32]) -> Result<String> {
        if audio.is_empty() {
            return Ok(String::new());
        }
        let bounds = split_at_quiet_points(audio, 30.0, 2.0, 0.1);
        let mut text = String::new();
        for (start, end) in bounds {
            if end - start < SAMPLE_RATE / 10 {
                continue;
            }
            let ids = self.transcribe_chunk(&audio[start..end])?;
            // Decode each chunk on its own, as `transcribe_longform` does, so a
            // chunk's leading word boundary does not become a space.
            text.push_str(&self.detokenize(&ids));
        }
        Ok(text)
    }

    fn transcribe_chunk(&mut self, audio: &[f32]) -> Result<Vec<usize>> {
        let feats = self.mel.compute(audio)?;
        anyhow::ensure!(feats.n_mels == N_MELS, "unexpected mel size");

        let audio_tensor = Tensor::from_array((
            ort::value::Shape::new([1i64, N_MELS as i64, feats.time as i64]),
            feats.data,
        ))
        .map_err(|e| anyhow::anyhow!("building the audio tensor: {e}"))?;
        let length_tensor =
            Tensor::from_array((ort::value::Shape::new([1i64]), vec![feats.length as i64]))
                .map_err(|e| anyhow::anyhow!("building the length tensor: {e}"))?;
        let outputs = self
            .session
            .run(ort::inputs![
                "audio_signal" => audio_tensor,
                "length" => length_tensor,
            ])
            .map_err(|e| anyhow::anyhow!("encoder: {e}"))?;

        let proj = {
            let (shape, data) = outputs["enc_proj"]
                .try_extract_tensor::<f32>()
                .map_err(|e| anyhow::anyhow!("encoder output: {e}"))?;
            anyhow::ensure!(
                shape.len() == 3 && shape[1] as usize == self.meta.hidden,
                "bad enc_proj shape"
            );
            (data.to_vec(), shape[2] as usize)
        };
        let enc_len = {
            let (_, data) = outputs["encoded_len"]
                .try_extract_tensor::<i64>()
                .map_err(|e| anyhow::anyhow!("encoder length: {e}"))?;
            data.first().copied().unwrap_or(0).max(0) as usize
        };
        drop(outputs);

        Ok(self.decode(&proj.0, proj.1, enc_len))
    }

    /// Greedy TDT decoding over `enc_proj` laid out `[hidden][frames]`.
    fn decode(&self, enc: &[f32], frames: usize, enc_len: usize) -> Vec<usize> {
        let meta = &self.meta;
        let blank = meta.blank_id;

        let mut state = State::new(meta.num_layers, meta.num_heads, meta.head_dim, meta.conv_size);
        let mut pred_proj = self.project_pred(&self.prednet_step(&mut state, blank));
        let mut out = Vec::new();
        let mut t = 0usize;
        let mut last_ts: isize = -1;
        let mut last_lasts = 0usize;

        while t < enc_len {
            let (label, dur_idx) = self.joint(enc, t, frames, &pred_proj);
            let mut dur = meta.durations[dur_idx];
            if dur == 0 && label == blank {
                dur = 1;
            }
            let mut time_label = t;
            t += dur;

            let mut label = label;
            while t < enc_len && label == blank {
                let (l, d) = self.joint(enc, t, frames, &pred_proj);
                label = l;
                dur = meta.durations[d];
                if dur == 0 && label == blank {
                    dur = 1;
                }
                time_label = t;
                t += dur;
            }

            let found = label != blank;
            if found {
                out.push(label);
                if last_ts == time_label as isize {
                    last_lasts += 1;
                } else {
                    last_lasts = 1;
                }
                last_ts = time_label as isize;
            }

            // Stage 3: advance the prediction network with the label. For a
            // blank label, NeMo restores the previous state.
            let mut next = state.clone();
            let next_pred = self.prednet_step(&mut next, label);
            let next_proj = self.project_pred(&next_pred);
            if found {
                state = next;
                pred_proj = next_proj;
            }

            // Stage 4: after `max_symbols` emissions at one frame, move on.
            if found && last_lasts >= meta.max_symbols && last_ts == t as isize {
                t += 1;
            }
        }

        out
    }

    /// Joint network argmax: (vocab label including blank, duration index).
    fn joint(&self, enc: &[f32], t: usize, frames: usize, pred_proj: &[f32]) -> (usize, usize) {
        let hidden_dim = self.meta.hidden;
        let n_out = self.meta.num_classes_with_blank;
        let vocab_with_blank = self.meta.blank_id + 1;
        let n_dur = self.meta.durations.len();

        let mut act = vec![0.0f32; hidden_dim];
        for d in 0..hidden_dim {
            let v = enc[d * frames + t] + pred_proj[d];
            act[d] = if v > 0.0 { v } else { 0.0 };
        }

        let mut best_label = 0usize;
        let mut best_score = f32::NEG_INFINITY;
        let mut best_dur = 0usize;
        let mut best_dur_score = f32::NEG_INFINITY;
        for d in 0..n_out {
            let row = &self.weights.joint_out_w[d * hidden_dim..(d + 1) * hidden_dim];
            let mut acc = self.weights.joint_out_b[d];
            for i in 0..hidden_dim {
                acc += row[i] * act[i];
            }
            if d < vocab_with_blank {
                if acc > best_score {
                    best_score = acc;
                    best_label = d;
                }
            } else {
                let idx = d - vocab_with_blank;
                if acc > best_dur_score {
                    best_dur_score = acc;
                    best_dur = idx.min(n_dur - 1);
                }
            }
        }
        (best_label, best_dur)
    }

    fn project_pred(&self, pred: &[f32]) -> Vec<f32> {
        let h = self.meta.hidden;
        let mut out = vec![0.0f32; h];
        matvec(&self.weights.joint_pred_w, pred, &mut out, h, h);
        for i in 0..h {
            out[i] += self.weights.joint_pred_b[i];
        }
        out
    }

    /// One KDA prediction-network step for `label`; advances `state`.
    fn prednet_step(&self, state: &mut State, label: usize) -> Vec<f32> {
        let h = self.meta.hidden;
        let nh = self.meta.num_heads;
        let hd = self.meta.head_dim;

        let mut x = self.weights.embed[label * h..(label + 1) * h].to_vec();
        let mut xn = vec![0.0f32; h];
        let mut q_in = vec![0.0f32; nh * hd];
        let mut k_in = vec![0.0f32; nh * hd];
        let mut v_in = vec![0.0f32; nh * hd];
        let mut q = vec![0.0f32; nh * hd];
        let mut k = vec![0.0f32; nh * hd];
        let mut v = vec![0.0f32; nh * hd];
        let mut qh = vec![0.0f32; nh * hd];
        let mut kh = vec![0.0f32; nh * hd];
        let mut out = vec![0.0f32; nh * hd];
        let mut o_out = vec![0.0f32; h];
        let mut g_raw = vec![0.0f32; nh * hd];
        let mut beta_raw = vec![0.0f32; nh];
        let mut gate_raw = vec![0.0f32; nh * hd];
        let mut proj = vec![0.0f32; 64];

        for (li, layer) in self.weights.layers.iter().enumerate() {
            layer_norm(&x, &layer.norm_w, &layer.norm_b, &mut xn, h);

            matvec(&layer.q_proj, &xn, &mut q_in, nh * hd, h);
            matvec(&layer.k_proj, &xn, &mut k_in, nh * hd, h);
            matvec(&layer.v_proj, &xn, &mut v_in, nh * hd, h);
            conv_step(&mut state.conv[li][0], &layer.q_conv, &q_in, &mut q, 4);
            conv_step(&mut state.conv[li][1], &layer.k_conv, &k_in, &mut k, 4);
            conv_step(&mut state.conv[li][2], &layer.v_conv, &v_in, &mut v, 4);

            // f_proj: 640 -> 64 -> 320
            matvec(&layer.f_proj_0, &xn, &mut proj, 64, h);
            matvec(&layer.f_proj_1, &proj, &mut g_raw, nh * hd, 64);
            matvec(&layer.b_proj, &xn, &mut beta_raw, nh, h);
            matvec(&layer.g_proj_0, &xn, &mut proj, 64, h);
            matvec(&layer.g_proj_1, &proj, &mut gate_raw, nh * hd, 64);
            for i in 0..nh * hd {
                gate_raw[i] += layer.g_proj_1_bias[i];
            }

            // L2-normalise q and k per head; q additionally carries the scale.
            for head in 0..nh {
                let base = head * hd;
                let mut nq = 0.0f32;
                let mut nk = 0.0f32;
                for d in 0..hd {
                    nq += q[base + d] * q[base + d];
                    nk += k[base + d] * k[base + d];
                }
                let inv_q = 1.0 / (nq + 1e-6).sqrt() * (hd as f32).powf(-0.5);
                let inv_k = 1.0 / (nk + 1e-6).sqrt();
                for d in 0..hd {
                    qh[base + d] = q[base + d] * inv_q;
                    kh[base + d] = k[base + d] * inv_k;
                }
            }

            let hstate = &mut state.h[li];
            let head_size = hd * hd;
            for head in 0..nh {
                let base = head * hd;
                // Decay the recurrent memory along the key dimension.
                for kk in 0..hd {
                    let g = -layer.a_log[head].exp()
                        * softplus(g_raw[base + kk] + layer.dt_bias[base + kk]);
                    let decay = g.exp();
                    let hbase = head * head_size + kk * hd;
                    for vv in 0..hd {
                        hstate[hbase + vv] *= decay;
                    }
                }
                let beta = sigmoid(beta_raw[head]);
                for vv in 0..hd {
                    let mut dot = 0.0f32;
                    for kk in 0..hd {
                        dot += hstate[head * head_size + kk * hd + vv] * kh[base + kk];
                    }
                    let val = (v[base + vv] - dot) * beta;
                    let mut o = 0.0f32;
                    for kk in 0..hd {
                        let idx = head * head_size + kk * hd + vv;
                        hstate[idx] += val * kh[base + kk];
                        o += hstate[idx] * qh[base + kk];
                    }
                    out[base + vv] = o;
                }
            }

            // RMSNorm with a sigmoid gate, then the output projection.
            for head in 0..nh {
                let base = head * hd;
                let mut sq = 0.0f32;
                for d in 0..hd {
                    sq += out[base + d] * out[base + d];
                }
                let rms = (sq / hd as f32 + 1e-5).sqrt();
                for d in 0..hd {
                    let gate = sigmoid(gate_raw[base + d]);
                    out[base + d] = out[base + d] / rms * layer.o_norm[d] * gate;
                }
            }
            matvec(&layer.o_proj, &out, &mut o_out, h, nh * hd);
            for i in 0..h {
                x[i] += o_out[i];
            }
        }

        layer_norm(&x, &self.weights.final_norm_w, &self.weights.final_norm_b, &mut xn, h);
        xn
    }

    fn detokenize(&self, ids: &[usize]) -> String {
        let mut text = String::new();
        let mut bytes: Vec<u8> = Vec::new();

        let flush = |text: &mut String, bytes: &mut Vec<u8>| {
            if !bytes.is_empty() {
                text.push_str(&String::from_utf8_lossy(bytes));
                bytes.clear();
            }
        };

        for &id in ids {
            let piece = self.vocab.get(id).map(String::as_str).unwrap_or("");
            if let Some(byte) = parse_byte_piece(piece) {
                bytes.push(byte);
            } else {
                flush(&mut text, &mut bytes);
                if piece == "<unk>" {
                    text.push(' ');
                } else {
                    text.push_str(&piece.replace('\u{2581}', " "));
                }
            }
        }
        flush(&mut text, &mut bytes);

        // SentencePiece's decoder removes extra whitespace.
        let mut cleaned = String::with_capacity(text.len());
        let mut last_space = true;
        for ch in text.chars() {
            if ch == ' ' {
                if !last_space {
                    cleaned.push(' ');
                    last_space = true;
                }
            } else {
                cleaned.push(ch);
                last_space = false;
            }
        }
        cleaned.trim_end().to_string()
    }
}

fn tensor_f32(st: &SafeTensors, name: &str) -> Result<Vec<f32>> {
    let t = st.tensor(name).with_context(|| format!("missing tensor {name}"))?;
    anyhow::ensure!(t.dtype() == safetensors::Dtype::F32, "tensor {name} is not f32");
    let data = t.data();
    Ok(data
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

impl Weights {
    fn load(st: &SafeTensors, layers: usize) -> Result<Self> {
        let get = |name: &str| tensor_f32(st, name);
        let mut out_layers = Vec::with_capacity(layers);
        for i in 0..layers {
            let p = format!("layer{i}.");
            out_layers.push(Layer {
                q_proj: get(&format!("{p}q_proj"))?,
                k_proj: get(&format!("{p}k_proj"))?,
                v_proj: get(&format!("{p}v_proj"))?,
                q_conv: get(&format!("{p}q_conv"))?,
                k_conv: get(&format!("{p}k_conv"))?,
                v_conv: get(&format!("{p}v_conv"))?,
                f_proj_0: get(&format!("{p}f_proj_0"))?,
                f_proj_1: get(&format!("{p}f_proj_1"))?,
                b_proj: get(&format!("{p}b_proj"))?,
                g_proj_0: get(&format!("{p}g_proj_0"))?,
                g_proj_1: get(&format!("{p}g_proj_1"))?,
                g_proj_1_bias: get(&format!("{p}g_proj_1_bias"))?,
                o_norm: get(&format!("{p}o_norm"))?,
                o_proj: get(&format!("{p}o_proj"))?,
                a_log: get(&format!("{p}A_log"))?,
                dt_bias: get(&format!("{p}dt_bias"))?,
                norm_w: get(&format!("norm{i}.weight"))?,
                norm_b: get(&format!("norm{i}.bias"))?,
            });
        }
        Ok(Self {
            embed: get("embed.weight")?,
            layers: out_layers,
            final_norm_w: get("final_norm.weight")?,
            final_norm_b: get("final_norm.bias")?,
            joint_pred_w: get("joint.pred.weight")?,
            joint_pred_b: get("joint.pred.bias")?,
            joint_out_w: get("joint.out.weight")?,
            joint_out_b: get("joint.out.bias")?,
        })
    }
}

impl State {
    fn new(layers: usize, heads: usize, head_dim: usize, conv_size: usize) -> Self {
        let hsize = heads * head_dim * head_dim;
        let csize = heads * head_dim * conv_size;
        Self {
            h: (0..layers).map(|_| vec![0.0; hsize]).collect(),
            conv: (0..layers)
                .map(|_| [vec![0.0; csize], vec![0.0; csize], vec![0.0; csize]])
                .collect(),
        }
    }
}

impl Clone for State {
    fn clone(&self) -> Self {
        Self { h: self.h.clone(), conv: self.conv.clone() }
    }
}

fn matvec(w: &[f32], x: &[f32], out: &mut [f32], rows: usize, cols: usize) {
    for r in 0..rows {
        let row = &w[r * cols..(r + 1) * cols];
        let mut acc = 0.0f32;
        for c in 0..cols {
            acc += row[c] * x[c];
        }
        out[r] = acc;
    }
}

fn layer_norm(x: &[f32], w: &[f32], b: &[f32], out: &mut [f32], dim: usize) {
    let mean = x[..dim].iter().sum::<f32>() / dim as f32;
    let var = x[..dim].iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / dim as f32;
    let inv = 1.0 / (var + 1e-5).sqrt();
    for i in 0..dim {
        out[i] = (x[i] - mean) * inv * w[i] + b[i];
    }
}

/// Causal depthwise convoy step with a SiLU, mirroring FLA's `ShortConvolution`.
fn conv_step(cache: &mut [f32], weight: &[f32], x: &[f32], out: &mut [f32], k: usize) {
    for c in 0..x.len() {
        let base = c * k;
        for j in 0..k - 1 {
            cache[base + j] = cache[base + j + 1];
        }
        cache[base + k - 1] = x[c];
        let mut acc = 0.0f32;
        for j in 0..k {
            acc += cache[base + j] * weight[base + j];
        }
        out[c] = acc * sigmoid(acc);
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        x.exp().ln_1p()
    }
}

fn parse_byte_piece(piece: &str) -> Option<u8> {
    let bytes = piece.as_bytes();
    if bytes.len() == 6 && bytes[0] == b'<' && bytes[1] == b'0' && bytes[2] == b'x' && bytes[5] == b'>' {
        let hex = std::str::from_utf8(&bytes[3..5]).ok()?;
        u8::from_str_radix(hex, 16).ok()
    } else {
        None
    }
}

/// Quietest-window cuts, as in `haas_asr.longform.split_at_quiet_points`.
fn split_at_quiet_points(audio: &[f32], chunk_s: f32, search_s: f32, window_s: f32) -> Vec<(usize, usize)> {
    let sr = SAMPLE_RATE as f32;
    let step = (chunk_s * sr) as usize;
    let search = (search_s * sr) as usize;
    let window = (window_s * sr) as usize;
    let mut bounds = Vec::new();
    let mut start = 0usize;
    while audio.len().saturating_sub(start) > step {
        let target = start + step;
        let low = (start + step / 2).max(target.saturating_sub(search));
        let high = (audio.len() - window).min(target + search);
        let mut cut = target;
        if high > low {
            let span = &audio[low..(high + window).min(audio.len())];
            cut = low + quietest_window(span, window) + window / 2;
        }
        bounds.push((start, cut));
        start = cut;
    }
    if start < audio.len() {
        bounds.push((start, audio.len()));
    }
    bounds
}

fn quietest_window(audio: &[f32], window: usize) -> usize {
    let mut energy = Vec::with_capacity(audio.len() + 1);
    energy.push(0.0f64);
    let mut acc = 0.0f64;
    for &v in audio {
        acc += (v as f64) * (v as f64);
        energy.push(acc);
    }
    let mut best = 0usize;
    let mut best_energy = f64::INFINITY;
    for i in 0..=audio.len().saturating_sub(window) {
        let e = energy[i + window] - energy[i];
        if e < best_energy {
            best_energy = e;
            best = i;
        }
    }
    best
}
