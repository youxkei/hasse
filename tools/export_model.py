"""Export Haas-ja-v1.0 for the Rust runtime.

Produces, in ``engine/models/``:

* ``frontend.onnx`` + ``frontend.onnx.data`` — the FastConformer encoder only.
  Inputs: log-mel ``[B, 128, T]`` and ``length``. Outputs: ``encoded``
  ``[B, 1024, T']`` and ``encoded_len``. The mel frontend is implemented in
  Rust (``app/src/mel.rs``) because torch.onnx cannot export ``torch.stft``.
* ``frontend.safetensors`` — the mel filterbank ``[128, 257]`` and the
  symmetric Hann window ``[400]``.
* ``decoder.safetensors`` — every weight the Rust TDT greedy decoder needs,
  in float32, under fixed names.
* ``vocab.json`` — the 3072 SentencePiece pieces, index = token id.
* ``meta.json`` — frontend and decoding constants.

Run with the engine venv:

    engine/.venv/Scripts/python.exe tools/export_model.py   # Windows
    engine/.venv/bin/python tools/export_model.py           # Linux

The export needs a CUDA machine (Linux or Windows) with NeMo 2.7.3,
flash-linear-attention 0.5.2, and a CUDA build of PyTorch. The artifacts are
platform-neutral; the macOS build uses them as they are.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import onnx
import torch
from safetensors.torch import save_file

import haas_asr

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "engine" / "models" / "Haas-ja-v1.0.nemo"
OUT_DIR = ROOT / "engine" / "models"
TMP_DIR = OUT_DIR / "_onnx_export"


class EncoderExport(torch.nn.Module):
    """Encoder plus the joint's encoder projection, so Rust gets [B, 640, T'].

    NeMo wraps forward with a kwargs-only check, so call it through a plain module.
    """

    def __init__(self, encoder, joint_enc):
        super().__init__()
        self.encoder = encoder
        self.joint_enc = joint_enc

    def forward(self, audio_signal, length):
        encoded, encoded_len = self.encoder(audio_signal=audio_signal, length=length)
        x = encoded.transpose(1, 2)  # [B, T, 1024]
        x = torch.nn.functional.linear(x, self.joint_enc.weight, self.joint_enc.bias)
        return x.transpose(1, 2), encoded_len  # [B, 640, T]


def export_encoder(model) -> None:
    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)
    TMP_DIR.mkdir(parents=True)

    wrapper = EncoderExport(model.encoder, model.joint.enc).eval()
    mel = torch.randn(1, 128, 800, device="cuda")
    length = torch.tensor([800], dtype=torch.long, device="cuda")
    with torch.no_grad():
        encoded, encoded_len = wrapper(mel, length)
    print(f"eager encoder: encoded {tuple(encoded.shape)} len {encoded_len.tolist()}")

    raw = TMP_DIR / "frontend.onnx"
    torch.onnx.export(
        wrapper,
        (mel, length),
        str(raw),
        input_names=["audio_signal", "length"],
        output_names=["enc_proj", "encoded_len"],
        dynamic_axes={
            "audio_signal": {0: "B", 2: "T"},
            "length": {0: "B"},
            "enc_proj": {0: "B", 2: "T_out"},
            "encoded_len": {0: "B"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

    # torch writes one external file per initializer; merge them into one.
    graph = onnx.load(str(raw))
    final = OUT_DIR / "frontend.onnx"
    # onnx appends external data; a stale file would double the size.
    for stale in (final, OUT_DIR / "frontend.onnx.data"):
        if stale.exists():
            stale.unlink()
    onnx.save_model(
        graph,
        str(final),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="frontend.onnx.data",
        size_threshold=1024,
    )
    shutil.rmtree(TMP_DIR)
    print(f"wrote {final} (+ .data)")


def export_frontend_assets(model) -> None:
    featurizer = model.preprocessor.featurizer
    save_file(
        {
            "fb": featurizer.fb.reshape(128, 257).to(torch.float32).cpu().contiguous(),
            "window": featurizer.window.to(torch.float32).cpu().contiguous(),
        },
        str(OUT_DIR / "frontend.safetensors"),
    )
    print("wrote frontend.safetensors (fb, window)")


def export_decoder(model) -> None:
    decoder = model.decoder
    prednet = decoder.prediction["dec_rnn"]
    joint = model.joint
    w: dict[str, torch.Tensor] = {}

    def put(name: str, t: torch.Tensor) -> None:
        w[name] = t.detach().to(torch.float32).cpu().contiguous()

    put("embed.weight", decoder.prediction["embed"].weight)
    for i, layer in enumerate(prednet.layers):
        p = f"layer{i}."
        put(p + "q_proj", layer.q_proj.weight)
        put(p + "k_proj", layer.k_proj.weight)
        put(p + "v_proj", layer.v_proj.weight)
        put(p + "q_conv", layer.q_conv1d.weight[:, 0, :])
        put(p + "k_conv", layer.k_conv1d.weight[:, 0, :])
        put(p + "v_conv", layer.v_conv1d.weight[:, 0, :])
        put(p + "f_proj_0", layer.f_proj[0].weight)
        put(p + "f_proj_1", layer.f_proj[1].weight)
        put(p + "b_proj", layer.b_proj.weight)
        put(p + "g_proj_0", layer.g_proj[0].weight)
        put(p + "g_proj_1", layer.g_proj[1].weight)
        put(p + "g_proj_1_bias", layer.g_proj[1].bias)
        put(p + "o_norm", layer.o_norm.weight)
        put(p + "o_proj", layer.o_proj.weight)
        put(p + "A_log", layer.A_log)
        put(p + "dt_bias", layer.dt_bias)
        put(f"norm{i}.weight", prednet.norms[i].weight)
        put(f"norm{i}.bias", prednet.norms[i].bias)
    put("final_norm.weight", prednet.final_norm.weight)
    put("final_norm.bias", prednet.final_norm.bias)
    put("joint.enc.weight", joint.enc.weight)
    put("joint.enc.bias", joint.enc.bias)
    put("joint.pred.weight", joint.pred.weight)
    put("joint.pred.bias", joint.pred.bias)
    put("joint.out.weight", joint.joint_net[2].weight)
    put("joint.out.bias", joint.joint_net[2].bias)

    path = OUT_DIR / "decoder.safetensors"
    save_file(w, str(path))
    print(f"wrote {path} ({len(w)} tensors)")


def export_meta(model) -> None:
    vocab = list(model.tokenizer.vocab)
    (OUT_DIR / "vocab.json").write_text(
        json.dumps(vocab, ensure_ascii=False), encoding="utf-8"
    )
    meta = {
        "model": "third-intelligence/Haas-ja-v1.0",
        "sample_rate": 16000,
        "vocab_size": 3072,
        "blank_id": 3072,
        "num_classes_with_blank": 3078,
        "durations": [0, 1, 2, 3, 4],
        "max_symbols": 10,
        "hidden": 640,
        "num_layers": 2,
        "num_heads": 5,
        "head_dim": 64,
        "head_v_dim": 64,
        "conv_size": 4,
        "encoder_dim": 1024,
        "frontend": "frontend.onnx",
        "mel": {
            "sample_rate": 16000,
            "n_fft": 512,
            "win_length": 400,
            "hop_length": 160,
            "n_mels": 128,
            "preemph": 0.97,
            "log_guard": 2 ** -24,
            "pad_to": 16,
            "pad_value": 0.0,
            "center": True,
            "pad_mode": "reflect",
        },
    }
    (OUT_DIR / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"wrote {OUT_DIR / 'vocab.json'} and meta.json")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = haas_asr.load_model(str(MODEL), dtype=torch.float32, device="cuda")
    export_encoder(model)
    export_frontend_assets(model)
    export_decoder(model)
    export_meta(model)


if __name__ == "__main__":
    main()
