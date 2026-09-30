"""
Convert a HuggingFace Whisper checkpoint (e.g. the output of
finetune_whisper.py) to the MLX format that `mlx_whisper` loads, so a
fine-tuned model runs on the Apple GPU inside the pipeline:

    python training/hf_to_mlx.py models/whisper-small-ami models/whisper-small-ami-mlx

Then:  ./run.sh pipeline audio.wav --model models/whisper-small-ami-mlx

Sanity check of the converter itself (converts the STOCK model and compares
every tensor with the published mlx-community conversion):

    python training/hf_to_mlx.py openai/whisper-small /tmp/stock-mlx --verify mlx-community/whisper-small-mlx

Notes:
- conv weights are transposed (torch: out,in,kernel -> MLX: out,kernel,in)
- k_proj has no bias in Whisper; the encoder's positional embedding is a fixed
  sinusoid in MLX (not a parameter), so it is not exported
- `alignment_heads` (which cross-attention heads give word timestamps) is not
  part of an HF state dict; it is copied from the stock MLX model of the same
  architecture, which is correct because fine-tuning does not move which heads
  were chosen by OpenAI for alignment
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict

import numpy as np

_BLOCK_RENAMES = [
    ("self_attn.q_proj", "attn.query"),
    ("self_attn.k_proj", "attn.key"),
    ("self_attn.v_proj", "attn.value"),
    ("self_attn.out_proj", "attn.out"),
    ("self_attn_layer_norm", "attn_ln"),
    ("encoder_attn.q_proj", "cross_attn.query"),
    ("encoder_attn.k_proj", "cross_attn.key"),
    ("encoder_attn.v_proj", "cross_attn.value"),
    ("encoder_attn.out_proj", "cross_attn.out"),
    ("encoder_attn_layer_norm", "cross_attn_ln"),
    ("fc1", "mlp1"),
    ("fc2", "mlp2"),
    ("final_layer_norm", "mlp_ln"),
]


def convert_state_dict(hf: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for name, w in hf.items():
        if name == "proj_out.weight" or name == "model.encoder.embed_positions.weight":
            continue                                   # tied to token embedding / fixed sinusoid
        n = name.removeprefix("model.")
        n = re.sub(r"\.layers\.(\d+)\.", r".blocks.\1.", n)
        for old, new in _BLOCK_RENAMES:
            n = n.replace(old, new)
        n = n.replace("encoder.layer_norm", "encoder.ln_post").replace("decoder.layer_norm", "decoder.ln")
        n = n.replace("decoder.embed_tokens", "decoder.token_embedding").replace(
            "decoder.embed_positions", "decoder.positional_embedding")
        if n.endswith("positional_embedding.weight"):
            n = n[: -len(".weight")]
        if n in ("encoder.conv1.weight", "encoder.conv2.weight"):
            w = w.transpose(0, 2, 1)
        out[n] = w
    return out


def load_hf(src: str) -> Dict[str, np.ndarray]:
    import torch
    from transformers import WhisperForConditionalGeneration

    model = WhisperForConditionalGeneration.from_pretrained(src, torch_dtype=torch.float32)
    return {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}, model.config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="HF model dir or repo id")
    ap.add_argument("dst", help="output directory")
    ap.add_argument("--alignment-from", default="mlx-community/whisper-small-mlx",
                    help="stock MLX repo of the same architecture (source of alignment_heads + config)")
    ap.add_argument("--verify", default=None, help="MLX repo to compare tensor-by-tensor against")
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    args = ap.parse_args()

    from huggingface_hub import snapshot_download

    state, cfg = load_hf(args.src)
    weights = convert_state_dict(state)

    ref_dir = Path(snapshot_download(args.alignment_from))
    ref = np.load(ref_dir / "weights.npz") if (ref_dir / "weights.npz").exists() else None
    if ref is None:
        sys.exit(f"{args.alignment_from}: expected weights.npz")
    weights["alignment_heads"] = ref["alignment_heads"]

    ref_cfg = json.loads((ref_dir / "config.json").read_text())
    derived = {
        "n_mels": cfg.num_mel_bins, "n_audio_ctx": cfg.max_source_positions, "n_audio_state": cfg.d_model,
        "n_audio_head": cfg.encoder_attention_heads, "n_audio_layer": cfg.encoder_layers,
        "n_vocab": cfg.vocab_size, "n_text_ctx": cfg.max_target_positions, "n_text_state": cfg.d_model,
        "n_text_head": cfg.decoder_attention_heads, "n_text_layer": cfg.decoder_layers,
    }
    for k, v in derived.items():
        if ref_cfg.get(k) != v:
            sys.exit(f"architecture mismatch with {args.alignment_from}: {k} {ref_cfg.get(k)} != {v}")

    dtype = np.dtype(args.dtype)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    np.savez(dst / "weights.npz", **{k: (v if v.dtype.kind == "i" else v.astype(dtype)) for k, v in weights.items()})
    (dst / "config.json").write_text(json.dumps(ref_cfg, indent=4))
    print(f"wrote {dst}  ({len(weights)} tensors, {args.dtype})")

    if args.verify:
        vdir = Path(snapshot_download(args.verify))
        vref = np.load(vdir / "weights.npz")
        missing = sorted(set(vref.files) - set(weights))
        extra = sorted(set(weights) - set(vref.files))
        print(f"tensors: converted {len(weights)}, reference {len(vref.files)} | missing {len(missing)} | extra {len(extra)}")
        worst = ("", 0.0)
        for name in sorted(set(vref.files) & set(weights)):
            a, b = weights[name].astype(np.float32), vref[name].astype(np.float32)
            if a.shape != b.shape:
                sys.exit(f"SHAPE MISMATCH {name}: {a.shape} vs {b.shape}")
            d = float(np.abs(a - b).max())
            if d > worst[1]:
                worst = (name, d)
        print(f"max abs difference over all shared tensors: {worst[1]:.2e} ({worst[0]})")
        if missing or extra:
            print("missing:", missing[:5], "extra:", extra[:5])
        ok = not missing and not extra and worst[1] < 5e-3
        print("VERIFIED: conversion matches the published MLX weights" if ok else "MISMATCH")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
