"""
Pure helpers for finetune_whisper.py (no model, no data needed; unit-tested):
latent-space augmentation of cached encoder activations, best-checkpoint
tracking, and save/restore of full trainer state.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import torch

STATE_FILE = "trainer_state.pt"


def augment_hidden(
    h: torch.Tensor, n_time: int = 2, time_w: int = 100, n_chan: int = 1, chan_w: int = 64,
    rng: Optional[random.Random] = None,
) -> torch.Tensor:
    """SpecAugment-style masking on post-encoder-layer hidden states [T, C]:
    zero `n_time` random time spans of `time_w` frames and `n_chan` random
    channel bands of `chan_w`. Returns a masked copy; the input is untouched.
    (The cache holds hidden states, not mel features, so this is latent-space
    masking, like wav2vec2's.)"""
    rng = rng or random
    h = h.clone()
    T, C = h.shape
    for _ in range(n_time):
        if T > time_w:
            t0 = rng.randint(0, T - time_w - 1)
            h[t0:t0 + time_w] = 0.0
    for _ in range(n_chan):
        if C > chan_w:
            c0 = rng.randint(0, C - chan_w - 1)
            h[:, c0:c0 + chan_w] = 0.0
    return h


class BestTracker:
    """Keeps the lowest validation WER seen; `update` says whether the new
    value is a strict improvement (=> overwrite the best checkpoint)."""

    def __init__(self, best: float = float("inf")):
        self.best = best

    def update(self, value: float) -> bool:
        if value < self.best:
            self.best = value
            return True
        return False


def save_trainer_state(out_dir: Path, step: int, optimizer, scheduler, best_wer: float) -> None:
    """Full state, so a resume continues the optimizer/schedule instead of
    restarting Adam's moments and the warmup."""
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"step": step, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "best_wer": best_wer},
        out_dir / STATE_FILE,
    )


def load_trainer_state(out_dir: Path) -> Optional[dict]:
    path = out_dir / STATE_FILE
    return torch.load(path, map_location="cpu") if path.exists() else None
