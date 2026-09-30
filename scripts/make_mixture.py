"""
Build a 2-speaker test recording with known ground truth from two
single-speaker clips, so Phases 2-4 can be tested with real (measurable)
overlap without needing a labelled corpus.

    python scripts/make_mixture.py a.wav b.wav --name demo --offset 8 --snr 0

Speaker A starts at t=0; speaker B starts at `--offset` seconds. If B starts
before A finishes, the two overlap. Writes to data/mixtures/<name>/:

    mix.wav      the mixture (what the pipeline is run on)
    s1.wav       A alone, aligned to the mixture timeline (separation ground truth)
    s2.wav       B alone, aligned to the mixture timeline
    truth.json   who speaks when, and the overlap window
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from audio_utils import load_audio, rms, save_wav, trim_silence  # noqa: E402
from config import DATA_DIR, TARGET_SR  # noqa: E402


def make_mixture(a: np.ndarray, b: np.ndarray, offset_s: float, snr_db: float, sr: int = TARGET_SR):
    """Returns (mix, s1, s2, truth). `snr_db` = level of A relative to B."""
    b = b * (rms(a) / max(rms(b), 1e-8)) * 10 ** (-snr_db / 20)

    off = int(round(offset_s * sr))
    total = max(len(a), off + len(b))
    s1 = np.zeros(total, dtype=np.float32)
    s2 = np.zeros(total, dtype=np.float32)
    s1[: len(a)] = a
    s2[off : off + len(b)] = b

    mix = s1 + s2
    peak = float(np.abs(mix).max())
    scale = 0.9 / peak if peak > 0 else 1.0
    mix, s1, s2 = mix * scale, s1 * scale, s2 * scale

    a_span = (0.0, len(a) / sr)
    b_span = (offset_s, offset_s + len(b) / sr)
    lo, hi = max(a_span[0], b_span[0]), min(a_span[1], b_span[1])
    truth = {
        "sample_rate": sr,
        "snr_db": snr_db,
        "turns": [
            {"speaker": "s1", "start": a_span[0], "end": a_span[1]},
            {"speaker": "s2", "start": b_span[0], "end": b_span[1]},
        ],
        "overlap": [lo, hi] if hi > lo else None,
    }
    return mix.astype(np.float32), s1, s2, truth


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("speaker_a")
    ap.add_argument("speaker_b")
    ap.add_argument("--name", default="mixture")
    ap.add_argument("--offset", type=float, default=None,
                    help="seconds at which B starts (default: 60%% through A, giving a partial overlap)")
    ap.add_argument("--snr", type=float, default=0.0, help="level of A relative to B, in dB")
    args = ap.parse_args()

    a, _ = trim_silence(load_audio(args.speaker_a), TARGET_SR)
    b, _ = trim_silence(load_audio(args.speaker_b), TARGET_SR)
    offset = args.offset if args.offset is not None else 0.6 * len(a) / TARGET_SR

    mix, s1, s2, truth = make_mixture(a, b, offset, args.snr)

    out = DATA_DIR / "mixtures" / args.name
    save_wav(out / "mix.wav", mix)
    save_wav(out / "s1.wav", s1)
    save_wav(out / "s2.wav", s2)
    (out / "truth.json").write_text(json.dumps(truth, indent=2))

    print(f"wrote {out}/  ({len(mix) / TARGET_SR:.1f}s)")
    for t in truth["turns"]:
        print(f"  {t['speaker']}: {t['start']:.1f}s -> {t['end']:.1f}s")
    ov = truth["overlap"]
    print(f"  overlap: {f'{ov[0]:.1f}s -> {ov[1]:.1f}s ({ov[1] - ov[0]:.1f}s)' if ov else 'none'}")


if __name__ == "__main__":
    main()
