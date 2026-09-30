"""
Phase 2 evaluation: run the separator on mixtures with known clean sources
and report SI-SDR (and improvement over the unprocessed mixture).

    # one mixture built by scripts/make_mixture.py
    python scripts/eval_separation.py --dir data/mixtures/demo

    # a LibriMix / WHAM-style subset (mix_clean|mix_both|mix_single / s1 / s2)
    python scripts/eval_separation.py --dir path/to/Libri2Mix/wav16k/min/dev --limit 20

    # explicit files
    python scripts/eval_separation.py --mix mix.wav --refs s1.wav s2.wav

If a mixture folder has truth.json with an overlap window, only that window
is evaluated -- mirroring how the pipeline uses the separator (overlap
windows only).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from audio_utils import load_audio, slice_audio  # noqa: E402
from config import TARGET_SR, pick_device  # noqa: E402
from metrics import pit_si_sdr, si_sdr  # noqa: E402
from separation import DEFAULT_MODEL, OverlapSeparator  # noqa: E402

Case = Tuple[str, Path, List[Path], Optional[Tuple[float, float]]]


def find_cases(root: Path, limit: Optional[int]) -> List[Case]:
    # Layout 1: a folder made by make_mixture.py
    if (root / "mix.wav").exists():
        window = None
        if (root / "truth.json").exists():
            ov = json.loads((root / "truth.json").read_text()).get("overlap")
            window = tuple(ov) if ov else None
        return [(root.name, root / "mix.wav", [root / "s1.wav", root / "s2.wav"], window)]

    # Layout 2: LibriMix-style tree
    mix_dir = next((root / d for d in ("mix_clean", "mix_both", "mix_single") if (root / d).is_dir()), None)
    if mix_dir is None or not (root / "s1").is_dir():
        sys.exit(f"{root}: expected mix.wav+s1.wav+s2.wav, or mix_clean|mix_both / s1 / s2 folders")
    cases = [
        (m.stem, m, [root / "s1" / m.name, root / "s2" / m.name], None)
        for m in sorted(mix_dir.glob("*.wav"))
    ]
    return cases[:limit] if limit else cases


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path)
    ap.add_argument("--mix", type=Path)
    ap.add_argument("--refs", type=Path, nargs="+")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    if args.dir:
        cases = find_cases(args.dir, args.limit)
    elif args.mix and args.refs:
        cases = [(args.mix.stem, args.mix, args.refs, None)]
    else:
        ap.error("give --dir, or --mix with --refs")

    separator = OverlapSeparator(args.model, device=pick_device())

    print(f"{'case':<28} {'SI-SDR s1':>10} {'SI-SDR s2':>10} {'mixture':>9} {'SI-SDRi':>9}")
    improvements = []
    for name, mix_path, ref_paths, window in cases:
        mix = load_audio(mix_path)
        refs = [load_audio(p) for p in ref_paths]
        if window:
            mix = slice_audio(mix, TARGET_SR, *window)
            refs = [slice_audio(r, TARGET_SR, *window) for r in refs]

        streams = separator.separate(mix, TARGET_SR)
        est = [s.audio for s in streams]
        scores, perm = pit_si_sdr(est, refs)
        baseline = float(np.mean([si_sdr(mix, r) for r in refs]))
        gain = float(np.mean(scores)) - baseline
        improvements.append(gain)
        print(f"{name[:28]:<28} {scores[0]:>10.2f} {scores[1]:>10.2f} {baseline:>9.2f} {gain:>9.2f}")

    if len(improvements) > 1:
        print(f"\nmean SI-SDRi over {len(improvements)} mixtures: {np.mean(improvements):.2f} dB")


if __name__ == "__main__":
    main()
