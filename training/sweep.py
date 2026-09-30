"""
Short learning-rate x trained-layer sweep before the long run (RETRAIN_TODO 6).

    python training/sweep.py                                   # 3 lr x 2 layer counts, 300 steps each
    python training/sweep.py --lrs 1e-5 2e-5 --layers 2 --steps 150

Each run is a normal finetune_whisper.py invocation into models/sweep/<name>;
its "val WER" log lines are parsed and the last/best values appended to
output/sweep_results.csv (one row per run, written as each run finishes, so
an interrupted sweep keeps what it has). Pick the winner by best_val_wer.

Note: the encoder cache depends on the layer count (enc_<split>_L<layer>.f16),
so each distinct --layers value builds its own cache (~2.3 MB per window).
"""

from __future__ import annotations

import argparse
import csv
import itertools
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parent.parent
_VAL = re.compile(r"^step (\d+) \| val loss ([\d.]+) \| val WER ([\d.]+)", re.MULTILINE)
FIELDS = ["name", "lr", "train_layers", "steps", "final_val_loss", "final_val_wer", "best_val_wer",
          "best_step", "minutes", "returncode"]


def parse_val_log(text: str) -> List[Tuple[int, float, float]]:
    """[(step, val_loss, val_wer)] from finetune_whisper.py output."""
    return [(int(s), float(l), float(w)) for s, l, w in _VAL.findall(text)]


def summarise(name: str, lr: float, layers: int, steps: int, log: str, minutes: float, rc: int) -> dict:
    vals = parse_val_log(log)
    row = dict(name=name, lr=lr, train_layers=layers, steps=steps, minutes=round(minutes, 1), returncode=rc,
               final_val_loss="", final_val_wer="", best_val_wer="", best_step="")
    if vals:
        best = min(vals, key=lambda v: v[2])
        row.update(final_val_loss=vals[-1][1], final_val_wer=vals[-1][2], best_val_wer=best[2], best_step=best[0])
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lrs", type=float, nargs="+", default=[5e-6, 1e-5, 2e-5])
    ap.add_argument("--layers", type=int, nargs="+", default=[2, 4])
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--val-every", type=int, default=100)
    ap.add_argument("--out-csv", default=str(ROOT / "output" / "sweep_results.csv"))
    ap.add_argument("extra", nargs=argparse.REMAINDER, help="after --, passed through to finetune_whisper.py")
    args = ap.parse_args()

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    new_file = not out_csv.exists()
    extra = [a for a in args.extra if a != "--"]

    for lr, layers in itertools.product(args.lrs, args.layers):
        name = f"lr{lr:g}_L{layers}"
        out_dir = ROOT / "models" / "sweep" / name
        cmd = [sys.executable, str(ROOT / "training" / "finetune_whisper.py"), "--lr", str(lr),
               "--train-layers", str(layers), "--steps", str(args.steps), "--val-every", str(args.val_every),
               "--save-every", str(args.steps), "--out", str(out_dir), *extra]
        print(f"=== {name}: {' '.join(cmd[1:])}", flush=True)
        t0 = time.time()
        lines = []
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=ROOT)
        for line in proc.stdout:                    # stream so progress stays visible
            print(line, end="", flush=True)
            lines.append(line)
        rc = proc.wait()
        row = summarise(name, lr, layers, args.steps, "".join(lines), (time.time() - t0) / 60, rc)
        with out_csv.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new_file:
                w.writeheader(); new_file = False
            w.writerow(row)
        print(f"=== {name}: best val WER {row['best_val_wer'] or 'n/a'} (rc {rc})", flush=True)


if __name__ == "__main__":
    main()
