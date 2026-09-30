"""
Download a subset of the AMI Meeting Corpus (utterance-level, single distant
microphone = 'sdm') from HuggingFace: edinburghcstr/ami, licence CC-BY-4.0,
not gated.

SDM is the far-field, one-mic-in-the-middle-of-the-room recording, which is
the closest public match to this project's "shared room audio" setting, and
its transcripts are verbatim (fillers such as UM / UH / MM are kept).

Shards are picked evenly across each split (not the first N) so the subset
covers many meetings and speakers. Files land in data/ami/.

    python training/download_ami.py [--train-shards 7] [--test-shards 2]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import DATA_DIR  # noqa: E402

REPO = "edinburghcstr/ami"
TOTALS = {"train": 27, "validation": 5, "test": 4}


def spaced(total: int, n: int) -> list[int]:
    n = min(n, total)
    return sorted({round(i * total / n) for i in range(n)}) if n else []


def main() -> None:
    from huggingface_hub import hf_hub_download

    ap = argparse.ArgumentParser()
    ap.add_argument("--train-shards", type=int, default=7)
    ap.add_argument("--val-shards", type=int, default=1)
    ap.add_argument("--test-shards", type=int, default=2)
    args = ap.parse_args()

    want = {"train": args.train_shards, "validation": args.val_shards, "test": args.test_shards}
    for split, n in want.items():
        for idx in spaced(TOTALS[split], n):
            name = f"sdm/{split}-{idx:05d}-of-{TOTALS[split]:05d}.parquet"
            path = hf_hub_download(REPO, name, repo_type="dataset", local_dir=DATA_DIR / "ami")
            print("ok", name, f"{Path(path).stat().st_size / 1e6:.0f} MB", flush=True)


if __name__ == "__main__":
    main()
