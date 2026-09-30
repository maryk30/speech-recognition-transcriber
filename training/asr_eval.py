"""
Evaluate an ASR model on held-out AMI (single distant mic) utterances.

    python training/asr_eval.py                                   # baseline: whisper-small (MLX)
    python training/asr_eval.py --model models/whisper-small-ami-mlx
    python training/asr_eval.py --backend ctranslate2 --model small

Reports, over the same sampled utterances:
    WER            Whisper's standard English normaliser on both sides
                   (lower-case, spelling and number normalisation, fillers
                   removed) -- the number to compare against published results
    verbatim WER   same but fillers count: keeping "um" is right, dropping it is an error
    filler recall  fraction of reference fillers (um, uh, hmm, mm ...) present in the output
    RTF            real-time factor of the transcription itself

Each rate comes with a 95% bootstrap interval (utterances resampled with
replacement, --n-boot times), so two models' numbers can be compared honestly.

Utterances are from the test split (meetings never used in training) and are
raw SDM clips, so they include the other speakers' overlapping speech and
room noise -- the same conditions the pipeline's ASR sees.
"""

from __future__ import annotations

import argparse
import io
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from audio_utils import normalize_rms  # noqa: E402
from bootstrap import bootstrap_ratio_ci  # noqa: E402
from config import DATA_DIR  # noqa: E402
from scoring import FILLERS, edit_distance, filler_recall  # noqa: E402

_TAGS = re.compile(r"<[^>]*>|\[[^\]]*\]")


def load_utterances(split: str, n: int, seed: int, min_s: float, max_s: float):
    files = sorted((DATA_DIR / "ami" / "sdm").glob(f"{split}-*.parquet"))
    if not files:
        sys.exit(f"no {split} shards in data/ami/sdm -- run training/download_ami.py")
    rows = []
    for f in files:
        t = pq.read_table(f, columns=["meeting_id", "text", "begin_time", "end_time", "audio"])
        for r in t.to_pylist():
            d = r["end_time"] - r["begin_time"]
            if min_s <= d <= max_s and r["text"].strip():
                rows.append(r)
    random.Random(seed).shuffle(rows)
    out = []
    for r in rows[:n]:
        audio, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
        assert sr == 16000
        if audio.ndim > 1:           # a few AMI clips are stereo
            audio = audio.mean(axis=1)
        out.append((r["meeting_id"], audio, r["text"]))
    return out


def get_normalizer():
    from transformers import WhisperTokenizer

    return WhisperTokenizer.from_pretrained("openai/whisper-small").normalize


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="small", help="Whisper size, HF repo, or local MLX model dir")
    ap.add_argument("--backend", default="mlx", choices=["mlx", "ctranslate2"])
    ap.add_argument("--split", default="test")
    ap.add_argument("-n", type=int, default=300, help="number of utterances")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-s", type=float, default=1.0)
    ap.add_argument("--max-s", type=float, default=20.0)
    ap.add_argument("--normalize", action="store_true", help="loudness-normalise each clip first (audio_utils.normalize_rms)")
    ap.add_argument("--pack", action="store_true", help="use batched/packed transcription (pipeline's clean-clip path)")
    ap.add_argument("--save", default=None, help="write per-utterance hypotheses to this JSON file")
    ap.add_argument("--n-boot", type=int, default=1000, help="bootstrap resamples for the 95%% intervals")
    args = ap.parse_args()

    from asr_baseline import make_transcriber

    data = load_utterances(args.split, args.n, args.seed, args.min_s, args.max_s)
    print(f"{len(data)} utterances, {sum(len(a) for _, a, _ in data) / 16000 / 60:.1f} min, "
          f"{len({m for m, _, _ in data})} meetings", flush=True)

    asr = make_transcriber(args.model, backend=args.backend, beam_size=1)
    norm = get_normalizer()

    clips = [normalize_rms(a) if args.normalize else a for _, a, _ in data]
    t0 = time.time()
    if args.pack and hasattr(asr, "transcribe_many"):
        results = asr.transcribe_many(clips)
    else:
        results = [asr.transcribe(c) for c in clips]
    wall = time.time() - t0
    hyps = [" ".join(u.text.strip() for u in utts) for utts in results]

    err = words = verr = vwords = 0
    f_found = f_total = 0
    per_utt = []
    for (meeting, _, ref), hyp in zip(data, hyps):
        ref_n, hyp_n = norm(_TAGS.sub(" ", ref)).split(), norm(_TAGS.sub(" ", hyp)).split()
        e = edit_distance(ref_n, hyp_n)
        err += e; words += len(ref_n)
        # verbatim: lower-case + strip punctuation only, so fillers are counted
        rv = re.sub(r"[^a-z0-9'\s-]", " ", ref.lower()).replace("-", " ").split()
        hv = re.sub(r"[^a-z0-9'\s-]", " ", hyp.lower()).replace("-", " ").split()
        ve = edit_distance(rv, hv)
        verr += ve; vwords += len(rv)
        found, total = filler_recall(ref, hyp)
        f_found += found; f_total += total
        per_utt.append({"meeting": meeting, "ref": ref.lower(), "hyp": hyp.strip(),
                        "errors": e, "words": len(ref_n), "v_errors": ve, "v_words": len(rv),
                        "fillers_found": found, "fillers_total": total})

    def ci(num_key: str, den_key: str) -> str:
        _, lo, hi = bootstrap_ratio_ci([r[num_key] for r in per_utt], [r[den_key] for r in per_utt],
                                       n_boot=args.n_boot)
        return f"[95% CI {lo:.1%}-{hi:.1%}]"

    audio_s = sum(len(c) for c in clips) / 16000
    print(f"\nmodel: {args.model} ({args.backend}{', packed' if args.pack else ''}{', level-normalised' if args.normalize else ''})")
    print(f"WER (standard, fillers ignored) {err / max(words, 1):7.1%}   ({err}/{words} words)  {ci('errors', 'words')}")
    print(f"WER (verbatim, fillers scored)  {verr / max(vwords, 1):7.1%}   ({verr}/{vwords} words)  {ci('v_errors', 'v_words')}")
    print(f"filler recall                   {f_found / max(f_total, 1):7.1%}   ({f_found}/{f_total})  {ci('fillers_found', 'fillers_total')}")
    print(f"real-time factor                {wall / audio_s:7.2f}   ({wall:.0f}s for {audio_s:.0f}s audio)")
    if args.save:
        Path(args.save).write_text(json.dumps(per_utt, indent=1))
    print("\nexamples:")
    for row in per_utt[:6]:
        print(f"  REF {row['ref']}\n  HYP {row['hyp'].lower()}\n")


if __name__ == "__main__":
    main()
