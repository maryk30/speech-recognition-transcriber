"""
Turn AMI utterance clips into ~28 s Whisper training windows.

AMI ships each utterance as its own short clip (mean 3 s), but Whisper always
encodes a padded 30 s window, so training on single utterances wastes ~90% of
the compute and never shows the model the multi-utterance windows it sees in
real use. All clips of a meeting come from the SAME distant microphone, so
the meeting timeline can be rebuilt by placing each clip at its begin_time.

WINDOWS ARE SINGLE-SPEAKER ONLY. Earlier versions of this script windowed
ALL speakers' utterances together in time order, so the training TARGET was
a merged, everyone's-words transcript of the mixed audio. But the pipeline's
ASR never sees that: at inference it only ever gets one speaker's clip --
either a clean non-overlap segment (src/pipeline.py: _single_jobs) or a
separated stream (_overlap_jobs). Training on merged multi-speaker targets
taught the model to transcribe whoever it could hear, which is why the
fine-tuned model's WER on single-speaker evaluation clips got WORSE, not
better (insertions roughly tripled). This version reuses the exact same
single-vs-overlap routing logic the pipeline uses at inference
(src/overlap.build_segments) to find single-speaker stretches, and only
windows utterances that fall inside one -- so a training pair is always
"this audio -> this one speaker's words", matching production exactly.
Overlapping stretches are excluded from base-ASR fine-tuning (they go
through the separator at inference, a different problem -- see
prepare_overlap_data.py for training data aimed at that stage instead).

    python training/prepare_ami.py --split train
    python training/prepare_ami.py --split validation

Output (data/ami/): windows_<split>.i16 (int16 audio, concatenated),
windows_<split>.json (index: offset, length, meeting, speaker, segments[start,end,text]).

Audio is loudness-normalised per window (rms 0.05) and stored as int16.

Text: lower-cased, no punctuation, verbatim fillers kept (um, uh, hmm, mm ...)
-- AMI's own convention. Consequence for later: a model fine-tuned on this
outputs unpunctuated lower-case text.

Windows whose text contains a pathological word-repeat run (see
`has_pathological_repeat`) are dropped -- these are rare AMI transcription
artifacts, not real disfluency, and training on them is a plausible source
of the fine-tuned model's own repetition-loop failures at inference.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from audio_utils import normalize_rms  # noqa: E402
from config import DATA_DIR  # noqa: E402
from diarization import DiarizedSegment  # noqa: E402
from overlap import build_segments  # noqa: E402

SR = 16000
MAX_WINDOW_S = 28.0      # Whisper's window is 30 s; leave headroom
PAD_S = 0.15
MAX_GAP_S = 5.0          # start a new window after a long silence
TS_STEP = 0.02           # Whisper timestamp resolution
MAX_WORD_REPEAT_RUN = 4  # a word repeated more than this many times running is an artifact, not disfluency
MIN_SEGMENT_MATCH = 0.9  # an utterance must overlap a single-speaker stretch by at least this fraction to count

_CLEAN = re.compile(r"[^a-z0-9'\- ]")


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", _CLEAN.sub(" ", text.lower())).strip()


def quantize(t: float) -> float:
    return round(round(t / TS_STEP) * TS_STEP, 2)


def has_pathological_repeat(text: str, max_run: int = MAX_WORD_REPEAT_RUN) -> bool:
    """True if some word repeats more than `max_run` times in a row. Real AMI
    disfluency ("the the the") rarely exceeds 2-3; longer runs are almost
    always a broken transcript alignment, and are exactly the shape of the
    decoding loops the fine-tuned model itself produced at inference, so they
    are excluded from training rather than taught."""
    words = text.split()
    run = 1
    for a, b in zip(words, words[1:]):
        run = run + 1 if a == b else 1
        if run > max_run:
            return True
    return False


def load_meetings(split: str) -> Dict[str, List[dict]]:
    """meeting_id -> every speaker's utterances (needed both to rebuild the
    shared-mic timeline and to classify single-vs-overlapping stretches)."""
    files = sorted((DATA_DIR / "ami" / "sdm").glob(f"{split}-*.parquet"))
    if not files:
        sys.exit(f"no {split} shards in data/ami/sdm -- run training/download_ami.py")
    meetings: Dict[str, List[dict]] = defaultdict(list)
    for f in files:
        for r in pq.read_table(f).to_pylist():
            text = clean_text(r["text"])
            if text:
                meetings[r["meeting_id"]].append(
                    {"speaker": r["speaker_id"], "begin": float(r["begin_time"]), "end": float(r["end_time"]),
                     "text": text, "bytes": r["audio"]["bytes"]}
                )
    return meetings


def timeline(utts: List[dict]) -> Tuple[np.ndarray, float]:
    """Rebuild the meeting audio (float32) from the clips; returns (audio, t0).
    (Clips are FLOAT wavs in [-1, 1]; reading them as int16 would truncate
    everything to zero, so stay in float until the final per-window scaling.)
    Shared across all speakers -- a single-speaker window's audio still
    carries whatever natural background bleed the room mic picked up, same
    as the pipeline's own non-overlap segment slicing at inference."""
    t0 = min(u["begin"] for u in utts)
    t1 = max(u["end"] for u in utts)
    buf = np.zeros(int((t1 - t0) * SR) + SR, dtype=np.float32)
    for u in utts:
        clip, sr = sf.read(io.BytesIO(u["bytes"]), dtype="float32")
        assert sr == SR
        if clip.ndim > 1:            # a few AMI clips are stereo
            clip = clip.mean(axis=1)
        i = int(round((u["begin"] - t0) * SR))
        buf[i : i + len(clip)] = clip[: len(buf) - i]
    return buf, t0


def single_speaker_utterances(utts: List[dict]) -> List[dict]:
    """Utterances that fall (almost entirely) inside a stretch where exactly
    one speaker is active, tagged with `_segment` = index of that stretch (so
    callers can tell "same speaker, same stretch" apart from "same speaker,
    different stretch, far apart in time"). Pure function of (speaker,
    begin, end) triples -- reuses the identical routing logic the live
    pipeline uses (src/overlap.build_segments), so training windows are
    built from exactly the audio the pipeline would itself call single-
    speaker."""
    turns = [DiarizedSegment(u["speaker"], u["begin"], u["end"]) for u in utts]
    segments = build_segments(turns)
    single = [s for s in segments if not s.is_overlapping]

    kept: List[dict] = []
    seg_i = 0
    for u in sorted(utts, key=lambda u: (u["begin"], u["end"])):
        while seg_i < len(single) and single[seg_i].end <= u["begin"]:
            seg_i += 1
        if seg_i >= len(single):
            break
        seg = single[seg_i]
        if seg.start > u["begin"]:
            continue    # utterance starts before this (or any later) single-speaker stretch
        overlap = max(0.0, min(seg.end, u["end"]) - max(seg.start, u["begin"]))
        if overlap >= MIN_SEGMENT_MATCH * (u["end"] - u["begin"]):
            kept.append({**u, "_segment": seg_i})
    return kept


def make_windows(utts: List[dict]) -> List[List[dict]]:
    """Greedy grouping of time-ordered, single-speaker-stretch utterances
    (see `single_speaker_utterances`) into <= MAX_WINDOW_S windows. A window
    never crosses a stretch boundary (`_segment` change), so every window is
    guaranteed single-speaker even though stretches from different times/
    speakers are windowed independently."""
    utts = sorted(utts, key=lambda u: (u["begin"], u["end"]))
    windows: List[List[dict]] = []
    cur: List[dict] = []
    cur_segment = None
    start = end = 0.0
    for u in utts:
        new_window = (
            not cur
            or u.get("_segment") != cur_segment
            or max(end, u["end"]) + PAD_S - start > MAX_WINDOW_S
            or u["begin"] - end > MAX_GAP_S
        )
        if new_window and cur:
            windows.append(cur)
            cur = []
        if not cur:
            start, end, cur_segment = u["begin"] - PAD_S, u["end"], u.get("_segment")
        else:
            end = max(end, u["end"])
        cur.append(u)
    if cur:
        windows.append(cur)
    return windows


def window_segments(win: List[dict], w_start: float, w_len: float) -> List[List]:
    """[[start, end, text], ...] relative to the window, monotonic, quantised."""
    segs: List[List] = []
    prev_end = 0.0
    for u in win:
        s = quantize(max(u["begin"] - w_start, prev_end))
        e = quantize(min(max(u["end"] - w_start, s + 0.1), w_len))
        if e <= s:
            continue
        segs.append([s, e, u["text"]])
        prev_end = e
    return segs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="train")
    args = ap.parse_args()

    out_dir = DATA_DIR / "ami"
    audio_path, index_path = out_dir / f"windows_{args.split}.i16", out_dir / f"windows_{args.split}.json"

    meetings = load_meetings(args.split)
    index, offset, n_utts, n_dropped_repeat = 0, 0, 0, 0
    index = []
    with open(audio_path, "wb") as fout:
        for meeting, utts in sorted(meetings.items()):
            buf, t0 = timeline(utts)
            solo = single_speaker_utterances(utts)
            for win in make_windows(solo):
                w_start = win[0]["begin"] - PAD_S
                w_end = max(u["end"] for u in win) + PAD_S
                a = max(0, int((w_start - t0) * SR))
                b = min(len(buf), int((w_end - t0) * SR))
                clip = buf[a:b]
                w_len = len(clip) / SR
                segs = window_segments(win, w_start, w_len)
                if not segs:
                    continue
                joined = " ".join(t for _, _, t in segs)
                if has_pathological_repeat(joined):
                    n_dropped_repeat += 1
                    continue
                # Per-window loudness normalisation; the pipeline applies the same
                # to every clip at inference (audio_utils.normalize_rms).
                clip = (normalize_rms(clip) * 32767).astype(np.int16)
                clip.tofile(fout)
                index.append({
                    "meeting": meeting, "speaker": win[0]["speaker"], "offset": offset,
                    "length": len(clip), "segments": segs,
                })
                offset += len(clip)
                n_utts += len(segs)
            print(f"{meeting}: {len(utts)} utterances, {len(solo)} single-speaker", flush=True)
    index_path.write_text(json.dumps(index))
    hours = offset / SR / 3600
    print(f"\n{args.split}: {len(index)} windows, {n_utts} utterances, {hours:.1f} h audio, "
          f"{len(meetings)} meetings, mean window {offset / SR / max(len(index), 1):.1f}s, "
          f"{n_dropped_repeat} windows dropped for pathological repeats")


if __name__ == "__main__":
    main()
