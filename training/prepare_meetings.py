"""
Cut real AMI meeting excerpts (single distant mic, 4 speakers each) plus a
ground-truth reference, for end-to-end scoring of the whole pipeline:

    python training/prepare_meetings.py                 # 3 default excerpts
    ./run.sh score data/ami_meetings/ES2004c_300s.wav --reference data/ami_meetings/ES2004c_300s.json

Audio comes from diarizers-community/ami (full-meeting recordings, CC-BY-4.0);
speaker turns and text come from the utterance-level edinburghcstr/ami test
shards already in data/ami/sdm (same annotations), so DER and cpWER are both
computable. Only meetings whose utterances are present in the downloaded
shards are usable. Excerpts start `--start` seconds in and last `--length`.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import DATA_DIR  # noqa: E402

DEFAULT_MEETINGS = ["ES2004c", "IS1009b", "EN2002b"]
_CLEAN = re.compile(r"[^a-z0-9'\- ]")


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", _CLEAN.sub(" ", text.lower())).strip()


def utterances_for(meeting: str):
    rows = []
    for f in sorted((DATA_DIR / "ami" / "sdm").glob("test-*.parquet")):
        t = pq.read_table(f, columns=["meeting_id", "speaker_id", "begin_time", "end_time", "text"])
        rows += [r for r in t.to_pylist() if r["meeting_id"] == meeting and clean(r["text"])]
    return sorted(rows, key=lambda r: r["begin_time"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("meetings", nargs="*", default=DEFAULT_MEETINGS)
    ap.add_argument("--start", type=float, default=300.0, help="excerpt start (s into the meeting)")
    ap.add_argument("--length", type=float, default=300.0, help="excerpt length (s)")
    args = ap.parse_args()

    out_dir = DATA_DIR / "ami_meetings"
    wanted = set(args.meetings)
    done = set()
    for shard in sorted((out_dir / "sdm").glob("test-*.parquet")):
        for row in pq.read_table(shard).to_pylist():
            meeting = row["audio"]["path"].split(".")[0]
            if meeting not in wanted or meeting in done:
                continue
            utts = utterances_for(meeting)
            if not utts:
                print(f"{meeting}: no utterance text in data/ami/sdm, skipped")
                continue
            audio, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            assert sr == 16000, sr
            lo, hi = args.start, min(args.start + args.length, len(audio) / sr)
            excerpt = audio[int(lo * sr) : int(hi * sr)]

            turns = [
                {"speaker": u["speaker_id"], "start": round(u["begin_time"] - lo, 3),
                 "end": round(u["end_time"] - lo, 3), "text": clean(u["text"])}
                for u in utts if u["begin_time"] >= lo and u["end_time"] <= hi
            ]
            # Complete speaker-turn annotation (from the full-meeting dataset) for DER:
            # the utterance-level data drops some speech, which shows up as false alarms.
            diar = [
                {"speaker": sp, "start": round(max(a, lo) - lo, 3), "end": round(min(b, hi) - lo, 3)}
                for a, b, sp in zip(row["timestamps_start"], row["timestamps_end"], row["speakers"])
                if b > lo and a < hi
            ]
            stem = f"{meeting}_{int(hi - lo)}s"
            sf.write(out_dir / f"{stem}.wav", excerpt, sr)
            (out_dir / f"{stem}.json").write_text(json.dumps({"meeting": meeting, "offset": lo, "turns": turns, "diar_turns": diar}, indent=1))

            events = sorted([(t["start"], t["end"]) for t in turns])
            overlap = sum(max(0.0, min(a[1], b[1]) - b[0]) for i, a in enumerate(events) for b in events[i + 1: i + 4] if b[0] < a[1])
            words = sum(len(t["text"].split()) for t in turns)
            print(f"{stem}: {len(excerpt) / sr:.0f}s, {len(turns)} turns, {len({t['speaker'] for t in turns})} speakers, "
                  f"{words} words, ~{overlap:.0f}s of overlapping speech")
            done.add(meeting)
    missing = wanted - done
    if missing:
        print("not found:", ", ".join(sorted(missing)))


if __name__ == "__main__":
    main()
