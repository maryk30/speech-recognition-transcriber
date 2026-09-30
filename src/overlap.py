"""
Phase 3 — VAD + overlap detection.

Turns the (possibly overlapping) speaker turns from diarization into a flat
timeline of routing segments:

    silence            -> dropped (that's the VAD part)
    one speaker        -> straight to ASR
    two+ speakers      -> speech separation first, then ASR per stream

Design note: pyannote's diarization model is itself built on a
frame-level segmentation network that already predicts "who is speaking
when, including simultaneously", so its turns give us VAD and overlap flags
from a single model pass -- no separate Silero-VAD / overlapped-speech
pipeline run is needed. (If overlap flags ever prove too noisy, pyannote's
dedicated overlapped-speech-detection pipeline can be added as a
cross-check without changing this module's interface.)

`build_segments` is pure logic and has unit tests (tests/test_overlap.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from diarization import DiarizedSegment

# Overlaps shorter than this are usually backchannels ("mm-hm") or boundary
# jitter; SepFormer is unreliable on such short clips, so they are folded
# into the neighbouring single-speaker segment instead.
DEFAULT_MIN_OVERLAP_S = 0.4
# Same-speaker segments separated by less than this are merged, so ASR sees
# whole utterances instead of fragments.
DEFAULT_MERGE_GAP_S = 0.4
# Drop single-speaker blips shorter than this.
DEFAULT_MIN_SPEECH_S = 0.15


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    speakers: Tuple[str, ...]  # sorted; length >= 2 means overlapping speech

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def is_overlapping(self) -> bool:
        return len(self.speakers) >= 2

    def as_tuple(self) -> Tuple[float, float, bool]:
        """The (start, end, is_overlapping) form named in the plan."""
        return (self.start, self.end, self.is_overlapping)


def _elementary_intervals(turns: Sequence[DiarizedSegment]) -> List[Segment]:
    """Cut the timeline at every turn boundary; each piece gets the set of
    speakers active throughout it. Silent pieces are skipped."""
    boundaries = sorted({t for turn in turns for t in (turn.start, turn.end)})
    pieces: List[Segment] = []
    for a, b in zip(boundaries, boundaries[1:]):
        mid = (a + b) / 2
        active = sorted({t.speaker_label for t in turns if t.start <= mid < t.end})
        if active:
            pieces.append(Segment(a, b, tuple(active)))
    return pieces


def _merge(segments: List[Segment], max_gap_s: float) -> List[Segment]:
    """Merge neighbours that have the same speaker set and are at most
    `max_gap_s` apart (silence in between is absorbed)."""
    merged: List[Segment] = []
    for seg in segments:
        if (
            merged
            and merged[-1].speakers == seg.speakers
            and seg.start - merged[-1].end <= max_gap_s
        ):
            merged[-1] = Segment(merged[-1].start, seg.end, seg.speakers)
        else:
            merged.append(seg)
    return merged


def _demote_short_overlaps(segments: List[Segment], min_overlap_s: float) -> List[Segment]:
    """Replace too-short overlap segments with a single speaker, preferring
    whoever was already talking just before (else just after)."""
    out: List[Segment] = list(segments)
    for i, seg in enumerate(out):
        if not seg.is_overlapping or seg.duration >= min_overlap_s:
            continue
        chosen = seg.speakers[0]
        prev_seg = out[i - 1] if i > 0 else None
        next_seg = out[i + 1] if i + 1 < len(out) else None
        if prev_seg and not prev_seg.is_overlapping and prev_seg.speakers[0] in seg.speakers:
            chosen = prev_seg.speakers[0]
        elif next_seg and not next_seg.is_overlapping and next_seg.speakers[0] in seg.speakers:
            chosen = next_seg.speakers[0]
        out[i] = Segment(seg.start, seg.end, (chosen,))
    return out


def build_segments(
    turns: Sequence[DiarizedSegment],
    min_overlap_s: float = DEFAULT_MIN_OVERLAP_S,
    merge_gap_s: float = DEFAULT_MERGE_GAP_S,
    min_speech_s: float = DEFAULT_MIN_SPEECH_S,
) -> List[Segment]:
    """Speaker turns (may overlap) -> chronological routing segments."""
    segments = _elementary_intervals(turns)
    segments = _merge(segments, merge_gap_s)
    segments = _demote_short_overlaps(segments, min_overlap_s)
    segments = _merge(segments, merge_gap_s)
    return [s for s in segments if s.is_overlapping or s.duration >= min_speech_s]


def detect_segments(
    audio_path: str, diarizer=None, **diarize_kwargs
) -> List[Segment]:
    """Phase 3 checkpoint: room audio in, (start, end, is_overlapping)
    segments out (see `Segment.as_tuple`)."""
    from audio_utils import load_audio
    from config import get_hf_token

    diarizer = diarizer or _default_diarizer(get_hf_token())
    audio = load_audio(audio_path)
    return build_segments(diarizer.diarize(audio, **diarize_kwargs))


def _default_diarizer(token: str):
    from diarization import Diarizer

    return Diarizer(hf_token=token)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Show speech / overlap segments for an audio file.")
    ap.add_argument("audio")
    ap.add_argument("--num-speakers", type=int, default=None)
    args = ap.parse_args()

    segs = detect_segments(args.audio, num_speakers=args.num_speakers)
    total = sum(s.duration for s in segs)
    over = sum(s.duration for s in segs if s.is_overlapping)
    print(f"{'start':>7} {'end':>7}  kind      speakers")
    for s in segs:
        kind = "OVERLAP" if s.is_overlapping else "single "
        print(f"{s.start:7.2f} {s.end:7.2f}  {kind}   {', '.join(s.speakers)}")
    print(f"\n{len(segs)} segments, {total:.1f}s speech, {over:.1f}s overlapping")
