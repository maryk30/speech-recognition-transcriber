"""
End-to-end scoring (Phases 4 and 6).

- DER: diarization error rate, overlap included, via pyannote.metrics.
- cpWER: concatenated-minimum-permutation WER -- the standard multi-speaker
  ASR metric. Each speaker's words are concatenated in time order, speakers
  are matched one-to-one to minimise total errors, so it penalises both
  wrong words AND words credited to the wrong person.
- Disfluency-aware: fillers ("um", "uh", ...) are dropped from BOTH
  reference and hypothesis before scoring, so keeping them is never
  penalised (plan: "don't use stock WER"). `filler_recall` reports
  separately how many of the reference fillers were actually retained.
- Speaker consistency: how much of each true speaker's speech carries a
  single hypothesis label (1.0 = never drifts / splits).

Everything except `der` is pure Python/numpy and unit-tested.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

FILLERS = frozenset({"um", "umm", "uh", "uhh", "er", "erm", "ah", "hmm", "hm", "mm", "mmm", "mhm"})

_PAUSE_MARKER = re.compile(r"\[pause[^\]]*\]", re.IGNORECASE)
_NON_WORD = re.compile(r"[^a-z0-9'\s-]")


@dataclass(frozen=True)
class Turn:
    speaker: str
    start: float
    end: float
    text: str = ""


def normalize(text: str, drop_fillers: bool = True) -> List[str]:
    """Lowercase, strip [pause] markers and punctuation, split into words."""
    text = _PAUSE_MARKER.sub(" ", text.lower())
    text = _NON_WORD.sub(" ", text.replace("-", " "))
    words = text.split()
    return [w for w in words if not (drop_fillers and w in FILLERS)]


def edit_distance(ref: Sequence[str], hyp: Sequence[str]) -> int:
    """Word-level Levenshtein distance (substitutions + deletions + insertions)."""
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


def wer(reference: str, hypothesis: str, drop_fillers: bool = True) -> float:
    ref, hyp = normalize(reference, drop_fillers), normalize(hypothesis, drop_fillers)
    if not ref:
        return 0.0 if not hyp else float(len(hyp))
    return edit_distance(ref, hyp) / len(ref)


def filler_recall(reference: str, hypothesis: str) -> Tuple[int, int]:
    """(reference fillers found in the hypothesis, total reference fillers),
    counted as a multiset so 3 reference "um"s and 1 hypothesis "um" = 1/3."""
    ref = Counter(w for w in normalize(reference, drop_fillers=False) if w in FILLERS)
    hyp = Counter(w for w in normalize(hypothesis, drop_fillers=False) if w in FILLERS)
    return sum((ref & hyp).values()), sum(ref.values())


def _text_by_speaker(turns: Iterable[Turn]) -> Dict[str, str]:
    out: Dict[str, List[str]] = {}
    for t in sorted(turns, key=lambda t: (t.start, t.end)):
        out.setdefault(t.speaker, []).append(t.text)
    return {spk: " ".join(parts) for spk, parts in out.items()}


def cp_wer(
    reference: Iterable[Turn], hypothesis: Iterable[Turn], drop_fillers: bool = True
) -> Tuple[float, Dict[str, str]]:
    """Returns (cpWER, {hypothesis speaker -> reference speaker})."""
    ref_text, hyp_text = _text_by_speaker(reference), _text_by_speaker(hypothesis)
    ref_names, hyp_names = list(ref_text), list(hyp_text)
    ref_words = [normalize(ref_text[n], drop_fillers) for n in ref_names]
    hyp_words = [normalize(hyp_text[n], drop_fillers) for n in hyp_names]

    # Pad to a square matrix with empty "speakers": an unmatched speaker then
    # costs all its words as deletions/insertions.
    size = max(len(ref_words), len(hyp_words))
    ref_words += [[]] * (size - len(ref_words))
    hyp_words += [[]] * (size - len(hyp_words))
    cost = np.array([[edit_distance(r, h) for h in hyp_words] for r in ref_words], dtype=float)

    rows, cols = linear_sum_assignment(cost)
    total_ref = sum(len(r) for r in ref_words)
    errors = float(cost[rows, cols].sum())
    mapping = {
        hyp_names[c]: ref_names[r]
        for r, c in zip(rows, cols)
        if r < len(ref_names) and c < len(hyp_names)
    }
    return (errors / total_ref if total_ref else 0.0), mapping


def speaker_consistency(
    reference: Iterable[Turn], hypothesis: Iterable[Turn], step: float = 0.05
) -> Dict[str, float]:
    """Per reference speaker: fraction of their speech time covered by their
    single most-used hypothesis label. Low values mean the label drifted or
    the speaker was split across clusters."""
    ref, hyp = list(reference), list(hypothesis)
    if not ref:
        return {}
    horizon = max(t.end for t in ref + hyp)
    grid = np.arange(0.0, horizon, step) + step / 2

    def active(turns: List[Turn], label: str) -> np.ndarray:
        mask = np.zeros(len(grid), dtype=bool)
        for t in turns:
            if t.speaker == label:
                mask |= (grid >= t.start) & (grid < t.end)
        return mask

    hyp_labels = sorted({t.speaker for t in hyp})
    hyp_masks = {lab: active(hyp, lab) for lab in hyp_labels}
    result: Dict[str, float] = {}
    for spk in sorted({t.speaker for t in ref}):
        mask = active(ref, spk)
        total = mask.sum()
        best = max((int((mask & m).sum()) for m in hyp_masks.values()), default=0)
        result[spk] = best / total if total else 1.0
    return result


def der(
    reference: Sequence[Turn], hypothesis: Sequence[Turn], collar: float = 0.25
) -> Dict[str, float]:
    """Diarization error rate with overlap scored (skip_overlap=False).
    Returns der / missed / false_alarm / confusion as fractions of reference speech."""
    from pyannote.core import Annotation, Segment
    from pyannote.metrics.diarization import DiarizationErrorRate

    def to_annotation(turns: Sequence[Turn]) -> Annotation:
        ann = Annotation()
        for t in turns:
            ann[Segment(t.start, t.end)] = t.speaker
        return ann

    metric = DiarizationErrorRate(collar=collar, skip_overlap=False)
    detail = metric(to_annotation(reference), to_annotation(hypothesis), detailed=True)
    total = detail["total"] or 1.0
    return {
        "der": detail["diarization error rate"],
        "missed": detail["missed detection"] / total,
        "false_alarm": detail["false alarm"] / total,
        "confusion": detail["confusion"] / total,
    }


def load_diarization_reference(path: str) -> List[Turn]:
    """Speaker turns for DER. A JSON reference may carry a separate, complete
    "diar_turns" list (e.g. the utterance-level AMI data omits some speech, so
    "turns" alone would inflate false alarms); otherwise the normal turns."""
    import json
    from pathlib import Path

    if Path(path).suffix.lower() == ".json":
        data = json.loads(Path(path).read_text())
        if data.get("diar_turns"):
            return [Turn(t["speaker"], float(t["start"]), float(t["end"])) for t in data["diar_turns"]]
    return load_reference(path)


def load_reference(path: str) -> List[Turn]:
    """Ground truth from JSON ({"turns": [{speaker,start,end,text?}]}) or RTTM."""
    import json
    from pathlib import Path

    p = Path(path)
    if p.suffix.lower() == ".rttm":
        turns = []
        for line in p.read_text().splitlines():
            f = line.split()
            if len(f) >= 8 and f[0] == "SPEAKER":
                start, dur = float(f[3]), float(f[4])
                turns.append(Turn(f[7], start, start + dur))
        return turns
    data = json.loads(p.read_text())
    return [
        Turn(t["speaker"], float(t["start"]), float(t["end"]), t.get("text", ""))
        for t in data["turns"]
    ]
