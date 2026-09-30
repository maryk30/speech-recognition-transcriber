"""
MeetEval-backed multi-speaker WER (cpWER and time-constrained tcpWER).

`scoring.cp_wer` is our own cpWER. This module cross-checks it against the
reference implementation (MeetEval, MIT) and adds tcpWER, which additionally
requires a hypothesis word to fall near the reference word in time -- the
right metric for a chronologically merged transcript. Text goes through
`scoring.normalize`, so fillers are handled exactly as in the rest of the
project (dropped on both sides unless `drop_fillers=False`).

Words carry no timestamps here (turns only), so MeetEval spreads a turn's
words over the turn (its pseudo word-level timing); tcpWER is therefore only
as precise as the turn boundaries.
"""

from __future__ import annotations

from typing import Dict, Iterable, List

from scoring import Turn, normalize

SESSION = "session"


def _seglst(turns: Iterable[Turn], drop_fillers: bool):
    from meeteval.io import SegLST

    segments: List[dict] = []
    for t in turns:
        words = normalize(t.text, drop_fillers)
        if not words:
            continue
        segments.append({
            "session_id": SESSION,
            "speaker": t.speaker,
            "words": " ".join(words),
            "start_time": float(t.start),
            "end_time": float(t.end),
        })
    return SegLST(segments)


def _summary(result) -> Dict[str, object]:
    return {
        "wer": result.error_rate,
        "errors": result.errors,
        "ref_words": result.length,
        "insertions": result.insertions,
        "deletions": result.deletions,
        "substitutions": result.substitutions,
        "assignment": dict((h, r) for r, h in result.assignment),
    }


def meeteval_cpwer(
    reference: Iterable[Turn], hypothesis: Iterable[Turn], drop_fillers: bool = True
) -> Dict[str, object]:
    from meeteval.wer import cp_word_error_rate

    ref, hyp = _seglst(reference, drop_fillers), _seglst(hypothesis, drop_fillers)
    return _summary(cp_word_error_rate(ref, hyp))


def meeteval_tcpwer(
    reference: Iterable[Turn],
    hypothesis: Iterable[Turn],
    collar: float = 5.0,
    drop_fillers: bool = True,
) -> Dict[str, object]:
    """collar: seconds a hypothesis word may be off from the reference word
    (MeetEval's convention; 5 s is the CHiME/NOTSOFAR default)."""
    from meeteval.wer import time_constrained_minimum_permutation_word_error_rate as tcp

    ref, hyp = _seglst(reference, drop_fillers), _seglst(hypothesis, drop_fillers)
    return _summary(tcp(ref, hyp, collar=collar))
