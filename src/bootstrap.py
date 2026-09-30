"""
Bootstrap confidence intervals for error rates, so results aren't bare point
estimates (3 meetings / 300 utterances is a small sample).

A rate like WER is a ratio of sums (total errors / total reference words), so
items are resampled with replacement and the *ratio of the resampled sums* is
recomputed -- not the mean of per-item rates, which over-weights short items.

For a single meeting's cpWER the items are fixed-length time blocks
(`block_counts`): turns inside a block are correlated, blocks much less so.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from scoring import Turn, edit_distance, normalize


def bootstrap_ratio_ci(
    numer: Sequence[float], denom: Sequence[float], n_boot: int = 1000,
    alpha: float = 0.05, seed: int = 0,
) -> Tuple[float, float, float]:
    """(point estimate, low, high) for sum(numer)/sum(denom), percentile
    bootstrap over items. Items with denom 0 still count (their errors are
    insertions against nothing)."""
    num, den = np.asarray(numer, float), np.asarray(denom, float)
    if len(num) != len(den):
        raise ValueError("numer and denom must have the same length")
    if len(num) == 0 or den.sum() == 0:
        return 0.0, 0.0, 0.0
    point = num.sum() / den.sum()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(num), size=(n_boot, len(num)))
    d = den[idx].sum(axis=1)
    ratios = np.where(d > 0, num[idx].sum(axis=1) / np.maximum(d, 1e-12), 0.0)
    lo, hi = np.quantile(ratios, [alpha / 2, 1 - alpha / 2])
    return float(point), float(lo), float(hi)


def block_counts(
    reference: Sequence[Turn], hypothesis: Sequence[Turn], mapping: Dict[str, str],
    block_s: float = 30.0, drop_fillers: bool = True,
) -> List[Tuple[int, int]]:
    """Per time block: (errors, reference words) under a FIXED hyp->ref speaker
    mapping (from cp_wer on the whole meeting). Turns go to the block holding
    their midpoint. Unmapped hypothesis speakers count as insertions.

    Summed over blocks this is close to, but can exceed, the whole-meeting
    cpWER: an alignment can't cross a block edge. Use it for the interval's
    width, and report the whole-meeting cpWER as the point estimate."""
    turns = list(reference) + list(hypothesis)
    if not turns:
        return []
    n_blocks = int(max(t.end for t in turns) // block_s) + 1
    ref_speakers = sorted({t.speaker for t in reference})

    def words_in(turns_, speaker_of, block):
        out: Dict[str, List[str]] = {}
        for t in sorted(turns_, key=lambda t: t.start):
            if int(((t.start + t.end) / 2) // block_s) == block:
                out.setdefault(speaker_of(t), []).extend(normalize(t.text, drop_fillers))
        return out

    counts = []
    for b in range(n_blocks):
        ref_w = words_in(reference, lambda t: t.speaker, b)
        hyp_w = words_in(hypothesis, lambda t: mapping.get(t.speaker, "__unmapped__" + t.speaker), b)
        errors = sum(edit_distance(ref_w.get(s, []), hyp_w.get(s, [])) for s in ref_speakers)
        errors += sum(len(w) for s, w in hyp_w.items() if s not in ref_speakers)
        counts.append((errors, sum(len(w) for w in ref_w.values())))
    return counts
