"""
Separation quality metrics (Phase 2): SI-SDR and its improvement over the
unprocessed mixture, with permutation-invariant matching since separator
output order is arbitrary.
"""

from __future__ import annotations

from itertools import permutations
from typing import List, Sequence, Tuple

import numpy as np


def si_sdr(estimate: np.ndarray, reference: np.ndarray, eps: float = 1e-8) -> float:
    """Scale-invariant SDR in dB (higher is better)."""
    n = min(len(estimate), len(reference))
    est = estimate[:n].astype(np.float64)
    ref = reference[:n].astype(np.float64)
    est = est - est.mean()
    ref = ref - ref.mean()
    target = (np.dot(est, ref) / (np.dot(ref, ref) + eps)) * ref
    noise = est - target
    return float(10 * np.log10((np.dot(target, target) + eps) / (np.dot(noise, noise) + eps)))


def pit_si_sdr(
    estimates: Sequence[np.ndarray], references: Sequence[np.ndarray]
) -> Tuple[List[float], Tuple[int, ...]]:
    """Best per-source SI-SDR over all estimate<->reference permutations.
    Returns (scores aligned to `references`, permutation used), where
    permutation[i] is the index of the estimate matched to references[i]."""
    best_scores: List[float] = []
    best_perm: Tuple[int, ...] = ()
    best_mean = -np.inf
    for perm in permutations(range(len(estimates)), len(references)):
        scores = [si_sdr(estimates[p], references[i]) for i, p in enumerate(perm)]
        if np.mean(scores) > best_mean:
            best_mean, best_scores, best_perm = float(np.mean(scores)), scores, perm
    return best_scores, best_perm
