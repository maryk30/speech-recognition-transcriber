import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from make_mixture import make_mixture  # noqa: E402


def test_mixture_is_sum_of_aligned_refs_and_reports_overlap():
    sr = 16000
    a = 0.3 * np.ones(sr * 4, dtype=np.float32)
    b = 0.3 * np.ones(sr * 3, dtype=np.float32)
    mix, s1, s2, truth = make_mixture(a, b, offset_s=2.0, snr_db=0.0, sr=sr)
    assert len(mix) == len(s1) == len(s2) == sr * 5
    assert np.allclose(mix, s1 + s2, atol=1e-5)
    assert truth["overlap"] == [2.0, 4.0]
    assert np.abs(s2[: sr * 2]).max() == 0          # B silent before its offset


def test_no_overlap_when_b_starts_after_a_ends():
    sr = 16000
    a = np.ones(sr, dtype=np.float32) * 0.2
    b = np.ones(sr, dtype=np.float32) * 0.2
    _, _, _, truth = make_mixture(a, b, offset_s=2.0, snr_db=0.0, sr=sr)
    assert truth["overlap"] is None


def test_snr_sets_relative_level():
    sr = 16000
    a = 0.3 * np.ones(sr * 2, dtype=np.float32)
    b = 0.3 * np.ones(sr * 2, dtype=np.float32)
    _, s1, s2, _ = make_mixture(a, b, offset_s=0.0, snr_db=6.0, sr=sr)
    ratio_db = 20 * np.log10(np.abs(s1).max() / np.abs(s2).max())
    assert abs(ratio_db - 6.0) < 0.1
