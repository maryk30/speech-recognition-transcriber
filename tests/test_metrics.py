import numpy as np

from metrics import pit_si_sdr, si_sdr


def _sig(seed, n=16000):
    return np.random.default_rng(seed).standard_normal(n).astype(np.float32)


def test_si_sdr_perfect_estimate_is_very_high():
    x = _sig(0)
    assert si_sdr(x, x) > 60


def test_si_sdr_is_scale_invariant():
    x = _sig(0)
    est = x + 0.3 * _sig(5)   # imperfect estimate, so the score is well away from the eps floor
    assert abs(si_sdr(3.0 * est, x) - si_sdr(est, x)) < 1e-3


def test_si_sdr_of_uncorrelated_signal_is_negative():
    assert si_sdr(_sig(1), _sig(2)) < -10


def test_pit_recovers_swapped_order():
    a, b = _sig(3), _sig(4)
    scores, perm = pit_si_sdr([b, a], [a, b])   # estimates arrive swapped
    assert perm == (1, 0)
    assert min(scores) > 60
