import pytest

from bootstrap import block_counts, bootstrap_ratio_ci
from scoring import Turn, cp_wer


def test_ci_brackets_point_and_is_ratio_of_sums():
    num, den = [1, 0, 5, 2, 0, 3], [10, 2, 20, 10, 5, 13]
    point, lo, hi = bootstrap_ratio_ci(num, den, n_boot=2000)
    assert point == pytest.approx(11 / 60)          # not the mean of per-item rates
    assert lo <= point <= hi and lo < hi


def test_ci_degenerate_cases():
    assert bootstrap_ratio_ci([], []) == (0.0, 0.0, 0.0)
    assert bootstrap_ratio_ci([0, 0], [5, 5]) == (0.0, 0.0, 0.0)


def test_ci_is_deterministic_for_a_seed():
    num, den = [1, 2, 3, 0], [4, 5, 6, 7]
    assert bootstrap_ratio_ci(num, den, seed=3) == bootstrap_ratio_ci(num, den, seed=3)


def test_ci_narrows_with_more_items():
    small = bootstrap_ratio_ci([1, 3] * 5, [10, 10] * 5)
    big = bootstrap_ratio_ci([1, 3] * 500, [10, 10] * 500)
    assert (big[2] - big[1]) < (small[2] - small[1])


REF = [Turn("A", 0, 10, "one two three"), Turn("B", 40, 50, "four five"), Turn("A", 70, 80, "six")]


def test_block_counts_sum_matches_cpwer_when_nothing_crosses_blocks():
    hyp = [Turn("X", 0, 10, "one two tree"), Turn("Y", 40, 50, "four five"), Turn("X", 70, 80, "")]
    w, mapping = cp_wer(REF, hyp)
    counts = block_counts(REF, hyp, mapping, block_s=30)
    assert len(counts) == 3
    assert sum(e for e, _ in counts) / sum(n for _, n in counts) == pytest.approx(w)


def test_block_counts_unmapped_speaker_is_insertions():
    hyp = [Turn("X", 0, 10, "one two three"), Turn("Z", 40, 50, "extra words")]
    counts = block_counts(REF[:1], hyp, {"X": "A"}, block_s=30)
    assert counts[0] == (0, 3) and counts[1] == (2, 0)


def test_paired_diff_detects_consistent_small_gain():
    from bootstrap import paired_bootstrap_diff
    # b is better by 1 error on every item; items vary a lot in difficulty
    a = [10, 2, 30, 5, 8, 20] * 20
    b = [x - 1 for x in a]
    d = [40, 10, 60, 20, 30, 50] * 20
    r = paired_bootstrap_diff(a, b, d)
    assert r["diff"] < 0 and r["high"] < 0 and r["p_b_not_better"] == 0.0


def test_paired_diff_no_difference():
    from bootstrap import paired_bootstrap_diff
    a = [3, 1, 4, 1, 5] * 10
    r = paired_bootstrap_diff(a, a, [10] * 50)
    assert r["diff"] == 0.0 and r["low"] == 0.0 and r["high"] == 0.0
