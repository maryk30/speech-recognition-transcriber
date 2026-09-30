import pytest

pytest.importorskip("meeteval")

from meeteval_scoring import meeteval_cpwer, meeteval_tcpwer  # noqa: E402
from scoring import Turn, cp_wer  # noqa: E402

REF = [
    Turn("A", 0.0, 2.0, "hello um world"),
    Turn("B", 2.0, 3.0, "yes"),
    Turn("A", 3.0, 5.0, "see you later"),
]


def test_perfect_hypothesis_is_zero_with_permuted_labels():
    hyp = [Turn("X", 0.0, 2.0, "hello world"), Turn("Y", 2.0, 3.0, "yes"), Turn("X", 3.0, 5.0, "see you later")]
    r = meeteval_cpwer(REF, hyp)
    assert r["wer"] == 0.0
    assert r["assignment"] == {"X": "A", "Y": "B"}


def test_matches_our_cpwer_on_errors():
    hyp = [Turn("X", 0.0, 2.0, "hello word"), Turn("Y", 2.0, 3.0, "yes"), Turn("X", 3.0, 5.0, "see you")]
    ours, _ = cp_wer(REF, hyp)
    assert meeteval_cpwer(REF, hyp)["wer"] == pytest.approx(ours)


def test_fillers_scored_when_asked():
    hyp = [Turn("X", 0.0, 2.0, "hello world"), Turn("Y", 2.0, 3.0, "yes"), Turn("X", 3.0, 5.0, "see you later")]
    assert meeteval_cpwer(REF, hyp, drop_fillers=False)["deletions"] == 1


def test_tcpwer_penalises_words_far_in_time():
    hyp = [Turn("A", 100.0, 102.0, "hello world"), Turn("B", 102.0, 103.0, "yes"), Turn("A", 103.0, 105.0, "see you later")]
    assert meeteval_cpwer(REF, hyp)["wer"] == 0.0
    assert meeteval_tcpwer(REF, hyp, collar=5.0)["wer"] > 0.0
