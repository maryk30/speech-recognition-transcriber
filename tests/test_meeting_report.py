import pytest

pytest.importorskip("meeteval")

from meeting_report import format_markdown, score_meeting  # noqa: E402
from scoring import Turn  # noqa: E402

REF = [Turn("A", 0.0, 4.0, "so um i think we should go"), Turn("B", 4.0, 6.0, "yeah okay")]


def test_perfect_hypothesis():
    hyp = [Turn("X", 0.0, 4.0, "so um i think we should go"), Turn("Y", 4.0, 6.0, "yeah okay")]
    diar = [Turn("A", 0.0, 4.0), Turn("B", 4.0, 6.0)]
    r = score_meeting(REF, hyp, diar, [Turn("X", 0.0, 4.0), Turn("Y", 4.0, 6.0)])
    assert r["cpwer"] == 0.0 and r["cpwer_fillers"] == 0.0 and r["tcpwer"] == 0.0
    assert r["der"] == pytest.approx(0.0)
    assert (r["filler_found"], r["filler_total"]) == (1, 1)


def test_dropped_filler_only_hurts_filler_scores():
    hyp = [Turn("X", 0.0, 4.0, "so i think we should go"), Turn("Y", 4.0, 6.0, "yeah okay")]
    r = score_meeting(REF, hyp)
    assert r["cpwer"] == 0.0 and r["cpwer_fillers"] > 0.0
    assert r["filler_found"] == 0 and "der" not in r


def test_our_cpwer_agrees_with_meeteval():
    hyp = [Turn("X", 0.0, 4.0, "so i think we go"), Turn("Y", 4.0, 6.0, "yeah")]
    r = score_meeting(REF, hyp)
    assert r["cpwer"] == pytest.approx(r["meeteval_cpwer"])


def test_markdown_table():
    hyp = [Turn("X", 0.0, 4.0, "so um i think we should go"), Turn("Y", 4.0, 6.0, "yeah okay")]
    row = score_meeting(REF, hyp)
    row.update(meeting="M1", asr="stock")
    md = format_markdown([row])
    assert "| M1 | stock | - | 0.0% | 0.0%-0.0% |" in md
