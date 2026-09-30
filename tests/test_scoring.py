from scoring import (Turn, cp_wer, der, edit_distance, filler_recall,
                     load_reference, normalize, speaker_consistency, wer)


def test_normalize_strips_punctuation_pause_markers_and_fillers():
    assert normalize("So, um, I was [pause 1.2s] thinking...") == ["so", "i", "was", "thinking"]
    assert normalize("Um yes", drop_fillers=False) == ["um", "yes"]
    assert normalize("don't stop") == ["don't", "stop"]


def test_edit_distance_basic():
    assert edit_distance("a b c".split(), "a b c".split()) == 0
    assert edit_distance("a b c".split(), "a x c".split()) == 1
    assert edit_distance("a b c".split(), "a c".split()) == 1
    assert edit_distance([], "a b".split()) == 2


def test_wer_ignores_fillers_on_both_sides():
    assert wer("so um we could go", "so we could go") == 0.0
    assert wer("so we could go", "so uh we could go") == 0.0
    assert wer("we could go", "we could stay") == 1 / 3


def test_filler_recall_is_multiset():
    assert filler_recall("um so um yes", "um so yes") == (1, 2)
    assert filler_recall("no fillers here", "anything") == (0, 0)


def test_cp_wer_perfect_and_label_permutation_invariant():
    ref = [Turn("A", 0, 2, "hello there"), Turn("B", 2, 4, "good morning")]
    hyp = [Turn("Speaker_2", 0, 2, "hello there"), Turn("Speaker_1", 2, 4, "good morning")]
    score, mapping = cp_wer(ref, hyp)
    assert score == 0.0
    assert mapping == {"Speaker_2": "A", "Speaker_1": "B"}


def test_cp_wer_penalises_words_credited_to_wrong_speaker():
    ref = [Turn("A", 0, 2, "hello there"), Turn("B", 2, 4, "good morning")]
    merged = [Turn("S1", 0, 4, "hello there good morning")]   # one cluster for both people
    score, _ = cp_wer(ref, merged)
    assert score > 0.4


def test_cp_wer_extra_hypothesis_speaker_counts_as_insertions():
    ref = [Turn("A", 0, 2, "hello there")]
    hyp = [Turn("S1", 0, 2, "hello there"), Turn("S2", 2, 3, "extra words")]
    assert cp_wer(ref, hyp)[0] == 1.0    # 2 insertions / 2 ref words


def test_speaker_consistency_detects_split_speaker():
    ref = [Turn("A", 0, 10)]
    steady = [Turn("S1", 0, 10)]
    drifting = [Turn("S1", 0, 6), Turn("S2", 6, 10)]
    assert speaker_consistency(ref, steady)["A"] > 0.99
    assert abs(speaker_consistency(ref, drifting)["A"] - 0.6) < 0.02


def test_der_perfect_and_confused():
    ref = [Turn("A", 0, 5), Turn("B", 5, 10)]
    assert der(ref, [Turn("x", 0, 5), Turn("y", 5, 10)], collar=0.0)["der"] < 1e-6
    swapped_half = [Turn("x", 0, 5), Turn("x", 5, 10)]
    res = der(ref, swapped_half, collar=0.0)
    assert abs(res["der"] - 0.5) < 1e-6 and abs(res["confusion"] - 0.5) < 1e-6


def test_load_reference_json_and_rttm(tmp_path):
    (tmp_path / "t.json").write_text('{"turns":[{"speaker":"A","start":0,"end":2,"text":"hi"}]}')
    assert load_reference(str(tmp_path / "t.json")) == [Turn("A", 0.0, 2.0, "hi")]
    (tmp_path / "t.rttm").write_text("SPEAKER f 1 1.5 2.0 <NA> <NA> alice <NA> <NA>\n")
    assert load_reference(str(tmp_path / "t.rttm")) == [Turn("alice", 1.5, 3.5)]
