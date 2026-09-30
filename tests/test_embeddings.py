import numpy as np

from embeddings import assign_streams, match_streams_to_speakers


def test_assign_streams_picks_best_one_to_one():
    sim = np.array([[0.1, 0.9], [0.8, 0.2]])
    assert sorted(assign_streams(sim)) == [(0, 1), (1, 0)]


def test_assign_streams_resolves_conflict_globally():
    # both streams prefer speaker 0, but the total is best if stream 1 takes speaker 1
    sim = np.array([[0.9, 0.8], [0.85, 0.1]])
    assert sorted(assign_streams(sim)) == [(0, 1), (1, 0)]


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def test_match_uses_voice_not_stream_order():
    centroids = {"Speaker_1": _unit([1, 0]), "Speaker_2": _unit([0, 1])}
    # stream 0 sounds like Speaker_2, stream 1 like Speaker_1
    streams = [_unit([0.1, 1]), _unit([1, 0.1])]
    out = match_streams_to_speakers(streams, ["Speaker_1", "Speaker_2"], centroids)
    assert out == {0: "Speaker_2", 1: "Speaker_1"}


def test_match_single_surviving_stream_still_matched_by_voice():
    centroids = {"Speaker_1": _unit([1, 0]), "Speaker_2": _unit([0, 1])}
    out = match_streams_to_speakers([_unit([0, 1])], ["Speaker_1", "Speaker_2"], centroids)
    assert out == {0: "Speaker_2"}


def test_match_without_centroids_falls_back_to_position():
    out = match_streams_to_speakers(
        [_unit([1, 0]), _unit([0, 1])], ["Speaker_1", "Speaker_2"], {}
    )
    assert out == {0: "Speaker_1", 1: "Speaker_2"}


def test_match_partial_centroids_pairs_leftover_positionally():
    centroids = {"Speaker_1": _unit([1, 0])}   # Speaker_2 never spoke alone
    streams = [_unit([0, 1]), _unit([1, 0.05])]
    out = match_streams_to_speakers(streams, ["Speaker_1", "Speaker_2"], centroids)
    assert out == {1: "Speaker_1", 0: "Speaker_2"}
