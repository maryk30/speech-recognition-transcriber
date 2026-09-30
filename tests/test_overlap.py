from diarization import DiarizedSegment as D
from overlap import build_segments


def spans(segs):
    return [(round(s.start, 2), round(s.end, 2), s.speakers) for s in segs]


def test_single_speaker_is_one_segment():
    segs = build_segments([D("Speaker_1", 1.0, 4.0)])
    assert spans(segs) == [(1.0, 4.0, ("Speaker_1",))]
    assert not segs[0].is_overlapping


def test_partial_overlap_splits_into_three():
    segs = build_segments([D("Speaker_1", 0, 5), D("Speaker_2", 3, 8)])
    assert spans(segs) == [
        (0, 3, ("Speaker_1",)),
        (3, 5, ("Speaker_1", "Speaker_2")),
        (5, 8, ("Speaker_2",)),
    ]
    assert [s.is_overlapping for s in segs] == [False, True, False]
    assert segs[1].as_tuple() == (3, 5, True)


def test_silence_is_dropped():
    segs = build_segments([D("Speaker_1", 0, 1), D("Speaker_2", 5, 6)])
    assert spans(segs) == [(0, 1, ("Speaker_1",)), (5, 6, ("Speaker_2",))]


def test_short_overlap_is_folded_into_previous_speaker():
    # 0.2s overlap < 0.4s minimum -> stays with Speaker_1, no separation
    segs = build_segments([D("Speaker_1", 0, 5), D("Speaker_2", 4.8, 8)])
    assert spans(segs) == [(0, 5, ("Speaker_1",)), (5, 8, ("Speaker_2",))]
    assert not any(s.is_overlapping for s in segs)


def test_same_speaker_small_gap_is_merged():
    segs = build_segments([D("Speaker_1", 0, 1), D("Speaker_1", 1.2, 2)])
    assert spans(segs) == [(0, 2, ("Speaker_1",))]


def test_same_speaker_large_gap_is_kept_apart():
    segs = build_segments([D("Speaker_1", 0, 1), D("Speaker_1", 3, 4)])
    assert len(segs) == 2


def test_tiny_blip_dropped():
    assert build_segments([D("Speaker_1", 0, 0.05)]) == []


def test_three_way_overlap_reports_all_speakers():
    segs = build_segments([D("Speaker_1", 0, 4), D("Speaker_2", 1, 4), D("Speaker_3", 1, 4)])
    assert segs[1].speakers == ("Speaker_1", "Speaker_2", "Speaker_3")


def test_output_is_chronological_and_non_overlapping_in_time():
    turns = [D("Speaker_2", 2, 6), D("Speaker_1", 0, 3), D("Speaker_3", 5, 9)]
    segs = build_segments(turns)
    for a, b in zip(segs, segs[1:]):
        assert a.end <= b.start + 1e-9
