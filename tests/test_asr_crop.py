from asr_baseline import Utterance, WordTiming, crop_utterances


def _utt(*words):
    ws = [WordTiming(w, s, e) for w, s, e in words]
    return Utterance(text="".join(w.word for w in ws), start=ws[0].start, end=ws[-1].end, words=ws)


def test_keeps_words_by_midpoint_and_rebuilds_text():
    u = _utt((" so", 0.0, 0.4), (" it", 0.5, 0.8), (" works", 0.9, 1.5), (" fine", 1.6, 2.0))
    out = crop_utterances([u], 0.45, 1.2)   # keeps " it" (mid .65) and " works" (mid 1.2)
    assert [w.word for w in out[0].words] == [" it", " works"]
    assert out[0].text == "it works"
    assert (out[0].start, out[0].end) == (0.5, 1.5)


def test_word_straddling_boundary_goes_to_exactly_one_side():
    u = _utt((" testing", 4.8, 5.3))            # midpoint 5.05
    left = crop_utterances([u], 0.0, 5.0)
    right = crop_utterances([u], 5.0, 10.0)
    assert len(left) + len(right) == 1
    assert right and not left


def test_utterance_with_no_surviving_words_is_dropped():
    u = _utt((" hello", 0.0, 0.3))
    assert crop_utterances([u], 5.0, 6.0) == []


# ---- packing ---------------------------------------------------------------

import numpy as np

from asr_baseline import pack_clips, transcribe_packed

SR = 16000


def test_pack_clips_fills_windows_in_order_with_gaps():
    windows = pack_clips([10, 10, 10], max_window_s=28.0, gap_s=1.0)
    assert windows == [[(0, 0.0), (1, 11.0)], [(2, 0.0)]]      # 10+1+10=21; a third would exceed 28


def test_pack_clips_oversized_clip_gets_its_own_window():
    assert pack_clips([40.0, 2.0], max_window_s=28.0) == [[(0, 0.0)], [(1, 0.0)]]


def _fake_asr(window):
    """One word per contiguous non-silent region, text = its amplitude."""
    active = np.abs(window) > 0
    words, i = [], 0
    while i < len(window):
        if active[i]:
            j = i
            while j < len(window) and active[j]:
                j += 1
            words.append(WordTiming(f" c{window[i]:.1f}", i / SR, j / SR))
            i = j
        else:
            i += 1
    return [Utterance("".join(w.word for w in words), words[0].start, words[-1].end, words)] if words else []


def test_transcribe_packed_returns_each_clip_its_own_words_with_local_times():
    clips = [np.full(SR, 0.1, np.float32), np.full(2 * SR, 0.2, np.float32), np.full(SR, 0.3, np.float32)]
    out = transcribe_packed(_fake_asr, clips, SR, max_window_s=28.0, gap_s=1.0)
    assert [u[0].text for u in out] == ["c0.1", "c0.2", "c0.3"]
    for utts, clip in zip(out, clips):
        w = utts[0].words[0]
        assert abs(w.start) < 1e-3 and abs(w.end - len(clip) / SR) < 1e-3   # times relative to the clip


def test_transcribe_packed_makes_one_call_per_window():
    calls = []
    def counting(window):
        calls.append(len(window)); return _fake_asr(window)
    clips = [np.full(2 * SR, 0.1, np.float32)] * 4
    transcribe_packed(counting, clips, SR, max_window_s=28.0)
    assert len(calls) == 1
