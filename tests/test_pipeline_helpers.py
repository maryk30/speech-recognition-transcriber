from overlap import Segment
from pipeline import CONTEXT_PAD_S, TranscriptLine, _continues


def test_render_format():
    line = TranscriptLine("SAMPATH", 72.4, "so I was thinking, um, we could")
    assert line.render() == '[01:12.4] SAMPATH: "so I was thinking, um, we could"'


def test_continues_only_when_neighbour_shares_speaker_and_is_adjacent():
    prev = Segment(0.0, 5.0, ("Speaker_2",))
    assert _continues(prev, "Speaker_2", 5.0)
    assert not _continues(prev, "Speaker_1", 5.0)            # different voice: nothing to dedupe
    assert not _continues(None, "Speaker_2", 5.0)
    assert not _continues(prev, "Speaker_2", 5.0 + CONTEXT_PAD_S + 1)   # too far to overlap the pad


def test_render_never_shows_sixty_seconds():
    assert TranscriptLine("S", 59.96, "x").render().startswith("[01:00.0]")
    assert TranscriptLine("S", 59.94, "x").render().startswith("[00:59.9]")
    assert TranscriptLine("S", 0.0, "x").render().startswith("[00:00.0]")


def test_filler_only_utterances_are_dropped_on_separated_audio_only():
    from asr_baseline import Utterance
    from pipeline import _usable

    lone_mm = Utterance("mm -hmm", 0.0, 0.5, [])
    assert not _usable(lone_mm, strict=True)                       # separated stream: hallucinated backchannel
    assert _usable(lone_mm, strict=False)                          # clean single-speaker audio: a real "mm-hmm"
    assert _usable(Utterance("um so yes", 0.0, 1.0, []), strict=True)   # fillers WITH words are kept


def test_line_span_never_ends_before_it_starts():
    from asr_baseline import Utterance
    from pipeline import _line_span

    # words Whisper placed entirely before the floor (the 198.04 s case seen on ES2004c)
    u = Utterance(text="i'm not a good person", start=0.1, end=0.3)
    start, end = _line_span(197.64, u, floor=198.04)
    assert start == 198.04 and end >= start
    # normal case unchanged
    assert _line_span(10.0, Utterance(text="hi", start=1.0, end=2.0), floor=-float("inf")) == (11.0, 12.0)
