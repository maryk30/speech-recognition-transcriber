"""Streaming logic tests with stub models: the audio itself encodes who is
speaking and when (sample = speaker + 0.001*t), so the stubs can 'diarize',
'embed' and 'transcribe' deterministically and the tests can check the
commit / label-stability / no-duplicate logic that real models can't
make reproducible."""
import numpy as np

from asr_baseline import Utterance, WordTiming
from diarization import DiarizedSegment
from pipeline import TranscriptionPipeline
from streaming import SpeakerRegistry, StreamingTranscriber

SR = 16000


def _unit(*v):
    v = np.array(v, dtype=float)
    return v / np.linalg.norm(v)


# ---- SpeakerRegistry --------------------------------------------------------

def test_registry_creates_then_reuses_speakers():
    reg = SpeakerRegistry(threshold=0.5)
    m1 = reg.assign({"a": _unit(1, 0), "b": _unit(0, 1)})
    assert set(m1.values()) == {"Speaker_1", "Speaker_2"}
    # next window: labels scrambled, same voices
    m2 = reg.assign({"x": _unit(0.05, 1), "y": _unit(1, 0.05)})
    assert m2["x"] == m1["b"] and m2["y"] == m1["a"]


def test_registry_new_voice_gets_new_label():
    reg = SpeakerRegistry(threshold=0.5)
    reg.assign({"a": _unit(1, 0, 0)})
    assert reg.assign({"z": _unit(0, 0, 1)}) == {"z": "Speaker_2"}


def test_registry_never_merges_two_speakers_in_one_window():
    reg = SpeakerRegistry(threshold=0.1)
    reg.assign({"a": _unit(1, 0)})
    m = reg.assign({"p": _unit(1, 0.1), "q": _unit(1, 0.2)})   # both resemble Speaker_1
    assert len(set(m.values())) == 2


def test_registry_numbers_new_speakers_in_order_of_appearance():
    reg = SpeakerRegistry()
    m = reg.assign({"late": _unit(1, 0), "later": _unit(0, 1)})
    assert m == {"late": "Speaker_1", "later": "Speaker_2"}


# ---- end-to-end streaming with stubs ----------------------------------------

TRUTH = [("A", 1, 0.0, 8.0), ("B", 2, 9.0, 16.0), ("A", 1, 17.0, 22.0)]  # (name, amp, start, end)


def make_audio(total_s=24.0):
    a = np.zeros(int(total_s * SR), dtype=np.float32)
    t = np.arange(len(a)) / SR
    for _, amp, s, e in TRUTH:
        m = (t >= s) & (t < e)
        a[m] = amp + 0.001 * t[m]
    return a


class StubDiarizer:
    def __init__(self):
        self.calls = 0
        self.tr = None   # set to the StreamingTranscriber under test

    def diarize(self, audio, sr, num_speakers=None, **kw):
        self.calls += 1
        bs, dur = self.tr._buffer_start, len(audio) / SR
        flip = self.calls % 2 == 0   # scramble local labels between windows
        out = []
        for _, amp, s, e in TRUTH:
            lo, hi = max(s - bs, 0.0), min(e - bs, dur)
            if hi > lo:
                label = f"SPEAKER_0{amp - 1 if not flip else 2 - amp}"
                out.append(DiarizedSegment(f"Speaker_{int(label[-1]) + 1}", lo, hi))
        return out


class StubEmbedder:
    def embed(self, audio, sr=SR):
        amp = int(round(float(audio.max())))
        return np.eye(4)[amp]


class StubTranscriber:
    """One word per 0.5s tick on an absolute grid, only where speech exists."""
    def transcribe(self, clip, language="en"):
        nz = np.flatnonzero(clip)
        if len(nz) == 0:
            return []
        t0 = (clip[nz[0]] - round(float(clip[nz[0]]))) / 0.001 - nz[0] / SR
        words = []
        g = np.ceil(t0 / 0.5) * 0.5
        while g + 0.4 <= t0 + len(clip) / SR:
            idx = int((g - t0 + 0.2) * SR)
            v = clip[idx] if 0 <= idx < len(clip) else 0
            if v != 0:
                words.append(WordTiming(f" {int(round(float(v)))}@{g:.1f}", g - t0, g - t0 + 0.4))
            g += 0.5
        if not words:
            return []
        return [Utterance("".join(w.word for w in words), words[0].start, words[-1].end, words)]


def build():
    p = object.__new__(TranscriptionPipeline)
    p.diarizer, p.transcriber, p._embedder = StubDiarizer(), StubTranscriber(), StubEmbedder()
    p.gallery, p.separate, p.log, p.last_turns, p._separator = None, False, (lambda m: None), [], None
    st = StreamingTranscriber(p, window_s=10, hop_s=2, guard_s=2)
    p.diarizer.tr = st
    return st


def run_stream(st, chunk_s=0.5):
    audio, lines = make_audio(), []
    for i in range(0, len(audio), int(chunk_s * SR)):
        st.feed(audio[i : i + int(chunk_s * SR)])
        if st.ready():
            lines += st.step()
    lines += st.flush()
    return lines


def test_every_word_emitted_exactly_once_in_order():
    lines = run_stream(build())
    words = [w for l in lines for w in l.text.split()]
    expected = []
    for _, amp, s, e in TRUTH:
        g = np.ceil(s / 0.5) * 0.5
        while g + 0.4 <= e:
            expected.append(f"{amp}@{g:.1f}")
            g += 0.5
    assert sorted(words) == sorted(expected)                 # nothing lost, nothing duplicated
    starts = [l.start for l in sorted(lines, key=lambda l: l.start)]
    assert starts == sorted(starts)


def test_labels_stay_consistent_despite_scrambled_window_labels():
    lines = run_stream(build())
    by_amp = {}
    for l in lines:
        amp = l.text.split()[0].split("@")[0]
        by_amp.setdefault(amp, set()).add(l.speaker)
    assert by_amp == {"1": {"Speaker_1"}, "2": {"Speaker_2"}}


def test_guard_delays_commit_and_buffer_is_trimmed():
    st = build()
    audio = make_audio()
    st.feed(audio[: 6 * SR])
    st.step()
    assert st._committed_until <= 6.0 - st.guard_s + 1e-6     # newest `guard_s` never committed
    st.feed(audio[6 * SR : 22 * SR])
    st.step()
    assert len(st._buffer) / SR <= st.window_s + 1e-6           # rolling window enforced


def test_registry_fixed_mapping_bypasses_voice_matching():
    reg = SpeakerRegistry(threshold=0.9)
    reg.assign({"a": _unit(1, 0)})
    # voice says "new person" (orthogonal), but temporal continuity says Speaker_1
    m = reg.assign({"z": _unit(0, 1)}, fixed={"z": "Speaker_1"})
    assert m == {"z": "Speaker_1"}


def test_registry_weak_print_never_corrupts_existing_speaker():
    reg = SpeakerRegistry(threshold=0.5)
    reg.assign({"a": _unit(1, 0)}, weights={"a": 10.0})
    before = reg.centroids["Speaker_1"].copy()
    reg.assign({"z": _unit(0, 1)}, weights={"z": 50.0}, fixed={"z": "Speaker_1"}, weak=frozenset({"z"}))
    assert np.allclose(reg.centroids["Speaker_1"], before)


def test_registry_weak_speaker_can_still_create_provisional_identity():
    reg = SpeakerRegistry()
    assert reg.assign({"z": _unit(0, 1)}, weak=frozenset({"z"})) == {"z": "Speaker_1"}


def test_speaker_seen_only_in_guard_region_is_not_registered_yet():
    st = build()
    audio = make_audio()
    st.feed(audio[: int(10.5 * SR)])     # B starts at 9.0; horizon = 10.5 - 2.0 = 8.5
    st.step()
    assert list(st.registry.centroids) == ["Speaker_1"]          # B not registered from the unstable tail
    st.feed(audio[int(10.5 * SR) : int(14 * SR)])
    st.step()                                                    # horizon 12.0: B now settled
    assert set(st.registry.centroids) == {"Speaker_1", "Speaker_2"}
