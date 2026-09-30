"""
Phase 7 — Streaming (near-live) transcription.

Reuses every stage of the offline `TranscriptionPipeline`, but works on a
rolling audio buffer instead of a whole file:

    feed(chunks) -> buffer
    step():  re-diarize the rolling window
             -> map its local speaker labels onto a persistent SpeakerRegistry
                (running voice prints), so Speaker_N stays the same person
                for the whole session
             -> take only audio older than `guard_s` ("committed" audio; the
                newest seconds can still change as more context arrives, so
                they are never emitted and later retracted)
             -> overlap routing / separation / verbatim ASR, exactly as offline
             -> return the new transcript lines

Latency = guard_s + processing time + waiting for the next hop. The hop is
adaptive: `ready()` fires when `hop_s` of new audio exists, but a step is
never started while another is running, so on slow hardware the system
degrades to bigger hops instead of falling further behind.

CLI (replays a file as if it were live and reports per-line latency):
    python src/streaming.py meeting.wav --speed 1
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

from audio_utils import load_audio, slice_audio
from config import TARGET_SR
from diarization import DiarizedSegment
from embeddings import assign_streams, cosine_similarity
from overlap import Segment, build_segments
from pipeline import AsrJob, TranscriptLine, TranscriptionPipeline

DEFAULT_WINDOW_S = 16.0     # audio kept for diarization context (diarization cost grows fast with this)
DEFAULT_HOP_S = 4.0         # minimum new audio between steps
DEFAULT_GUARD_S = 3.0       # newest audio held back until diarization settles
# Cosine similarity needed to call a window-local speaker "the same person"
# as a registered one. Different people are usually well below this, and
# same-speaker centroids from a few seconds of audio well above; tune on data.
DEFAULT_SIM_THRESHOLD = 0.4
# Seconds a window-local speaker must overlap a previously labelled speaker
# (in the audio the two windows share) to inherit that label.
MIN_TEMPORAL_OVERLAP_S = 0.4
_MAX_WEIGHT_S = 120.0       # cap so long sessions still adapt to a speaker's voice drift


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-8)


class SpeakerRegistry:
    """Session-long speaker identities. Each window's diarization produces
    arbitrary local labels; `assign` maps them to persistent global labels by
    voice similarity, creating a new Speaker_N when nobody matches."""

    def __init__(self, threshold: float = DEFAULT_SIM_THRESHOLD):
        self.threshold = threshold
        self._sums: Dict[str, np.ndarray] = {}     # weighted sum of unit embeddings
        self._weights: Dict[str, float] = {}

    @property
    def centroids(self) -> Dict[str, np.ndarray]:
        return {k: _unit(v) for k, v in self._sums.items()}

    def assign(
        self,
        local: Dict[str, np.ndarray],
        weights: Optional[Dict[str, float]] = None,
        fixed: Optional[Dict[str, str]] = None,
        weak: frozenset = frozenset(),
    ) -> Dict[str, str]:
        """local: window label -> embedding (iteration order = order of first
        appearance, which decides numbering of brand-new speakers).

        fixed: labels already resolved by other evidence (temporal continuity
               with the previous window); they skip voice matching.
        weak:  labels whose print came from overlapped speech only. They may
               be matched or create a provisional speaker, but never update
               an existing speaker's voice print (a mixture would corrupt it).

        Returns window label -> global label. One-to-one: two people in the
        same window can never be merged into one identity."""
        existing = set(self._sums)
        mapping: Dict[str, str] = dict(fixed or {})
        free = [l for l in local if l not in mapping]
        known = [k for k in self._sums if k not in set(mapping.values())]

        if known and free:
            cents = self.centroids
            sim = np.array(
                [[cosine_similarity(local[l], cents[k]) for k in known] for l in free]
            )
            for i, j in assign_streams(sim):
                if sim[i, j] >= self.threshold:
                    mapping[free[i]] = known[j]

        for label in local:
            if label not in mapping:
                new = f"Speaker_{len(self._sums) + 1}"
                self._sums[new] = np.zeros_like(local[label])
                self._weights[new] = 0.0
                mapping[label] = new

        for label, glob in mapping.items():
            if label not in local or (label in weak and glob in existing):
                continue
            w = (weights or {}).get(label, 1.0)
            self._sums[glob] = self._sums[glob] + _unit(local[label]) * w
            self._weights[glob] += w
            if self._weights[glob] > _MAX_WEIGHT_S:
                scale = _MAX_WEIGHT_S / self._weights[glob]
                self._sums[glob] = self._sums[glob] * scale
                self._weights[glob] = _MAX_WEIGHT_S
        return mapping


class StreamingTranscriber:
    def __init__(
        self,
        pipeline: TranscriptionPipeline,
        window_s: float = DEFAULT_WINDOW_S,
        hop_s: float = DEFAULT_HOP_S,
        guard_s: float = DEFAULT_GUARD_S,
        sim_threshold: float = DEFAULT_SIM_THRESHOLD,
        num_speakers: Optional[int] = None,
        log: Optional[Callable[[str], None]] = None,
    ):
        self.p = pipeline
        self.window_s, self.hop_s, self.guard_s = window_s, hop_s, guard_s
        self.num_speakers = num_speakers
        self.log = log or pipeline.log
        self.registry = SpeakerRegistry(sim_threshold)

        self._lock = threading.Lock()
        self._pending: List[np.ndarray] = []
        self._pending_samples = 0

        self._buffer = np.zeros(0, dtype=np.float32)
        self._buffer_start = 0.0          # absolute time of _buffer[0]
        self._committed_until = 0.0       # everything before this has been emitted
        self._last_committed: Optional[Segment] = None
        self._total_at_last_step = 0.0
        self._prev_turns: List[DiarizedSegment] = []   # last window's turns: absolute time, global labels

    # -- input ----------------------------------------------------------------

    def feed(self, samples: np.ndarray) -> None:
        """Append mono float32 16 kHz audio. Thread-safe; call from the audio thread."""
        with self._lock:
            self._pending.append(np.asarray(samples, dtype=np.float32))
            self._pending_samples += len(samples)

    @property
    def total_time(self) -> float:
        """Seconds of audio received so far (buffered + pending)."""
        with self._lock:
            return self._buffer_start + (len(self._buffer) + self._pending_samples) / TARGET_SR

    def ready(self) -> bool:
        return self.total_time - self._total_at_last_step >= self.hop_s

    def _ingest(self) -> None:
        with self._lock:
            if self._pending:
                self._buffer = np.concatenate([self._buffer, *self._pending])
                self._pending, self._pending_samples = [], 0

    # -- processing -------------------------------------------------------------

    def flush(self) -> List[TranscriptLine]:
        """End of stream: commit everything, including the guarded tail."""
        return self.step(final=True)

    def step(self, final: bool = False) -> List[TranscriptLine]:
        """Process buffered audio; returns newly committed lines (absolute times)."""
        self._ingest()
        buf_end = self._buffer_start + len(self._buffer) / TARGET_SR
        self._total_at_last_step = buf_end
        horizon = buf_end if final else buf_end - self.guard_s
        if horizon - self._committed_until < 0.5:
            return []

        t0 = time.time()
        audio = self._buffer
        # The speaker-count hint is for the whole session; any single window may
        # contain fewer people, and forcing an exact count on it makes pyannote
        # split one voice into phantom speakers. So it is only an upper bound.
        turns = self.p.diarizer.diarize(audio, TARGET_SR, max_speakers=self.num_speakers)
        t_diar = time.time() - t0
        if not turns:
            self._advance(horizon, buf_end)
            return []

        t1 = time.time()
        mapping = self._register_speakers(audio, turns, horizon)
        t_embed = time.time() - t1

        # Global labels, absolute times, clipped to the commit window.
        window = []
        for t in turns:
            lo = max(t.start + self._buffer_start, self._committed_until)
            hi = min(t.end + self._buffer_start, horizon)
            if hi > lo:
                window.append(DiarizedSegment(mapping[t.speaker_label], lo, hi))
        segments = build_segments(window)

        t2 = time.time()
        lines = self._transcribe(audio, segments, final, horizon)
        t_asr = time.time() - t2
        self._advance(horizon, buf_end)
        self.log(
            f"step: committed to {horizon:.1f}s, {len(lines)} lines, "
            f"{time.time() - t0:.1f}s compute for {buf_end - self._buffer_start:.0f}s window "
            f"(diarize {t_diar:.1f}s, embed {t_embed:.1f}s, separate+ASR {t_asr:.1f}s)"
        )
        self.log(f"  speakers this window: {mapping}")
        return lines

    def _register_speakers(
        self, audio: np.ndarray, turns: List[DiarizedSegment], horizon: float
    ) -> Dict[str, str]:
        """Window-local diarizer label -> persistent global label.

        Two kinds of evidence, in order of trust:
        1. Temporal continuity: consecutive windows share most of their audio,
           so a local speaker who occupies the same moments as a previously
           labelled speaker IS that speaker. Robust even when voice prints are
           poor (e.g. someone heard only inside overlaps).
        2. Voice similarity, for speakers with no temporal anchor (newly
           arrived, or returning after silence).
        """
        bs = self._buffer_start
        # A speaker who only exists beyond the commit horizon is inside the
        # guard region, where diarization is still unstable. Registering an
        # identity from it risks a phantom duplicate, so wait until they show
        # up in committed audio.
        settled = {
            t.speaker_label
            for t in turns
            if t.end + bs > self._committed_until and t.start + bs < horizon
        }
        by_speaker: Dict[str, List[DiarizedSegment]] = {}
        for t in turns:
            if t.speaker_label in settled:
                by_speaker.setdefault(t.speaker_label, []).append(t)
        order = sorted(by_speaker, key=lambda l: min(t.start for t in by_speaker[l]))

        clean = build_segments(turns)
        local: Dict[str, np.ndarray] = {}
        weights: Dict[str, float] = {}
        weak = set()
        for label in order:
            solo = [s for s in clean if not s.is_overlapping and s.speakers == (label,)]
            spans = sorted(solo or by_speaker[label], key=lambda s: s.end - s.start, reverse=True)
            chunks, total = [], 0.0
            for s in spans:
                chunks.append(slice_audio(audio, TARGET_SR, s.start, s.end))
                total += s.end - s.start
                if total >= 20.0:
                    break
            local[label] = self.p.embedder.embed(np.concatenate(chunks), TARGET_SR)
            weights[label] = total if solo else total * 0.3
            if not solo:
                weak.add(label)

        fixed = self._temporal_mapping(by_speaker, bs)
        mapping = self.registry.assign(local, weights, fixed=fixed, weak=frozenset(weak))
        self._prev_turns = [
            DiarizedSegment(mapping[t.speaker_label], t.start + bs, t.end + bs)
            for t in turns
            if t.speaker_label in mapping
        ]
        return mapping

    def _temporal_mapping(
        self, by_speaker: Dict[str, List[DiarizedSegment]], buffer_start: float
    ) -> Dict[str, str]:
        """Match local labels to previous-window global labels by how many
        seconds they are active at the same moments."""
        if not self._prev_turns:
            return {}
        globals_ = sorted({t.speaker_label for t in self._prev_turns})
        labels = list(by_speaker)
        overlap = np.zeros((len(labels), len(globals_)))
        for i, label in enumerate(labels):
            for t in by_speaker[label]:
                for p in self._prev_turns:
                    shared = min(t.end + buffer_start, p.end) - max(t.start + buffer_start, p.start)
                    if shared > 0:
                        overlap[i, globals_.index(p.speaker_label)] += shared
        return {
            labels[i]: globals_[j]
            for i, j in assign_streams(overlap)
            if overlap[i, j] >= MIN_TEMPORAL_OVERLAP_S
        }

    def _transcribe(
        self,
        audio: np.ndarray,
        segments: List[Segment],
        final: bool,
        horizon: float,
    ) -> List[TranscriptLine]:
        bs = self._buffer_start
        names = self.p.gallery.resolve(self.registry.centroids) if self.p.gallery else {}
        display = lambda label: names.get(label, label)  # noqa: E731

        def rel(s: Segment) -> Segment:
            return Segment(s.start - bs, s.end - bs, s.speakers)

        jobs: List[AsrJob] = []
        for i, seg in enumerate(segments):
            prev = segments[i - 1] if i > 0 else self._last_committed
            if i + 1 < len(segments):
                nxt = segments[i + 1]
            elif final:
                nxt = None
            else:
                # More audio follows: assume every speaker may continue, so
                # overlap streams crop this edge (the next step re-covers it).
                nxt = Segment(horizon, horizon + 1.0, seg.speakers)

            seg_rel = rel(seg)
            if seg.is_overlapping and self.p.separate:
                jobs += self.p._overlap_jobs(
                    audio, seg_rel, rel(prev) if prev else None, rel(nxt) if nxt else None,
                    self.registry.centroids, display,
                )
            else:
                jobs += self.p._single_jobs(audio, seg_rel, display)

        # One batched ASR pass for the whole step (see asr_baseline.transcribe_packed).
        lines = self.p._run_jobs(jobs)
        for line in lines:
            line.start += bs
            line.end += bs

        if segments:
            self._last_committed = segments[-1]
        lines.sort(key=lambda l: l.start)
        return lines

    def _advance(self, horizon: float, buf_end: float) -> None:
        self._committed_until = max(self._committed_until, horizon)
        keep_from = max(self._buffer_start, buf_end - self.window_s)
        if keep_from > self._committed_until:
            self.log("warning: processing is falling behind; uncommitted audio was dropped")
        cut = int((keep_from - self._buffer_start) * TARGET_SR)
        if cut > 0:
            self._buffer = self._buffer[cut:]
            self._buffer_start += cut / TARGET_SR


def replay_file(
    transcriber: StreamingTranscriber,
    path: str,
    speed: float = 1.0,
    chunk_s: float = 0.5,
    on_lines: Optional[Callable[[List[TranscriptLine], float], None]] = None,
) -> List[TranscriptLine]:
    """Feed a recording through `transcriber` as if it were live. speed=1 is
    real time; speed=0 feeds as fast as possible (latency figures are then
    meaningless). Calls on_lines(lines, latency_s) for each committed batch."""
    audio = load_audio(path)
    chunk = int(chunk_s * TARGET_SR)
    all_lines: List[TranscriptLine] = []
    start = time.time()

    def emit(lines: List[TranscriptLine]) -> None:
        if not lines:
            return
        all_lines.extend(lines)
        if on_lines:
            audio_time = (time.time() - start) * speed if speed else float("nan")
            on_lines(lines, audio_time - max(l.end for l in lines))

    for i in range(0, len(audio), chunk):
        transcriber.feed(audio[i : i + chunk])
        if speed:
            # Real-time pacing: wait until this chunk "would have" arrived.
            due = start + (i + chunk) / TARGET_SR / speed
            time.sleep(max(0.0, due - time.time()))
        if transcriber.ready():
            emit(transcriber.step())
    emit(transcriber.flush())
    return sorted(all_lines, key=lambda l: l.start)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Replay a recording as a live stream.")
    ap.add_argument("audio")
    ap.add_argument("--speed", type=float, default=1.0, help="1 = real time, 0 = as fast as possible")
    ap.add_argument("--model", default="small", help="Whisper size; tiny/base are much faster")
    ap.add_argument("--beam-size", type=int, default=1, help="1 = greedy (fast, default for streaming)")
    ap.add_argument("--asr-backend", default="auto", choices=["auto", "mlx", "ctranslate2"])
    ap.add_argument("--num-speakers", type=int, default=None, help="upper bound on speakers in the session")
    ap.add_argument("--no-separation", action="store_true")
    ap.add_argument("--hop", type=float, default=DEFAULT_HOP_S)
    ap.add_argument("--guard", type=float, default=DEFAULT_GUARD_S)
    ap.add_argument("--window", type=float, default=DEFAULT_WINDOW_S)
    ap.add_argument("--sim-threshold", type=float, default=DEFAULT_SIM_THRESHOLD)
    args = ap.parse_args()

    pipe = TranscriptionPipeline(
        asr_model_size=args.model, separate=not args.no_separation, beam_size=args.beam_size,
        asr_backend=args.asr_backend,
    )
    st = StreamingTranscriber(
        pipe, window_s=args.window, hop_s=args.hop, guard_s=args.guard,
        sim_threshold=args.sim_threshold, num_speakers=args.num_speakers,
    )

    latencies: List[float] = []

    def show(lines: List[TranscriptLine], latency: float) -> None:
        for l in lines:
            print(l.render(), flush=True)
        if latency == latency:   # not NaN
            latencies.append(latency)
            print(f"    (emitted {latency:.1f}s after the audio was spoken)", flush=True)

    replay_file(st, args.audio, speed=args.speed, on_lines=show)
    if latencies:
        print(f"\nlatency: mean {np.mean(latencies):.1f}s, max {np.max(latencies):.1f}s")
