"""
Speaker embeddings (SpeechBrain ECAPA-TDNN) and the two things we use them
for in the pipeline:

1. Per-speaker centroids from a diarized recording -- used to work out which
   separated stream belongs to which diarized speaker (Phase 2/3 wiring),
   and to match against enrolled voices (Phase 5).
2. `assign_streams`: Hungarian matching of separated streams to speakers.

Why this exists: a separator such as SepFormer returns its outputs in an
ARBITRARY order -- output 0 may be Speaker_1 in one overlap window and
Speaker_2 in the next. Matching each stream's voice against the speaker's
clean, non-overlapping speech resolves that.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from audio_utils import resample, slice_audio
from config import MODELS_DIR, TARGET_SR
from overlap import Segment

MIN_EMBED_SAMPLES = int(0.4 * TARGET_SR)


class SpeakerEmbedder:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self._model = None  # lazy-loaded

    def _get_model(self):
        if self._model is None:
            from speechbrain.inference.speaker import EncoderClassifier

            self._model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=str(MODELS_DIR / "ecapa"),
                run_opts={"device": self.device},
            )
        return self._model

    def embed(self, audio: np.ndarray, sample_rate: int = TARGET_SR) -> np.ndarray:
        """Fixed-length (192-d) voice embedding for a mono waveform."""
        import torch

        audio = resample(audio, sample_rate, TARGET_SR)
        if len(audio) < MIN_EMBED_SAMPLES:
            audio = np.pad(audio, (0, MIN_EMBED_SAMPLES - len(audio)))
        wav = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self._get_model().encode_batch(wav)
        return emb.squeeze().cpu().numpy()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def speaker_centroids(
    audio: np.ndarray,
    sample_rate: int,
    segments: Sequence[Segment],
    embedder: SpeakerEmbedder,
    max_seconds: float = 30.0,
    min_segment_s: float = 0.5,
) -> Dict[str, np.ndarray]:
    """One embedding per diarized speaker, computed from their CLEAN
    (non-overlapping) speech only -- longest segments first, up to
    `max_seconds`. Speakers with no clean speech get no centroid."""
    by_speaker: Dict[str, List[Segment]] = {}
    for seg in segments:
        if not seg.is_overlapping and seg.duration >= min_segment_s:
            by_speaker.setdefault(seg.speakers[0], []).append(seg)

    centroids: Dict[str, np.ndarray] = {}
    for label, segs in by_speaker.items():
        segs = sorted(segs, key=lambda s: s.duration, reverse=True)
        chunks, total = [], 0.0
        for s in segs:
            chunks.append(slice_audio(audio, sample_rate, s.start, s.end))
            total += s.duration
            if total >= max_seconds:
                break
        centroids[label] = embedder.embed(np.concatenate(chunks), sample_rate)
    return centroids


def assign_streams(similarity: np.ndarray) -> List[tuple[int, int]]:
    """similarity[i, j] = how well stream i matches speaker j. Returns the
    (stream_idx, speaker_idx) pairs of the one-to-one assignment that
    maximises total similarity (streams/speakers beyond min(n, m) are left
    unassigned)."""
    rows, cols = linear_sum_assignment(-similarity)
    return [(int(r), int(c)) for r, c in zip(rows, cols)]


def match_streams_to_speakers(
    stream_embeddings: Sequence[np.ndarray],
    speakers: Sequence[str],
    centroids: Dict[str, np.ndarray],
) -> Dict[int, str]:
    """Map separated-stream index -> speaker label for one overlap window.

    Streams are matched by voice against every window speaker that has a
    clean-speech centroid. Anything left over (a speaker with no centroid,
    e.g. someone who never spoke alone) is paired positionally with the
    remaining streams -- a last resort, since separator output order is
    arbitrary, but better than dropping the speech.
    """
    known = [s for s in speakers if s in centroids]
    result: Dict[int, str] = {}

    if known and len(stream_embeddings):
        sim = np.array(
            [[cosine_similarity(e, centroids[s]) for s in known] for e in stream_embeddings]
        )
        result = {i: known[j] for i, j in assign_streams(sim)}

    leftover_speakers = [s for s in speakers if s not in result.values()]
    leftover_streams = [i for i in range(len(stream_embeddings)) if i not in result]
    result.update(zip(leftover_streams, leftover_speakers))
    return result
