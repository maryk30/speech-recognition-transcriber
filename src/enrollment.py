"""
Phase 5 — Optional enrollment: let people register a short voice sample so
their generic Speaker_N label is replaced by their real name. Anyone who
isn't enrolled (or doesn't match confidently) simply stays Speaker_N.

Kept apart from diarization on purpose: diarization must never depend on
naming.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from audio_utils import load_audio
from config import TARGET_SR
from embeddings import SpeakerEmbedder, cosine_similarity

# ECAPA cosine similarity between DIFFERENT people is typically well below
# 0.3, and same-speaker scores on real meeting audio often land in the
# 0.5-0.8 range, so 0.75 (the original plan value) risks missing real
# matches. 0.5 is a starting point -- tune it on your own enrollments.
DEFAULT_SIMILARITY_THRESHOLD = 0.5


class EnrollmentGallery:
    def __init__(
        self,
        embedder: Optional[SpeakerEmbedder] = None,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ):
        self.embedder = embedder or SpeakerEmbedder()
        self.similarity_threshold = similarity_threshold
        self._speakers: Dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self._speakers)

    def enroll(self, name: str, audio_path: str) -> None:
        """Register a person from a few seconds of their own clean speech."""
        self._speakers[name] = self.embedder.embed(load_audio(audio_path), TARGET_SR)

    def resolve(self, centroids: Dict[str, np.ndarray]) -> Dict[str, str]:
        """Map diarized label -> enrolled name, one-to-one, keeping only
        matches at or above the threshold (best scores claimed first, so two
        clusters can never both be given the same name)."""
        candidates = sorted(
            (
                (cosine_similarity(emb, enrolled), label, name)
                for label, emb in centroids.items()
                for name, enrolled in self._speakers.items()
            ),
            reverse=True,
        )
        label_to_name: Dict[str, str] = {}
        used_names = set()
        for score, label, name in candidates:
            if score < self.similarity_threshold:
                break
            if label in label_to_name or name in used_names:
                continue
            label_to_name[label] = name
            used_names.add(name)
        return label_to_name
