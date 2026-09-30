"""
Phase 4 — Diarization: assign a consistent Speaker_N label to every stretch
of speech, for ANY set of speakers, with no prior training on those people.

Model: pyannote/speaker-diarization-3.1 (free, but "gated" on HuggingFace --
needs a free account, a one-time license accept, and an access token; see
README.md).

Important for Phase 3: pyannote's `speaker_diarization` output keeps
OVERLAPPING turns (two speakers active at the same time get two overlapping
turns), which is exactly the signal `overlap.py` uses to decide where
speech separation is needed. Optional real-name matching lives in
enrollment.py, and is deliberately decoupled from this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from config import TARGET_SR

DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"


@dataclass
class DiarizedSegment:
    speaker_label: str   # "Speaker_1", "Speaker_2", ... assigned by clustering
    start: float
    end: float


class Diarizer:
    def __init__(self, hf_token: str, device: str = "cpu"):
        # Imported lazily so importing this module stays cheap.
        import torch
        from pyannote.audio import Pipeline

        self.pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, token=hf_token)
        self.pipeline.to(torch.device(device))

    def diarize(
        self,
        audio: np.ndarray,
        sample_rate: int = TARGET_SR,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> List[DiarizedSegment]:
        """audio: mono float32 waveform. Passing the waveform (rather than a
        file path) means pyannote never needs torchcodec/ffmpeg to decode.
        Speaker-count hints are optional but improve DER when known."""
        import torch

        waveform = torch.from_numpy(audio).float().unsqueeze(0)  # (channel, time)
        output = self.pipeline(
            {"waveform": waveform, "sample_rate": sample_rate},
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )

        segments: List[DiarizedSegment] = []
        for turn, _, speaker in output.speaker_diarization.itertracks(yield_label=True):
            # pyannote's raw labels look like "SPEAKER_00" -- normalize to
            # our "Speaker_1"-style, 1-indexed, output format.
            idx = int(speaker.split("_")[-1]) + 1
            segments.append(
                DiarizedSegment(speaker_label=f"Speaker_{idx}", start=turn.start, end=turn.end)
            )
        segments.sort(key=lambda s: (s.start, s.end))
        return segments
