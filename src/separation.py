"""
Phase 2 — Speech separation for overlapping speech.

When overlap detection (overlap.py) flags a segment as containing 2+
simultaneous speakers, this module splits the mixed waveform into isolated
per-speaker estimates BEFORE the segment goes to ASR. Non-overlapping
segments skip this step entirely (it's expensive, and running it on clean
single-speaker audio can hurt quality).

Model: SepFormer via SpeechBrain. Free, open weights, speaker-independent
by construction (trained on synthetic mixtures of arbitrary speaker pairs),
so it generalises to any speaker set. It separates exactly TWO sources --
consistent with the project scope (brief 2-speaker overlaps).

Available checkpoints (all 8 kHz, resampling is handled here):
    speechbrain/sepformer-libri2mix  clean LibriSpeech-style mixtures (default)
    speechbrain/sepformer-wsj02mix   clean WSJ0-2mix
    speechbrain/sepformer-whamr      noisy + reverberant mixtures

Default chosen by measurement (scripts/eval_separation.py) on a 2-voice TTS
conversation: libri2mix 19.0 dB SI-SDRi vs wsj02mix 7.6 vs whamr 1.5. That
test is clean audio, so re-run the eval on real room recordings before
trusting it there -- whamr is the one to try if the room is noisy/echoey.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from audio_utils import resample, rms
from config import MODELS_DIR, TARGET_SR

DEFAULT_MODEL = "speechbrain/sepformer-libri2mix"


@dataclass
class SeparatedStream:
    """One isolated speaker estimate produced from a mixed segment."""
    audio: np.ndarray          # mono float32 waveform at `sample_rate`
    sample_rate: int
    source_index: int          # 0, 1 -- ARBITRARY order; match to speakers by voice
    origin_start: float        # start time (s) in the ORIGINAL recording
    origin_end: float          # end time (s) in the ORIGINAL recording

    @property
    def rms(self) -> float:
        return rms(self.audio)


class OverlapSeparator:
    """
    Thin wrapper around a SpeechBrain SepFormer checkpoint. Kept separate
    from the pipeline so Conv-TasNet, DPRNN, or a 3+-speaker model can be
    swapped in later without touching anything else.
    """

    def __init__(self, source: str = DEFAULT_MODEL, device: str = "cpu"):
        # Imported lazily so this module can be imported (and its data
        # structures tested) without loading speechbrain.
        from speechbrain.inference.separation import SepformerSeparation

        self.model = SepformerSeparation.from_hparams(
            source=source,
            savedir=str(MODELS_DIR / source.split("/")[-1]),
            run_opts={"device": device},
        )
        self.device = device
        self.model_sr = int(getattr(self.model.hparams, "sample_rate", 8000))

    def separate(
        self,
        mixed_audio: np.ndarray,
        sample_rate: int,
        origin_start: float = 0.0,
        origin_end: float | None = None,
    ) -> List[SeparatedStream]:
        """
        mixed_audio: mono float32 waveform of ONE overlapping segment
                     (already trimmed to the overlap window; keep these
                     short, a few seconds, for speed and quality).
        Returns two streams at the INPUT sample rate. Both are scaled by
        one shared factor, so their relative loudness is preserved (a
        near-silent stream stays near-silent instead of being amplified
        into noise that ASR would hallucinate words from).
        """
        import torch

        if origin_end is None:
            origin_end = origin_start + len(mixed_audio) / sample_rate

        model_input = resample(mixed_audio, sample_rate, self.model_sr)
        wav = torch.from_numpy(model_input).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            est = self.model.separate_batch(wav)  # (1, T, n_src)
        est = est.squeeze(0).cpu().numpy()        # (T, n_src)

        streams = [resample(est[:, i], self.model_sr, sample_rate) for i in range(est.shape[1])]

        peak = max((float(np.abs(s).max()) for s in streams), default=0.0)
        if peak > 0:
            scale = 0.95 / peak
            streams = [(s * scale).astype(np.float32) for s in streams]

        return [
            SeparatedStream(
                audio=s,
                sample_rate=sample_rate,
                source_index=i,
                origin_start=origin_start,
                origin_end=origin_end,
            )
            for i, s in enumerate(streams)
        ]


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    from audio_utils import load_audio, save_wav, slice_audio
    from config import OUTPUT_DIR, pick_device

    ap = argparse.ArgumentParser(
        description="Separate a 2-speaker mixture into two streams (Phase 2 checkpoint)."
    )
    ap.add_argument("audio")
    ap.add_argument("--start", type=float, default=0.0, help="segment start (s)")
    ap.add_argument("--end", type=float, default=None, help="segment end (s)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--transcribe", action="store_true", help="also run verbatim ASR on each stream")
    args = ap.parse_args()

    audio = load_audio(args.audio)
    end = args.end if args.end is not None else len(audio) / TARGET_SR
    segment = slice_audio(audio, TARGET_SR, args.start, end)

    separator = OverlapSeparator(args.model, device=pick_device())
    streams = separator.separate(segment, TARGET_SR, args.start, end)

    out_dir = OUTPUT_DIR / "separated"
    stem = Path(args.audio).stem.replace(" ", "_")
    transcriber = None
    if args.transcribe:
        from asr_baseline import VerbatimTranscriber

        transcriber = VerbatimTranscriber()

    for s in streams:
        out = out_dir / f"{stem}_src{s.source_index}.wav"
        save_wav(out, s.audio, s.sample_rate)
        print(f"source {s.source_index}: rms={s.rms:.4f} -> {out}")
        if transcriber:
            for u in transcriber.transcribe(s.audio):
                print(f'   [{args.start + u.start:6.1f}s] "{u.as_display_text()}"')
