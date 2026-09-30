"""
Phase 1 — Verbatim ASR baseline.

Goal: transcribe a single-speaker (or pre-isolated) audio stream WITHOUT
Whisper's default behavior of smoothing over filler words ("um", "uh") and
without collapsing pauses. This is the building block every later stage
(separation, diarization, chronological assembly) will call per-stream.

Model: faster-whisper (CTranslate2 port of OpenAI Whisper). Free, open
weights, runs on CPU or GPU, no API key required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple, Union

import config  # noqa: F401  (must come first: sets huggingface_hub timeouts / offline mode)
import numpy as np
from audio_utils import normalize_rms
from faster_whisper import WhisperModel

# faster-whisper's default is suppress_tokens=[-1], which EXPANDS to
# Whisper's built-in non-speech token list (this is part of what makes it
# drop fillers). An empty list turns suppression off entirely so "um"/"uh"/
# repetitions can survive into the transcript, as the project spec requires.
VERBATIM_SUPPRESS_TOKENS: List[int] = []

# Optional decoding prompt written in a disfluent style; Whisper tends to
# imitate the style of its prompt, which nudges it to keep fillers. Off by
# default because on very short or silent audio a prompt can leak into the
# output -- enable with VerbatimTranscriber(verbatim_prompt=True) and compare.
VERBATIM_PROMPT = "Umm, so, uh, I was like, you know, thinking... hmm, well, okay."

# A segment whose text compresses better than this is a repetition loop.
MAX_COMPRESSION_RATIO = 2.8

# Gap (seconds) between two consecutive words above which we insert an
# explicit pause marker in the output.
DEFAULT_PAUSE_THRESHOLD_S = 0.6


@dataclass
class WordTiming:
    word: str
    start: float
    end: float


@dataclass
class Utterance:
    """One ASR segment, verbatim, with word-level timing and pause markers."""
    text: str
    start: float
    end: float
    words: List[WordTiming] = field(default_factory=list)
    no_speech_prob: float = 0.0   # Whisper's own "this is not speech" estimate
    avg_logprob: float = 0.0      # mean token log-prob; very low => likely hallucination

    def as_display_text(self, pause_threshold_s: float = DEFAULT_PAUSE_THRESHOLD_S) -> str:
        """Re-render the segment text with explicit [pause Xs] markers
        wherever the gap between two words exceeds the threshold."""
        if not self.words:
            return self.text.strip()

        parts: List[str] = []
        for i, w in enumerate(self.words):
            parts.append(w.word)
            if i < len(self.words) - 1:
                gap = self.words[i + 1].start - w.end
                if gap >= pause_threshold_s:
                    parts.append(f"[pause {gap:.1f}s]")
        return " ".join(p.strip() for p in parts if p.strip())


def crop_utterances(utterances: List[Utterance], start: float, end: float) -> List[Utterance]:
    """Keep only the words whose midpoint lies in [start, end] (times
    relative to the transcribed clip), rebuilding each utterance from what
    survives. Lets callers transcribe a segment WITH extra audio context
    (so words at the edges aren't chopped) while still assigning every word
    to exactly one segment."""
    cropped: List[Utterance] = []
    for utt in utterances:
        if not utt.words:
            if start <= (utt.start + utt.end) / 2 <= end:
                cropped.append(utt)
            continue
        kept = [w for w in utt.words if start <= (w.start + w.end) / 2 <= end]
        if not kept:
            continue
        cropped.append(
            Utterance(
                text="".join(w.word for w in kept).strip(),
                start=kept[0].start,
                end=kept[-1].end,
                words=kept,
                no_speech_prob=utt.no_speech_prob,
                avg_logprob=utt.avg_logprob,
            )
        )
    return cropped


def shift_utterances(utterances: List[Utterance], dt: float) -> List[Utterance]:
    """Copy of `utterances` with every timestamp moved by `dt` seconds."""
    return [
        Utterance(
            text=u.text,
            start=u.start + dt,
            end=u.end + dt,
            words=[WordTiming(w.word, w.start + dt, w.end + dt) for w in u.words],
            no_speech_prob=u.no_speech_prob,
            avg_logprob=u.avg_logprob,
        )
        for u in utterances
    ]


def pack_clips(
    lengths_s: Sequence[float], max_window_s: float = 28.0, gap_s: float = 1.0
) -> List[List[Tuple[int, float]]]:
    """Greedy left-to-right packing of clips into Whisper-sized windows.
    Returns windows, each a list of (clip_index, offset_s within the window)."""
    windows: List[List[Tuple[int, float]]] = []
    current: List[Tuple[int, float]] = []
    used = 0.0
    for i, length in enumerate(lengths_s):
        if current and used + gap_s + length > max_window_s:
            windows.append(current)
            current, used = [], 0.0
        offset = used + (gap_s if current else 0.0)
        current.append((i, offset))
        used = offset + length
    if current:
        windows.append(current)
    return windows


def transcribe_packed(
    transcribe: Callable[[np.ndarray], List[Utterance]],
    clips: Sequence[np.ndarray],
    sample_rate: int = 16000,
    max_window_s: float = 28.0,
    gap_s: float = 1.0,
) -> List[List[Utterance]]:
    """Transcribe many short clips with as few Whisper calls as possible.

    Whisper always encodes a padded 30 s window, so on the Apple GPU a call
    costs about the same for a 1 s clip as for a 25 s one (measured: 0.77 s vs
    0.89 s). Packing clips into one window separated by silence and splitting
    the words back out by timestamp turns N calls into ~1. Each clip gets the
    words whose midpoint falls inside its own slot, with times relative to the
    clip start, exactly as if it had been transcribed alone.
    """
    results: List[List[Utterance]] = [[] for _ in clips]
    windows = pack_clips([len(c) / sample_rate for c in clips], max_window_s, gap_s)
    gap = np.zeros(int(gap_s * sample_rate), dtype=np.float32)
    for window in windows:
        parts: List[np.ndarray] = []
        for pos, (idx, _) in enumerate(window):
            if pos:
                parts.append(gap)
            parts.append(np.asarray(clips[idx], dtype=np.float32))
        utts = transcribe(np.concatenate(parts))
        for idx, offset in window:
            duration = len(clips[idx]) / sample_rate
            slot = crop_utterances(utts, offset, offset + duration)
            results[idx] = shift_utterances(slot, -offset)
    return results


class VerbatimTranscriber:
    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        verbatim_prompt: bool = False,
        beam_size: int = 5,
    ) -> None:
        """
        model_size: tiny | base | small | medium | large-v3
                    (start with "small" for CPU speed while developing;
                     switch to "large-v3" for the accent-robustness numbers
                     you'll want in the final report)
        device:     "cpu" or "cuda"
        compute_type: "int8" is fastest/lightest on CPU; use "float16" on GPU
        """
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.verbatim_prompt = verbatim_prompt
        self.beam_size = beam_size   # 1 = greedy: much faster (streaming), slightly less accurate

    def transcribe_many(self, clips: Sequence[np.ndarray]) -> List[List[Utterance]]:
        # CPU cost scales with audio length, so packing would gain nothing here.
        return [self.transcribe(c) for c in clips]

    def transcribe(
        self,
        audio: Union[str, np.ndarray],
        language: Optional[str] = "en",
    ) -> List[Utterance]:
        """audio: a file path, or a mono float32 16 kHz numpy array (used by
        the pipeline so per-segment slices never touch disk). Utterance times
        are relative to the start of `audio`."""
        segments, info = self.model.transcribe(
            audio,
            language=language,
            beam_size=self.beam_size,
            word_timestamps=True,
            vad_filter=True,           # trims leading/trailing silence per segment
            suppress_tokens=VERBATIM_SUPPRESS_TOKENS,
            condition_on_previous_text=False,  # avoids Whisper "cleaning up" based on prior context
            temperature=0.0,
            initial_prompt=VERBATIM_PROMPT if self.verbatim_prompt else None,
        )

        utterances: List[Utterance] = []
        for seg in segments:
            words = [
                WordTiming(word=w.word, start=w.start, end=w.end)
                for w in (seg.words or [])
            ]
            utt = Utterance(
                text=seg.text,
                start=seg.start,
                end=seg.end,
                words=words,
                no_speech_prob=seg.no_speech_prob,
                avg_logprob=seg.avg_logprob,
            )
            utterances.append(utt)

        return utterances


class MLXTranscriber:
    """Same interface as VerbatimTranscriber, but runs Whisper on the Apple
    GPU through MLX. Measured on this project's clips (Whisper small, 4.4 s
    of audio): 0.45 s vs 2.3 s for the CPU/CTranslate2 path, with an identical
    transcript -- this is what makes the live pipeline fast enough.

    Greedy decoding only (MLX has no beam search); `beam_size` is accepted for
    interface compatibility and ignored.
    """

    def __init__(
        self,
        model_size: str = "small",
        verbatim_prompt: bool = False,
        beam_size: int = 1,
        **_ignored,
    ) -> None:
        import mlx_whisper  # noqa: F401  (fail early if not installed)

        self.repo = mlx_repo(model_size)
        self.verbatim_prompt = verbatim_prompt
        self._mlx = mlx_whisper
        # Warm-up: load weights and compile kernels now, not on the first live chunk.
        self.transcribe(np.zeros(16000, dtype=np.float32))

    def transcribe_many(self, clips: Sequence[np.ndarray]) -> List[List[Utterance]]:
        """Pack all clips into as few 30 s Whisper windows as possible."""
        return transcribe_packed(self.transcribe, clips)

    def transcribe(
        self, audio: Union[str, np.ndarray], language: Optional[str] = "en"
    ) -> List[Utterance]:
        max_tokens = 224          # Whisper's default cap
        if isinstance(audio, np.ndarray):
            # Speech (incl. timestamp tokens) never exceeds ~8 tokens per second, so
            # cap generation by clip length: a repetition loop then dies in a few
            # tokens instead of running to 224 on every temperature retry.
            max_tokens = int(min(224, 8 * len(audio) / 16000 + 24))
            # Same per-clip loudness normalisation the AMI fine-tuning used
            # (measured harmless for the stock model: 38.7% vs 38.8% WER).
            audio = normalize_rms(audio.astype(np.float32, copy=False))
        result = self._mlx.transcribe(
            audio,
            path_or_hf_repo=self.repo,
            language=language,
            word_timestamps=True,
            condition_on_previous_text=False,
            # Greedy first; if the result is repetitive (compression ratio) or
            # low-confidence, retry at a slightly higher temperature. Fine-tuned
            # models otherwise fall into loops ("a f making a f making ...").
            temperature=(0.0, 0.2, 0.4),
            sample_len=max_tokens,
            compression_ratio_threshold=2.4,
            logprob_threshold=-1.0,
            suppress_tokens=VERBATIM_SUPPRESS_TOKENS,
            initial_prompt=VERBATIM_PROMPT if self.verbatim_prompt else None,
        )
        utterances: List[Utterance] = []
        for seg in result["segments"]:
            if seg.get("compression_ratio", 0.0) > MAX_COMPRESSION_RATIO:
                continue   # still a repetition loop after the fallback: drop, don't emit garbage
            words = [
                WordTiming(word=w["word"], start=w["start"], end=w["end"])
                for w in seg.get("words", [])
            ]
            utterances.append(
                Utterance(
                    text=seg["text"],
                    start=seg["start"],
                    end=seg["end"],
                    words=words,
                    no_speech_prob=seg.get("no_speech_prob", 0.0),
                    avg_logprob=seg.get("avg_logprob", 0.0),
                )
            )
        return utterances


def mlx_repo(model_size: str) -> str:
    """'small' -> 'mlx-community/whisper-small-mlx'; a path or 'org/repo' is used as-is."""
    if "/" in model_size or Path(model_size).exists():
        return model_size
    return f"mlx-community/whisper-{model_size}-mlx"


def mlx_available() -> bool:
    import importlib.util
    import platform

    return (
        platform.system() == "Darwin"
        and platform.machine() == "arm64"
        and importlib.util.find_spec("mlx_whisper") is not None
    )


def make_transcriber(
    model_size: str = "small",
    backend: str = "auto",
    device: str = "cpu",
    compute_type: str = "int8",
    verbatim_prompt: bool = False,
    beam_size: int = 5,
):
    """backend: 'mlx' (Apple GPU), 'ctranslate2' (CPU/CUDA), or 'auto' (mlx if available)."""
    if backend == "auto":
        backend = "mlx" if mlx_available() else "ctranslate2"
    if backend == "mlx":
        return MLXTranscriber(model_size, verbatim_prompt=verbatim_prompt, beam_size=beam_size)
    return VerbatimTranscriber(
        model_size=model_size,
        device=device,
        compute_type=compute_type,
        verbatim_prompt=verbatim_prompt,
        beam_size=beam_size,
    )


def transcribe_file_to_lines(
    audio_path: str,
    speaker_label: str = "Speaker_1",
    model_size: str = "small",
) -> List[str]:
    """Convenience wrapper matching the target output format:
    [mm:ss.s] NAME: "speech, with [pause] markers preserved"
    """
    transcriber = VerbatimTranscriber(model_size=model_size)
    utterances = transcriber.transcribe(audio_path)

    lines = []
    for u in utterances:
        mm = int(u.start // 60)
        ss = u.start - mm * 60
        text = u.as_display_text()
        lines.append(f'[{mm:02d}:{ss:04.1f}] {speaker_label}: "{text}"')
    return lines


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python asr_baseline.py <audio_file> [speaker_label] [model_size]")
        sys.exit(1)

    audio_file = sys.argv[1]
    label = sys.argv[2] if len(sys.argv) > 2 else "Speaker_1"
    size = sys.argv[3] if len(sys.argv) > 3 else "small"

    for line in transcribe_file_to_lines(audio_file, speaker_label=label, model_size=size):
        print(line)
