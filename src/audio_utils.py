"""
Small numpy-only audio helpers shared by every phase (no torch import, so
this module is cheap to import and easy to unit test).
"""

from __future__ import annotations

import shutil
import subprocess
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from config import TARGET_SR


def load_audio(path: str | Path, sr: int = TARGET_SR) -> np.ndarray:
    """Load any audio file as mono float32 at `sr`.

    WAV/FLAC/OGG go through soundfile; anything else (m4a, mp3, ...) falls
    back to the ffmpeg binary if one is on PATH.
    """
    path = str(path)
    try:
        audio, file_sr = sf.read(path, dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        return resample(audio, file_sr, sr)
    except (sf.LibsndfileError, RuntimeError):
        pass

    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            f"Cannot decode {path}: soundfile doesn't support this format and "
            "ffmpeg is not installed. Convert to WAV first."
        )
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"],
        capture_output=True,
        check=True,
    )
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def resample(audio: np.ndarray, orig_sr: int, new_sr: int) -> np.ndarray:
    if orig_sr == new_sr:
        return audio.astype(np.float32, copy=False)
    g = gcd(orig_sr, new_sr)
    out = resample_poly(audio, new_sr // g, orig_sr // g)
    return out.astype(np.float32)


def save_wav(path: str | Path, audio: np.ndarray, sr: int = TARGET_SR) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr)


def rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))


def normalize_rms(audio: np.ndarray, target_rms: float = 0.05, max_gain: float = 50.0) -> np.ndarray:
    """Scale a clip to a nominal loudness before ASR.

    Far-field room microphones are quiet (AMI's distant mic peaks around
    0.02), and Whisper's log-mel features are not fully level-invariant, so
    each clip is brought to the same rms. Gain is capped so a clip of pure
    noise is not blown up, and the result is clipped to avoid overflow.
    """
    level = rms(audio)
    if level <= 1e-8:
        return audio
    gain = min(target_rms / level, max_gain)
    return np.clip(audio * gain, -0.99, 0.99).astype(np.float32)


def slice_audio(
    audio: np.ndarray, sr: int, start: float, end: float, pad: float = 0.0
) -> np.ndarray:
    """audio[start-pad : end+pad] in seconds, clamped to the array bounds."""
    lo = max(0, int(round((start - pad) * sr)))
    hi = min(len(audio), int(round((end + pad) * sr)))
    return audio[lo:hi]


def trim_silence(
    audio: np.ndarray, sr: int, threshold_db: float = -40.0, frame_s: float = 0.02
) -> tuple[np.ndarray, float]:
    """Trim leading/trailing silence (frames below `threshold_db` relative to
    the loudest frame). Returns (trimmed_audio, seconds_trimmed_from_start)."""
    frame = max(1, int(frame_s * sr))
    n = len(audio) // frame
    if n == 0:
        return audio, 0.0
    frames = audio[: n * frame].reshape(n, frame)
    level = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    if level.max() <= 0:
        return audio, 0.0
    active = np.where(level >= level.max() * 10 ** (threshold_db / 20))[0]
    lo, hi = active[0] * frame, (active[-1] + 1) * frame
    return audio[lo:hi], lo / sr
