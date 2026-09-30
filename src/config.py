"""
Shared paths and environment loading, so every script behaves the same
whether it is launched via ./run.sh, `python src/x.py`, or pytest.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
MODELS_DIR = ROOT / "models"

# Every model in the pipeline (Whisper, pyannote, ECAPA) consumes 16 kHz mono.
TARGET_SR = 16000


def configure_hub() -> None:
    """Keep model loading from hanging on a flaky network.

    Every model here is cached after its first download, but loading still
    makes hub requests to check for updates. If the hub can't be reached
    quickly, switch huggingface_hub to offline mode (cached files only)
    instead of letting each request wait out long retries. Explicit settings
    win: set HF_HUB_OFFLINE=0/1 yourself to force either behaviour.
    """
    import socket

    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "5")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")
    if "HF_HUB_OFFLINE" in os.environ:
        return
    try:
        socket.create_connection(("huggingface.co", 443), timeout=3).close()
    except OSError:
        os.environ["HF_HUB_OFFLINE"] = "1"
        print("note: huggingface.co unreachable, using cached models only", file=sys.stderr)


def load_env() -> None:
    """Read KEY=VALUE lines from <root>/.env into os.environ (no overwrite)."""
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def get_hf_token() -> str:
    load_env()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN is not set. Add HF_TOKEN=hf_... to .env (see README.md)."
        )
    return token


def pick_device() -> str:
    """Device for the torch models (diarization, separation, embeddings):
    cuda > mps > cpu. Override with DLPBL_DEVICE=cpu|mps|cuda.

    mps was benchmarked against cpu on this project's models and gave
    identical diarization turns and separated audio at roughly 2x speed.
    (faster-whisper/CTranslate2 has no mps backend, so ASR stays on cpu.)"""
    import torch

    load_env()
    forced = os.environ.get("DLPBL_DEVICE")
    if forced:
        return forced
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# Runs on first import of this module. Entry points import `config` before any
# model library so the hub settings are in place when huggingface_hub reads them.
configure_hub()
