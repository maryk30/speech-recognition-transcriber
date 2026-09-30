"""
Stream an audio file to a running live server exactly the way the browser
does (16 kHz int16 PCM over a WebSocket), and print the lines that come back.
Useful for testing the server without a microphone.

    ./run.sh serve                                  # terminal 1
    ./run.sh stream data/mixtures/tts_demo/mix.wav  # terminal 2

    --speed 1   real-time pacing (default)      --speed 0   as fast as possible
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import aiohttp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from audio_utils import load_audio  # noqa: E402
from config import TARGET_SR  # noqa: E402


async def stream(url: str, path: str, speed: float, num_speakers: int | None, chunk_s: float = 0.5) -> int:
    audio = load_audio(path)
    pcm = (np_clip(audio) * 32767).astype("<i2")
    chunk = int(chunk_s * TARGET_SR)
    lines = 0

    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(url, max_msg_size=8 * 1024 * 1024) as ws:
            start = time.time()

            async def receiver() -> None:
                nonlocal lines
                async for msg in ws:
                    if msg.type != aiohttp.WSMsgType.TEXT:
                        continue
                    data = json.loads(msg.data)
                    if data["type"] == "line":
                        lines += 1
                        print(f"{data['rendered']}   (+{time.time() - start:.1f}s wall)", flush=True)
                    elif data["type"] == "status":
                        print(f"    lag {data['lag_s']}s", flush=True)
                    elif data["type"] == "error":
                        print("server error:", data["message"], file=sys.stderr)
                    elif data["type"] == "done":
                        return

            first = await ws.receive_json()          # {"type": "ready"} (or an error)
            if first["type"] != "ready":
                raise SystemExit(f"server refused the session: {first}")
            await ws.send_json({"type": "start", "num_speakers": num_speakers})
            recv = asyncio.create_task(receiver())   # only after the handshake: aiohttp forbids concurrent receive()

            for i in range(0, len(pcm), chunk):
                await ws.send_bytes(pcm[i : i + chunk].tobytes())
                if speed:
                    await asyncio.sleep(chunk_s / speed)
            await ws.send_json({"type": "stop"})
            await asyncio.wait_for(recv, timeout=300)
    return lines


def np_clip(a):
    import numpy as np

    return np.clip(a, -1.0, 1.0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio")
    ap.add_argument("--url", default="ws://127.0.0.1:8000/ws")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--num-speakers", type=int, default=None)
    args = ap.parse_args()
    n = asyncio.run(stream(args.url, args.audio, args.speed, args.num_speakers))
    print(f"\n{n} lines received")
