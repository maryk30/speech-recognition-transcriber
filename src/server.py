"""
Phase 7 — live transcription server + demo.

    ./run.sh demo             open http://localhost:8000

Two ways to feed a session, over the same WebSocket (/ws):

  1. Live input: the browser captures the microphone (or a file you pick),
     downsamples to 16 kHz mono and streams 16-bit PCM.
  2. Demo: `/ws?sample=<id>&model=stock|tuned&seconds=60` -- the SERVER plays
     one of the bundled recordings into the pipeline at real-time pace while
     the browser plays the same audio (/audio/<id>), so the audience hears the
     speech and watches the transcript trail it by the real end-to-end latency.

Client -> server (live input only):
    text   {"type": "start", "num_speakers": 3}     optional; upper bound on speakers
    binary little-endian int16 PCM, 16 kHz mono, any chunk size
    text   {"type": "stop"}                         flush the tail and finish
Server -> client (JSON text):
    {"type": "ready"}  {"type": "started", "duration": s}   (demo: audio playback should start now)
    {"type": "line", "speaker", "start", "end", "text", "rendered", "overlap"}
    {"type": "status", "received_s", "committed_s", "lag_s"}
    {"type": "done"}   /   {"type": "error", "message"}
REST:
    GET /api/samples          bundled recordings + available models
    GET /api/reference/<id>   ground-truth turns (speaker, start, end, text), when known
    GET /audio/<id>           the recording

One live session at a time: the models are large and shared.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf
from aiohttp import WSMsgType, web

from audio_utils import load_audio
from config import DATA_DIR, MODELS_DIR, ROOT, TARGET_SR
from pipeline import TranscriptLine, TranscriptionPipeline
from streaming import StreamingTranscriber

WEB_DIR = ROOT / "web"

# Curated demo recordings: id -> (label, note). Anything not listed here is
# ignored, so degenerate test files never show up in the menu.
_DEMO_LABELS = {
    "tts_demo": ("Two synthetic voices, 2.5 s overlap", "easy: clean audio, one interruption"),
    "real_plus_tts": ("Real voice + synthetic voice, 6.5 s overlap", "a real recording overlapped with a second speaker"),
    "EN2002b_300s": ("AMI meeting EN2002b (4 speakers)", "hard: heavy overlap, distant microphone"),
    "ES2004c_300s": ("AMI meeting ES2004c (4 speakers)", "hard: distant microphone, little overlap"),
    "IS1009b_300s": ("AMI meeting IS1009b (4 speakers)", "hard: distant microphone, similar voices"),
}


def line_message(line: TranscriptLine) -> dict:
    return {
        "type": "line",
        "speaker": line.speaker,
        "start": round(line.start, 2),
        "end": round(line.end, 2),
        "text": line.text,
        "rendered": line.render(),
        "overlap": line.from_overlap,
    }


def load_reference_turns(path: Optional[Path]) -> List[dict]:
    if not path or not Path(path).exists():
        return []
    turns = json.loads(Path(path).read_text()).get("turns", [])
    return [
        {"speaker": t["speaker"], "start": float(t["start"]), "end": float(t["end"]), "text": t.get("text", "")}
        for t in turns
        if t.get("text")
    ]


def discover_samples() -> Dict[str, dict]:
    """Bundled recordings that exist on disk, keyed by id."""
    candidates = {p.parent.name: (p, None) for p in (DATA_DIR / "mixtures").glob("*/mix.wav")}
    for p in (DATA_DIR / "ami_meetings").glob("*.wav"):
        candidates[p.stem] = (p, DATA_DIR / "ami_meetings" / f"{p.stem}.json")
    truth = {"tts_demo": DATA_DIR / "samples" / "tts_demo_truth.json"}

    found: Dict[str, dict] = {}
    for sample_id, (label, note) in _DEMO_LABELS.items():
        if sample_id not in candidates:
            continue
        path, ref = candidates[sample_id]
        ref = ref or truth.get(sample_id)
        turns = load_reference_turns(ref)
        found[sample_id] = {
            "id": sample_id,
            "label": label,
            "note": note,
            "path": path,
            "duration": round(sf.info(str(path)).duration, 1),
            "reference": ref if turns else None,
            "speakers": len({t["speaker"] for t in turns}) or None,
        }
    return found


async def index(request: web.Request) -> web.StreamResponse:
    return web.FileResponse(WEB_DIR / "index.html")


async def api_samples(request: web.Request) -> web.Response:
    app = request.app
    return web.json_response({
        "samples": [
            {**{k: v for k, v in s.items() if k not in ("path", "reference")}, "has_reference": bool(s["reference"])}
            for s in app["samples"].values()
        ],
        "models": list(app["transcribers"]),
    })


async def api_reference(request: web.Request) -> web.Response:
    sample = request.app["samples"].get(request.match_info["sample_id"])
    if not sample:
        raise web.HTTPNotFound()
    return web.json_response(load_reference_turns(sample["reference"]))


async def api_audio(request: web.Request) -> web.StreamResponse:
    sample = request.app["samples"].get(request.match_info["sample_id"])
    if not sample:
        raise web.HTTPNotFound()
    return web.FileResponse(sample["path"])


async def websocket(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse(max_msg_size=8 * 1024 * 1024)
    await ws.prepare(request)
    app = request.app

    async def refuse(message: str) -> web.WebSocketResponse:
        await ws.send_json({"type": "error", "message": message})
        await ws.close()
        return ws

    if app["busy"]:
        return await refuse("another session is already running")
    sample_id = request.query.get("sample")
    sample = app["samples"].get(sample_id) if sample_id else None
    if sample_id and not sample:
        return await refuse(f"unknown sample {sample_id!r}")
    model = request.query.get("model", "stock")
    if model not in app["transcribers"]:
        return await refuse(f"model {model!r} is not available")

    app["busy"] = True
    app["pipeline"].transcriber = app["transcribers"][model]      # safe: one session at a time

    loop = asyncio.get_running_loop()
    session = StreamingTranscriber(app["pipeline"], **app["stream_kwargs"])
    if sample and sample["speakers"]:
        session.num_speakers = sample["speakers"]
    finished, stop = asyncio.Event(), asyncio.Event()

    async def emit(lines: List[TranscriptLine]) -> None:
        if ws.closed:
            return
        for line in lines:
            await ws.send_json(line_message(line))
        received = session.total_time
        await ws.send_json({
            "type": "status",
            "received_s": round(received, 1),
            "committed_s": round(session._committed_until, 1),
            "lag_s": round(received - session._committed_until, 1),
        })

    async def worker() -> None:
        # Steps run in a thread so the event loop keeps receiving audio.
        try:
            while not finished.is_set():
                if session.ready():
                    await emit(await loop.run_in_executor(None, session.step))
                else:
                    await asyncio.sleep(0.1)
        except Exception as exc:   # model failure or client vanished: report, never wedge the server
            print(f"session worker stopped: {type(exc).__name__}: {exc}", file=sys.stderr)
            if not ws.closed:
                await ws.send_json({"type": "error", "message": f"processing failed: {exc}"})
            finished.set()
            stop.set()

    async def replay() -> None:
        """Demo mode: play the recording into the session at real-time pace."""
        seconds = float(request.query.get("seconds", 0)) or None
        speed = float(request.query.get("speed", 1.0)) or 1.0
        audio = await loop.run_in_executor(None, load_audio, str(sample["path"]))
        if seconds:
            audio = audio[: int(seconds * TARGET_SR)]
        await ws.send_json({"type": "started", "duration": round(len(audio) / TARGET_SR, 1)})
        chunk_s = 0.5
        chunk = int(chunk_s * TARGET_SR)
        start = loop.time()
        for i in range(0, len(audio), chunk):
            session.feed(audio[i : i + chunk])
            await asyncio.sleep(max(0.0, start + (i + chunk) / TARGET_SR / speed - loop.time()))
        stop.set()

    task = asyncio.create_task(worker())
    feeder = None
    try:
        await ws.send_json({"type": "ready"})
        if sample:
            feeder = asyncio.create_task(replay())
        while not stop.is_set() and not ws.closed:
            try:
                msg = await asyncio.wait_for(ws.receive(), timeout=0.3)
            except asyncio.TimeoutError:
                continue
            if msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSING, WSMsgType.CLOSED, WSMsgType.ERROR):
                break
            if sample:
                continue                # demo sessions ignore client audio
            if msg.type == WSMsgType.BINARY:
                session.feed(np.frombuffer(msg.data, dtype="<i2").astype(np.float32) / 32768.0)
            elif msg.type == WSMsgType.TEXT:
                data = json.loads(msg.data)
                if data.get("type") == "start" and data.get("num_speakers"):
                    session.num_speakers = int(data["num_speakers"])
                elif data.get("type") == "stop":
                    break
    finally:
        if feeder:
            feeder.cancel()
        finished.set()
        try:
            await task                  # let an in-flight step complete
            if not ws.closed:
                await emit(await loop.run_in_executor(None, session.flush))
                await ws.send_json({"type": "done"})
                await ws.close()
        except Exception as exc:
            print(f"session cleanup: {type(exc).__name__}: {exc}", file=sys.stderr)
        finally:
            app["busy"] = False         # always release, whatever happened above
    return ws


def create_app(
    pipeline: TranscriptionPipeline,
    tuned_transcriber=None,
    **stream_kwargs,
) -> web.Application:
    app = web.Application()
    app["pipeline"] = pipeline
    app["transcribers"] = {"stock": pipeline.transcriber}
    if tuned_transcriber is not None:
        app["transcribers"]["tuned"] = tuned_transcriber
    app["stream_kwargs"] = stream_kwargs
    app["busy"] = False
    app["samples"] = discover_samples()
    app.router.add_get("/", index)
    app.router.add_get("/ws", websocket)
    app.router.add_get("/api/samples", api_samples)
    app.router.add_get("/api/reference/{sample_id}", api_reference)
    app.router.add_get("/audio/{sample_id}", api_audio)
    return app


def main(argv: Optional[List[str]] = None) -> None:
    import argparse

    from asr_baseline import make_transcriber
    from enrollment import EnrollmentGallery
    from streaming import DEFAULT_GUARD_S, DEFAULT_HOP_S, DEFAULT_SIM_THRESHOLD, DEFAULT_WINDOW_S

    ap = argparse.ArgumentParser(description="Live multi-speaker transcription server and demo.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default="small", help="Whisper size for the 'stock' model; tiny/base are much faster")
    ap.add_argument("--tuned", default=str(MODELS_DIR / "whisper-small-ami-mlx"),
                    help="fine-tuned MLX model dir, offered as 'tuned' in the demo when it exists")
    ap.add_argument("--beam-size", type=int, default=1)
    ap.add_argument("--asr-backend", default="auto", choices=["auto", "mlx", "ctranslate2"])
    ap.add_argument("--no-separation", action="store_true")
    ap.add_argument("--hop", type=float, default=DEFAULT_HOP_S)
    ap.add_argument("--guard", type=float, default=DEFAULT_GUARD_S)
    ap.add_argument("--window", type=float, default=DEFAULT_WINDOW_S)
    ap.add_argument("--sim-threshold", type=float, default=DEFAULT_SIM_THRESHOLD)
    ap.add_argument("--enroll", action="append", default=[], metavar="NAME=WAV")
    args = ap.parse_args(argv)

    gallery = None
    if args.enroll:
        gallery = EnrollmentGallery()
        for item in args.enroll:
            name, _, path = item.partition("=")
            gallery.enroll(name, path)

    pipeline = TranscriptionPipeline(
        asr_model_size=args.model,
        beam_size=args.beam_size,
        asr_backend=args.asr_backend,
        separate=not args.no_separation,
        gallery=gallery,
    )
    tuned = None
    if args.asr_backend not in ("auto", "mlx"):
        print("fine-tuned model needs the MLX backend; demo will offer the stock model only", file=sys.stderr)
    elif Path(args.tuned).exists():
        print(f"loading fine-tuned model: {args.tuned}", file=sys.stderr)
        tuned = make_transcriber(args.tuned, backend="mlx")
    else:
        print(f"no fine-tuned model at {args.tuned}; demo will offer the stock model only", file=sys.stderr)

    app = create_app(
        pipeline, tuned,
        window_s=args.window, hop_s=args.hop, guard_s=args.guard, sim_threshold=args.sim_threshold,
    )
    print(f"{len(app['samples'])} demo recordings | models: {list(app['transcribers'])}", file=sys.stderr)
    print(f"open http://{args.host}:{args.port}", file=sys.stderr)
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
