# Multi-Speaker Live Transcription Pipeline

Speaker-independent pipeline for transcribing overlapping multi-speaker
conversations from shared room audio, with verbatim output (fillers and
pauses kept), optional name enrollment, and a generic `Speaker_N` fallback.

```
[00:04.6] Speaker_1: "Wait actually I don't think that works"
[00:05.1] Speaker_2: "is taking a lot longer than we expected."
[00:07.6] Krishiv:   "This is a test for our DL pipelines."
```

Everything runs locally on free, open-weight models. No paid APIs.

## Status

| Phase | What | Module | Status |
|---|---|---|---|
| 1 | Verbatim ASR (fillers + `[pause Xs]`) | `src/asr_baseline.py` | Working |
| 2 | Speech separation of overlaps (SepFormer) | `src/separation.py` | Working, SI-SDR measured on synthetic data |
| 3 | VAD + overlap detection | `src/overlap.py` | Working, checked against ground truth on synthetic data |
| 4 | Diarization (`Speaker_N`) | `src/diarization.py` | Working; scoring harness ready (`./run.sh score`), **DER not yet measured on real meetings** |
| 5 | Optional enrollment (real names) | `src/enrollment.py` | Working; threshold not yet tuned |
| 6 | Full offline pipeline | `src/pipeline.py`, `src/scoring.py` | Working end to end; DER / cpWER / filler recall / consistency scorer ready, waiting for real ground-truth data |
| 7 | Live / streaming | `src/streaming.py`, `src/server.py`, `web/index.html` | Working, but **not real-time on this CPU** (see Live mode) |

## Quick start

```bash
./setup.sh                 # once: creates .venv, installs requirements, creates .env
# put your HuggingFace token in .env  ->  HF_TOKEN=hf_...   (see below)

./run.sh test              # unit tests, no models or network needed
./run.sh pipeline data/mixtures/tts_demo/mix.wav --num-speakers 2
```

`./run.sh` is the single entry point:

| Command | What it does |
|---|---|
| `./run.sh pipeline <audio> [--num-speakers N] [--enroll NAME=clip.wav] [--no-separation] [--sep-model ...] [--out file]` | Full pipeline (default command) |
| `./run.sh overlap <audio>` | Phase 3: print speech / overlap segments |
| `./run.sh separate <audio> [--start S --end E] [--transcribe]` | Phase 2: split a 2-speaker mix into two streams (`output/separated/`) |
| `./run.sh asr <audio> [label] [model]` | Phase 1: single-speaker verbatim ASR |
| `./run.sh mix <a.wav> <b.wav> --name X --offset S` | Build a 2-speaker test mixture with ground truth in `data/mixtures/X/` |
| `./run.sh eval --dir <folder>` | Phase 2: SI-SDR / SI-SDRi of the separator (mixture folder or LibriMix tree) |
| `./run.sh score <audio> --reference truth.json` | Phases 4/6: DER, cpWER (fillers ignored and scored), filler recall, speaker consistency. Reference is JSON or RTTM |
| `./run.sh live <audio> [--speed 1]` | Phase 7: replay a file through the streaming pipeline and report per-line latency |
| `./run.sh demo` | Demo: server + browser page with bundled recordings, stock vs fine-tuned switch, ground-truth column |
| `./run.sh serve` | Same server without opening the browser (live microphone / file input) |
| `./run.sh stream <audio>` | Phase 7: stream a file to the running server (no microphone needed) |
| `./run.sh test` | Unit tests |

Any audio format works: WAV/FLAC directly, m4a/mp3 via the `ffmpeg` binary.

## How the pipeline works

```
room audio
  -> diarization             pyannote: consistent Speaker_N turns, overlaps preserved
  -> overlap detection/VAD   flat timeline: silence dropped, single vs overlapping segments
  -> speaker centroids       ECAPA voice print per speaker from their CLEAN speech
  -> single segments         verbatim ASR directly
  -> overlapping segments    SepFormer -> 2 streams -> match streams to speakers by voice
                             (separator output order is arbitrary) -> verbatim ASR per stream
  -> enrollment (optional)   Speaker_N -> real name if voice matches an enrolled clip
  -> chronological merge     interruptions appear as separate, correctly ordered lines
```

Details worth knowing:
- Every segment is transcribed with 0.4 s of extra context and cropped back
  to the segment by word midpoint, so words at window edges aren't chopped
  and none are duplicated.
- Overlap flags come from the diarization model itself (one pass gives both
  speech activity and simultaneous speakers), not from a separate VAD run.
- A separated stream far quieter than the other is treated as leakage and
  skipped, and low-confidence output on separated audio is filtered, because
  Whisper hallucinates words on residue.

## Demo

```bash
./run.sh demo          # starts the server and opens http://127.0.0.1:8000
```

Pick a recording and a model, press **Play demo**. The server feeds the
recording into the live pipeline at real-time pace while your browser plays
the same audio, so you hear the speech and watch the transcript trail it by
the system's real end-to-end delay. On screen:

- speaker-coloured lines, an **overlap** tag on lines recovered by separation,
  `um`/`uh` highlighted, `[pause Xs]` dimmed;
- live stats: seconds behind the audio, lines, speakers found, overlap lines,
  fillers kept;
- a **ground-truth column** revealed in step with the audio (for the AMI
  excerpts and the synthetic conversation);
- a model switch between **stock Whisper-small** and the **fine-tuned model**
  (offered when `models/whisper-small-ami-mlx` exists), so an audience can
  compare filler retention on the same audio.

Bundled recordings: two synthetic/real overlap demos and three 5-minute
4-person AMI meeting excerpts (`./run.sh` builds them via
`training/prepare_meetings.py`). Good things to show: `tts_demo` with the
fine-tuned model (um kept, interruption separated), then an AMI excerpt to
show the honest hard case. A scripted start: `http://127.0.0.1:8000/?autostart=tts_demo&model=tuned`.
The "Live input" section under the page runs the microphone or a file of your own.

Caveats to state when demoing: the fine-tuned model prints lower-case text
without punctuation; on noisy stretches it can invent a stray `mm`; lag is
typically 6-10 s; separation handles two simultaneous speakers only.

## Live mode

`./run.sh serve`, then open http://localhost:8000. The page streams your
microphone (or a replayed recording) to the server over a WebSocket and shows
lines as they are committed, with per-speaker colours and an "overlap" tag.
Browser noise suppression is deliberately switched off: it attenuates the
quieter of two simultaneous talkers, which is the speech the separator needs.

How it works (`src/streaming.py`): a rolling 16 s window is re-diarized every
~4 s; only audio older than a 3 s guard is committed, so lines are never
retracted. Speaker labels stay stable across windows using (1) temporal
continuity with the previous window and (2) voice-print similarity for
speakers with no temporal anchor. A speaker seen only inside the guard
region is not registered until they appear in committed audio.

**Latency, measured** (Apple M3, 21 s two-speaker clip with overlap, replayed
at real-time pace): mean 6-8 s, max 8-11 s, and it no longer grows during
overlaps. Down from mean 14.5 s / max 25 s before the changes below.

What made it real-time:
- **Whisper on the Apple GPU via MLX** (`--asr-backend mlx`, the default on
  Apple silicon): 0.45 s vs 2.34 s for 4.4 s of audio, same transcript.
  Greedy decoding only (MLX has no beam search); `--asr-backend ctranslate2`
  keeps beam search on CPU/CUDA.
- **MPS for diarization, separation and speaker embeddings** (about 2x faster
  than CPU with identical output; `DLPBL_DEVICE=cpu` to override).
- **Packed ASR calls**: a Whisper call costs about 0.8 s whatever the clip
  length (it always encodes a padded 30 s window), so all clean single-speaker
  clips of a step are packed into one window with silence gaps and split back
  by word timestamps. Separated overlap streams are NOT packed: doing so
  measurably lost words in overlaps ("This is a test for our DL pipelines"
  disappeared), so they are transcribed alone.
- **Offline-safe model loading**: `config.configure_hub()` bounds hub timeouts
  and falls back to cached models if huggingface.co is unreachable, so a flaky
  network can't stall startup.

The browser microphone path is written and its JS helpers are unit-tested
in node, but it has not been exercised with a real microphone.

## Layout

```
src/         pipeline code (one module per phase, plus config / audio_utils / embeddings / metrics / scoring)
web/         live transcript page (single HTML file, no build step)
scripts/     make_mixture.py, eval_separation.py, evaluate.py, stream_client.py
tests/       unit tests (pure logic; no models needed)
data/        your recordings; data/samples = TTS clips; data/mixtures = generated (git-ignored)
models/      downloaded model weights (git-ignored)
output/      generated audio / transcripts (git-ignored)
PLAN.md      phase-by-phase build plan and current status
```

## HuggingFace access (one-time, free)

pyannote's models are "gated": free, but each HuggingFace account must accept
the license itself.

1. Sign up at huggingface.co and create a token (Settings -> Access Tokens, "read").
2. Click "Agree and access repository" on **all three**:
   `pyannote/speaker-diarization-3.1`, `pyannote/segmentation-3.0`,
   `pyannote/speaker-diarization-community-1`.
3. Put the token in `.env` as `HF_TOKEN=hf_...`.

SpeechBrain models (SepFormer, ECAPA) and faster-whisper need no account.

## Known limitations

- Separation is 2-source only. With 3+ simultaneous speakers only two are
  recovered (the pipeline logs a warning). Matches the project scope: mostly
  one speaker, brief two-speaker overlaps.
- Overlap quality is bounded by diarization: if pyannote misses that a
  second person is talking, that speech is transcribed as the first speaker's.
- Speakers who never talk alone have no clean voice print, so stream-to-speaker
  matching for them falls back to positional order.
- Separator default (`sepformer-libri2mix`) was chosen on clean synthetic
  audio; on noisy/echoey room recordings compare against `--sep-model
  speechbrain/sepformer-whamr` using `./run.sh eval`.
