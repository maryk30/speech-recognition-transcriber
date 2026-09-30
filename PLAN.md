# Multi-Speaker Live Transcription Pipeline — Implementation Plan

Speaker-independent, multimodal (acoustic + linguistic + spatial) system
for transcribing overlapping multi-speaker conversations from shared room
audio (no per-person mics), with optional name enrollment and a generic
`Speaker_N` fallback. Output format:

```
[00:12.4] SAMPATH: "so I was thinking, um, we could—"
[00:13.1] Speaker_7: "wait, actually—"
[00:13.6] SAMPATH: "—we could push the deadline"
```

This document is the step-by-step build plan. Follow it phase by phase —
each phase produces something runnable/testable before you move to the
next, so you always have a working checkpoint.

> **Picking this up in a cloud session?** Start at
> [`training/RETRAIN_TODO.md`](training/RETRAIN_TODO.md) — it's the current
> to-do list (fixing the fine-tuned model's training data, then retraining
> with more data and proper eval) and has the context you need.

---

## 0. Prerequisites (do this first, once)

1. **Environment**: use a machine/notebook with normal internet access and
   a few GB of free disk — your own laptop, Google Colab (free GPU tier),
   or Modal (modal.com, free compute credits). Don't use a locked-down
   sandbox — several models below need to download weights from
   HuggingFace Hub.
2. **Python**: 3.10+.
3. **Free HuggingFace account + token** (needed for the diarization model
   in Phase 4):
   - Sign up at huggingface.co (free).
   - Visit `huggingface.co/pyannote/speaker-diarization-3.1` and click
     "Agree and access repository" (one-time license click-through, no
     payment).
   - Settings → Access Tokens → New token → "read" scope is enough.
   - `export HF_TOKEN=hf_...` in your shell before running anything that
     touches pyannote.
4. **Install base dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   (Installs `faster-whisper`, `soundfile`, `numpy` immediately; `torch`,
   `speechbrain`, `pyannote.audio` are heavier — install them when you
   reach the phase that needs them, so early phases stay fast to set up.)

No step in this plan uses a paid API. Everything runs on downloaded open
model weights.

---

## Phase 1 — Verbatim ASR baseline (single speaker, no overlap)

**Goal**: prove the transcription core works and correctly preserves
filler words and pauses, before adding any multi-speaker complexity.

**Steps**:
1. `pip install faster-whisper soundfile numpy`
2. Get or record a short single-speaker test clip (a few sentences,
   include some "um"/"uh" and a deliberate pause). You can also generate
   one offline for free with `espeak-ng` + `sox` if you don't have a mic
   handy — useful for quick sanity checks, not for real accuracy testing.
3. Run `src/asr_baseline.py <audio_file>` and inspect the output:
   - Confirm filler words ("um", "uh") appear in the transcript instead of
     being silently dropped.
   - Confirm `[pause Xs]` markers appear where you left gaps.
4. **Checkpoint**: you have a working, verbatim, pause-aware single-speaker
   transcriber. Everything after this reuses it per-speaker-stream.

**Common issues**: if fillers still get stripped, double check
`condition_on_previous_text=False` and the `suppress_tokens` setting are
actually taking effect for your `faster-whisper` version.

---

## Phase 2 — Speech separation for overlapping segments

**Status: DONE (2026-09-20).** Run: `./run.sh separate <audio>`, `./run.sh eval --dir <folder>`.
Deviations from the steps below: default model is `sepformer-libri2mix`
(19.0 dB SI-SDRi vs 7.6 wsj02mix vs 1.5 whamr on a clean 2-voice TTS test;
12.4 dB on a real-voice + TTS overlap); test mixtures are built with
`./run.sh mix` (ground truth included) instead of downloading LibriMix, though
`eval_separation.py` also accepts a LibriMix tree. Still to do: SI-SDR on real
room recordings / a LibriMix subset for the report.

**Goal**: when two people talk at once, split the mixed waveform into
isolated per-speaker estimates before ASR ever sees it.

**Steps**:
1. `pip install speechbrain torch torchaudio`
2. Download a small overlap test set: **LibriMix** (2-speaker synthetic
   mixtures) is the standard sanity-check dataset — small subset is enough
   to start.
3. Load the pretrained SepFormer model (`src/separation.py`,
   `speechbrain/sepformer-whamr`) and run it on a handful of LibriMix
   mixtures.
4. Evaluate separation quality with **SI-SDR** (SpeechBrain includes
   utilities for this) — this tells you how cleanly the two voices were
   pulled apart, independent of ASR.
5. Feed a separated stream into your Phase 1 transcriber and manually
   check the transcript makes sense (no need for automated WER yet).
6. **Checkpoint**: given a short mixed-speech clip, you can produce two
   isolated audio streams and transcribe each one separately.

**Note**: only run separation on segments actually flagged as
overlapping (Phase 3 gives you that flag) — running it on clean
single-speaker audio wastes compute and can hurt quality.

---

## Phase 3 — Overlap detection + VAD

**Status: DONE (2026-09-20).** Run: `./run.sh overlap <audio>`; logic in
`src/overlap.py` (`build_segments`, unit-tested).
Deviation from the steps below: overlap flags and speech/silence boundaries
are derived from pyannote's diarization turns (its segmentation network
already models simultaneous speakers), so no separate Silero-VAD or
overlapped-speech-detection run is needed. On the synthetic mixture the
detected overlap was 5.01-7.56s vs ground truth 5.0-7.5s. Overlaps < 0.4s
are folded into the neighbouring speaker (too short to separate reliably).

**Goal**: decide, for each moment of audio, whether it's silence,
one speaker, or overlapping speech — this routes audio to the right
downstream path (skip separation vs. run separation).

**Steps**:
1. `pip install pyannote.audio` (uses the same `HF_TOKEN` as Phase 4).
2. Run `silero-vad` (or pyannote's VAD) over a test recording to strip
   silence.
3. Run pyannote's overlapped-speech-detection model over the same
   recording; inspect where it flags overlap.
4. Wire the two together: VAD → speech segments → overlap flag per
   segment → route overlapping segments to Phase 2, everything else
   straight to Phase 1's transcriber.
5. **Checkpoint**: a single function that takes room audio and emits a
   list of `(start, end, is_overlapping)` segments.

---

## Phase 4 — Diarization (consistent Speaker_N labels)

**Goal**: assign a consistent speaker identity (`Speaker_1`, `Speaker_2`,
...) to every segment — same person always gets the same label,
throughout the session, for a completely arbitrary set of speakers.

**Steps**:
1. Confirm `HF_TOKEN` is set and you've accepted the model license
   (Prerequisites, step 3).
2. Run `src/diarization.py`'s `Diarizer` on a real multi-speaker test
   recording — start with a 2-3 speaker clip before jumping to 10.
   Good test sets: **AMI Meeting Corpus** or **CHiME-6** (both have
   ground-truth speaker labels so you can measure accuracy).
3. Compute **DER (Diarization Error Rate)** against the ground truth —
   `pyannote.metrics` has this built in.
4. Test with increasing speaker counts (3 → 5 → 10) and record how DER
   degrades — this number matters a lot for your final report/demo
   scoping.
5. **Checkpoint**: given room audio, you get back consistent `Speaker_N`
   labeled segments, with a measured DER on a known benchmark.

---

## Phase 5 — Optional enrollment (real names)

**Goal**: let people optionally register their voice ahead of time so
their `Speaker_N` label gets replaced with their real name; unenrolled
speakers stay as `Speaker_N`.

**Steps**:
1. Use `speechbrain/spkrec-ecapa-voxceleb` (wired in
   `src/embeddings.py` / `src/enrollment.py`'s `EnrollmentGallery`) to extract an embedding
   from a short (3-5s) clean enrollment clip per person.
2. For each diarized cluster from Phase 4, compute its centroid embedding
   and compare against the enrollment gallery via cosine similarity.
3. Tune the similarity threshold on a held-out set: too low → wrong name
   matches; too high → real matches get missed and fall back to
   `Speaker_N` unnecessarily. Default is now 0.5 (0.75 risks missing real matches on noisy audio) - adjust based on your
   own test enrollments.
4. **Checkpoint**: enrolled speakers show up by name; everyone else shows
   up as `Speaker_N`, with no crash or wrong assumptions either way.

---

## Phase 6 — Full offline pipeline

**Goal**: combine everything into one file-in, transcript-out pipeline
with correct chronological ordering (this is what makes interruptions
show up as separate, correctly interleaved lines).

**Steps**:
1. Run `src/pipeline.py <room_audio.wav>` end to end on a real multi-
   speaker recording.
2. Verify: overlapping segments got separated, each stream got
   transcribed verbatim, diarization labels are consistent, enrolled
   names (if any) show up correctly, and all lines are sorted by true
   start time.
3. Compute end-to-end metrics on a benchmark set (AMI/LibriCSS):
   - **DER** (with overlap counted)
   - **WER**, disfluency-aware (don't penalize retained fillers)
   - **Speaker consistency** (does a label ever drift mid-session)
4. **Checkpoint**: a working offline demo — feed in a recorded multi-
   speaker meeting, get back a readable, correctly-ordered transcript.

---

## Phase 7 — Real-time streaming version

**Status: BUILT, not real-time on this hardware (2026-09-20).** Code:
`src/streaming.py` (StreamingTranscriber, SpeakerRegistry), `src/server.py`
(aiohttp WebSocket server), `web/index.html`. Deviations from the steps
below: the frontend is one dependency-free HTML page instead of React, and
the whole window is re-diarized each hop (16 s window, 4 s hop, 3 s guard)
rather than incrementally, with labels stabilised by temporal continuity +
voice prints. Measured latency 14.5 s mean / 25 s max (target was "a few
seconds"); the bottleneck is Whisper-small on CPU. Next: try `--model base`
(needs a download), an Apple-GPU Whisper, or CUDA; then test with a real mic.

**Goal**: turn the offline pipeline into a live system with acceptable
latency (target: a few seconds from speech to displayed line, not true
sub-second — see the scoping note at the bottom).

**Steps**:
1. Replace the "read whole file" loop with a sliding buffer over a live
   mic/array feed (e.g. 3s windows, 1s hop).
2. Run VAD + overlap detection per window; only re-diarize/re-separate
   the new audio, not the whole session, for latency.
3. Maintain diarization state across windows (running cluster centroids)
   so `Speaker_N` labels stay consistent as the buffer slides forward —
   this is the trickiest part of going from offline to streaming.
4. Stream transcript lines out over a WebSocket to a simple frontend
   (React, since that's a stack you already know) that appends new lines
   as they arrive, sorted by timestamp.
5. **Checkpoint**: a live demo — speak into a mic (or play a multi-speaker
   recording as if live) and watch labeled transcript lines appear with a
   few seconds of delay.
6. **Optional**: deploy the heavier separation/diarization steps as Modal
   serverless functions if your local machine can't keep up in real time;
   use n8n only if you want external orchestration (routing transcript
   lines to Slack/a dashboard/a database) — neither is required for the
   core pipeline to work.

---

## Datasets reference

| Purpose | Dataset |
|---|---|
| Speech separation training/eval | LibriMix, WHAM! |
| Diarization eval | AMI Meeting Corpus, CHiME-6, LibriCSS |
| Enrollment/embeddings | VoxCeleb1/2 |
| Accent robustness | Common Voice (accented subsets), EdAcc |
| Disfluency/verbatim retention | Switchboard, Fisher English |

## Evaluation summary

| Metric | Measures | Tool |
|---|---|---|
| DER | Diarization accuracy, overlap-inclusive | `pyannote.metrics` |
| WER (disfluency-aware) | ASR accuracy without penalizing kept fillers | custom scoring, don't use stock WER |
| SI-SDR | Separation quality | `speechbrain`/`asteroid` utilities |
| Speaker consistency | Label drift across a session | manual/scripted check |
| Latency | Mic → displayed line delay | timestamp logging in Phase 7 |

## Scoping note (be explicit about this in your report)

True simultaneous 10-way overlap separation is research-frontier, not a
solved problem even in SOTA systems. Scope your headline demo to "10
people present, realistic conversational overlap (mostly 1 speaker, brief
2-speaker overlaps during interruptions)" rather than promising robust
10-way simultaneous separation — that's how real conversations behave
anyway, and it's an honest, achievable target for a project at this
level.

---

## Local environment status (as of 2026-09-20)

- The `venv/` folder in the project root is a broken copy from another
  machine (its `pyvenv.cfg` points at `/Users/sampathbageyawadi/...`, no
  python binary). It is unused and can be deleted. The working environment is
  `.venv/` (Python 3.11), created by `./setup.sh`.
- Installed and verified: `faster-whisper`, `torch 2.14`, `torchaudio`,
  `speechbrain 1.1.1`, `pyannote.audio 4.0.7`. No `ffmpeg@7` needed anymore:
  audio is decoded in-process and passed to pyannote as a waveform.
- `diarization.py` uses pyannote 4.x API (`token=`, `output.speaker_diarization`).
  The three gated-license requirement is documented in README.md.
- File layout is now `src/`, `scripts/`, `tests/`, `data/` (see README.md).
- `pipeline.py` now runs Phases 1-6 together: diarization, overlap routing,
  separation with voice-based stream-to-speaker matching, optional enrollment,
  chronological merge. 32 unit tests pass (`./run.sh test`).

### Verified results so far

| Check | Data | Result |
|---|---|---|
| Overlap detection | TTS 2-voice, true overlap 5.0-7.5s | detected 5.01-7.56s |
| Separation (SI-SDRi) | TTS 2-voice | 19.0 dB (libri2mix) |
| Separation (SI-SDRi) | real voice + TTS, 6.5s overlap | 12.4 dB |
| Enrollment | real voice enrolled, mixed with TTS | matched correctly, other speaker stayed Speaker_N |
| Filler retention | TTS "so I was thinking, um, ..." | "um" kept after `suppress_tokens=[]` fix |

### Streaming results (Phase 7)

| Check | Result |
|---|---|
| Stub-model tests (commit logic, label stability, no duplicates/losses) | pass; 3 deliberate code mutations each make a test fail |
| Label stability on real-voice + TTS clip | Speaker_1 / Speaker_2 stable across all windows after fixing 3 issues (max_speakers hint instead of exact count; temporal anchoring; deferring speakers seen only in the guard region) |
| Server: abort mid-stream, then new session | recovers (was permanently "busy" before the cleanup fix) |
| Latency at real-time pacing | mean 14.5 s, max 25 s: NOT real-time on this CPU |
| Pipeline-level scoring (`tts_demo`, offline) | DER 0.0%, cpWER 0.0% with separation vs 45.1% without; filler recall 1/1 |

### Open items

- **Streaming latency** (above): needs a faster ASR path or hardware.
- **Web page microphone path** untested with a real mic (JS helpers unit-tested in node; page renders in headless Chrome).
- **Real multi-speaker audio** has not been tested: everything above uses
  one real voice plus TTS/synthetic mixes. Needed before Phase 4/6 numbers
  mean anything (DER, WER, speaker consistency).
- Diarization can miss overlap (seen once: second speaker unseen for ~1s),
  which the pipeline then attributes to one speaker.
- Enrollment threshold (0.5) untuned; enrolled-speaker test used one person.
- Filler retention only checked on TTS; verify on a real "um/uh" recording,
  and compare `--verbatim-prompt`.
- Intermittent `recursive_mutex lock failed` at interpreter exit seen once
  (after output was printed; not reproduced in 3 later runs). Suspected cause:
  torchcodec (pulled in by pyannote) loads Homebrew ffmpeg alongside PyAV's own
  copy - the harmless-looking "Class AVFFrameReceiver is implemented in both"
  warnings.

## Verbatim fine-tuning and demo (2026-09-21)

Built after Phase 7: public-data fine-tuning of Whisper-small on AMI SDM, a converter to
MLX, evaluation tooling, and a demo mode. Results are in `output/` and in `paper/main.pdf`.

### What was built
- `training/download_ami.py`, `prepare_ami.py` (meeting timelines -> ~28 s windows with
  Whisper timestamp tokens), `finetune_whisper.py` (cached-encoder fine-tuning of the top
  2 encoder layers + decoder), `hf_to_mlx.py` (verified against the stock MLX weights:
  479 tensors, max diff 0), `asr_eval.py`, `prepare_meetings.py`, `run_all.sh`.
- Demo: `./run.sh demo` (server plays bundled recordings while the browser plays the audio;
  stock vs fine-tuned switch; ground-truth column; live stats).

### Findings worth remembering
- Stock Whisper-small keeps 0 of 96 fillers on AMI even with suppression off.
- The fine-tuned model repeats itself on short clips; a temperature fallback plus a token
  cap proportional to clip length (8 tokens/s + 24) fixed most of it: utterance WER
  47.6% -> 42.5%, filler recall 60% -> 74% (stock: 38.8% WER, 0% fillers).
- On separated overlap streams the fine-tuned model hallucinates lone "mm"/"mm-hmm";
  filler-only utterances are dropped on separated audio (`pipeline._usable`).
- My first DER reference (utterance-level AMI data) omitted 78 s of speech in ES2004c and
  gave a spurious 57% DER; `diar_turns` from the complete annotation fixes it (9.4%).
- Training on MPS: driver memory grows with micro-batch (5.0/6.2/8.4 GB at 1/2/4);
  batch 4 swapped a 17 GB Mac to a crawl. Use batch 2 x accum 4.
- Overnight the Mac ran ~10x slower than normal (evals took ~3.8 h instead of 3 min);
  cause unknown (sleep/thermal), so wall-clock "real-time factor" figures from that run are void.

### Open
- Meeting-level scores (`output/meeting_results.txt`) are on 3 excerpts only.
- Fine-tuned output is lower-case and unpunctuated (punctuation restoration not built).
- Microphone input in the browser is still untested with real hardware.
