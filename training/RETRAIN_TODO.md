# Retraining TODO — cloud handoff

Context: this project fine-tunes Whisper-small on AMI meeting audio for
verbatim output (kept fillers). The first fine-tuning run (see
`PLAN.md`, section "Verbatim fine-tuning and demo") worked but had a real
problem: training windows merged **every** speaker's words into one
chronological transcript, while the live pipeline's ASR only ever sees
**one** speaker's audio at inference (a clean non-overlap segment, or a
SepFormer-separated stream). That mismatch is the reason the fine-tuned
model's utterance-level WER got *worse* than stock Whisper (42.5% vs
38.8%) even though filler recall improved a lot (74% vs 0%). This doc is
the fix list, in the order to do it. Moved to a cloud session because the
first run overloaded a 17 GB M3 laptop (swapping, one 10x-slowdown
incident of unknown cause).

Read `README.md` for the general project layout and `PLAN.md` for full
history before starting. Run `./setup.sh` then `./run.sh test` first
(60+ unit tests, no models/network needed) to confirm the environment is
sane before touching any of this.

## Already done (local, before the move to cloud)

- [x] **`training/prepare_ami.py` rewritten.** Windows are now built only
  from single-speaker stretches, using the exact same routing logic the
  live pipeline uses at inference (`src/overlap.build_segments`), so a
  training pair is always "this audio -> this one speaker's words". Also
  drops windows with a pathological word-repeat run
  (`has_pathological_repeat`) — likely source of the fine-tuned model's own
  repetition loops at inference.
- [x] **Items 2-5 below are implemented** (2026-09-30, cloud session, verified
  only with unit tests + a tiny random-weight Whisper smoke test -- no real
  training run yet): `--resume` with full trainer state, `--weight-decay`
  (default 0.01), latent masking on the train set only (`training/train_utils.py`),
  and best-checkpoint selection by validation WER (`<out>` = best WER,
  `<out>-last` = newest + `trainer_state.pt`). The old `WindowDataset` no longer
  exists in the script. Extra deps: `pip install -r requirements-train.txt`.
- [x] **Item 6 driver** (`training/sweep.py`) and **item 10** (bootstrap CIs, `src/bootstrap.py`) are
  implemented and unit-tested; the sweep has not been run.
- [ ] Still **not done**: items 1, 7-9, 11-12 and running 6 (need HuggingFace/AMI access and compute).
  Re-download AMI, rebuild windows with the new script, rebuild the encoder cache,
  then proceed.

## How to run it (training machine: 13" MacBook Air, M3, 16 GB, fanless)

```bash
./setup.sh && .venv/bin/pip install -r requirements-train.txt
./run.sh test                                   # must pass first
./training/run_all.sh                           # data -> cache -> stock baseline -> train -> convert -> eval
SWEEP=1 STAGES="sweep" ./training/run_all.sh    # optional: pick LR / TRAIN_LAYERS first
```

Every stage skips itself when its output exists and training resumes from
`models/whisper-small-ami-last/trainer_state.pt`, so an interrupted run is just
re-run. The Air has no fan and will throttle under hours of load (likely the
unexplained 10x slowdown last time): plugged in, hard surface, lid open. The
script refuses to build the encoder cache without enough free disk (~2.3 GB per
1,000 windows); lower `TRAIN_SHARDS` if it stops there.

## TODO, in order

1. **Re-run data prep with the new single-speaker windowing.**
   ```bash
   python training/download_ami.py --train-shards 14 --val-shards 2 --test-shards 3
   python training/prepare_ami.py --split train
   python training/prepare_ami.py --split validation
   ```
   Shard counts roughly doubled vs. the first run (was 7/1/2) for more
   meeting diversity — adjust down if disk/time is tight (`TOTALS` in
   `download_ami.py` shows the max per split: train 27, validation 5,
   test 4). Expect noticeably *fewer* total windows than the old run even
   with more shards, because overlapping stretches are now excluded —
   that's expected and correct, not a bug.

2. **Save/restore full training state, not just weights**
   (`training/finetune_whisper.py`). Currently `model.save_pretrained()`
   only saves weights; a resume after interruption restarts the optimizer
   from scratch, which is what happened last time. Add, at every
   checkpoint save (around the `model.save_pretrained(out_dir)` call):
   ```python
   torch.save({"step": step, "optimizer": opt.state_dict(), "scheduler": sched.state_dict()},
              out_dir / "trainer_state.pt")
   ```
   Add a `--resume` flag to `main()`: if set and `(Path(args.out) /
   "trainer_state.pt").exists()`, load the model from `args.out` instead
   of `args.base`, and after building `opt`/`sched`, load their state
   dicts and set the starting step from the saved one before entering the
   training loop.

3. **Add weight decay.** In `finetune_whisper.py::main()`, change
   `torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)` to
   `weight_decay=0.01` (or expose `--weight-decay`, default 0.01). No
   other code changes needed.

4. **Turn on augmentation on the path actually used.** The old
   `WindowDataset(augment=True)` is dead code — training goes through
   `CachedDataset`, which has none. Add masking directly to the cached
   encoder activations (analogous to wav2vec2's latent-space masking,
   since these are post-encoder hidden states, not raw mel features):
   ```python
   def augment_hidden(h: torch.Tensor, n_time=2, time_w=100, n_chan=1, chan_w=64) -> torch.Tensor:
       h = h.clone()
       T, C = h.shape
       for _ in range(n_time):
           if T > time_w:
               t0 = random.randint(0, T - time_w - 1)
               h[t0:t0 + time_w] = 0.0
       for _ in range(n_chan):
           if C > chan_w:
               c0 = random.randint(0, C - chan_w - 1)
               h[:, c0:c0 + chan_w] = 0.0
       return h
   ```
   Apply it in `CachedDataset.__getitem__` only when `self.augment` is
   True; construct the train dataset with `augment=True` and the
   validation dataset with `augment=False`. Delete the now-unused
   `WindowDataset` class (and its `augment`/`--speed` gain-jitter code) so
   it stops looking like an active feature.

5. **Select the checkpoint by validation WER, not just loss.** Add a
   small greedy-decode-and-score function that runs on maybe 20-40
   validation windows every `--val-every` steps (reuse `src/scoring.wer`
   against each window's reference text), log it next to val loss, and
   only overwrite the saved checkpoint when it improves on the best-seen
   WER so far (keep a `best_wer` variable across the loop; still save a
   `-last` checkpoint too if you want a fallback). This avoids blindly
   trusting the final step regardless of whether it overfit.

6. **Small LR / layer-count sweep before the long run.** Wrap `main()`'s
   body (or write a thin driver script `training/sweep.py`) over something
   like `lr in [5e-6, 1e-5, 2e-5]` x `train_layers in [2, 4]`, ~150-300
   steps each, logging final val WER (from step 5) to
   `output/sweep_results.csv`. Pick the winner before committing to the
   long run. This can run unattended, similar to how
   `training/run_all.sh` already chains steps.

7. **Full training run with all of the above wired in.** Expect this to
   take a while — scale `--steps` to the new (larger, filtered) dataset
   size; log progress the same way the original run did
   (`step N/TOTAL | loss ... | eta ...`).

8. **Convert to MLX and sanity-check first.**
   ```bash
   python training/hf_to_mlx.py models/whisper-small-ami models/whisper-small-ami-mlx
   ```
   `training/hf_to_mlx.py` already has a `--verify` mode against the stock
   MLX weights (confirmed exact match: 479 tensors, max diff 0, in the
   first run) — no changes needed there.

9. **Re-check whether the inference-time patches are still needed.**
   `src/asr_baseline.py`'s `MLXTranscriber` currently has a temperature
   fallback + a token cap proportional to clip length
   (`8 * len(audio)/16000 + 24`) to stop repetition loops, and
   `src/pipeline.py::_usable` drops filler-only utterances on separated
   streams (patch for the old model hallucinating lone "mm" on separator
   residue). Once steps 1 and 4 are done, re-run
   `training/asr_eval.py` with the guard effectively disabled (temporarily
   set `temperature=0.0` only, no retry, and a large `sample_len`) and see
   if loops still happen. If they're gone, that's a stronger result (a
   model that doesn't need the guard) — but don't remove the guard code
   itself; it's a legitimate safety net either way.

10. **Bootstrap confidence intervals**, so numbers aren't bare point
    estimates:
    - `training/asr_eval.py`: bootstrap-resample the per-utterance results
      (N=1000, with replacement) and report a 95% CI on WER and filler
      recall alongside the point estimate.
    - `scripts/evaluate.py`: do the same at the turn/word level for cpWER
      per meeting.

11. **Expand meeting-level eval past 3 excerpts.** `training/
    prepare_meetings.py` already takes arbitrary meeting IDs — pull 3-5
    more from the held-out test shards (check which meetings are in the
    downloaded `data/ami_meetings/sdm/*.parquet` and
    `data/ami/sdm/test-*.parquet` shards; with `--test-shards 3` from step
    1 there should be more than the original 3 meetings available). Re-run
    the meeting-scoring loop (see `training/run_all.sh` for the pattern:
    for each meeting x {stock, tuned}, run `scripts/evaluate.py` and
    collect DER / cpWER / filler recall into one summary file).

12. **Update `PLAN.md`** with the final numbers (a new dated section,
    following the style of the existing "Verbatim fine-tuning and demo"
    entry) once 1-11 are done. **Do not touch `paper/main.tex`** — the
    user said the paper will be redone separately once these numbers are
    final.

## Notes for whoever runs this

- The `TranscriptionPipeline`'s `asr_backend="auto"` picks MLX (Apple
  GPU) only on Apple silicon; on a cloud Linux box it'll fall back to
  `ctranslate2` on CPU (or CUDA if available — check `src/config.py:
  pick_device()`). Training itself (`training/finetune_whisper.py`) uses
  plain PyTorch/`pick_device()` too, so it should pick up CUDA
  automatically if the cloud environment has a GPU; MPS-specific bits
  (`torch.mps.empty_cache()` calls) are guarded by `device == "mps"` and
  are harmless no-ops elsewhere.
- `data/ami/` and `data/ami_meetings/sdm/` are gitignored (large,
  regenerable from the public CC-BY-4.0 dataset) — re-run the download
  scripts, don't expect them to already be present after cloning.
- `HF_TOKEN` needs to be in `.env` (gitignored) — see README's
  "HuggingFace access" section for the free one-time account setup.
- 60+ unit tests in `tests/` cover pure logic (no models/network); run
  `./run.sh test` after any code change before spending compute on a real
  training run.
