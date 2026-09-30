# Run of 2026-09-30 (user's M3 MacBook Air) — produced by the OLD training/run_all.sh

Evidence it is the pre-roadmap script (from `main`): it resumed an earlier checkpoint for
1150 steps (= old defaults TOTAL_STEPS 1350 - DONE_STEPS 200), and the output has the old
section headers, no confidence intervals and no tcpWER. So the fine-tuned rows below are
**not** the planned retrain: no weight decay, no augmentation, no val-WER checkpoint
selection, and the earlier checkpoint it resumed from may have been trained on the old
merged-speaker windows. **Stock rows are a valid baseline** (they don't depend on training).

## Utterance level (300 held-out AMI SDM utterances, level-normalised, MLX)

| model | WER (fillers ignored) | verbatim WER | filler recall | RTF |
|---|---|---|---|---|
| stock whisper-small | 39.7% (1296/3265) | 42.3% (1362/3222) | 0.0% (0/96) | 0.26 |
| fine-tuned (old script) | 43.6% (1424/3265) | 47.4% (1526/3222) | 69.8% (67/96) | 0.36 |

## Full pipeline on the 3 AMI excerpts (4 speakers, separation on)

DER does not depend on the ASR model, so it is the same for both rows.

| excerpt | DER (miss / FA / confusion) | cpWER stock | cpWER tuned | filler recall stock | tuned |
|---|---|---|---|---|---|
| ES2004c | 9.4% (4.9 / 2.5 / 2.0) | 60.6% | 65.3% | 1/14 | 14/14 |
| IS1009b | 25.4% (2.9 / 1.3 / 21.2) | 55.7% | 52.4% | 1/39 | 33/39 |
| EN2002b | 23.5% (15.4 / 1.6 / 6.5) | 55.6% | 61.0% | 0/30 | 21/30 |

Per-speaker label consistency (same for both): ES2004c 72-94%, IS1009b 52/53/86/88%,
EN2002b 72-83%.

## Reading
- Same trade-off as the first fine-tune (PLAN.md 2026-09-21): fillers go from ~0% to ~70%,
  but word accuracy gets ~4 points worse at utterance level and worse on 2 of 3 meetings.
- The fine-tuned model is ~40% slower (RTF 0.36 vs 0.26), consistent with the repetition
  guard's temperature-fallback retries firing -- a sign of repetition loops.
- cpWER (55-61%) is far above utterance WER (~40%). The ASR is not the main cause:
  IS1009b has 21% speaker confusion (two speakers ~52% consistent, i.e. merged/split),
  EN2002b misses 15% of speech (41% overlap, 24 s of 3+ speakers), and ES2004c's text
  reference lacks 86 s of speech, so correct words there count as insertions
  (results/reference_stats.md).
