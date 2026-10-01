# Retrain of 2026-09-30/10-01 (M3 MacBook Air, new training/run_all.sh)

Single-speaker windows, weight decay 0.01, latent masking, checkpoint chosen by
validation WER, fresh start (not resumed from the old-recipe model). Numbers as
pasted from the run log; per-utterance files (output/utt_*.json) not yet in the repo.

## Utterance level — 300 held-out AMI SDM utterances, 16 meetings, same set for both

| model | WER (fillers ignored) | verbatim WER | filler recall |
|---|---|---|---|
| stock whisper-small | 33.9% [30.3-38.3] | 37.7% [34.3-41.8] | 2.3% (3/128) [0.0-5.3] |
| fine-tuned (new recipe) | **31.5%** [27.6-35.9] | **32.2%** [28.2-36.4] | **64.8%** (83/128) [53.4-74.8] |

First time the fine-tune beats stock on word accuracy *and* keeps fillers
(old recipe: 39.7% -> 43.6%, on a different sample). The per-model intervals
overlap; whether the 2.4-point WER gain is real needs the paired test
(`scripts/compare_utt.py output/utt_stock.json output/utt_tuned.json`).

## Full pipeline — 3 AMI excerpts (DER is ASR-independent)

| excerpt | DER | cpWER stock | cpWER tuned | tcpWER stock | tcpWER tuned | fillers stock | tuned |
|---|---|---|---|---|---|---|---|
| ES2004c | 9.4% | 58.7% [48.8-84.8] | 61.9% [46.7-89.3] | 62.9% | 67.1% | 1/14 | 14/14 |
| IS1009b | 25.4% | 56.0% [32.3-90.4] | 53.8% [23.9-91.3] | 58.4% | 56.0% | 1/39 | 30/39 |
| EN2002b | 23.5% | 55.0% [50.9-63.4] | 56.3% [54.3-70.5] | 57.1% | 62.6% | 0/30 | 21/30 |

Meeting-level differences are well inside the intervals: three 5-minute
excerpts cannot separate the two models. The utterance-level comparison is the
one that can. Why meeting cpWER doesn't follow the utterance gain is open:
candidates are the separated-stream path (where the model was not trained), the
ES2004c reference gap (78 s of speech without text, see reference_stats.md), and
diarization (IS1009b 21% confusion). Re-running the meeting stage now also
records insertions/deletions/substitutions and saves the transcripts
(output/meetings/<tag>/) so this can be checked rather than guessed.
