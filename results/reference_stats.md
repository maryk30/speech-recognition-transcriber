# AMI excerpt references — sanity check (2026-09-30)

Computed from `data/ami_meetings/*_300s.{json,wav}` (no models). All three are
300.0 s, 16 kHz mono, 4 speakers; speaker sets of `turns` and `diar_turns` agree;
no out-of-bounds or empty turns.

| excerpt | speech | overlap | 3+ speakers | words | fillers | text turns missing from `diar_turns` (speaker-s) | `diar_turns` speech with no text turn (speaker-s; wall-clock) |
|---|---|---|---|---|---|---|---|
| ES2004c | 262 s | 26 s (10%) | 1 s | 610 | 14 | 3.4 s | 86.3 s; 78.4 s |
| IS1009b | 275 s | 13 s (5%) | 0 s | 859 | 39 | 0.9 s | 7.9 s; 7.7 s |
| EN2002b | 260 s | 106 s (41%) | 24 s | 1072 | 30 | 11.8 s | 6.2 s; 1.3 s |

What this means for the baseline:
- **ES2004c**: the text turns omit 78 s of speech (wall-clock; 86 speaker-seconds) that `diar_turns` has. This is the
  gap behind the old spurious 57% DER; DER must use `diar_turns`
  (`scoring.load_diarization_reference` already does). cpWER on this excerpt
  under-counts reference words for those 78 s, so its hypothesis words there
  score as insertions — read ES2004c's cpWER with that in mind.
- **EN2002b** is the hard case: 41% overlap and 24 s of 3+ simultaneous speakers,
  which the 2-source separator cannot fully recover by design.
- **Only 83 reference fillers in total** (14 + 39 + 30). Filler recall on these
  excerpts will have wide intervals; utterance-level `training/asr_eval.py`
  (hundreds of utterances) is the better filler measure.
