#!/usr/bin/env bash
# Unattended: finish fine-tuning, convert to MLX, and evaluate against the stock model.
#
#   ./training/run_all.sh
#
# Safe to start while training is already running (it waits for it) or when it
# is not (it resumes from the last saved checkpoint). Keeps the Mac awake
# (caffeinate), writes everything to output/overnight_<timestamp>.log and a
# short summary to output/overnight_summary.txt. A failing step is logged and
# the script carries on with the next one.
#
#   SKIP_TRAIN=1 ./training/run_all.sh     only convert + evaluate
#   TOTAL_STEPS=1350 DONE_STEPS=200        used only when resuming (see below)

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [ -z "${CAFFEINATED:-}" ]; then
  CAFFEINATED=1 exec caffeinate -dimsu "$0" "$@"
fi

mkdir -p output
LOG="output/overnight_$(date +%Y%m%d_%H%M).log"
SUMMARY="output/overnight_summary.txt"
exec > >(tee -a "$LOG") 2>&1

set -a; [ -f .env ] && source .env; set +a
export PYTHONPATH=src TOKENIZERS_PARALLELISM=false
PY=.venv/bin/python
CKPT=models/whisper-small-ami
MLX_OUT=models/whisper-small-ami-mlx
TOTAL_STEPS="${TOTAL_STEPS:-1350}"
DONE_STEPS="${DONE_STEPS:-200}"       # steps already covered by the checkpoint being resumed from

stamp() { echo; echo "===== $(date '+%H:%M:%S')  $* ====="; }
: > "$SUMMARY"

# ---------------------------------------------------------------- 1. training
if [ -z "${SKIP_TRAIN:-}" ]; then
  if pgrep -f "training/finetune_whisper.py" >/dev/null; then
    stamp "training already running -- waiting for it to finish"
    while pgrep -f "training/finetune_whisper.py" >/dev/null; do sleep 30; done
  elif grep -qs "saved .* @ step ${TOTAL_STEPS}" /tmp/train.log; then
    stamp "training already finished (step ${TOTAL_STEPS} checkpoint exists)"
  elif [ -f "$CKPT/model.safetensors" ]; then
    REMAIN=$((TOTAL_STEPS - DONE_STEPS))
    stamp "resuming from $CKPT for $REMAIN more steps"
    rm -rf models/whisper-small-ami-resume && cp -R "$CKPT" models/whisper-small-ami-resume
    $PY training/finetune_whisper.py --base models/whisper-small-ami-resume \
        --steps "$REMAIN" --warmup 20 --lr 1e-5 --log-every 10 --out "$CKPT"
  else
    stamp "no checkpoint -- training from scratch"
    $PY training/finetune_whisper.py --steps "$TOTAL_STEPS" --lr 1e-5 --log-every 10 --out "$CKPT"
  fi
fi
grep -E "val loss|saved" /tmp/train.log 2>/dev/null | tail -8

# ---------------------------------------------------------------- 2. convert
stamp "converting to MLX"
$PY training/hf_to_mlx.py "$CKPT" "$MLX_OUT" || { echo "CONVERSION FAILED"; exit 1; }

# ------------------------------------------- 3. utterance-level ASR evaluation
for M in small "$MLX_OUT"; do
  stamp "AMI utterances (300, held-out meetings): $M"
  $PY training/asr_eval.py --model "$M" -n 300 --normalize --save "output/utt_$(basename "$M").json" \
    2>&1 | grep -E "^model|^WER|^filler|^real-time" | tee -a "$SUMMARY"
done

# ------------------------------------------------- 4. full-meeting pipeline scoring
for MEET in ES2004c_300s IS1009b_300s EN2002b_300s; do
  for M in small "$MLX_OUT"; do
    stamp "pipeline on $MEET with ASR model: $M"
    echo "--- $MEET  |  ASR: $M" >> "$SUMMARY"
    ./.venv/bin/python scripts/evaluate.py "data/ami_meetings/$MEET.wav" \
        --reference "data/ami_meetings/$MEET.json" --num-speakers 4 --model "$M" \
      2>&1 | tee "output/meeting_${MEET}_$(basename "$M").txt" \
           | grep -E "^DER|^consistency|^cpWER|^filler recall" | tee -a "$SUMMARY"
  done
done

stamp "ALL DONE"
echo; echo "Summary ($SUMMARY):"; cat "$SUMMARY"
