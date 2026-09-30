#!/usr/bin/env bash
# The whole retrain, end to end, on the 16 GB MacBook Air (M3). Every stage is
# skipped if its output already exists, and training resumes from its last
# saved state, so after a crash, sleep or reboot just run it again.
#
#   ./training/run_all.sh                 everything below, in order
#   STAGES="baseline" ./training/run_all.sh          just some stages
#   STAGES="train convert eval" ./training/run_all.sh
#   SWEEP=1 ./training/run_all.sh         also run the lr x layer sweep before training
#
# Stages: data  cache  baseline  sweep  train  convert  eval
#
# The MacBook Air has no fan: under a long load it throttles, which is the likely
# cause of the "~10x slower overnight" run in PLAN.md. Keep it plugged in, on a hard
# surface, lid open, nothing else heavy running; watch the s/step figure in the log.
# Memory: micro-batch 2 x accum 4 (batch 4 swapped the 16 GB machine last time).
#
# Log: output/run_<timestamp>.log. Numbers: results/ (committed) + output/ (not).

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [ "$(uname)" = "Darwin" ] && [ -z "${CAFFEINATED:-}" ]; then
  CAFFEINATED=1 exec caffeinate -dimsu "$0" "$@"      # keep the Mac awake for the whole run
fi

mkdir -p output results
LOG="output/run_$(date +%Y%m%d_%H%M).log"
exec > >(tee -a "$LOG") 2>&1

set -a; [ -f .env ] && source .env; set +a
export PYTHONPATH=src TOKENIZERS_PARALLELISM=false
PY=.venv/bin/python
CKPT=models/whisper-small-ami
MLX_OUT=models/whisper-small-ami-mlx
TOTAL_STEPS="${TOTAL_STEPS:-1400}"
LR="${LR:-1e-5}"
TRAIN_LAYERS="${TRAIN_LAYERS:-2}"
STAGES="${STAGES:-data cache baseline sweep train convert eval}"

want() { [[ " $STAGES " == *" $1 "* ]]; }
stamp() { echo; echo "===== $(date '+%Y-%m-%d %H:%M:%S')  $* ====="; }
die() { echo "FAILED: $*"; exit 1; }

[ -n "${HF_TOKEN:-}" ] || die "HF_TOKEN missing -- put it in .env (see README 'HuggingFace access')"

# ------------------------------------------------------------------ data
if want data; then
  if [ -f data/ami/windows_train.json ] && [ -f data/ami/windows_validation.json ]; then
    stamp "data: windows already built (delete data/ami/windows_* to rebuild)"
  else
    stamp "data: download AMI shards + build single-speaker windows"
    $PY training/download_ami.py --train-shards "${TRAIN_SHARDS:-14}" --val-shards 2 --test-shards 3 || die download
    $PY training/prepare_ami.py --split train || die "prepare train"
    $PY training/prepare_ami.py --split validation || die "prepare validation"
  fi
fi

# ------------------------------------------------------------------ cache (disk check first)
if want cache; then
  N=$($PY -c "import json;print(sum(len(json.load(open(f'data/ami/windows_{s}.json'))) for s in ('train','validation')))")
  NEED_GB=$(( N * 2304 / 1000000 + 1 ))              # 1500 x 768 float16 = 2.3 MB per window
  FREE_GB=$(df -g . 2>/dev/null | awk 'NR==2{print $4}' || df -BG . | awk 'NR==2{gsub("G","",$4);print $4}')
  stamp "cache: $N windows -> ~${NEED_GB} GB encoder cache per layer setting; ${FREE_GB} GB free"
  [ "$FREE_GB" -gt $(( NEED_GB + 10 )) ] || die "not enough disk (need ${NEED_GB} GB + 10 GB headroom); use fewer TRAIN_SHARDS"
  $PY training/finetune_whisper.py --cache-only --train-layers "$TRAIN_LAYERS" || die cache
fi

# ------------------------------------------------------------------ baseline (stock model, before any training)
if want baseline; then
  if grep -qs "^filler recall" results/utt_stock.txt; then
    stamp "baseline: results/utt_stock.txt exists (utterance level)"
  else
    stamp "baseline: stock Whisper-small, utterance level (300 held-out AMI utterances)"
    $PY training/asr_eval.py --model small -n 300 --normalize --save output/utt_stock.json \
      | tee results/utt_stock.txt || die "asr_eval stock"
  fi
  if [ -f results/baseline_stock.json ]; then
    stamp "baseline: results/baseline_stock.json exists (meetings)"
  else
    stamp "baseline: stock Whisper-small, full pipeline on the 3 AMI excerpts"
    $PY scripts/baseline.py --tag stock || die "baseline stock"
  fi
fi

# ------------------------------------------------------------------ sweep (optional)
if want sweep && [ -n "${SWEEP:-}" ]; then
  stamp "sweep: lr x train-layers, 300 steps each -> output/sweep_results.csv"
  $PY training/sweep.py --steps 300 || die sweep
  echo "Pick the best row, then rerun with LR=... TRAIN_LAYERS=... STAGES=\"train convert eval\""
  exit 0
fi

# ------------------------------------------------------------------ train (resumable)
if want train; then
  stamp "train: $TOTAL_STEPS steps, lr $LR, top $TRAIN_LAYERS encoder layers (resumes if interrupted)"
  $PY training/finetune_whisper.py --steps "$TOTAL_STEPS" --lr "$LR" --train-layers "$TRAIN_LAYERS" \
      --log-every 10 --out "$CKPT" --resume || die train
fi

# ------------------------------------------------------------------ convert
if want convert; then
  stamp "convert: $CKPT (best val WER) -> $MLX_OUT"
  $PY training/hf_to_mlx.py "$CKPT" "$MLX_OUT" || die convert
fi

# ------------------------------------------------------------------ eval (tuned vs stock, same data)
if want eval; then
  stamp "eval: fine-tuned, utterance level"
  $PY training/asr_eval.py --model "$MLX_OUT" -n 300 --normalize --save output/utt_tuned.json \
    | tee results/utt_tuned.txt || die "asr_eval tuned"
  stamp "eval: fine-tuned, full pipeline on the 3 AMI excerpts"
  $PY scripts/baseline.py --model "$MLX_OUT" --tag tuned || die "baseline tuned"
  stamp "SUMMARY"
  grep -hE "^model|^WER|^filler" results/utt_stock.txt results/utt_tuned.txt
  cat results/baseline_stock.md results/baseline_tuned.md 2>/dev/null
fi

stamp "done (log: $LOG)"
