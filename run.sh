#!/usr/bin/env bash
# One entry point for everything in the project.
#
#   ./run.sh [pipeline] <audio>       full transcription pipeline (default command)
#   ./run.sh overlap   <audio>        Phase 3: show speech / overlap segments
#   ./run.sh separate  <audio>        Phase 2: split a 2-speaker mix into two streams
#   ./run.sh asr       <audio>        Phase 1: single-speaker verbatim ASR
#   ./run.sh mix   <a.wav> <b.wav>    build a 2-speaker test mixture with ground truth
#   ./run.sh eval  --dir <folder>     Phase 2: SI-SDR evaluation of the separator
#   ./run.sh score <audio> --reference truth.json   Phases 4/6: DER, cpWER, filler recall
#   ./run.sh demo                     start the demo (server + browser page, bundled recordings)
#   ./run.sh serve                    same server without opening the browser
#   ./run.sh stream <audio>           Phase 7: stream a file to the running server (no mic needed)
#   ./run.sh live  <audio>            Phase 7: replay a file through the streaming pipeline, no server
#   ./run.sh test                     unit tests (no models / network needed)
#
# Extra flags are passed straight through, e.g.:
#   ./run.sh pipeline data/mixtures/demo/mix.wav --num-speakers 2 --out output/demo.txt
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

if [ ! -x .venv/bin/python ]; then
  echo "No .venv found -- run ./setup.sh first." >&2
  exit 1
fi

# .env is read by src/config.py too; sourcing here keeps HF_TOKEN visible to
# any subprocess as well.
if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
PY=.venv/bin/python

cmd="${1:-pipeline}"
case "$cmd" in
  pipeline|overlap|separate|asr|mix|eval|score|demo|serve|stream|live|test) shift ;;
  *) cmd="pipeline" ;;   # backwards compatible: ./run.sh file.wav
esac

case "$cmd" in
  pipeline) exec $PY src/pipeline.py "${@:-data/test_utterance.wav}" ;;
  overlap)  exec $PY src/overlap.py "$@" ;;
  separate) exec $PY src/separation.py "$@" ;;
  asr)      exec $PY src/asr_baseline.py "$@" ;;
  mix)      exec $PY scripts/make_mixture.py "$@" ;;
  eval)     exec $PY scripts/eval_separation.py "$@" ;;
  score)    exec $PY scripts/evaluate.py "$@" ;;
  demo)     (until curl -s -o /dev/null http://127.0.0.1:8000/; do sleep 2; done; open http://127.0.0.1:8000) &
            exec $PY src/server.py "$@" ;;
  serve)    exec $PY src/server.py "$@" ;;
  stream)   exec $PY scripts/stream_client.py "$@" ;;
  live)     exec $PY src/streaming.py "$@" ;;
  test)     exec $PY -m pytest tests -q "$@" ;;
esac
