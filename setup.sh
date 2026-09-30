#!/usr/bin/env bash
# One-time environment setup (safe to re-run). Creates .venv and installs
# requirements. Needs Python 3.10-3.13 (3.11 recommended).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PYBIN=""
for candidate in python3.11 python3.12 python3.10 python3.13; do
  if command -v "$candidate" >/dev/null 2>&1; then PYBIN="$candidate"; break; fi
done
[ -n "$PYBIN" ] || { echo "No suitable Python (3.10-3.13) found on PATH." >&2; exit 1; }

[ -d .venv ] || "$PYBIN" -m venv .venv
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install -r requirements.txt

if [ ! -f .env ]; then
  echo "HF_TOKEN=" > .env
  echo "Created .env -- paste your HuggingFace token after HF_TOKEN= (see README.md)."
fi
echo "Setup complete. Try: ./run.sh test"
