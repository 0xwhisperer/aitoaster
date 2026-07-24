#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d venv ]; then
  python3 -m venv venv
  ./venv/bin/pip install --quiet --upgrade pip
  ./venv/bin/pip install --quiet -r requirements.txt
fi

# load .env if present (currently just FIXER_WATERMARK_SEED) - never committed,
# see .gitignore. Absence is fine; watermark.py falls back to a built-in
# default seed with a clear warning rather than failing.
if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

exec ./venv/bin/python3 -m app.server
