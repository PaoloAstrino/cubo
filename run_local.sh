#!/usr/bin/env bash
# Simple cross-platform local startup for unskilled users (Linux/macOS)
set -euo pipefail

echo "== CUBO Local Quickstart (bash) =="
# Create virtualenv if missing
if [ ! -d ".venv" ]; then
  echo "Creating virtualenv..."
  python -m venv .venv
fi

# Activate
# shellcheck source=/dev/null
. .venv/bin/activate

# Install (editable) and frontend deps (idempotent)
python -m pip install -e .
npm run install:frontend

# Start fullstack (uses scripts/start_fullstack.py)
python scripts/start_fullstack.py
