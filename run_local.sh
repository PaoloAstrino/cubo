#!/usr/bin/env bash
# Simple cross-platform local startup for unskilled users (Linux/macOS)
set -euo pipefail

echo "== CUBO Local Quickstart (bash) =="

# Prefer python3, fall back to python
PY=python3
if ! command -v $PY >/dev/null 2>&1; then
  if command -v python >/dev/null 2>&1; then
    PY=python
  else
    echo "Error: Python not found. Please install Python 3.11+"
    exit 1
  fi
fi

# Check Python version >= 3.11
echo "Checking Python version..."
if ! $PY -c "import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)" >/dev/null 2>&1; then
  echo "Error: Python 3.11 or higher is required. Found: $($($PY -V 2>&1))"
  exit 1
fi

# Check available disk space (require at least 5GB)
echo "Checking available disk space..."
avail_kb=$(df -Pk . | awk 'NR==2 {print $4}')
min_kb=$((5 * 1024 * 1024))
if [ "$avail_kb" -lt "$min_kb" ]; then
  echo "Error: Not enough free disk space. At least 5GB required. Available: ${avail_kb}KB"
  exit 1
fi

# Create virtualenv if missing
if [ ! -d ".venv" ]; then
  echo "Creating virtualenv..."
  $PY -m venv .venv
fi

# Activate
# shellcheck source=/dev/null
. .venv/bin/activate

# Ensure pip/up-to-date
$PY -m pip install --upgrade pip setuptools wheel

# Install package in editable mode
echo "Installing package in editable mode..."
$PY -m pip install -e .

# Frontend deps: prefer pnpm, fall back to npm
# Check Node version (require >= 18)
if command -v node >/dev/null 2>&1; then
  node_major=$(node -v | sed 's/^v//' | awk -F. '{print $1}')
  if [ "$node_major" -lt 18 ]; then
    echo "Error: Node.js v18 or higher is required. Found: $(node -v)"
    exit 1
  fi
else
  echo "Error: Node.js not found. Please install Node.js v18+ (https://nodejs.org/)"
  exit 1
fi

if command -v pnpm >/dev/null 2>&1; then
  echo "Using pnpm to install frontend deps..."
  pnpm install --prefix frontend
else
  if command -v npm >/dev/null 2>&1; then
    echo "pnpm not found; using npm to install frontend deps..."
    npm install --prefix frontend
  else
    echo "Error: neither pnpm nor npm were found. Please install Node.js and npm (https://nodejs.org/)."
    exit 1
  fi
fi

# Start fullstack (uses scripts/start_fullstack.py)
echo ""
echo "Starting CUBO full stack..."
$PY scripts/start_fullstack.py "$@"
