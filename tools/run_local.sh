#!/usr/bin/env bash
# Simple cross-platform local startup for unskilled users (Linux/macOS)
set -euo pipefail

# Default: quiet mode. Use --verbose or -v to enable louder output
VERBOSE=0
for arg in "$@"; do
  case "$arg" in
    -v|--verbose)
      VERBOSE=1
      shift
      ;;
  esac
done

log() { if [ "$VERBOSE" -eq 1 ]; then echo "$@"; fi }
err() { echo "$@" >&2; }

log "== CUBO Local Quickstart (bash) =="

# Prefer python3, fall back to python
PY=python3
if ! command -v $PY >/dev/null 2>&1; then
  if command -v python >/dev/null 2>&1; then
    PY=python
  else
    err "Error: Python not found. Please install Python 3.11+"
    exit 1
  fi
fi

# Check Python version >= 3.11
log "Checking Python version..."
if ! $PY -c "import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)" >/dev/null 2>&1; then
  err "Error: Python 3.11 or higher is required. Found: $($($PY -V 2>&1))"
  exit 1
fi

# Check available disk space (require at least 5GB)
log "Checking available disk space..."
avail_kb=$(df -Pk . | awk 'NR==2 {print $4}')
min_kb=$((5 * 1024 * 1024))
if [ "$avail_kb" -lt "$min_kb" ]; then
  err "Error: Not enough free disk space. At least 5GB required. Available: ${avail_kb}KB"
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
if [ "$VERBOSE" -eq 1 ]; then
  $PY -m pip install --upgrade pip setuptools wheel
else
  $PY -m pip install -q --upgrade pip setuptools wheel
fi

# Install package in editable mode
log "Installing package in editable mode..."
if [ "$VERBOSE" -eq 1 ]; then
  $PY -m pip install -e .
else
  $PY -m pip install -q -e .
fi

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
  log "Using pnpm to install frontend deps..."
  if [ "$VERBOSE" -eq 1 ]; then
    pnpm install --prefix frontend
  else
    pnpm install --silent --prefix frontend
  fi
else
  if command -v npm >/dev/null 2>&1; then
    log "pnpm not found; using npm to install frontend deps..."
    if [ "$VERBOSE" -eq 1 ]; then
      npm install --prefix frontend
    else
      npm install --silent --prefix frontend
    fi
  else
    err "Error: neither pnpm nor npm were found. Please install Node.js and npm (https://nodejs.org/)."
    exit 1
  fi
fi

# Start fullstack (uses tools/start_fullstack.py)
log ""
log "Starting CUBO full stack..."
# Export verbosity env so start_fullstack.py can adjust its own verbosity
if [ "$VERBOSE" -eq 1 ]; then
  export CUBO_VERBOSE=1
  $PY tools/start_fullstack.py "$@"
else
  export CUBO_VERBOSE=0
  # Keep start script quiet by default
  CUBO_VERBOSE=0 $PY tools/start_fullstack.py "$@" >/dev/null 2>&1 &
fi
