#!/usr/bin/env bash
set -euo pipefail

# Quickstart script: clone a repository and run its local quickstart (run_local.sh)
# Usage: quickstart.sh <git-repo-url> [target-dir]

REPO_URL=${1:-}
TARGET_DIR=${2:-}

if [ -z "$REPO_URL" ]; then
  echo "Usage: $0 <git-repository-url> [target-dir]"
  echo "Example: $0 https://github.com/your-username/cubo.git"
  exit 2
fi

if ! command -v git >/dev/null 2>&1; then
  echo "Error: git not found in PATH. Please install git first (https://git-scm.com/downloads)."
  exit 3
fi

if [ -z "$TARGET_DIR" ]; then
  # derive dir from repo name
  TARGET_DIR=$(basename -s .git "$REPO_URL")
fi

if [ -d "$TARGET_DIR" ]; then
  echo "Directory '$TARGET_DIR' already exists. Pulling latest changes..."
  cd "$TARGET_DIR"
  git pull --rebase || true
else
  echo "Cloning $REPO_URL into $TARGET_DIR..."
  git clone "$REPO_URL" "$TARGET_DIR"
  cd "$TARGET_DIR"
fi

if [ ! -f "./scripts/run_local.sh" ]; then
  echo "Error: scripts/run_local.sh not found in repository. This quickstart expects the project to include ./scripts/run_local.sh"
  exit 4
fi

echo "Launching local dev environment with ./scripts/run_local.sh"
chmod +x ./scripts/run_local.sh
./scripts/run_local.sh
