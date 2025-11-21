#!/bin/bash
# Quick start script for CUBO full stack

set -e

echo "============================================================"
echo "CUBO Full Stack Startup"
echo "============================================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python not found. Please install Python 3.11+"
    exit 1
fi

# Check if pnpm is installed
if ! command -v pnpm &> /dev/null; then
    echo "Error: pnpm not found. Installing globally..."
    npm install -g pnpm
fi

# Install backend dependencies
echo "Installing backend dependencies..."
pip install -r requirements.txt

# Install frontend dependencies
echo "Installing frontend dependencies..."
cd frontend
pnpm install
cd ..

# Start full stack
echo ""
echo "Starting CUBO full stack..."
python3 scripts/start_fullstack.py
