<#
Simple PowerShell local startup for unskilled users (Windows PowerShell / PowerShell Core)
Usage: .\run_local.ps1
#>

$ErrorActionPreference = 'Stop'
Write-Host "== CUBO Local Quickstart (PowerShell) =="

if (-not (Test-Path -Path '.venv')) {
    Write-Host 'Creating virtualenv...'
    python -m venv .venv
}

# Activate virtualenv
& .\.venv\Scripts\Activate.ps1

# Install editable package and frontend deps
python -m pip install -e .
npm run install:frontend

# Start fullstack (uses scripts/start_fullstack.py)
python scripts/start_fullstack.py
