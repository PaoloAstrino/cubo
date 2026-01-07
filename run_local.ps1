<#
Simple PowerShell local startup for unskilled users (Windows PowerShell / PowerShell Core)
Usage: .\run_local.ps1
#>

$ErrorActionPreference = 'Stop'
Write-Host "== CUBO Local Quickstart (PowerShell) =="

# Preflight checks
Write-Host "Checking Python version..."
$py = (Get-Command python -ErrorAction SilentlyContinue) ? 'python' : 'python3'
if (-not (Get-Command $py -ErrorAction SilentlyContinue)) {
    Write-Error "Error: Python not found. Please install Python 3.11+"
    exit 1
}
try {
    & $py -c "import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)" 2>$null
} catch {
    Write-Error "Error: Python 3.11 or higher is required."
    exit 1
}

Write-Host "Checking Node.js version..."
if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
    Write-Error "Error: Node.js not found. Please install Node.js v18+"
    exit 1
}
$v = (node -v).TrimStart('v')
$major = [int]($v.Split('.')[0])
if ($major -lt 18) {
    Write-Error "Error: Node.js v18 or higher is required. Found: v$v"
    exit 1
}

Write-Host "Checking available disk space..."
$cwd = Get-Location
$driveLetter = $cwd.Path.Split(':')[0]
$drive = Get-PSDrive -Name $driveLetter -ErrorAction SilentlyContinue
if (-not $drive) { $drive = Get-PSDrive -PSProvider FileSystem | Select-Object -First 1 }
if ($drive.Free -lt 5GB) {
    Write-Error "Error: Not enough disk space. At least 5GB required. Available: $([math]::Round($drive.Free/1GB,2)) GB"
    exit 1
}

if (-not (Test-Path -Path '.venv')) {
    Write-Host 'Creating virtualenv...'
    $py -m venv .venv
}

# Activate virtualenv
& .\.venv\Scripts\Activate.ps1

# Install editable package and frontend deps
Write-Host "Installing package and frontend dependencies..."
python -m pip install -e .

if (Get-Command pnpm -ErrorAction SilentlyContinue) {
    pnpm install --prefix frontend
} elseif (Get-Command npm -ErrorAction SilentlyContinue) {
    npm install --prefix frontend
} else {
    Write-Error "Error: neither pnpm nor npm were found. Please install Node.js and npm (https://nodejs.org/)"
    exit 1
}

# Start fullstack (uses scripts/start_fullstack.py)
python scripts/start_fullstack.py
