<#
Simple PowerShell local startup for unskilled users (Windows PowerShell / PowerShell Core)
Usage: .\run_local.ps1
#>

param(
    [switch]$Verbose,
    [switch]$CleanInstall
)

$ErrorActionPreference = 'Stop'

# Change to the project root directory (parent of scripts/)
$ScriptDir = Split-Path -Parent $PSCommandPath
Set-Location "$ScriptDir\.."

# Quiet by default; pass -Verbose to be loud
$IsVerbose = $Verbose -or ($env:CUBO_VERBOSE -eq '1')

if ($IsVerbose) { Write-Host "== CUBO Local Quickstart (PowerShell) ==" }

# Helper function for logging
function Log-Info($msg) {
    if ($IsVerbose) { Write-Host "✓ $msg" -ForegroundColor Green }
}

function Log-Warning($msg) {
    Write-Host "⚠ $msg" -ForegroundColor Yellow
}

function Log-Error($msg) {
    Write-Host "✗ $msg" -ForegroundColor Red
}

# System Checks
Log-Info "Checking if Python is ready..."
if (Get-Command python -ErrorAction SilentlyContinue) {
    $py = 'python'
} else {
    $py = 'python3'
}

if (-not (Get-Command $py -ErrorAction SilentlyContinue)) {
    Log-Error "We couldn't find Python. Please install Python 3.11 or newer from https://python.org"
    exit 1
}
try {
    $pyVersion = & $py --version 2>&1
    & $py -c "import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Log-Error "CUBO requires a newer version of Python (3.11+). Your current version is: $pyVersion"
        exit 1
    }
    Log-Info "Great! Python is ready ($pyVersion)."
}
catch {
    Log-Error "Something went wrong checking Python: $_"
    exit 1
}

Log-Info "Checking if Node.js is ready..."
if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
    Log-Error "We couldn't find Node.js. Please install Node.js v18 or newer from https://nodejs.org"
    exit 1
}
try {
    $nodeVersion = node -v
    $major = [int]($nodeVersion.TrimStart('v').Split('.')[0])
    if ($major -lt 18) {
        Log-Error "CUBO requires Node.js v18 or higher. Your current version is: $nodeVersion"
        exit 1
    }
    Log-Info "Great! Node.js is ready ($nodeVersion)."
}
catch {
    Log-Error "Something went wrong checking Node.js: $_"
    exit 1
}

Log-Info "Checking disk space..."
try {
    $cwd = Get-Location
    $driveLetter = $cwd.Path.Split(':')[0]
    $drive = Get-PSDrive -Name $driveLetter -ErrorAction SilentlyContinue
    if (-not $drive) { $drive = Get-PSDrive -PSProvider FileSystem | Select-Object -First 1 }
    $freeGB = [math]::Round($drive.Free / 1GB, 2)
    if ($drive.Free -lt 5GB) {
        Log-Error "We need a bit more disk space. Please free up at least 5GB (Available: $freeGB GB)."
        exit 1
    }
    Log-Info "Disk space looks good ($freeGB GB available)."
}
catch {
    Log-Warning "Could not verify disk space: $_"
}

# Clean up corrupted packages if requested
if ($CleanInstall -and (Test-Path -Path '.venv')) {
    Log-Warning "Clean install requested. Resetting the local environment..."
    Remove-Item -Path '.venv' -Recurse -Force -ErrorAction SilentlyContinue
}

# Create or verify virtualenv
if (-not (Test-Path -Path '.venv')) {
    Log-Info "Setting up a local environment for CUBO..."
    try {
        python -m venv .venv
        if ($LASTEXITCODE -ne 0) {
            Log-Error "Failed to create the local environment."
            exit 1
        }
    }
    catch {
        Log-Error "Failed to create the local environment: $_"
        exit 1
    }
}
else {
    Log-Info "Found existing local environment."
}

# Activate virtualenv
Log-Info "Preparing the environment..."
try {
    & .\.venv\Scripts\Activate.ps1
    if ($LASTEXITCODE -ne 0) {
        Log-Error "Failed to activate the environment."
        exit 1
    }
}
catch {
    Log-Error "Failed to activate the environment: $_"
    exit 1
}

# Clean up corrupted pip packages (packages starting with ~)
try {
    $corruptedPkgs = Get-ChildItem -Path ".venv\Lib\site-packages" -Filter "~*" -Directory -ErrorAction SilentlyContinue
    if ($corruptedPkgs) {
        Log-Warning "Found some incomplete files, cleaning them up..."
        foreach ($pkg in $corruptedPkgs) {
            Remove-Item -Path $pkg.FullName -Recurse -Force -ErrorAction SilentlyContinue
            Log-Info "Cleaned up: $($pkg.Name)"
        }
    }
}
catch {
    Log-Warning "Could not check for incomplete files: $_"
}

# Install Python package with retry logic
Log-Info "Installing CUBO core components (this might take a few minutes)..."
$maxRetries = 3
$retryCount = 0
$pipSuccess = $false

while (-not $pipSuccess -and $retryCount -lt $maxRetries) {
    try {
        if ($retryCount -gt 0) {
            Log-Warning "Retrying installation ($($retryCount)/$($maxRetries))..."
            Start-Sleep -Seconds 2
        }
        
        if ($IsVerbose) {
            python -m pip install -e .
        }
        else {
            python -m pip install -q -e . 2>&1 | Out-Null
        }
        
        if ($LASTEXITCODE -eq 0) {
            $pipSuccess = $true
            Log-Info "Core components installed successfully."
        }
        else {
            throw "pip install failed with exit code $LASTEXITCODE"
        }
    }
    catch {
        $retryCount++
        if ($retryCount -ge $maxRetries) {
            Log-Error "Failed to install core components after $maxRetries attempts."
            Log-Error "Error details: $_"
            Log-Error "Try running with -Verbose for more info."
            exit 1
        }
    }
}

# Check npm/pnpm and install frontend dependencies
$npmCmd = $null
if (Get-Command pnpm -ErrorAction SilentlyContinue) {
    $npmCmd = "pnpm"
    Log-Info "Using pnpm for user interface setup."
}
elseif (Get-Command npm -ErrorAction SilentlyContinue) {
    $npmCmd = "npm"
    Log-Info "Using npm for user interface setup."
}
else {
    Log-Error "Neither pnpm nor npm found. Please install Node.js from https://nodejs.org"
    exit 1
}

# Install frontend dependencies with retry
Log-Info "Setting up the user interface (this might take a few minutes)..."
$retryCount = 0
$npmSuccess = $false

while (-not $npmSuccess -and $retryCount -lt $maxRetries) {
    try {
        if ($retryCount -gt 0) {
            Log-Warning "Retrying setup ($($retryCount)/$($maxRetries))..."
            Start-Sleep -Seconds 2
        }
        
        if ($npmCmd -eq "pnpm") {
            if ($IsVerbose) {
                pnpm install --prefix frontend
            }
            else {
                pnpm install --silent --prefix frontend 2>&1 | Out-Null
            }
        }
        else {
            if ($IsVerbose) {
                npm install --prefix frontend
            }
            else {
                npm install --prefix frontend 2>&1 | Out-Null
            }
        }
        
        if ($LASTEXITCODE -eq 0) {
            $npmSuccess = $true
            Log-Info "User interface setup complete."
        }
        else {
            throw "npm/pnpm install failed with exit code $LASTEXITCODE"
        }
    }
    catch {
        $retryCount++
        if ($retryCount -ge $maxRetries) {
            Log-Error "Failed to set up the user interface after $maxRetries attempts."
            Log-Error "Error details: $_"
            Log-Error "Try running with -Verbose for more info."
            exit 1
        }
    }
}

# Check for port conflicts and offer to kill existing processes
Log-Info "Checking network ports..."
$portsToCheck = @(
    @{Port = 8000; Name = "Backend API" },
    @{Port = 3000; Name = "Frontend UI" },
    @{Port = 11434; Name = "Ollama AI" }
)
$conflictingProcesses = @()

foreach ($portInfo in $portsToCheck) {
    try {
        $port = $portInfo.Port
        $name = $portInfo.Name
        $netstatOutput = netstat -ano | Select-String ":$port\s" | Select-String "LISTENING"
        
        if ($netstatOutput) {
            # Robustly get the first matching line content
            $line = $netstatOutput | Select-Object -ExpandProperty Line -First 1
            if ($line -match '\s+(\d+)\s*$') {
                $processId = [int]$Matches[1]
                $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
                if ($process) {
                    $conflictingProcesses += @{
                        Port        = $port
                        Name        = $name
                        PID         = $processId
                        ProcessName = $process.ProcessName
                    }
                }
            }
        }
    }
    catch {
        # Ignore errors in port checking
    }
}

if ($conflictingProcesses.Count -gt 0) {
    Log-Warning "We found some programs using ports that CUBO needs:"
    foreach ($proc in $conflictingProcesses) {
        Write-Host "  - Port $($proc.Port) ($($proc.Name)) is used by: $($proc.ProcessName) (PID: $($proc.PID))" -ForegroundColor Yellow
    }
    
    $response = Read-Host "`nCan we stop these programs for you so CUBO can run? (Y/n)"
    if ($response -notmatch '^[Nn]') {
        Log-Info "Stopping conflicting programs..."
        foreach ($proc in $conflictingProcesses) {
            try {
                taskkill /F /PID $($proc.PID) /T 2>&1 | Out-Null
                Log-Info "Stopped $($proc.ProcessName) on port $($proc.Port)"
            }
            catch {
                Log-Warning "Could not stop $($proc.ProcessName): $_"
            }
        }
        Start-Sleep -Seconds 2
    }
    else {
        Log-Warning "CUBO might fail to start if these ports are busy. We'll try anyway."
    }
}

# Check for port conflicts and offer to kill existing processes
# ... (existing port check code) ...

# Check Ollama and Model Readiness
$targetModel = $env:CUBO_LLM_MODEL
if (-not $targetModel) { $targetModel = "llama3.2:latest" }
& "tools\debugging\check_ollama.ps1" -ModelName $targetModel

# Start fullstack (uses scripts/start_fullstack.py)
Log-Info "Launching CUBO..."
Log-Info "Please wait while we start the services..."

try {
    if ($IsVerbose) {
        $env:CUBO_VERBOSE = '1'
        Write-Host "`n=== Starting services in verbose mode ===" -ForegroundColor Cyan
        Write-Host "Press Ctrl+C to stop all services`n" -ForegroundColor Cyan
        python scripts/start_fullstack.py
    }
    else {
        $env:CUBO_VERBOSE = '0'
        
        # Start process and wait a bit to see if it crashes immediately
        $process = Start-Process -FilePath "python" -ArgumentList "scripts/start_fullstack.py" -WindowStyle Hidden -PassThru
        Start-Sleep -Seconds 5
        
        if ($process.HasExited) {
            Log-Error "CUBO failed to start. Please check logs/backend.err and logs/frontend.err for details."
            exit 1
        }
        
        # Verify services are actually responding
        $healthOk = $false
        try {
            Start-Sleep -Seconds 3
            $health = Invoke-RestMethod -Uri "http://localhost:8000/api/health" -TimeoutSec 5 -ErrorAction SilentlyContinue
            if ($health.status -eq "healthy") {
                $healthOk = $true
            }
        }
        catch {
            # Health check failed, will warn below
        }
        
        if ($healthOk) {
            Write-Host "`n✓ CUBO is running!" -ForegroundColor Green
            Write-Host "  Open this link in your browser: http://localhost:3000" -ForegroundColor Cyan
            Write-Host "`n  (To stop CUBO, close this window or use Task Manager to stop python.exe)" -ForegroundColor Gray
        }
        else {
            Write-Host "`n✓ CUBO is starting up..." -ForegroundColor Green
            Write-Host "  It might take a few more seconds." -ForegroundColor Gray
            Write-Host "  Open this link in your browser: http://localhost:3000" -ForegroundColor Cyan
        }
    }
}
catch {
    Log-Error "Failed to start CUBO: $_"
    Log-Error "Check logs in the logs/ directory for details"
    exit 1
}
