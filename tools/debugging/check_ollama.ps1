<#
Check if Ollama is running and has the required model.
Called by run_local.ps1
#>

param(
    [string]$ModelName = "llama3.2:latest"
)

$ErrorActionPreference = 'Stop'

Write-Host "`n-- Checking Ollama Intelligence --" -ForegroundColor Cyan

# 1. Check if Ollama is reachable
$ollmaIsRunning = $false
try {
    $tags = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -Method Get -TimeoutSec 2 -ErrorAction Stop
    Write-Host "✓ Ollama service is running." -ForegroundColor Green
    $ollmaIsRunning = $true
}
catch {
    Write-Host "⚠ Ollama is not running. Attempting to start it..." -ForegroundColor Yellow
    
    # Try to start Ollama
    try {
        # Check if ollama command exists
        if (Get-Command ollama -ErrorAction SilentlyContinue) {
            Write-Host "  Starting Ollama service..." -ForegroundColor Cyan
            Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden -PassThru | Out-Null
            
            # Wait for Ollama to start
            $maxWait = 30
            $waited = 0
            while ($waited -lt $maxWait) {
                try {
                    $tags = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -Method Get -TimeoutSec 2 -ErrorAction Stop
                    Write-Host "✓ Ollama service started successfully." -ForegroundColor Green
                    $ollmaIsRunning = $true
                    break
                }
                catch {
                    Start-Sleep -Seconds 1
                    $waited++
                }
            }
            
            if (-not $ollmaIsRunning) {
                throw "Ollama did not start within 30 seconds"
            }
        }
        else {
            Write-Host "✗ Ollama is not installed or not found in PATH." -ForegroundColor Red
            Write-Host "  Please download and install Ollama from https://ollama.com" -ForegroundColor Yellow
            Write-Host "  Then run this script again.`n"
            exit 1
        }
    }
    catch {
        Write-Host "✗ Failed to start Ollama: $_" -ForegroundColor Red
        Write-Host "  Please manually start Ollama:" -ForegroundColor Yellow
        Write-Host "  1. Download/Install Ollama from https://ollama.com (if you haven't)" -ForegroundColor Yellow
        Write-Host "  2. Run 'ollama serve' in a terminal" -ForegroundColor Yellow
        Write-Host "  3. Then try running this script again.`n"
        exit 1
    }
}

# 2. Check if the specific model exists
$hasModel = $false
if ($tags.models) {
    foreach ($m in $tags.models) {
        if ($m.name -eq $ModelName -or $m.name -eq ($ModelName -replace ":latest", "")) {
            $hasModel = $true
            break
        }
    }
}

if ($hasModel) {
    Write-Host "✓ AI Model '$ModelName' is ready." -ForegroundColor Green
}
else {
    Write-Host "⚠ AI Model '$ModelName' is missing." -ForegroundColor Yellow
    Write-Host "  CUBO needs this model file (the 'brain') to understand your documents."
    Write-Host "  The file is approximately 2.0 GB and is downloaded once."
    
    $response = Read-Host "  Do you want to download '$ModelName' now? [Y/n]"
    
    if ($response -notmatch '^[Nn]') {
        Write-Host "  Downloading... (Please wait, speed depends on your internet)" -ForegroundColor Cyan
        # Run pull command directly so the user sees the progress bar
        ollama pull $ModelName
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✓ AI Model downloaded successfully!" -ForegroundColor Green
        }
        else {
            Write-Host "✗ Download failed." -ForegroundColor Red
            Write-Host "  Please try running 'ollama pull $ModelName' manually in a terminal." -ForegroundColor Red
            exit 1
        }
    }
    else {
        Write-Host "  Skipping download." -ForegroundColor Gray 
        Write-Host "  Warning: CUBO will launch, but it won't be able to answer questions until you install a model." -ForegroundColor Red
    }
}
Write-Host ""
