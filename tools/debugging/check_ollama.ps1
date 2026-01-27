<#
Check if Ollama is running and has the required model.
Called by run_local.ps1
#>

param(
    [string]$ModelName = "llama3.2:latest"
)

$ErrorActionPreference = 'Stop'

Write-Host "`n-- Checking Ollama Intelligence --" -ForegroundColor Cyan

# 1. Check if Ollama is installed and reachable
$ollmaIsRunning = $false
try {
    # Try ollama list first (works if ollama serve is running)
    $ollamaOutput = & ollama list 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Ollama service is running." -ForegroundColor Green
        $ollmaIsRunning = $true
    }
    else {
        throw "ollama list failed"
    }
}
catch {
    Write-Host "Ollama is not running." -ForegroundColor Yellow
    
    # Try to start Ollama
    try {
        # Check if ollama command exists
        if (Get-Command ollama -ErrorAction SilentlyContinue) {
            Write-Host "  Starting Ollama service (this may take 15-45 seconds on first run)..." -ForegroundColor Cyan
            
            # Start Ollama in background with error redirection
            $process = Start-Process -FilePath "ollama" -ArgumentList "serve" `
                -WindowStyle Hidden -PassThru `
                -RedirectStandardError "$env:TEMP\ollama_error.log" `
                -RedirectStandardOutput "$env:TEMP\ollama_output.log"
            
            # Wait for Ollama to start (longer timeout - it takes a while to initialize GPU)
            $maxWait = 120
            $waited = 0
            $pollInterval = 2  # Poll every 2 seconds (not 1)
            $lastUpdate = 0
            
            while ($waited -lt $maxWait) {
                try {
                    $response = Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" `
                        -Method Get -TimeoutSec 2 -ErrorAction Stop -SkipHttpErrorCheck
                    Write-Host "Ollama service started successfully." -ForegroundColor Green
                    $ollmaIsRunning = $true
                    break
                }
                catch {
                    # Check if process died
                    if ($process.HasExited) {
                        $errorLog = Get-Content "$env:TEMP\ollama_error.log" -Raw 2>$null
                        throw "Ollama process exited unexpectedly. Process may not have proper permissions or dependencies.`n$errorLog"
                    }
                    
                    Start-Sleep -Seconds $pollInterval
                    $waited += $pollInterval
                    
                    # Give user feedback every 15 seconds
                    if ($waited - $lastUpdate -ge 15) {
                        Write-Host "  Still initializing GPU... ($waited seconds elapsed)" -ForegroundColor DarkGray
                        $lastUpdate = $waited
                    }
                }
            }
            
            if (-not $ollmaIsRunning) {
                $process | Stop-Process -Force -ErrorAction SilentlyContinue
                Write-Host "`nOllama took too long to start (>120 seconds)" -ForegroundColor Red
                throw "Ollama startup timeout - it may need to initialize GPU or models"
            }
        }
        else {
            Write-Host "Ollama is not installed or not found in PATH." -ForegroundColor Red
            Write-Host "  Please download and install Ollama from https://ollama.com" -ForegroundColor Yellow
            Write-Host "  Then run this script again.`n"
            exit 1
        }
    }
    catch {
        Write-Host "Failed to start Ollama: $_" -ForegroundColor Red
        Write-Host "  MANUAL FIX: Open a new terminal and run this command:" -ForegroundColor Yellow
        Write-Host "    ollama serve" -ForegroundColor Cyan
        Write-Host "  Then try this script again in another terminal.`n" -ForegroundColor Yellow
        exit 1
    }
}

# 2. Get available models from ollama list
$availableModels = @()
$selectedModel = $null

try {
    $ollamaListOutput = & ollama list 2>&1
    
    # Parse the output to extract model names (skip header line)
    $lines = $ollamaListOutput -split "`n" | Where-Object { $_ -and $_ -notmatch "^NAME" }
    
    foreach ($line in $lines) {
        if ($line -match "^(\S+)") {
            $modelName = $Matches[1]
            if ($modelName) {
                $availableModels += $modelName
            }
        }
    }
}
catch {
    Write-Host "Could not check installed models" -ForegroundColor Yellow
}

# If models are available, let user choose
if ($availableModels.Count -gt 0) {
    Write-Host "Found $($availableModels.Count) AI model(s) available:" -ForegroundColor Green
    
    # Always show menu if multiple models available
    if ($availableModels.Count -gt 1) {
        Write-Host ""
        for ($i = 0; $i -lt $availableModels.Count; $i++) {
            Write-Host "  [$($i + 1)] $($availableModels[$i])" -ForegroundColor Cyan
        }
        Write-Host ""
        
        $choice = Read-Host "  Select a model (enter number)"
        $choiceIndex = [int]$choice - 1
        
        if ($choiceIndex -ge 0 -and $choiceIndex -lt $availableModels.Count) {
            $selectedModel = $availableModels[$choiceIndex]
        }
        else {
            Write-Host "✗ Invalid selection." -ForegroundColor Red
            exit 1
        }
    }
    else {
        # Only one model, use it directly
        $selectedModel = $availableModels[0]
    }
    
    if ($selectedModel) {
        Write-Host "AI Model '$selectedModel' is ready." -ForegroundColor Green
    }
}
else {
    Write-Host "⚠ No AI models found." -ForegroundColor Yellow
    Write-Host "  CUBO needs a model file (the 'brain') to understand your documents."
    Write-Host "  The file is approximately 2.0 GB and is downloaded once."
    
    $response = Read-Host "  Do you want to download '$ModelName' now? [Y/n]"
    
    if ($response -notmatch '^[Nn]') {
        Write-Host "  Downloading... (Please wait, speed depends on your internet)" -ForegroundColor Cyan
        # Run pull command directly so the user sees the progress bar
        ollama pull $ModelName
        if ($LASTEXITCODE -eq 0) {
            Write-Host "AI Model downloaded successfully!" -ForegroundColor Green
            $selectedModel = $ModelName
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

# Export the selected model for use by the caller
$env:CUBO_SELECTED_MODEL = $selectedModel
Write-Host ""
