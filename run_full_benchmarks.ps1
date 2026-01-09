#!/usr/bin/env pwsh
# Automated Benchmark Suite Runner - ROBUST VERSION
# Runs all evaluation scripts sequentially and saves results
# ESTIMATED RUNTIME: 2-3 HOURS (Full) | 20-30 MIN (Quick)

param(
    [switch]$Quick,
    [string]$Dataset = "nfcorpus",
    [switch]$Yes
)

$ErrorActionPreference = "Stop" # Stop on errors to avoid "fake" completions
$env:PYTHONPATH = "C:\Users\paolo\Desktop\cubo"

# Create results directory
if (!(Test-Path "results")) {
    New-Item -ItemType Directory -Path "results" | Out-Null
}

# Log file
$logFile = "results/benchmark_run_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
Start-Transcript -Path $logFile

try {
    # Prepare query subset if quick mode
    $queryFile = "data/beir/$Dataset/queries.jsonl"
    $corpusFile = "data/beir/$Dataset/corpus.jsonl"
    $qrelsFile = "data/beir/$Dataset/qrels/test.tsv"

    # Validate dataset files exist
    if (!(Test-Path $corpusFile)) {
        throw "Corpus file not found: $corpusFile"
    }
    if (!(Test-Path $queryFile)) {
        throw "Queries file not found: $queryFile"
    }

    if ($Quick) {
        Write-Host "Creating subset of 50 queries for quick mode..." -ForegroundColor Yellow
        Get-Content $queryFile -TotalCount 50 | Out-File "data/beir/$Dataset/queries_quick50.jsonl"
        $queryFile = "data/beir/$Dataset/queries_quick50.jsonl"
    }

    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "BENCHMARK DATASET: $Dataset" -ForegroundColor Cyan
    if ($Quick) {
        Write-Host "MODE: Quick (50 queries)" -ForegroundColor Cyan
        Write-Host "Estimated runtime: 20-30 minutes" -ForegroundColor Yellow
    }
    else {
        Write-Host "MODE: Full benchmark" -ForegroundColor Cyan
        Write-Host "Estimated runtime: 1-3 hours" -ForegroundColor Yellow
    }
    Write-Host "Start Time: $(Get-Date)" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""

    # Confirmation prompt unless --Yes flag is set
    if (!$Yes) {
        $confirmation = Read-Host "Continue with benchmark run? (Y/N)"
        if ($confirmation -ne 'Y' -and $confirmation -ne 'y') {
            Write-Host "Benchmark cancelled by user." -ForegroundColor Yellow
            exit 0
        }
    }

    # Check if scripts exist
    $requiredScripts = @("scripts/run_beir_adapter.py", "scripts/run_reranker_eval.py", "scripts/system_metrics.py", "scripts/run_ablation.py")
    foreach ($s in $requiredScripts) {
        if (!(Test-Path $s)) {
            throw "Required script missing: $s"
        }
    }

    # ===== STEP 1: Index Corpus (Baseline) =====
    Write-Host "[1/5] Indexing $Dataset corpus..." -ForegroundColor Yellow
    $step1Start = Get-Date
    python scripts/run_beir_adapter.py `
        --corpus $corpusFile `
        --queries $queryFile `
        --reindex `
        --output "results/${Dataset}_bench.json" `
        --index-dir "results/${Dataset}_bench_index" `
        --use-optimized --laptop-mode `
        --top-k 50
    $step1End = Get-Date
    $step1Duration = ($step1End - $step1Start).TotalSeconds
    Write-Host "      ✓ Completed in $([math]::Round($step1Duration, 1))s" -ForegroundColor Green
    Write-Host ""

    # ===== STEP 2: Reranker Evaluation =====
    Write-Host "[2/5] Running Reranker Evaluation..." -ForegroundColor Yellow
    $step2Start = Get-Date
    python scripts/run_reranker_eval.py `
        --index-dir "results/${Dataset}_bench_index" `
        --queries $queryFile `
        --top-k 10
    $step2End = Get-Date
    $step2Duration = ($step2End - $step2Start).TotalSeconds
    Write-Host "      ✓ Completed in $([math]::Round($step2Duration, 1))s" -ForegroundColor Green
    Write-Host ""

    # ===== STEP 3: System Metrics =====
    Write-Host "[3/5] Collecting System Metrics..." -ForegroundColor Yellow
    $step3Start = Get-Date
    python scripts/system_metrics.py `
        --corpus $corpusFile `
        --queries $queryFile `
        --index-dir "results/${Dataset}_sys_metrics" `
        --top-k 10
    $step3End = Get-Date
    $step3Duration = ($step3End - $step3Start).TotalSeconds
    Write-Host "      ✓ Completed in $([math]::Round($step3Duration, 1))s" -ForegroundColor Green
    Write-Host ""

    # ===== STEP 4: Ablation Study =====
    Write-Host "[4/5] Running Ablation Study (Dense/Hybrid/BM25)..." -ForegroundColor Yellow
    $step4Start = Get-Date
    python scripts/run_ablation.py `
        --dataset $Dataset `
        --queries $queryFile `
        --top-k 50 `
        --index-dir results `
        --output-dir results
    $step4End = Get-Date
    $step4Duration = ($step4End - $step4Start).TotalSeconds
    Write-Host "      ✓ Completed in $([math]::Round($step4Duration, 1))s" -ForegroundColor Green
    Write-Host ""

    # ===== STEP 5: Metrics Calculation =====
    Write-Host "[5/5] Finalizing metrics..." -ForegroundColor Yellow
    $step5Start = Get-Date
    foreach ($mode in @("dense", "hybrid", "bm25")) {
        $runFile = "results/beir_run_${Dataset}_topk50_${mode}.json"
        if (Test-Path $runFile) {
            if (Test-Path $qrelsFile) {
                python scripts/calculate_beir_metrics.py `
                    --results $runFile `
                    --qrels $qrelsFile `
                    --k 10
            }
            else {
                Write-Host "      ⚠ Skipping metrics for $mode (qrels not found: $qrelsFile)" -ForegroundColor Yellow
            }
        }
    }
    $step5End = Get-Date
    $step5Duration = ($step5End - $step5Start).TotalSeconds
    Write-Host "      ✓ Completed in $([math]::Round($step5Duration, 1))s" -ForegroundColor Green
    Write-Host ""

    # ===== SUMMARY =====
    $totalDuration = ($step5End - $step1Start).TotalSeconds
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "BENCHMARK SUITE COMPLETED SUCCESSFULLY" -ForegroundColor Cyan
    Write-Host "Total Duration: $([math]::Round($totalDuration / 60, 1)) minutes" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan

}
catch {
    Write-Host ""
    Write-Host "!!! ERROR DETECTED !!!" -ForegroundColor Red
    Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Benchmark stopped prematurely." -ForegroundColor Red
    Write-Host "Check the log file for details: $logFile" -ForegroundColor Gray
}
finally {
    Stop-Transcript
}
