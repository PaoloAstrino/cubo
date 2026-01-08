#!/usr/bin/env pwsh
# Automated Benchmark Suite Runner - ROBUST VERSION
# Runs all evaluation scripts sequentially and saves results
# ESTIMATED RUNTIME: 2-3 HOURS (Full) | 20-30 MIN (Quick)

param(
    [switch]$Quick
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
    $queryFile = "data/beir/nfcorpus/queries.jsonl"
    if ($Quick) {
        Write-Host "Creating subset of 50 queries for quick mode..." -ForegroundColor Yellow
        Get-Content $queryFile -TotalCount 50 | Out-File "data/beir/nfcorpus/queries_quick50.jsonl"
        $queryFile = "data/beir/nfcorpus/queries_quick50.jsonl"
    }

    Write-Host "========================================" -ForegroundColor Cyan
    if ($Quick) {
        Write-Host "STARTING QUICK BENCHMARK (50 queries)" -ForegroundColor Cyan
        Write-Host "Estimated runtime: 20-30 minutes" -ForegroundColor Yellow
    }
    else {
        Write-Host "STARTING FULL BENCHMARK SUITE (323 queries)" -ForegroundColor Cyan
        Write-Host "Estimated runtime: 2-3 HOURS" -ForegroundColor Yellow
    }
    Write-Host "Start Time: $(Get-Date)" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""

    # Check if scripts exist
    $requiredScripts = @("scripts/run_beir_adapter.py", "scripts/run_reranker_eval.py", "scripts/system_metrics.py", "scripts/run_ablation.py")
    foreach ($s in $requiredScripts) {
        if (!(Test-Path $s)) {
            throw "Required script missing: $s"
        }
    }

    # ===== STEP 1: Index NFCorpus (Baseline) =====
    Write-Host "[1/5] Indexing NFCorpus corpus (~3,633 docs)..." -ForegroundColor Yellow
    $step1Start = Get-Date
    python scripts/run_beir_adapter.py `
        --corpus data/beir/nfcorpus/corpus.jsonl `
        --queries $queryFile `
        --reindex `
        --output results/nfcorpus_bench.json `
        --index-dir results/nfcorpus_bench_index `
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
        --index-dir results/nfcorpus_bench_index `
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
        --corpus data/beir/nfcorpus/corpus.jsonl `
        --queries $queryFile `
        --index-dir results/nfcorpus_sys_metrics `
        --top-k 10
    $step3End = Get-Date
    $step3Duration = ($step3End - $step3Start).TotalSeconds
    Write-Host "      ✓ Completed in $([math]::Round($step3Duration, 1))s" -ForegroundColor Green
    Write-Host ""

    # ===== STEP 4: Ablation Study =====
    Write-Host "[4/5] Running Ablation Study (Dense/Hybrid/BM25)..." -ForegroundColor Yellow
    $step4Start = Get-Date
    python scripts/run_ablation.py `
        --dataset nfcorpus `
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
        $runFile = "results/beir_run_nfcorpus_topk50_${mode}.json"
        if (Test-Path $runFile) {
            python scripts/calculate_beir_metrics.py `
                --results $runFile `
                --qrels data/beir/nfcorpus/qrels/test.tsv `
                --k 10
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
