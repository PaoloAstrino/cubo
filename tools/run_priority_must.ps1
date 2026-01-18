# Orchestrate priority (M1-M4) experimental runs
# - Creates a run manifest
# - Launches: concurrency (BG), retrieval breakdown (BG), sensitivity (BG)
# - Runs baselines sequentially (BM25 -> SPLADE -> e5) to avoid OOM
# - Collects logs and writes a summary manifest

param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# --- Run identifiers & paths ---
$ts = Get-Date -Format yyyyMMdd-HHmmss
$gitSha = (& git rev-parse --short HEAD) -replace "\s+", ""
$runId = "$ts-$gitSha"
$base = Resolve-Path -Path .
$resultsDir = Join-Path $base "results"
$logsDir = Join-Path $resultsDir "logs"
$manifestsDir = Join-Path $resultsDir "manifests"

New-Item -Path $resultsDir -ItemType Directory -Force | Out-Null
New-Item -Path $logsDir -ItemType Directory -Force | Out-Null
New-Item -Path $manifestsDir -ItemType Directory -Force | Out-Null

# --- Manifest (start) ---
$manifest = [ordered]@{
    run_id     = $runId
    git_sha    = $gitSha
    branch     = (& git rev-parse --abbrev-ref HEAD) -replace "\s+", ""
    start_time = (Get-Date).ToString("o")
    host       = $env:COMPUTERNAME
    python     = (python -c "import sys; print(sys.version.splitlines()[0])") -replace "\r|\n", ""
    commands   = @()
}

$manifestPath = Join-Path $manifestsDir "$runId-manifest.json"
$manifest | ConvertTo-Json -Depth 4 | Out-File -FilePath $manifestPath -Encoding UTF8
Write-Output "[RUN] $runId — manifest written to $manifestPath"

# Ensure venv is active in this session
if (Test-Path ".\.venv\Scripts\Activate.ps1") {
    . .\.venv\Scripts\Activate.ps1
}

# Utility to run a command and append to manifest.commands
function Record-Command($cmd) {
    $m = Get-Content -Raw $manifestPath | ConvertFrom-Json
    $m.commands += $cmd
    $m | ConvertTo-Json -Depth 6 | Out-File -FilePath $manifestPath -Encoding UTF8
}

# --- Background (concurrent) jobs ---
$indexDir = "data/beir_index_scifact"

# Concurrency (long)
$cmd_concurrency = "python tools/benchmark_concurrency_real.py --index-dir $indexDir --num-workers 4 --queries-per-worker 100 --concurrent-ingestion --ingestion-docs data/smoke --output results/concurrency_scifact_full.json"
Record-Command $cmd_concurrency
$log = Join-Path $logsDir "$runId-concurrency.log"
Write-Output "[PROC-start] concurrency -> $log"
$proc_concurrency = Start-Process -FilePath powershell -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-Command', $cmd_concurrency) -RedirectStandardOutput $log -RedirectStandardError $log -PassThru
$procs = @($proc_concurrency) 

# Retrieval breakdown (medium)
$cmd_breakdown = "python tools/profile_retrieval_breakdown_real.py --index-dir $indexDir --queries data/beir/scifact/queries.jsonl --num-samples 200 --top-k 10 --output results/retrieval_breakdown_scifact.json"
Record-Command $cmd_breakdown
$log = Join-Path $logsDir "$runId-breakdown.log"
Write-Output "[PROC-start] breakdown -> $log"
$proc_breakdown = Start-Process -FilePath powershell -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-Command', $cmd_breakdown) -RedirectStandardOutput $log -RedirectStandardError $log -PassThru
$procs += $proc_breakdown 

# Sensitivity grid (medium)
$cmd_sensitivity = "python tools/sensitivity_analysis_real.py --index-dir $indexDir --queries data/beir/scifact/queries.jsonl --nprobe-values 1,5,10,20,50,100 --num-samples 50 --top-k 10 --output results/sensitivity_scifact.json"
Record-Command $cmd_sensitivity
$log = Join-Path $logsDir "$runId-sensitivity.log"
Write-Output "[PROC-start] sensitivity -> $log"
$proc_sensitivity = Start-Process -FilePath powershell -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-Command', $cmd_sensitivity) -RedirectStandardOutput $log -RedirectStandardError $log -PassThru
$procs += $proc_sensitivity 

# --- Baselines (sequential to avoid OOM) ---
# BM25 (Pyserini)
$cmd_bm25 = "python tools/run_pyserini_baseline.py --dataset scifact --data-dir data/beir --memory-limit 16GB --output results/baselines/scifact/bm25.json"
Record-Command $cmd_bm25
$log_bm25 = Join-Path $logsDir "$runId-bm25.log"
Write-Output "[SEQ] Running BM25 -> $log_bm25"
& cmd /c "$cmd_bm25" 2>&1 | Tee-Object -FilePath $log_bm25

# SPLADE (CPU-only)
$cmd_splade = "python tools/run_splade_baseline.py --dataset scifact --data-dir data/beir --cpu-only --max-memory 15GB --output results/baselines/scifact/splade.json"
Record-Command $cmd_splade
$log_splade = Join-Path $logsDir "$runId-splade.log"
Write-Output "[SEQ] Running SPLADE -> $log_splade"
& cmd /c "$cmd_splade" 2>&1 | Tee-Object -FilePath $log_splade

# e5 + IVFPQ
$cmd_e5 = "python tools/run_e5_ivfpq_baseline.py --dataset scifact --data-dir data/beir --memory-limit 16GB --output results/baselines/scifact/e5.json"
Record-Command $cmd_e5
$log_e5 = Join-Path $logsDir "$runId-e5.log"
Write-Output "[SEQ] Running e5 -> $log_e5"
& cmd /c "$cmd_e5" 2>&1 | Tee-Object -FilePath $log_e5

# --- Wait for background jobs to finish ---
Write-Output "[ORCH] Waiting for background jobs (concurrency, breakdown, sensitivity) to finish..."
Get-Job | Where-Object { $_.Name -like "*_$runId" } | Wait-Job -Timeout 86400

# Collect job outputs (if any)
$jobs = Get-Job | Where-Object { $_.Name -like "*_$runId" }
$jobSummaries = @()
foreach ($j in $jobs) {
    $state = $j.State
    $output = Receive-Job -Job $j -Keep
    $jobSummaries += [pscustomobject]@{ name = $j.Name; state = $state }
}

# Finalize manifest
$m = Get-Content -Raw $manifestPath | ConvertFrom-Json
$m.end_time = (Get-Date).ToString("o")
$m.jobs = $jobSummaries
$m | ConvertTo-Json -Depth 6 | Out-File -FilePath $manifestPath -Encoding UTF8

Write-Output "[DONE] Priority run $runId finished (or launched). Manifest: $manifestPath"
Write-Output "To monitor: Get-Job | Where-Object { \'Name -like \"*$runId\"\' } ; Get-Content results/logs/$runId-concurrency.log -Tail 200 -Wait"
