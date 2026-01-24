<#
Sequential experiment runner (implements Option B from OVERNIGHT_RUN.md).
- Runs M1-M4 sequentially to avoid OOM on a 16GB laptop.
- Writes per-step logs to results/logs/, JSON outputs to results/.
- Writes a run manifest to results/manifests/ for reproducibility.

Usage (recommended):
    powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\run_all_experiments.ps1

This script is designed to be safe: it records failures but continues to the next
experiment so you get as many artifacts as possible from a single overnight run.
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Continue'

# --- Setup ---
$ts = Get-Date -Format yyyyMMdd-HHmmss
$gitSha = (& git rev-parse --short HEAD) -replace '\s+',''
$runId = "$ts-$gitSha"
$root = (Resolve-Path -Path .).Path
$results = Join-Path $root 'results'
$logs = Join-Path $results 'logs'
$manifests = Join-Path $results 'manifests'

New-Item -Path $results -ItemType Directory -Force | Out-Null
New-Item -Path $logs -ItemType Directory -Force | Out-Null
New-Item -Path $manifests -ItemType Directory -Force | Out-Null

$manifestPath = Join-Path $manifests "$runId-manifest.json"
$manifest = [ordered]@{
    run_id = $runId
    git_sha = $gitSha
    start_time = (Get-Date).ToString('o')
    host = $env:COMPUTERNAME
}
# Capture python version separately to avoid nested-quoting/parser issues
$pyver = (& python -c 'import sys; print(sys.version.splitlines()[0])') -replace '\r|\n',''
$manifest = [ordered]@{
    run_id = $runId
    git_sha = $gitSha
    start_time = (Get-Date).ToString('o')
    host = $env:COMPUTERNAME
    python = $pyver
    steps = @()
}
$manifest | ConvertTo-Json -Depth 4 | Out-File -FilePath $manifestPath -Encoding UTF8
Write-Output "[RUN] $runId — manifest: $manifestPath"

# Activate venv if present (safe cross-shell path construction)
$activatePath = Join-Path $root '.venv\Scripts\Activate.ps1'
if (Test-Path $activatePath) {
    . $activatePath
}

function Run-Step {
    Param(
        [string]$name,
        [string]$cmd,
        [string]$outJsonPath,
        [string]$logFilePath,
        [int]$timeoutSeconds = 0
    )

    Write-Output "[STEP] $name"
    New-Item -Path (Split-Path $logFilePath) -ItemType Directory -Force | Out-Null
    $start = Get-Date

    try {
        if ($timeoutSeconds -gt 0) {
            $proc = Start-Process -FilePath powershell -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-Command',$cmd) -RedirectStandardOutput $logFilePath -RedirectStandardError $logFilePath -PassThru
            $finished = $proc.WaitForExit($timeoutSeconds * 1000)
            if (-not $finished) {
                Write-Warning "Step '$name' exceeded timeout ($($timeoutSeconds)) seconds; killing process."
                $proc | Stop-Process -Force
            }
        } else {
            & powershell -NoProfile -ExecutionPolicy Bypass -Command $cmd 2>&1 | Tee-Object -FilePath $logFilePath
        }
        $rc = $LASTEXITCODE
    } catch {
        Write-Warning "Step '$name' failed: $_"
        $rc = 1
    }

    $end = Get-Date
    $entry = [ordered]@{
        name = $name
        command = $cmd
        start_time = $start.ToString('o')
        end_time = $end.ToString('o')
        exit_code = $rc
        output = if (Test-Path $outJsonPath) { (Get-Item $outJsonPath).FullName } else { $null }
        log = (Get-Item $logFilePath).FullName
    }

    $m = Get-Content -Raw $manifestPath | ConvertFrom-Json
    $m.steps += $entry
    $m | ConvertTo-Json -Depth 6 | Out-File -FilePath $manifestPath -Encoding UTF8

    return $entry
}

# --- Experiment commands (follow OVERNIGHT_RUN.md Option B) ---
$indexDir = 'data/faiss_test'

# M1: Concurrency
$step1_cmd = "python tools/benchmark_concurrency_real.py --index-dir $indexDir --num-workers 4 --queries-per-worker 100 --concurrent-ingestion --ingestion-docs data/smoke --output results/concurrency.json"
$step1_log = Join-Path $logs "$runId-concurrency.log"
$step1_out = Join-Path $results 'concurrency.json'
Run-Step 'M1 concurrency' $step1_cmd $step1_out $step1_log

# M2: Profiling (breakdown + sensitivity)
$step2_cmd = "python tools/profile_retrieval_breakdown_real.py --index-dir $indexDir --queries data/beir/scifact/queries.jsonl --num-samples 200 --output results/breakdown.json"
$step2_log = Join-Path $logs "$runId-breakdown.log"
$step2_out = Join-Path $results 'breakdown.json'
Run-Step 'M2 retrieval_breakdown' $step2_cmd $step2_out $step2_log

$step2b_cmd = "python tools/sensitivity_analysis_real.py --index-dir $indexDir --queries data/beir/scifact/queries.jsonl --nprobe-values 1,5,10,20,50 --num-samples 50 --output results/sensitivity.json"
$step2b_log = Join-Path $logs "$runId-sensitivity.log"
$step2b_out = Join-Path $results 'sensitivity.json'
Run-Step 'M2 sensitivity' $step2b_cmd $step2b_out $step2b_log

# M3: Baselines (sequential)
$baselineDir = Join-Path $results 'baselines' ; New-Item -Path $baselineDir -ItemType Directory -Force | Out-Null

$step3a_out = Join-Path $baselineDir 'scifact.bm25.json'
$step3a_cmd = "python tools/run_pyserini_baseline.py --dataset scifact --data-dir data/beir --memory-limit 16GB --output '$step3a_out'"
$step3a_log = Join-Path $logs "$runId-bm25.log"
Run-Step 'M3 baseline_bm25' $step3a_cmd $step3a_out $step3a_log

$step3b_out = Join-Path $baselineDir 'scifact.splade.json'
$step3b_cmd = "python tools/run_splade_baseline.py --dataset scifact --data-dir data/beir --cpu-only --max-memory 15GB --output '$step3b_out'"
$step3b_log = Join-Path $logs "$runId-splade.log"
Run-Step 'M3 baseline_splade' $step3b_cmd $step3b_out $step3b_log

$step3c_out = Join-Path $baselineDir 'scifact.e5.json'
$step3c_cmd = "python tools/run_e5_ivfpq_baseline.py --dataset scifact --data-dir data/beir --memory-limit 16GB --output '$step3c_out'"
$step3c_log = Join-Path $logs "$runId-e5.log"
Run-Step 'M3 baseline_e5' $step3c_cmd $step3c_out $step3c_log

# M4: Multilingual
$step4_cmd = "python tools/run_multilingual_eval.py --dataset miracl-de --queries data/multilingual/miracl-de/queries.jsonl --use-compound-splitter --index-dir data/legal_de --output results/multilingual_with.json"
$step4_log = Join-Path $logs "$runId-multilingual-with.log"
$step4_out = Join-Path $results 'multilingual_with.json'
Run-Step 'M4 miracl_de_with_splitter' $step4_cmd $step4_out $step4_log

$step4b_cmd = "python tools/run_multilingual_eval.py --dataset miracl-de --queries data/multilingual/miracl-de/queries.jsonl --index-dir data/legal_de --output results/multilingual_without.json"
$step4b_log = Join-Path $logs "$runId-multilingual-without.log"
$step4b_out = Join-Path $results 'multilingual_without.json'
Run-Step 'M4 miracl_de_without_splitter' $step4b_cmd $step4b_out $step4b_log

# --- Finalize manifest ---
$m = Get-Content -Raw $manifestPath | ConvertFrom-Json
$m.end_time = (Get-Date).ToString('o')
$m | ConvertTo-Json -Depth 6 | Out-File -FilePath $manifestPath -Encoding UTF8

Write-Output "[DONE] Sequential run finished (or paused on errors). Manifest: $manifestPath"
$pattern = Join-Path $logs "$runId-*.log"
Write-Output "Tail logs: Get-Content $pattern -Tail 200 -Wait"
