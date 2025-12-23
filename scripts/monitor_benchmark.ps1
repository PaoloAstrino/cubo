Param(
    [string]$RunDir = "results",
    [int]$IntervalSeconds = 10,
    [int]$TailLines = 20,
    [switch]$Follow,
    [string]$ApiUrl = "http://localhost:8000/api/health",
    [switch]$UseGPing
)

function Get-Last-Run {
    param($runParent)
    Get-ChildItem -Path $runParent -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1
}

function Show-Process-Info {
    # Use CIM to capture full commandlines reliably
    $proc = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match "rag_benchmark|benchmark_runner|run_rag_tests" -or $_.Name -match "python" }
    if (-not $proc) { Write-Host "No python benchmark process found." -ForegroundColor Yellow; return }
    Write-Host "Running processes (python) :" -ForegroundColor Green
    $proc | ForEach-Object {
        $pid = $_.ProcessId
        $creation = [Management.ManagementDateTimeConverter]::ToDateTime($_.CreationDate)
        $runtime = (Get-Date) - $creation
        [PSCustomObject]@{
            ProcessId   = $pid
            RunTime     = ('{0:hh\:mm\:ss}' -f $runtime)
            Name        = $_.Name
            CommandLine = $_.CommandLine
        }
    } | Format-Table -AutoSize
}

function Tail-File($path, $lines = $TailLines) {
    if (Test-Path $path) {
        try {
            Get-Content -Path $path -Tail $lines -ErrorAction SilentlyContinue
        }
        catch { Write-Host "Could not tail $path : $_" -ForegroundColor Red }
    }
    else { Write-Host "File not found: $path" -ForegroundColor Yellow }
}

function Show-Run-Stats($runDir) {
    # show size of files
    Write-Host "\nRun Directory: $runDir" -ForegroundColor Cyan
    if (-not (Test-Path $runDir)) { Write-Host "Run directory doesn't exist" -ForegroundColor Yellow; return }

    Get-ChildItem -Path $runDir -File | Select-Object Name, @{Name = 'SizeMB'; Expression = { [math]::Round($_.Length / 1MB, 2) } }, LastWriteTime | Format-Table -AutoSize

    # test_results.json
    $resultFile = Join-Path $runDir 'test_results.json'
    if (Test-Path $resultFile) {
        try {
            $json = Get-Content $resultFile -Raw | ConvertFrom-Json -ErrorAction Stop
            $meta = $json.metadata
            Write-Host "\nSummary (from test_results.json):" -ForegroundColor Green
            Write-Host "  Mode: $($meta.mode)  Total: $($meta.total_questions)  Success Rate: $([math]::Round($meta.success_rate*100,2))%"
            if ($meta.questions_by_difficulty) {
                foreach ($k in $meta.questions_by_difficulty.PSObject.Properties.Name) {
                    $stats = $meta.questions_by_difficulty.$k
                    Write-Host "   $k : total=$($stats.total), success_rate=$([math]::Round($stats.success_rate*100,2))%"
                }
            }
        }
        catch { Write-Host "Failed to parse test_results.json: $_" -ForegroundColor Red }
    }
    else { Write-Host "No test_results.json found in run dir" -ForegroundColor Yellow }

    # Show any ingestion/test stdout/stderr
    $stdout = Join-Path $runDir 'test_stdout.log'
    $stderr = Join-Path $runDir 'test_stderr.log'
    $ingestStd = Join-Path $runDir 'ingest_stdout.log'
    $ingestErr = Join-Path $runDir 'ingest_stderr.log'

    Write-Host "\nLatest stdout from test runner (last $TailLines lines):" -ForegroundColor Green
    Tail-File $stdout
    Write-Host "\nLatest stderr from test runner (last $TailLines lines):" -ForegroundColor Green
    Tail-File $stderr
    Write-Host "\nLatest ingest stdout (last $TailLines lines):" -ForegroundColor Green
    Tail-File $ingestStd
    Write-Host "\nLatest ingest stderr (last $TailLines lines):" -ForegroundColor Green
    Tail-File $ingestErr

    # Show FAISS index files and sizes
    $faissFolder = Join-Path -Path (Get-Location).Path -ChildPath 'data\faiss'
    if (Test-Path $faissFolder) {
        Write-Host "\nFAISS Files and DB count:" -ForegroundColor Cyan
        Get-ChildItem -Path $faissFolder -File | Select-Object Name, @{Name = 'SizeMB'; Expression = { [math]::Round($_.Length / 1MB, 2) } }, LastWriteTime | Format-Table -AutoSize
        $dbPath = Join-Path $faissFolder 'documents.db'
        if (Test-Path $dbPath) {
            try {
                $count = & python -c "import sqlite3; conn=sqlite3.connect('data/faiss/documents.db'); c=conn.cursor(); c.execute('SELECT COUNT(*) FROM documents'); print(c.fetchone()[0]); conn.close()" 2>$null
                Write-Host "Documents in DB: $count" -ForegroundColor Green
            }
            catch { Write-Host "Could not query documents.db: $_" -ForegroundColor Red }
        }
    }
}

function Check-Api-Health($url) {
    try {
        $start = Get-Date
        $resp = Invoke-WebRequest -Uri $url -Method Get -UseBasicParsing -TimeoutSec 5 -ErrorAction Stop
        $end = Get-Date
        $lat = ($end - $start).TotalMilliseconds
        Write-Host "API Health: HTTP $($resp.StatusCode)  - Latency: $([math]::Round($lat,1)) ms" -ForegroundColor Green
        return @{ status = $true; code = $resp.StatusCode; latency = $lat }
    }
    catch [System.Net.WebException] {
        Write-Host "API Health: Unreachable ($($_.Exception.Message))" -ForegroundColor Red
        return @{ status = $false; error = $_.Exception.Message }
    }
    catch {
        Write-Host "API Health: Error ($($_.Exception.Message))" -ForegroundColor Red
        return @{ status = $false; error = $_.Exception.Message }
    }
}

function Launch-GPing {
    param($url)
    $uri = [uri]$url
    $host = $uri.Host
    $gpingCmd = Get-Command gping -ErrorAction SilentlyContinue
    if (-not $gpingCmd) {
        Write-Host "gping not found in PATH. Install gping (https://github.com/orf/gping) or run it manually: gping $host" -ForegroundColor Yellow
        return $false
    }
    # Do not start if already running
    $existing = Get-Process -ErrorAction SilentlyContinue | Where-Object { $_.ProcessName -eq 'gping' -or ($_.Path -and $_.Path -like '*gping*') }
    if ($existing) {
        Write-Host "gping already running (PID: $($existing.Id)), not starting a new instance." -ForegroundColor Cyan
        return $true
    }
    Write-Host "Starting gping on host: $host" -ForegroundColor Cyan
    Start-Process -FilePath "gping" -ArgumentList $host -NoNewWindow -WindowStyle Normal
    return $true
}

# Main loop
$monitorRun = $null
if ($RunDir -eq 'results') {
    $last = Get-Last-Run -runParent 'results/benchmark_runs'
    if ($last) { $monitorRun = $last.FullName } else { $monitorRun = 'results' }
}
else { $monitorRun = $RunDir }

Write-Host "Monitoring run: $monitorRun" -ForegroundColor Cyan

if (-not $Follow) {
    Show-Process-Info
    Show-Run-Stats -runDir $monitorRun
    if ($ApiUrl) { Check-Api-Health $ApiUrl }
    exit 0
}

while ($true) {
    Clear-Host
    Write-Host "Monitor: $(Get-Date)  (Run: $monitorRun)" -ForegroundColor Green
    Show-Process-Info
    Show-Run-Stats -runDir $monitorRun
    if ($ApiUrl) { Check-Api-Health $ApiUrl }
    if ($UseGPing) { Launch-GPing -url $ApiUrl }
    Start-Sleep -Seconds $IntervalSeconds
}
