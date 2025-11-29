# CUBO Overnight BEIR Benchmark Script
# Run this script to: 1) Build FAISS index from BEIR, 2) Run benchmarks
# Results are saved and can be analyzed later

$ErrorActionPreference = "Stop"
$startTime = Get-Date

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "CUBO Overnight BEIR Benchmark" -ForegroundColor Cyan
Write-Host "Started: $startTime" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Set working directory
Set-Location "c:\Users\paolo\Desktop\cubo"

# Activate virtual environment
& .\.venv\Scripts\Activate.ps1

# Set PYTHONPATH
$env:PYTHONPATH = "c:\Users\paolo\Desktop\cubo"

# Create results directory
$resultsDir = "results\overnight_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
New-Item -ItemType Directory -Force -Path $resultsDir | Out-Null

# Log file
$logFile = "$resultsDir\overnight_log.txt"

function Log-Message {
    param($message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logEntry = "[$timestamp] $message"
    Write-Host $logEntry
    Add-Content -Path $logFile -Value $logEntry
}

# =============================================================================
# PHASE 0: QUICK VALIDATION (runs in ~2 minutes)
# =============================================================================
Log-Message "=== PHASE 0: Quick Validation Before Long Run ==="
Log-Message "Testing with 100 documents to verify everything works..."

try {
    # Test 1: Verify parquet exists
    if (-not (Test-Path "data/beir/beir_corpus.parquet")) {
        throw "beir_corpus.parquet not found! Run the conversion first."
    }
    Log-Message "[OK] beir_corpus.parquet exists"

    # Test 2: Verify build script works with small sample
    Log-Message "Testing index build with 100 docs (dry-run)..."
    python -c @"
import pandas as pd
import sys
sys.path.insert(0, '.')

# Load small sample
df = pd.read_parquet('data/beir/beir_corpus.parquet').head(100)
print(f'Loaded {len(df)} test documents')

# Test embedding generation
from src.cubo.embeddings.embedding_generator import EmbeddingGenerator
gen = EmbeddingGenerator(batch_size=32)
embeddings = gen.encode(df['text'].tolist()[:10], batch_size=32)
print(f'Generated {len(embeddings)} embeddings, dim={len(embeddings[0])}')

# Test FAISS index creation
from src.cubo.indexing.faiss_index import FAISSIndexManager
from pathlib import Path
import tempfile
with tempfile.TemporaryDirectory() as tmpdir:
    manager = FAISSIndexManager(dimension=len(embeddings[0]), index_dir=Path(tmpdir))
    manager.build_indexes(embeddings, [str(i) for i in range(len(embeddings))])
    hits = manager.search(embeddings[0], k=3)
    print(f'Test search returned {len(hits)} hits')

print('=== VALIDATION PASSED ===')
"@ 2>&1 | Tee-Object -Append -FilePath $logFile

    if ($LASTEXITCODE -ne 0) { throw "Validation failed!" }
    Log-Message "[OK] Index build test passed"

    # Test 3: Verify benchmark script can load
    Log-Message "Testing benchmark imports..."
    python -c @"
import sys
sys.path.insert(0, '.')
from benchmarks.retrieval.rag_benchmark import RAGTester
from benchmarks.utils.metrics import IRMetricsEvaluator
print('Benchmark imports OK')
"@ 2>&1 | Tee-Object -Append -FilePath $logFile

    if ($LASTEXITCODE -ne 0) { throw "Benchmark import failed!" }
    Log-Message "[OK] Benchmark imports verified"

    # Test 4: Verify ground truth loads
    Log-Message "Testing ground truth loading..."
    python -c @"
import json
with open('data/beir/ground_truth.json') as f:
    gt = json.load(f)
print(f'Ground truth has {len(gt)} queries')
with open('data/beir_with_ground_truth.json', encoding='utf-8') as f:
    qs = json.load(f)
print(f'Questions file has {len(qs.get(\"questions\", {}))} categories')
print('Ground truth loading OK')
"@ 2>&1 | Tee-Object -Append -FilePath $logFile

    if ($LASTEXITCODE -ne 0) { throw "Ground truth loading failed!" }
    Log-Message "[OK] Ground truth verified"

    Log-Message ""
    Log-Message "=============================================="
    Log-Message "ALL VALIDATIONS PASSED! Safe to run overnight."
    Log-Message "=============================================="
    Log-Message ""
    
    # Ask for confirmation before long run
    Write-Host "`nValidation complete. The full indexing will take 2-4 hours." -ForegroundColor Yellow
    $confirm = Read-Host "Start overnight run now? (y/n)"
    if ($confirm -ne "y" -and $confirm -ne "Y") {
        Log-Message "User cancelled. Exiting."
        exit 0
    }

}
catch {
    Log-Message "VALIDATION FAILED: $_"
    Write-Host "`nFix the issues above before running overnight." -ForegroundColor Red
    exit 1
}

# =============================================================================
# PHASE 1: BUILD FAISS INDEX (the long part)
# =============================================================================
Log-Message "=== PHASE 1: Building FAISS Index from BEIR Corpus ==="
Log-Message "This will take 2-4 hours for 57,638 documents..."

try {
    # Build FAISS index from BEIR corpus
    python src/cubo/scripts/build_faiss_index.py `
        --parquet data/beir/beir_corpus.parquet `
        --id-column chunk_id `
        --text-column text `
        --index-dir faiss_index_beir `
        --batch-size 64 `
        --hot-fraction 0.25 `
        --nlist 128 2>&1 | Tee-Object -Append -FilePath $logFile
    
    Log-Message "FAISS index build completed successfully!"
}
catch {
    Log-Message "ERROR during index build: $_"
    exit 1
}

# Populate documents.db for the new index
Log-Message "=== Populating documents.db ==="
python -c @"
import json
import sqlite3
import pandas as pd

df = pd.read_parquet('data/beir/beir_corpus.parquet')
conn = sqlite3.connect('faiss_index_beir/documents.db')
cur = conn.cursor()
cur.execute('CREATE TABLE IF NOT EXISTS documents (id TEXT PRIMARY KEY, content TEXT NOT NULL, metadata TEXT NOT NULL)')
for _, row in df.iterrows():
    meta = json.dumps({'source': 'beir_corpus'})
    cur.execute('INSERT OR REPLACE INTO documents (id, content, metadata) VALUES (?, ?, ?)', 
                (str(row['chunk_id']), row['text'], meta))
conn.commit()
print(f'Populated {len(df)} documents')
conn.close()
"@ 2>&1 | Tee-Object -Append -FilePath $logFile

Log-Message "=== PHASE 2: Running Benchmarks ==="

# Update config to use BEIR index
Log-Message "Updating config to use faiss_index_beir..."
python -c @"
import json
with open('config.json', 'r') as f:
    cfg = json.load(f)
cfg['faiss_index_dir'] = './faiss_index_beir'
cfg['vector_store_path'] = './faiss_index_beir'
with open('config.json', 'w') as f:
    json.dump(cfg, f, indent=2)
print('Config updated')
"@ 2>&1 | Tee-Object -Append -FilePath $logFile

# Run benchmark
Log-Message "Running BEIR IR metrics benchmark..."
try {
    python benchmarks/retrieval/rag_benchmark.py `
        --questions data/beir_with_ground_truth.json `
        --ground-truth data/beir/ground_truth.json `
        --mode retrieval-only `
        --skip-index `
        --output "$resultsDir/beir_benchmark_results.json" 2>&1 | Tee-Object -Append -FilePath $logFile
    
    Log-Message "Benchmark completed successfully!"
}
catch {
    Log-Message "ERROR during benchmark: $_"
}

# Summary
$endTime = Get-Date
$duration = $endTime - $startTime
Log-Message "========================================" 
Log-Message "Overnight run completed!"
Log-Message "Duration: $duration"
Log-Message "Results saved to: $resultsDir"
Log-Message "========================================"

# Copy final results
Copy-Item "$resultsDir\beir_benchmark_results.json" "results\latest_beir_results.json" -ErrorAction SilentlyContinue

Write-Host "`nDone! Check $resultsDir for results." -ForegroundColor Green
