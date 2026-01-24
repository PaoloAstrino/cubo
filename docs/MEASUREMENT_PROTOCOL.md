    # CUBO Measurement Protocol

**Date:** January 24, 2026  
**Status:** OFFICIAL MEASUREMENT STANDARD  
**Applies to:** All baseline experiments, latency/throughput measurements, resource monitoring

---

## Overview

This document defines the **exact procedures, hardware configuration, and timing methodology** for all CUBO baseline experiments. Following this protocol ensures reproducible results across different machines and enables fair comparison between systems (BM25, E5-small, RRF, CUBO, etc.).

**Key Principle:** Measure on consumer hardware with realistic constraints, not ideal conditions. Report all overhead (model loading, I/O, fusion) in latency numbers.

---

## Part 1: Hardware Configuration

### 1.1 Target Hardware Profile

CUBO's design targets **two canonical configurations**:

#### Configuration A: Office Laptop (Entry-Level)
```
CPU:     Intel Core i5-1135G7 (4 cores, 8 threads, 2.4–4.2 GHz)
RAM:     16 GB DDR4-3200 (single or dual channel)
Disk:    512 GB NVMe SSD (PCIe Gen3, ~3,500 MB/s read)
GPU:     Intel Iris Xe (shared memory, no discrete GPU)
OS:      Windows 11 Pro (22H2) or Ubuntu 22.04 LTS
```

**Use this for:** Latency measurements, memory profiling, single-user load profile

#### Configuration B: Entry Gaming Laptop (Higher-End Consumer)
```
CPU:     AMD Ryzen 5 5600H (6 cores, 12 threads, 3.3–4.6 GHz)
RAM:     16 GB DDR4-3200
Disk:    512 GB NVMe SSD (PCIe Gen4, ~4,500 MB/s read)
GPU:     NVIDIA RTX 4050 (6 GB VRAM, optional for baselines)
OS:      Ubuntu 22.04 LTS or Windows 11
```

**Use this for:** Throughput measurements, concurrent load profile

**If using different hardware:** Document exact specs and note deviations in result tables.

---

### 1.2 Pre-Measurement System Setup

Run these commands **BEFORE** any baseline experiments:

#### Step 1: Disable CPU Turbo Boost (Critical for Reproducibility)

**Windows 11:**
```powershell
# Run as Administrator
# Check current turbo setting
Get-WmiObject -Namespace root\cimv2 -Class Win32_ProcessorSetting | Select-Object *

# Disable turbo boost via registry (requires restart)
Set-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\Power" `
  -Name "MachineProcessorIdlePolicy" -Value 1

# Verify: Check Windows Device Manager → Processor Power Settings
# CPU should show "2.4 GHz" fixed (not "2.4–4.2 GHz" variable)
```

**Ubuntu/Linux:**
```bash
# Disable turbo boost on Intel
echo "1" | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo

# Disable turbo boost on AMD
echo "0" | sudo tee /sys/devices/system/cpu/cpufreq/boost

# Verify
cat /sys/devices/system/cpu/intel_pstate/no_turbo  # Should output "1"

# Make permanent (reboot required):
sudo bash -c 'echo "vm.mmap_min_addr = 0" >> /etc/sysctl.conf'
```

#### Step 2: Pin CPU Frequency to Base Clock

**Windows 11:**
```powershell
# Use ThrottleStop or RwEverything to lock frequency
# OR use power profile settings:
powercfg /SETACTIVE SCHEME_CURRENT
powercfg /CHANGE SCHEME_CURRENT SUB_PROCESSOR PERFBOOSTMODE 0  # Disable performance boost
```

**Ubuntu/Linux:**
```bash
# Pin to base frequency (2.4 GHz for i5-1135G7)
for cpu in /sys/devices/system/cpu/cpu*/; do
  echo "powersave" | sudo tee "$cpu/cpufreq/scaling_governor"
  echo "2400000" | sudo tee "$cpu/cpufreq/scaling_max_freq"  # In kHz
done

# Verify all CPUs locked to 2.4 GHz
watch -n 0.1 'cat /proc/cpuinfo | grep MHz'
```

#### Step 3: Check RAM & Memory Availability

```bash
# Ensure >20 GB free RAM (to avoid swapping)
# Windows:
Get-CimInstance Win32_OperatingSystem | Select-Object @{N="AvailableMemory(GB)"; E={$_.FreePhysicalMemory/1MB}}

# Linux:
free -h
# Output should show "Avail" > 20 GB
```

#### Step 4: Measure Disk Performance

```bash
# Windows: CrystalDiskInfo or:
# Benchmark read speed
$disk = Get-PhysicalDisk | Where-Object {$_.DeviceId -eq "Disk 0"}
Write-Host "Disk Speed: $($disk.BusType) - $(($disk.Size/1GB)GB total)"

# Linux: fio (flexible I/O tester)
sudo apt-get install fio

# 4KB random read benchmark (representative of FAISS I/O pattern)
fio --name=random-read --ioengine=libaio --iodepth=32 \
    --rw=randread --bs=4K --size=1G --numjobs=1 \
    --filename=/mnt/test_file --group_reporting --output=disk_bench.txt

# Typical results:
# NVMe Gen3: 15,000–25,000 IOPS
# NVMe Gen4: 40,000–80,000 IOPS
# SATA SSD: 3,000–5,000 IOPS
```

---

### 1.3 System State Checklist

**BEFORE every measurement run, verify:**

```bash
# 1. CPU frequency pinned (run on both Windows & Linux)
# Windows: Task Manager → Performance → CPU → Speed shows "2.4 GHz" (not variable)
# Linux: watch -n 0.1 'cat /proc/cpuinfo | grep MHz'
✓ Check: All CPUs show fixed base frequency

# 2. No background processes consuming resources
# Windows:
Get-Process | Sort-Object WorkingSet -Descending | Select-Object -First 10

# Linux:
ps aux --sort=-%mem | head -10

✓ Check: Only OS + Python interpreter in top 10 (nothing >5% CPU or 1GB RAM)

# 3. Sufficient free RAM
# Windows: (see above) > 20 GB free
# Linux: free -h | grep Mem (Avail > 20 GB)
✓ Check: Free RAM > 20 GB

# 4. Network disconnected (if testing locally)
# Windows: Settings → Network → Status → Disconnect
# Linux: sudo nmcli radio wifi off && sudo nmcli radio wwan off
✓ Check: `ping google.com` fails or times out

# 5. No antivirus scans running
# Windows: Windows Defender → Virus & threat protection → Check "Last scan"
✓ Check: No active scans (reschedule if needed)
```

---

## Part 2: Cache Setup Procedures

### 2.1 Cold Cache Startup (First-Query Measurement)

Cold cache simulates a fresh system boot with no residual data in memory.

#### Procedure:

```bash
# Step 1: Restart the machine
# Windows: Restart-Computer -Force
# Linux: sudo reboot

# Step 2: Wait for system stabilization (2–3 minutes)
# Step 3: Open terminal
# Step 4: Clear OS page cache (Linux only; Windows auto-flushes on restart)

# Linux:
sudo sync
echo 3 | sudo tee /proc/sys/vm/drop_caches

# Step 5: Run single query immediately
python -c "
import subprocess
import time
start = time.perf_counter_ns()
result = run_query('test query')
end = time.perf_counter_ns()
print(f'Cold start latency: {(end - start) / 1e6:.2f} ms')
"

# Step 6: Log the result as "First Query (Cold)"
```

**Expected behavior:**
- First query is 1.5–3× slower than steady state (due to model loading, cache misses, I/O)
- Subsequent queries are faster as caches warm

#### Example Results:
```
CUBO cold start (first query): 340 ms
CUBO steady state (query 11–50): 234 ms (45% faster)
```

---

### 2.2 Warm Cache Steady State (Typical User Experience)

Warm cache represents normal usage after system has been running for a few minutes.

#### Procedure:

```bash
# Step 1: After cold start above, run 10 dummy "warm-up" queries
for i in {1..10}; do
    python -c "run_query('warmup query $i')"
done

# Step 2: Clear timing logs (start fresh)
# Step 3: Run 100–200 test queries (e.g., from SciFact dev set)
# Step 4: Collect timing for queries 11–110 (skip first 10)
# Step 5: Report p50, p95, p99 from queries 11–110

python benchmark.py \
  --dataset scifact \
  --num_queries 200 \
  --warmup_queries 10 \
  --measure_from_query 11 \
  --report_percentiles 50,95,99
```

**Expected behavior:**
- Queries 11–50: stabilize around baseline latency
- Queries 51–200: remain flat (steady state)
- No GC pauses visible in raw latencies

---

### 2.3 Mixed Cache State (Realistic Varied Workload)

Optional: Simulate varied query patterns (different cache hit rates).

#### Procedure:

```bash
# Run 50 unique queries (diverse vocabulary → varied cache behavior)
# Don't flush cache; let OS manage naturally
# Report distribution: min, p25, p50, p75, p95, p99, max

python benchmark.py \
  --dataset scifact \
  --num_queries 50 \
  --cache_mode mixed \
  --report_distribution
```

---

## Part 3: Timing Methodology

### 3.1 What to Measure (and What NOT to)

#### Measure: **End-to-End Latency** (what users experience)

```python
import time

def measure_query_latency(query_text, system):
    """Measure total time from query input to results returned."""
    start = time.perf_counter_ns()
    
    # INCLUDE all overhead:
    # - Query embedding (if needed)
    # - Index search (FAISS, BM25)
    # - Result fusion (RRF, etc.)
    # - Reranking (if enabled)
    # - JSON serialization
    
    results = system.retrieve(query_text, top_k=100)
    
    end = time.perf_counter_ns()
    latency_ms = (end - start) / 1e6
    
    return latency_ms, results
```

#### DO NOT Measure (and why):

```python
# ❌ WRONG: Exclude I/O
latency = measure_faiss_search_only()  # Misleading, ignores 90% of overhead

# ❌ WRONG: Use time.time() (affected by system clock adjustments)
import time
t0 = time.time()  # Can jump backwards on NTP sync
# ... do work ...
t1 = time.time()

# ✅ CORRECT: Use perf_counter (monotonic clock)
import time
t0 = time.perf_counter_ns()  # Nanosecond precision, not affected by clock adjustments
# ... do work ...
t1 = time.perf_counter_ns()
latency_ms = (t1 - t0) / 1e6
```

---

### 3.2 Percentile Reporting Standard

**Always report three percentiles:** p50, p95, p99

| Percentile | Interpretation | Why Matters |
|-----------|-----------------|------------|
| **p50** | Median latency | Typical user experience |
| **p95** | 95th percentile | Acceptable worst case for 95% of users |
| **p99** | 99th percentile | Outlier GC pauses, disk I/O spikes |

#### Reporting Format:

```
System: CUBO (SciFact, warm cache, n=100 queries)
p50 latency: 234 ms
p95 latency: 456 ms
p99 latency: 612 ms
Range: 198–890 ms
Std dev: ±52 ms
```

#### Python Implementation:

```python
import numpy as np

def report_latencies(latencies_ms, system_name, config):
    """Generate standardized latency report."""
    p50 = np.percentile(latencies_ms, 50)
    p95 = np.percentile(latencies_ms, 95)
    p99 = np.percentile(latencies_ms, 99)
    
    report = f"""
System: {system_name} ({config})
n queries: {len(latencies_ms)}
p50 latency: {p50:.1f} ms
p95 latency: {p95:.1f} ms
p99 latency: {p99:.1f} ms
Range: {min(latencies_ms):.1f}–{max(latencies_ms):.1f} ms
Std dev: ±{np.std(latencies_ms):.1f} ms
Mean: {np.mean(latencies_ms):.1f} ms
    """
    return report
```

---

### 3.3 Queries Per Second (QPS) / Throughput

For concurrent load testing, measure **queries processed per second**.

#### Single-Threaded QPS:

```python
import time

def measure_qps(system, queries, duration_sec=60):
    """Measure QPS in single-threaded mode."""
    count = 0
    start = time.perf_counter()
    
    while (time.perf_counter() - start) < duration_sec:
        query = queries[count % len(queries)]
        results = system.retrieve(query, top_k=100)
        count += 1
    
    elapsed = time.perf_counter() - start
    qps = count / elapsed
    return qps
```

#### Multi-Worker QPS:

```python
from concurrent.futures import ThreadPoolExecutor
import time

def measure_concurrent_qps(system, queries, num_workers=4, duration_sec=60):
    """Measure QPS with concurrent workers."""
    
    def worker():
        count = 0
        while (time.perf_counter() - start) < duration_sec:
            query = queries[count % len(queries)]
            system.retrieve(query, top_k=100)
            count += 1
        return count
    
    start = time.perf_counter()
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(worker) for _ in range(num_workers)]
        total_queries = sum(f.result() for f in futures)
    
    elapsed = time.perf_counter() - start
    qps = total_queries / elapsed
    return qps
```

#### Reporting Format:

```
Single-threaded QPS: 4.3 queries/sec (p50 latency 234 ms)
4-worker concurrent QPS: 9.2 queries/sec (avg per-worker latency 434 ms)
```

---

## Part 4: Load Profiles

Define **two standard load profiles** for all baseline experiments:

### 4.1 Single-Threaded Profile (Latency Focus)

**Use case:** Measure individual query latency (what each user experiences)

#### Configuration:

```python
load_profile = {
    "type": "single-threaded",
    "num_queries": 100,
    "warmup_queries": 10,
    "measure_from": 11,
    "cache_state": "warm",
    "report_metrics": ["p50", "p95", "p99", "mean", "std", "min", "max"]
}
```

#### Procedure:

```bash
# Run baseline systems in single-threaded mode
python benchmark.py \
  --system cubo \
  --system bm25 \
  --system e5-small \
  --system rrf \
  --profile single-threaded \
  --num_queries 100 \
  --warmup 10 \
  --output results/latency_comparison.json
```

#### Expected Results Table:

```
System      | p50 (ms) | p95 (ms) | p99 (ms) | Mean (ms) | Std (ms)
BM25        | 12       | 25       | 45       | 15        | 8
E5-small    | 187      | 234      | 289      | 201       | 31
RRF         | 198      | 312      | 412      | 215       | 48
CUBO        | 234      | 456      | 612      | 278       | 89
```

---

### 4.2 Concurrent Workers Profile (Throughput Focus)

**Use case:** Measure system performance under concurrent load

#### Configuration:

```python
load_profile = {
    "type": "concurrent",
    "num_workers": [1, 2, 4],  # Test with increasing workers
    "duration_per_config": 60,  # seconds
    "cache_state": "warm",
    "measure_metrics": ["qps", "avg_latency", "p95_latency", "cpu_usage", "memory_usage"]
}
```

#### Procedure:

```bash
# Run all systems with varying worker counts
for workers in 1 2 4; do
  python benchmark.py \
    --system cubo \
    --profile concurrent \
    --num_workers $workers \
    --duration 60 \
    --output results/throughput_${workers}w.json
done
```

#### Expected Results Table:

```
System | Workers | QPS   | Avg Latency (ms) | p95 Latency (ms) | CPU % | RAM (GB)
BM25   | 1       | 83    | 12               | 25               | 45    | 0.5
BM25   | 2       | 156   | 13               | 28               | 85    | 0.6
BM25   | 4       | 198   | 20               | 50               | 95    | 0.7
---
CUBO   | 1       | 4.3   | 234              | 456              | 35    | 12.1
CUBO   | 2       | 6.8   | 294              | 567              | 65    | 12.3
CUBO   | 4       | 9.2   | 434              | 890              | 95    | 12.8
```

---

## Part 5: Memory & Resource Monitoring

### 5.1 Peak Memory Measurement

Track memory usage during query execution.

```python
import psutil
import subprocess
import os

def measure_peak_memory(system, num_queries=100):
    """Monitor peak memory usage during query execution."""
    
    process = psutil.Process(os.getpid())
    
    peak_rss = 0
    peak_uss = 0
    
    for i in range(num_queries):
        query = get_test_query(i)
        results = system.retrieve(query, top_k=100)
        
        # Check memory after each query
        mem_info = process.memory_info()
        peak_rss = max(peak_rss, mem_info.rss / 1e9)  # Convert to GB
        peak_uss = max(peak_uss, mem_info.uss / 1e9)
    
    return {
        "peak_rss": peak_rss,  # Resident Set Size (physical RAM)
        "peak_uss": peak_uss,  # Unique Set Size (not shared)
    }
```

#### Reporting Format:

```
CUBO Memory Usage (100 queries, warm cache):
Peak RSS: 12.1 GB (includes Python, models, caches)
Peak USS: 8.3 GB (unique memory, not shared with OS)
Heap fragmentation: 1.4 GB (estimate)
```

---

### 5.2 CPU & I/O Monitoring

```bash
# Linux: Monitor system-wide metrics during benchmark
dstat -tcsm --disk --net 10 > dstat.log &
DSTAT_PID=$!

# Run benchmark
python benchmark.py --system cubo --num_queries 100

# Stop monitoring
kill $DSTAT_PID

# Analyze: dstat_log shows CPU%, memory growth, disk I/O patterns
```

---

## Part 6: Reproducibility Checklist

Before **every** baseline run, verify:

### Pre-Run Checklist

```markdown
## BASELINE RUN CHECKLIST

### Hardware Preparation
- [ ] CPU frequency pinned to base clock (no turbo boost)
- [ ] Confirmed all CPUs locked (watch -n 0.1 'cat /proc/cpuinfo | grep MHz')
- [ ] Free RAM > 20 GB (free -h || Get-CimInstance Win32_OperatingSystem)
- [ ] Disk IOPS measured and logged (~4KB random read test)
- [ ] Network disconnected (ping google.com fails)
- [ ] No background processes (top 10 processes show <5% CPU each)
- [ ] Antivirus scan completed, not running

### Cache Configuration
- [ ] Cache mode documented: [ ] Cold [ ] Warm [ ] Mixed
- [ ] If cold: machine restarted, page cache cleared, logged first-query time
- [ ] If warm: 10 warmup queries run, measurements start from query 11
- [ ] Cache state verified before each test (ps aux | grep -i cache, etc.)

### Timing Setup
- [ ] Using time.perf_counter_ns() (monotonic, not affected by clock skew)
- [ ] Latencies in milliseconds (not seconds or microseconds)
- [ ] Collecting p50, p95, p99, mean, std, min, max
- [ ] n≥100 queries (for statistical significance)
- [ ] End-to-end latency (including all overhead)

### Load Profile
- [ ] Single-threaded: [ ] Yes [ ] No
- [ ] Concurrent workers: [ ] Yes (num_workers=__) [ ] No
- [ ] Duration per config: [ ] 60 sec (or specify: __)
- [ ] Results logged to JSON with timestamp

### Documentation
- [ ] System spec documented (CPU, RAM, disk, OS version)
- [ ] Date and time of run recorded
- [ ] All parameters saved in result file (num_queries, warmup, etc.)
- [ ] Disk IOPS baseline included in results
- [ ] Peak memory measurements collected

### Post-Run Analysis
- [ ] Results reviewed for anomalies (p50 > p95? latency = 0?)
- [ ] Comparison to prior runs (is variance expected?)
- [ ] System state verified (CPU usage, no unexpected processes)
- [ ] Results saved with unique filename (results/baseline_SYSTEM_DATETIME.json)
```

---

## Part 7: Running Baseline Experiments

### 7.1 BM25 Baseline (Pyserini)

```bash
# Setup
pip install pyserini

# Index SciFact corpus (one-time)
python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input data/scifact_corpus/ \
  --index indexes/scifact_bm25

# Run queries with timing
python scripts/run_bm25_baseline.py \
  --index indexes/scifact_bm25 \
  --queries data/scifact_queries.jsonl \
  --output results/bm25_latency.json \
  --warmup 10 \
  --num_queries 100
```

### 7.2 E5-Small Dense Baseline

```bash
# Setup
pip install sentence-transformers faiss-cpu

# Create index (one-time)
python scripts/create_e5_index.py \
  --corpus data/scifact_corpus/ \
  --model intfloat/e5-small-v2 \
  --index indexes/scifact_e5 \
  --quantization 8bit

# Run queries with timing
python scripts/run_dense_baseline.py \
  --index indexes/scifact_e5 \
  --queries data/scifact_queries.jsonl \
  --output results/e5_latency.json \
  --warmup 10 \
  --num_queries 100
```

### 7.3 Hybrid RRF Baseline

```bash
# Run RRF fusion combining BM25 + E5
python scripts/run_rrf_baseline.py \
  --bm25_index indexes/scifact_bm25 \
  --dense_index indexes/scifact_e5 \
  --queries data/scifact_queries.jsonl \
  --output results/rrf_latency.json \
  --k 60 \
  --warmup 10 \
  --num_queries 100
```

### 7.4 Comparison Table Generation

```bash
python scripts/generate_comparison_table.py \
  --results results/bm25_latency.json \
             results/e5_latency.json \
             results/rrf_latency.json \
             results/cubo_latency.json \
  --output results/comparison_table.md
```

Expected output:

```markdown
# Baseline Comparison (SciFact, Warm Cache, n=100)

| System    | p50 (ms) | p95 (ms) | p99 (ms) | Peak RAM (GB) | nDCG@10 |
|-----------|----------|----------|----------|---------------|---------|
| BM25      | 12       | 25       | 45       | 0.5           | 0.645   |
| E5-small  | 187      | 234      | 289      | 9.2           | 0.501   |
| RRF       | 198      | 312      | 412      | 9.3           | 0.671   |
| CUBO      | 234      | 456      | 612      | 12.1          | 0.668   |
```

---

## Part 8: Deviation Handling

### What if Results Seem Anomalous?

**Problem: p50 latency is 0 ms**
- ❌ Measurement error (timing code issue)
- ✅ Add verbose logging: `timing_debug=True`
- ✅ Rerun with explicit per-query timing output
- ✅ Check if result caching is interfering (disable cache, retry)

**Problem: p95 < p50 (impossible)**
- ❌ Data collection error
- ✅ Check percentile calculation: `np.percentile(data, 95)` (should be ≥ p50)
- ✅ Rerun with clean data

**Problem: Latency varies wildly between runs (std > mean)**
- ✓ This is normal if GC pauses occur (expected for Python)
- ✓ Report as-is: "High variance due to Python GC; see individual GC pause times"
- ✓ Increase n_queries to stabilize (n=500 instead of 100)

**Problem: RAM usage different from expected**
- ✓ Check for memory leaks: `watch -n 1 'ps aux | grep python'`
- ✓ Document actual vs expected in results
- ✓ Don't exclude high values; they're real

---

## Part 9: Baseline Scripts

Create these Python scripts for standardized measurements:

### scripts/run_bm25_baseline.py

```python
#!/usr/bin/env python3
"""BM25 baseline measurement with standardized protocol."""

import json
import time
import numpy as np
from pyserini.search.lucene import LuceneSearcher
import argparse

def measure_bm25(index_path, queries_file, num_queries=100, warmup=10):
    """Run BM25 queries with timing."""
    searcher = LuceneSearcher(index_path)
    
    with open(queries_file) as f:
        queries = [json.loads(line)['text'] for line in f][:num_queries + warmup]
    
    latencies = []
    
    # Warmup phase
    for i in range(warmup):
        query = queries[i]
        results = searcher.search(query, k=100)
    
    # Measurement phase
    for i in range(warmup, len(queries)):
        query = queries[i]
        
        start = time.perf_counter_ns()
        results = searcher.search(query, k=100)
        end = time.perf_counter_ns()
        
        latency_ms = (end - start) / 1e6
        latencies.append(latency_ms)
    
    return latencies

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--num_queries", type=int, default=100)
    args = parser.parse_args()
    
    latencies = measure_bm25(args.index, args.queries, args.num_queries, args.warmup)
    
    result = {
        "system": "BM25",
        "num_queries": len(latencies),
        "warmup_queries": args.warmup,
        "p50": float(np.percentile(latencies, 50)),
        "p95": float(np.percentile(latencies, 95)),
        "p99": float(np.percentile(latencies, 99)),
        "mean": float(np.mean(latencies)),
        "std": float(np.std(latencies)),
        "min": float(np.min(latencies)),
        "max": float(np.max(latencies)),
        "latencies_ms": latencies,
    }
    
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Results saved to {args.output}")
    print(f"p50: {result['p50']:.1f} ms, p95: {result['p95']:.1f} ms, p99: {result['p99']:.1f} ms")
```

---

## Part 10: Version Control & Reproducibility

### 10.1 Document Exact Software Versions

```bash
# Create versions.txt before every run
pip freeze > results/versions_DATETIME.txt
python --version >> results/versions_DATETIME.txt
uname -a >> results/versions_DATETIME.txt  # Linux/Mac
systeminfo | findstr /C:"System Boot Time" >> results/versions_DATETIME.txt  # Windows
```

### 10.2 Archive Complete Experiment

```bash
# After measurement, save everything
tar -czf results/experiment_SYSTEM_DATETIME.tar.gz \
  results/*.json \
  results/versions_DATETIME.txt \
  scripts/
```

---

## Summary

This protocol ensures **reproducible, fair baseline comparisons**. Key points:

✅ **CPU pinned** to base frequency (no turbo boost variability)  
✅ **n≥100 queries** measured (statistical significance)  
✅ **p50, p95, p99 reported** (not cherry-picked single numbers)  
✅ **Cold and warm cache** measured separately  
✅ **End-to-end latency** including all overhead  
✅ **Hardware documented** (CPU, RAM, disk, OS)  
✅ **Version control** of all software packages  

**Ready to run baselines.** Follow this protocol for all BM25, E5-small, RRF, and CUBO measurements.

---

*Last updated: January 24, 2026*  
*Status: OFFICIAL - Use for all baseline experiments*
