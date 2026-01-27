"""Fallback sensitivity sweep using DocumentRetriever when scaffold retriever isn't available."""

import json
import statistics
import time
from pathlib import Path

from cubo.retrieval.retriever import DocumentRetriever

index_dir = "data/beir_index_scifact"
queries_file = "data/beir/scifact/queries_quick50.jsonl"
output = "results/sensitivity_scifact_canonical_quick_fixed.json"

queries = [json.loads(l) for l in open(queries_file) if l.strip()]
queries = queries[:20]

retriever = DocumentRetriever(index_dir)
collection = getattr(retriever, "collection", None)

nprobe_values = [1, 2, 4, 8]
results = []
for nprobe in nprobe_values:
    latencies = []
    try:
        if collection and hasattr(collection, "index") and collection.index:
            collection.index.nprobe = nprobe
    except Exception:
        pass
    for q in queries:
        text = q.get("text") or q.get("query")
        if not text:
            continue
        start = time.perf_counter()
        _ = retriever.retrieve(text, top_k=10)
        elapsed = (time.perf_counter() - start) * 1000
        latencies.append(elapsed)
    if latencies:
        res = {
            "nprobe": nprobe,
            "num_queries": len(latencies),
            "latency_mean_ms": statistics.mean(latencies),
            "latency_median_ms": statistics.median(latencies),
            "latency_std_ms": statistics.pstdev(latencies),
            "latency_min_ms": min(latencies),
            "latency_max_ms": max(latencies),
            "latency_p95_ms": sorted(latencies)[int(0.95 * len(latencies)) - 1],
            "latency_p99_ms": sorted(latencies)[int(0.99 * len(latencies)) - 1],
        }
        results.append(res)

analysis = {
    "nprobe_range": {"min": min(nprobe_values), "max": max(nprobe_values)},
    "latency_range": {
        "min": min(r["latency_mean_ms"] for r in results),
        "max": max(r["latency_mean_ms"] for r in results),
    },
    "latency_increase_factor": max(r["latency_mean_ms"] for r in results)
    / min(r["latency_mean_ms"] for r in results),
    "monotonic_increase": all(
        results[i]["latency_mean_ms"] <= results[i + 1]["latency_mean_ms"]
        for i in range(len(results) - 1)
    ),
    "recommended_nprobe": results[0]["nprobe"] if results else None,
}

out = {
    "config": {
        "index_dir": index_dir,
        "queries_file": queries_file,
        "num_samples": len(queries),
        "nprobe_values": nprobe_values,
        "top_k": 10,
    },
    "results": results,
    "analysis": analysis,
}

Path(output).parent.mkdir(parents=True, exist_ok=True)
with open(output, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
print("wrote", output)
