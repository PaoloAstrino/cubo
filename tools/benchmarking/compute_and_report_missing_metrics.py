#!/usr/bin/env python3
import glob
import json
import os
import re
import subprocess

runs = sorted(
    [r for r in glob.glob("results/beir_run_*.json") if not r.endswith("_metrics_k10.json")]
)
if not runs:
    print("No run files found.")
    exit(0)

for r in runs:
    # skip some helper files
    if r.endswith("_metrics_k10.json"):
        continue
    metrics_fn = r.replace(".json", "_metrics_k10.json")
    if os.path.exists(metrics_fn):
        print(f"SKIP (exists): {metrics_fn}")
        continue
    # infer dataset
    m = re.match(r"results/beir_run_(.+?)_topk", os.path.basename(r))
    if m:
        dataset = m.group(1)
    else:
        m2 = re.match(r"results/beir_run_(.+?)\.json", os.path.basename(r))
        dataset = m2.group(1) if m2 else "unknown"
    # find qrels
    qrels_candidate = f"data/beir/{dataset}/qrels/test.tsv"
    if not os.path.exists(qrels_candidate):
        # try variants
        found = None
        for path in glob.glob("data/beir/**/qrels/test.tsv", recursive=True):
            if dataset in path:
                found = path
                break
        if not found:
            # fallback: if only one qrels in dataset dir, try
            # try exact dataset name match with undersc/ dash replacements
            alt = dataset.replace("-", "_")
            for path in glob.glob("data/beir/**/qrels/test.tsv", recursive=True):
                if alt in path:
                    found = path
                    break
        qrels_candidate = found
    if not qrels_candidate or not os.path.exists(qrels_candidate):
        print(f"WARNING: qrels missing for dataset {dataset}; skipping {r}")
        continue
    print(f"Computing metrics for {os.path.basename(r)} using qrels {qrels_candidate}...")
    try:
        subprocess.run(
            [
                "python",
                "tools/calculate_beir_metrics.py",
                "--results",
                r,
                "--qrels",
                qrels_candidate,
                "--k",
                "10",
            ],
            check=True,
        )
        # read metrics
        with open(metrics_fn, "r", encoding="utf-8") as mf:
            mdata = json.load(mf)
        print(json.dumps({"dataset": dataset, "run": os.path.basename(r), "metrics": mdata}))
    except subprocess.CalledProcessError as e:
        print(f"ERROR computing metrics for {r}: {e}")

print("Done.")
