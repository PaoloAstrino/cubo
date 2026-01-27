#!/usr/bin/env python3
"""Check coherence between docs/eval/evaluation_antigravity.md and results/aggregated_metrics.csv

Prints a summary of matches/mismatches for Recall@10 values.
"""
import re
import sys
from pathlib import Path

import pandas as pd

md_path = Path("docs/eval/evaluation_antigravity.md")
csv_path = Path("results/aggregated_metrics.csv")

if not md_path.exists():
    print("ERROR: docs/eval/evaluation_antigravity.md not found")
    sys.exit(1)
if not csv_path.exists():
    print("ERROR: results/aggregated_metrics.csv not found")
    sys.exit(1)

md = md_path.read_text(encoding="utf-8")
# extract Results Summary table block
m = re.search(r"## Results Summary\n\n(.*?)\n\n---", md, re.S)
if not m:
    print("ERROR: Cannot find Results Summary block in docs/eval/evaluation_antigravity.md")
    sys.exit(1)

table = m.group(1)
lines = [l for l in table.splitlines() if l.strip().startswith("|")]
# header line index
if len(lines) < 3:
    print("ERROR: Results table seems too small")
    sys.exit(1)

header = lines[0]
rows = lines[2:]  # skip header and separator

# parse rows into (dataset, recall)
md_entries = []
for r in rows:
    parts = [p.strip() for p in r.split("|") if p.strip()]
    if len(parts) < 5:
        continue
    dataset = parts[0]
    recall_str = parts[4]
    try:
        recall = float(recall_str)
    except Exception:
        # sometimes format like 1.0000 or with symbols; strip non-numeric
        recall = (
            float(re.sub(r"[^0-9.]", "", recall_str)) if re.search(r"[0-9]", recall_str) else None
        )
    md_entries.append((dataset, recall))

# load aggregated metrics
df = pd.read_csv(csv_path)
# create mapping dataset -> best recall (prefer topk50 dense rows where present)
agg_map = {}
for _, row in df.iterrows():
    src = str(row.get("source_file", ""))
    # infer dataset key
    m = re.search(r"beir_run_([a-z0-9_]+)", src)
    key = m.group(1) if m else src
    key = key.lower()
    # take recall_at_k
    rcv = float(row.get("recall_at_k") or 0.0)
    # if key already present, keep highest recall (coarse)
    if key in agg_map:
        if rcv > agg_map[key]:
            agg_map[key] = rcv
    else:
        agg_map[key] = rcv

# matching logic
print("Coherence check results:")
ok = []
mismatches = []
missing = []
for ds, md_rec in md_entries:
    ds_key = ds.lower()
    # normalize names: allow prefix match
    candidates = [k for k in agg_map.keys() if k.startswith(ds_key) or ds_key in k]
    if not candidates:
        missing.append((ds, md_rec))
        continue
    # pick candidate with maximum recall
    cand = max(candidates, key=lambda k: agg_map[k])
    agg_rec = agg_map[cand]
    if md_rec is None:
        missing.append((ds, md_rec))
    else:
        if abs(md_rec - agg_rec) > 1e-3:
            mismatches.append((ds, md_rec, agg_rec, cand))
        else:
            ok.append((ds, md_rec))

for ds, v in ok:
    print(f" OK: {ds} = {v:.4f}")
for ds, mdv in missing:
    print(f" MISSING: {ds} has no aggregated entry (table value {mdv})")
for ds, mdv, agg, cand in mismatches:
    print(f" MISMATCH: {ds} table={mdv:.4f} aggregated({cand})={agg:.4f}")

print("\nDone.")
