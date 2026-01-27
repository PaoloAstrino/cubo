#!/usr/bin/env python3
"""Compare the best RRF settings in docs/eval/rrf_sensitivity_summary.md
with entries in docs/eval/evaluation_antigravity.md and print a small report.
"""
import re
from pathlib import Path

sfile = Path("docs/eval/rrf_sensitivity_summary.md").read_text(encoding="utf-8")
afile = Path("docs/eval/evaluation_antigravity.md").read_text(encoding="utf-8")

best_current = {}
for section in sfile.split("\n\n"):
    if section.startswith("## "):
        lines = section.splitlines()
        ds = lines[0].strip().lstrip("## ").strip()
        for l in lines:
            if l.strip().startswith("| rrf_"):
                parts = [p.strip() for p in l.strip().strip("|").split("|")]
                tag = parts[0]
                recall = float(parts[4])
                mrr = float(parts[5])
                ndcg = float(parts[6])
                best_current[ds.lower()] = (tag, recall, mrr, ndcg)
                break

best_prev = {}
for line in afile.splitlines():
    m = re.search(r"\*\*(\w+)\*\* best: (rrf_[^\s]+) \(Recall@10=([0-9.]+), nDCG=([0-9.]+)\)", line)
    if m:
        ds = m.group(1).lower()
        tag = m.group(2)
        recall = float(m.group(3))
        ndcg = float(m.group(4))
        best_prev[ds] = (tag, recall, ndcg)

report = []
for ds, cur in best_current.items():
    prev = best_prev.get(ds)
    if prev:
        ptag, precall, pndcg = prev
        tag, recall, mrr, ndcg = cur
        if abs(recall - precall) < 1e-9 and abs(ndcg - pndcg) < 1e-9:
            status = "no change"
        else:
            status = f"recall {precall:.4f} -> {recall:.4f}, nDCG {pndcg:.4f} -> {ndcg:.4f}"
    else:
        status = "no previous baseline found"
    report.append((ds, cur[0], recall, ndcg, status))

for r in report:
    print(f"{r[0]}: best={r[1]} (Recall={r[2]:.4f}, nDCG={r[3]:.4f}) -> {r[4]}")
