#!/usr/bin/env python3
"""Update `evaluation_antigravity.md` with new metrics from ablation/reranker/system runs."""
import argparse
import json
from pathlib import Path

REPORT = Path("evaluation_antigravity.md")


def append_section(title, content):
    with open(REPORT, "a", encoding="utf-8") as f:
        f.write("\n---\n\n")
        f.write(f"## {title}\n\n")
        f.write(content)
        f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", help="Path to ablation JSON summary or directory")
    parser.add_argument("--reranker", help="Path to reranker eval JSON")
    parser.add_argument("--system", help="Path to system metrics JSON")
    args = parser.parse_args()

    if args.ablation:
        # If directory, list JSONs
        p = Path(args.ablation)
        content = ""
        if p.is_dir():
            for j in sorted(p.glob("*.json")):
                content += f"- Results file: `{j}`\n"
        else:
            content += f"- Results file: `{p}`\n"
        append_section("Ablation Results", content)

    if args.reranker:
        with open(args.reranker, "r", encoding="utf-8") as f:
            r = json.load(f)
        content = f"- No rerank peak RSS: {r['no_rerank']['peak_rss'] / (1024**2):.1f} MB\n"
        content += f"- With rerank peak RSS: {r['with_rerank']['peak_rss'] / (1024**2):.1f} MB\n"
        content += f"- No rerank p50/p95 (s): {r['no_rerank']['latency']['p50']:.3f}/{r['no_rerank']['latency']['p95']:.3f}\n"
        content += f"- With rerank p50/p95 (s): {r['with_rerank']['latency']['p50']:.3f}/{r['with_rerank']['latency']['p95']:.3f}\n"
        append_section("Reranker Effect", content)

    if args.system:
        with open(args.system, "r", encoding="utf-8") as f:
            s = json.load(f)
        idx = s["indexing"]
        q = s["query"]
        content = f"- Indexing time: {idx.get('time_s', 0):.1f}s, peak RSS: {idx.get('peak_rss', 0)/(1024**2):.1f} MB, indexed: {idx.get('indexed_count', 'N/A')}\n"
        content += f"- Query p50/p95 (s): {q.get('p50', 0):.3f}/{q.get('p95', 0):.3f}, peak RSS: {q.get('peak_rss', 0)/(1024**2):.1f} MB, total_time: {q.get('total', 0):.1f}s\n"
        append_section("System Metrics", content)

    print("Report updated (appended)")
