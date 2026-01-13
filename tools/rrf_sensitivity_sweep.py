#!/usr/bin/env python3
"""RRF sensitivity sweep: fuse existing dense and bm25 runs with varying k and weights.
Writes run files and computes metrics via calculate_beir_metrics.py, and produces a summary markdown.
"""
import json
import subprocess
from pathlib import Path
from itertools import product
import math

DATASETS = ["arguana", "fiqa", "nfcorpus", "scifact"]
K_VALUES = [20, 60, 120]
WEIGHTS = [0.7, 1.0, 1.3]
TOP_K = 50
RESULTS_DIR = Path("results")
DOCS_DIR = Path("docs/eval")
SUMMARY_FILE = DOCS_DIR / "rrf_sensitivity_summary.md"

# Helper to load run json (qid -> {doc_id: score}) and convert to list format expected by rrf_fuse
def load_run_as_list(run_path):
    with open(run_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # data expected as mapping qid-> {doc_id: score}
    out = {}
    for qid, hits in data.items():
        # Skip any metadata keys (starting with underscore)
        if str(qid).startswith("_"):
            continue
        # hits should be a dict of doc_id->score, otherwise skip
        if not isinstance(hits, dict):
            continue
        # filter out non-numeric values and sort by score desc
        pairs = []
        for doc_id, score in hits.items():
            try:
                scoref = float(score)
            except Exception:
                continue
            pairs.append((doc_id, scoref))
        pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
        out[qid] = [{"doc_id": doc_id, "similarity": float(score)} for doc_id, score in pairs]
    return out


def run_sweep_for_dataset(dataset):
    dense_file = RESULTS_DIR / f"beir_run_{dataset}_topk50_dense.json"
    bm25_file = RESULTS_DIR / f"beir_run_{dataset}_topk50_bm25.json"
    qrels = Path(f"data/beir/{dataset}/qrels/test.tsv")
    if not dense_file.exists() or not bm25_file.exists() or not qrels.exists():
        print(f"Skipping {dataset}: missing run or qrels")
        return []

    dense = load_run_as_list(dense_file)
    bm25 = load_run_as_list(bm25_file)

    combos = []

    for k, sw, bw in product(K_VALUES, WEIGHTS, WEIGHTS):
        tag = f"rrf_k{k}_sw{sw}_bw{bw}".replace('.', 'p')
        out_run = RESULTS_DIR / f"beir_run_{dataset}_topk50_hybrid_{tag}.json"
        out_metrics = RESULTS_DIR / f"beir_run_{dataset}_topk50_hybrid_{tag}_metrics_k10.json"

        # prepare fused run: for each qid, fuse lists using local rrf implementation
        fused_results = {}
        for qid in dense.keys():
            # get lists (fallback empty)
            dense_list = dense.get(qid, [])
            bm25_list = bm25.get(qid, [])
            # call internal rrf fuser by running a small subprocess that imports fusion (avoid duplicating logic)
            # But simpler: implement local rank-based scoring here
            scores = {}
            # BM25 ranks
            for idx, entry in enumerate(bm25_list[:TOP_K]):
                rank = idx + 1
                doc = entry.get('doc_id')
                scores[doc] = scores.get(doc, 0.0) + bw * (1.0 / (k + rank))
            # Semantic ranks
            for idx, entry in enumerate(dense_list[:TOP_K]):
                rank = idx + 1
                doc = entry.get('doc_id')
                scores[doc] = scores.get(doc, 0.0) + sw * (1.0 / (k + rank))
            # sort by fused score
            ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            fused_results[qid] = {doc: float(score) for doc, score in ordered[:TOP_K]}

        # write run json
        with open(out_run, 'w', encoding='utf-8') as f:
            json.dump(fused_results, f, indent=2)

        # compute metrics via existing script
        cmd = ["python", "tools/calculate_beir_metrics.py", "--results", str(out_run), "--qrels", str(qrels), "--k", "10"]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            print(f"Metric computation failed for {dataset} {tag}")
            continue

        # load metrics
        if out_metrics.exists():
            m = json.load(open(out_metrics, 'r', encoding='utf-8'))
            combos.append({"dataset": dataset, "tag": tag, "k": k, "sw": sw, "bw": bw, "recall": m.get('recall_at_k'), "mrr": m.get('mrr'), "ndcg": m.get('ndcg')})
        else:
            print(f"Missing metrics file for {dataset} {tag}")

    # sort combos by recall desc
    combos.sort(key=lambda x: float(x['recall'] or 0.0), reverse=True)
    return combos


def main():
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    summary_lines = ["# RRF Sensitivity Sweep Results\n", "This file summarizes results of RRF hyperparameter sweeps (k, semantic_weight, bm25_weight) conducted on dense/bm25 topk50 runs.\n"]

    overall = []
    for ds in DATASETS:
        print(f"Running sweep for {ds}...")
        combos = run_sweep_for_dataset(ds)
        if not combos:
            summary_lines.append(f"\n## {ds} - no results\n")
            continue
        summary_lines.append(f"\n## {ds}\n")
        summary_lines.append("| tag | k | sw | bw | Recall@10 | MRR | nDCG@10 |\n")
        summary_lines.append("|---|---:|---:|---:|---:|---:|---:|\n")
        for c in combos[:10]:  # top 10
            summary_lines.append(f"| {c['tag']} | {c['k']} | {c['sw']} | {c['bw']} | {c['recall']:.4f} | {c['mrr']:.4f} | {c['ndcg']:.4f} |\n")
        overall.append((ds, combos[0] if combos else None))

    # write summary
    with open(SUMMARY_FILE, 'w', encoding='utf-8') as f:
        f.writelines(summary_lines)

    # append short recommendations into canonical doc
    canon = Path('docs/eval/evaluation_antigravity.md')
    if canon.exists():
        text = canon.read_text(encoding='utf-8')
        appended = '\n\n## RRF Sensitivity Sweep (summary)\n\n'
        for ds, best in overall:
            if best:
                appended += f"- **{ds}** best: {best['tag']} (Recall@10={best['recall']:.4f}, nDCG={best['ndcg']:.4f})\n"
            else:
                appended += f"- **{ds}**: no sweep results available\n"
        canon.write_text(text + appended, encoding='utf-8')

    print('Sweep complete. Summary written to', SUMMARY_FILE)

if __name__ == '__main__':
    main()
