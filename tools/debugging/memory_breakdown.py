#!/usr/bin/env python3
"""Instrument indexing steps to capture per-component memory breakdown.

Usage: python tools/memory_breakdown.py --corpus data/beir/nfcorpus/corpus.jsonl --limit 1000

This script performs representative steps in-process and records RSS at checkpoints:
- baseline
- after embedding model load
- after computing embeddings for the sample
- after building FAISS index
- after saving FAISS index
- after building BM25 index (if available)
- after LLM initialization (if available)

The output is saved to results/memory_breakdown_*.json and a stacked bar plot under paper/figs.
"""
import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import psutil

# Ensure repo root is on path so imports work when script is invoked directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR = Path("paper/figs")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def rss_mb():
    return psutil.Process(os.getpid()).memory_info().rss / (1024**2)


def load_sample_texts(corpus_path, limit):
    texts = []
    ids = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            try:
                import json as _j

                j = _j.loads(line)
                ids.append(str(j.get("_id", i)))
                txt = (j.get("title", "") + " " + j.get("text", "")).strip()
                texts.append(txt)
            except Exception:
                continue
    return ids, texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--limit", type=int, default=1000)
    args = parser.parse_args()

    ids, texts = load_sample_texts(args.corpus, args.limit)
    print(f"Loaded {len(texts)} sample docs")

    m = {"baseline_mb": rss_mb(), "checkpoints": []}

    # 1) Embedding model load
    try:
        from cubo.embeddings.embedding_generator import EmbeddingGenerator

        t0 = time.time()
        eg = EmbeddingGenerator()
        t1 = time.time()
        gc.collect()
        m["checkpoints"].append(
            {"name": "embedding_model_loaded", "rss_mb": rss_mb(), "time_s": t1 - t0}
        )
        print("Embedding model loaded, rss_mb=", m["checkpoints"][-1]["rss_mb"])
    except Exception as e:
        print("Embedding model load failed:", e)
        eg = None

    # 2) Compute embeddings for sample
    embedding_bytes = 0
    if eg and texts:
        try:
            t0 = time.time()
            embeddings = eg.encode(texts, batch_size=eg.batch_size)
            t1 = time.time()
            gc.collect()
            m["checkpoints"].append(
                {"name": "embeddings_computed", "rss_mb": rss_mb(), "time_s": t1 - t0}
            )
            print("Embeddings computed, rss_mb=", m["checkpoints"][-1]["rss_mb"])
            # Estimate embedding memory as size of structure (rough)
            embedding_bytes = sum([len(e) * 8 for e in embeddings])
        except Exception as e:
            print("Embedding computation failed:", e)
            embeddings = None
    else:
        embeddings = None

    # 3) Build FAISS index
    try:
        from cubo.indexing.faiss_index import FAISSIndexManager

        t0 = time.time()
        dim = len(embeddings[0]) if embeddings else 768
        fm = FAISSIndexManager(dimension=dim, index_dir=Path("results/tmp_faiss"))
        fm.build_indexes(embeddings or [[0.0] * dim], ids or ["0"])
        t1 = time.time()
        gc.collect()
        m["checkpoints"].append({"name": "faiss_built", "rss_mb": rss_mb(), "time_s": t1 - t0})
        print("FAISS built, rss_mb=", m["checkpoints"][-1]["rss_mb"])

        # Save
        t0 = time.time()
        fm.save(Path("results/tmp_faiss"))
        t1 = time.time()
        gc.collect()
        m["checkpoints"].append({"name": "faiss_saved", "rss_mb": rss_mb(), "time_s": t1 - t0})
        print("FAISS saved, rss_mb=", m["checkpoints"][-1]["rss_mb"])
    except Exception as e:
        print("FAISS build/save failed:", e)

    # 4) Build BM25 (if available)
    try:
        from cubo.indexing.tiered_index_manager import TieredIndexManager

        t0 = time.time()
        manager = TieredIndexManager(
            dimension=len(embeddings[0]) if embeddings else 768,
            index_dir=Path("results/tmp_index_with_bm25"),
        )
        docs = [{"id": i, "text": t} for i, t in zip(ids, texts)]
        manager.bm25_store.index_documents(docs)
        t1 = time.time()
        gc.collect()
        m["checkpoints"].append({"name": "bm25_built", "rss_mb": rss_mb(), "time_s": t1 - t0})
        print("BM25 built, rss_mb=", m["checkpoints"][-1]["rss_mb"])
    except Exception as e:
        print("BM25 build failed (may require Whoosh):", e)

    # 5) LLM init (best-effort)
    try:
        from cubo.processing.llm_local import LocalResponseGenerator

        t0 = time.time()
        llm = LocalResponseGenerator()
        t1 = time.time()
        gc.collect()
        m["checkpoints"].append({"name": "llm_init", "rss_mb": rss_mb(), "time_s": t1 - t0})
        print("LLM init (local) rss_mb=", m["checkpoints"][-1]["rss_mb"])
    except Exception as e:
        print("Local LLM init failed (llama_cpp may be missing):", e)

    # Summarize
    m["final_rss_mb"] = rss_mb()
    out_file = RESULTS_DIR / f"memory_breakdown_{int(time.time())}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2)
    print("Wrote memory breakdown to", out_file)

    # Create a simple stacked bar chart if we have checkpoints
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = [c["name"] for c in m["checkpoints"]]
        values = [c["rss_mb"] for c in m["checkpoints"]]
        # Convert to incremental component sizes (delta from previous)
        baseline = m["baseline_mb"]
        deltas = [v - baseline for v in values]
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(["components"], deltas, bottom=0)
        # Add labels for each component vertically
        ypos = 0
        for nm, dv in zip(names, deltas):
            ax.text(
                0,
                ypos + dv / 2,
                f"{nm}: {dv:.1f}MB",
                va="center",
                ha="center",
                color="white",
                fontsize=8,
            )
            ypos += dv
        ax.set_ylabel("Memory (MB)")
        plt.title("Memory breakdown (sample run)")
        plt.tight_layout()
        out_png = PLOTS_DIR / "memory_breakdown_sample.png"
        plt.savefig(out_png, dpi=200)
        print("Wrote plot to", out_png)
    except Exception as e:
        print("Plotting failed:", e)


if __name__ == "__main__":
    main()
