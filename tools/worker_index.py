#!/usr/bin/env python3
"""Worker script to index a corpus and report timing.
Usage: python tools/worker_index.py --corpus data/beir/nfcorpus/corpus.jsonl --index-dir results/beir_index_nfcorpus_test --limit 1000
"""
import argparse
import time
from cubo.adapters.beir_adapter import CuboBeirAdapter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', required=True)
    parser.add_argument('--index-dir', required=True)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()

    adapter = CuboBeirAdapter(lightweight=False)
    start = time.time()
    count = adapter.index_corpus(args.corpus, args.index_dir, limit=(args.limit or None))
    end = time.time()

    metrics = {'indexed_count': count, 'time_s': end - start}
    out = args.index_dir.rstrip('/') + '/index_metrics.json'
    import json
    import os
    os.makedirs(args.index_dir, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved index metrics to {out}")

if __name__ == '__main__':
    main()
