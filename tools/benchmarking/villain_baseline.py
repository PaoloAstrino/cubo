import json
import os
import time
from typing import Dict, List

import psutil


def get_ram_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # MB


def run_baseline_villain(corpus_path: str, limit: int = 100000):
    print("--- Starting Baseline 'Villain' Test (In-Memory) ---")
    print(f"Loading corpus from {corpus_path}...")

    start_time = time.time()
    memory_store: List[Dict] = []

    try:
        with open(corpus_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= limit:
                    break

                doc = json.loads(line)
                # Naive in-memory storage of the whole document + some 'index' overhead
                memory_store.append(doc)

                if i > 0 and i % 10000 == 0:
                    usage = get_ram_usage()
                    print(f"Loaded {i} docs. Current RAM: {usage:.2f} MB")

                    # Simulate simple search overhead (inverted index mock)
                    # if usage > 12000: # 12GB threshold for a laptop
                    #    print("CAUTION: RAM usage critical!")

        total_time = time.time() - start_time
        final_usage = get_ram_usage()
        print("--- SUCCESS (or luck) ---")
        print(f"Total Loaded: {len(memory_store)} docs")
        print(f"Final RAM: {final_usage:.2f} MB")
        print(f"Time: {total_time:.2f} seconds")

    except MemoryError:
        print("!!! CRASH: Out of Memory !!!")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    beir_corpus = "data/beir/beir_corpus.jsonl"
    if os.path.exists(beir_corpus):
        run_baseline_villain(beir_corpus)
    else:
        print(f"Corpus not found at {beir_corpus}")
