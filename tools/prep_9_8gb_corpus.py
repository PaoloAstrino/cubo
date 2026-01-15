#!/usr/bin/env python3
"""Prepare a reproducible ~9.8 GB corpus by concatenating BEIR corpus files.

Usage:
  # Create a small test corpus (fast)
  python tools/prep_9_8gb_corpus.py --out data/ingest/ingest_9_8gb_corpus_small.jsonl --target-bytes 100000 --test

  # Create full corpus (may take long and disk space)
  python tools/prep_9_8gb_corpus.py --out data/ingest/ingest_9_8gb_corpus.jsonl --target-gb 9.8

Notes:
- The script concatenates BEIR corpora in deterministic order, records a SHA256 checksum, and writes metadata for reproducibility.
"""

import argparse
import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BEIR_DIR = REPO_ROOT / "data" / "beir"
OUT_DIR = REPO_ROOT / "data" / "ingest"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def iter_beir_corpus_files():
    # Find corpus.jsonl files under data/beir/*/corpus.jsonl
    items = sorted(BEIR_DIR.glob("**/corpus.jsonl"))
    for p in items:
        yield p


def sha256_of_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def build_corpus(out_path: Path, target_bytes: int, test: bool = False):
    # Concatenate all BEIR corpus files deterministically
    files = list(iter_beir_corpus_files())
    if not files:
        raise RuntimeError(f"No BEIR corpus files found under {BEIR_DIR}")

    print(f"Found {len(files)} BEIR corpus files. Using them in sorted order.")

    # Create a temp concatenated seed
    seed_path = out_path.with_suffix('.seed.jsonl')
    with open(seed_path, 'w', encoding='utf-8') as out_f:
        for fpath in files:
            with open(fpath, 'r', encoding='utf-8') as r:
                for line in r:
                    out_f.write(line)

    seed_size = seed_path.stat().st_size
    print(f"Seed size: {seed_size} bytes")

    # If test mode, just copy seed and truncate
    if test or target_bytes <= seed_size:
        # For test, we may write only the first N bytes or lines
        if test:
            # write first 1000 lines
            with open(seed_path, 'r', encoding='utf-8') as r, open(out_path, 'w', encoding='utf-8') as out_f:
                for i, line in enumerate(r):
                    if i >= 1000:
                        break
                    out_f.write(line)
        else:
            seed_path.replace(out_path)
        checksum = sha256_of_file(out_path)
        meta = {
            'source_files': [str(p.relative_to(REPO_ROOT)) for p in files],
            'seed_size_bytes': seed_size,
            'out_size_bytes': out_path.stat().st_size,
            'sha256': checksum,
            'method': 'concatenate-seed',
            'note': 'test mode or target smaller than seed'
        }
        meta_path = out_path.with_suffix('.json')
        with open(meta_path, 'w', encoding='utf-8') as mf:
            json.dump(meta, mf, indent=2)
        print(f"Wrote test corpus {out_path} and metadata {meta_path}")
        return out_path, meta

    # For large target, duplicate seed deterministically until reach target
    with open(out_path, 'wb') as out_f, open(seed_path, 'rb') as seed_f:
        written = 0
        while written < target_bytes:
            seed_f.seek(0)
            for chunk in iter(lambda: seed_f.read(65536), b''):
                out_f.write(chunk)
                written += len(chunk)
                if written >= target_bytes:
                    break
    checksum = sha256_of_file(out_path)
    meta = {
        'source_files': [str(p.relative_to(REPO_ROOT)) for p in files],
        'seed_size_bytes': seed_size,
        'target_bytes': target_bytes,
        'out_size_bytes': out_path.stat().st_size,
        'sha256': checksum,
        'method': 'concatenate-and-duplicate'
    }
    meta_path = out_path.with_suffix('.json')
    with open(meta_path, 'w', encoding='utf-8') as mf:
        json.dump(meta, mf, indent=2)
    print(f"Built corpus {out_path} size {out_path.stat().st_size} bytes (target {target_bytes})")
    return out_path, meta


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=OUT_DIR / 'ingest_9_8gb_corpus.jsonl')
    p.add_argument('--target-gb', type=float, default=9.8, help='Target size in GB')
    p.add_argument('--target-bytes', type=int, default=None, help='Target size in bytes (overrides target-gb)')
    p.add_argument('--test', action='store_true', help='Create a small test corpus (fast)')
    return p.parse_args()


def main():
    args = parse_args()
    if args.target_bytes:
        target = args.target_bytes
    else:
        target = int(args.target_gb * 1024 ** 3)
    if args.test:
        # test mode: small target
        target = 100000
    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    build_corpus(out_path, target, test=args.test)

if __name__ == '__main__':
    main()
