"""Create minimal scaffold mappings and embeddings for a given index dir.

Usage: python tools/_create_scaffold_for_index.py --index-dir data/legal_de --num-scaffolds 40
"""
import argparse
import json
import numpy as np
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--index-dir", required=True)
parser.add_argument("--num-scaffolds", type=int, default=40)
parser.add_argument("--dim", type=int, default=768)
args = parser.parse_args()

index_dir = Path(args.index_dir)
corpus = index_dir / 'corpus.jsonl'
if not corpus.exists():
    print('corpus.jsonl not found in', index_dir)
    raise SystemExit(1)

ids = []
with open(corpus, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            try:
                obj = json.loads(line)
            except Exception:
                continue
            _id = obj.get('doc_id') or obj.get('_id') or obj.get('id')
            if _id:
                ids.append(_id)

num_scaffolds = min(args.num_scaffolds, max(1, len(ids)))
per = max(1, len(ids)//num_scaffolds)
scaffold_map = {}
for s in range(num_scaffolds):
    start = s*per
    chunk_ids = ids[start:start+per]
    if not chunk_ids:
        break
    scaffold_map[f'scaffold_{s}'] = chunk_ids

index_dir.mkdir(parents=True, exist_ok=True)
with open(index_dir / 'scaffold_mappings.json', 'w', encoding='utf-8') as f:
    json.dump(scaffold_map, f, indent=2, ensure_ascii=False)

# embeddings
emb = np.random.normal(size=(len(scaffold_map), args.dim)).astype('float32')
emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
np.save(index_dir / 'scaffold_embeddings.npy', emb)
print(f'WROTE scaffold_mappings.json ({len(scaffold_map)} entries) and scaffold_embeddings.npy (shape={emb.shape}) to {index_dir}')
