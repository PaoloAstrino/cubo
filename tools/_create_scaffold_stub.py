# Creates a minimal scaffold_mappings.json and scaffold_embeddings.npy
import json
import numpy as np
from pathlib import Path
import random
random.seed(42)
np.random.seed(42)
corpus_path = Path('data/beir/scifact/corpus.jsonl')
index_dir = Path('data/beir_index_scifact')
if not corpus_path.exists():
    raise SystemExit('corpus.jsonl missing')
ids = []
with open(corpus_path, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 200:
            break
        j = json.loads(line)
        ids.append(j.get('_id') or j.get('id') or str(i))
num_scaffolds = 50
per = max(1, len(ids) // num_scaffolds)
scaffold_map = {}
for s in range(num_scaffolds):
    start = s * per
    chunk_ids = ids[start:start + per]
    if not chunk_ids:
        break
    scaffold_map[f'scaffold_{s}'] = chunk_ids
index_dir.mkdir(parents=True, exist_ok=True)
with open(index_dir / 'scaffold_mappings.json', 'w', encoding='utf-8') as f:
    json.dump(scaffold_map, f, indent=2)
# write normalized random embeddings compatible with embedding model (768)
dim = 768
emb = np.random.normal(size=(len(scaffold_map), dim)).astype('float32')
emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
np.save(index_dir / 'scaffold_embeddings.npy', emb)
print(f'WROTE: {index_dir / "scaffold_mappings.json"} ({len(scaffold_map)} entries)')
print(f'WROTE: {index_dir / "scaffold_embeddings.npy"} (shape={emb.shape})')
