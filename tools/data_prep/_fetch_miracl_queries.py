"""Fetch MIRACL queries (miracl-de) using the `datasets` library or fall back to creating a placeholder.

Writes: data/multilingual/miracl-de/queries.jsonl

Usage: python tools/_fetch_miracl_queries.py --dataset miracl-de --limit 200
"""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="miracl-de")
parser.add_argument("--limit", type=int, default=200)
args = parser.parse_args()

out_dir = Path('data/multilingual') / args.dataset
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / 'queries.jsonl'

# Try to fetch via datasets
fetched = False
try:
    from datasets import load_dataset
    # Map friendly names like 'miracl-de' -> HF config 'de'
    cfg = args.dataset
    if args.dataset.startswith('miracl-'):
        cfg = args.dataset.split('miracl-')[-1]
    print(f'Trying to load MIRACL config: {cfg}')
    # Some MIRACL configs require trust_remote_code=True
    # MIRACL uses 'dev' and 'testB' splits; try 'test' then fall back
    try:
        ds = load_dataset('MIRACL/miracl', cfg, split='test', trust_remote_code=True)
    except Exception:
        try:
            ds = load_dataset('MIRACL/miracl', cfg, split='testB', trust_remote_code=True)
        except Exception:
            ds = load_dataset('MIRACL/miracl', cfg, split='dev', trust_remote_code=True)
    # ds usually has fields 'id' and 'text' or 'query'
    count = 0
    with open(out_path, 'w', encoding='utf-8') as f:
        for ex in ds:
            if count >= args.limit:
                break
            q = {}
            # normalize field names
            if 'query' in ex:
                q['query'] = ex['query']
            elif 'text' in ex:
                q['query'] = ex['text']
            else:
                # pick first string field
                q['query'] = next((v for k, v in ex.items() if isinstance(v, str)), '')
            q['_id'] = ex.get('id') or ex.get('_id') or str(count)
            if q['query']:
                f.write(json.dumps(q, ensure_ascii=False) + '\n')
                count += 1
    if count > 0:
        print(f'WROTE {count} queries to {out_path} from MIRACL')
        fetched = True
except Exception as e:
    print('Failed to fetch MIRACL via datasets:', e)

# Fallback: use SciFact queries as placeholder
if not fetched:
    print('FALLBACK: creating placeholder from SciFact queries')
    src = Path('data/beir/scifact/queries.jsonl')
    if not src.exists():
        print('SciFact queries are missing too; cannot create placeholder')
        raise SystemExit(1)
    count = 0
    with open(src, 'r', encoding='utf-8') as sf, open(out_path, 'w', encoding='utf-8') as f:
        for line in sf:
            if count >= args.limit:
                break
            if line.strip():
                obj = json.loads(line)
                q = {'_id': obj.get('_id') or obj.get('id') or str(count), 'query': obj.get('text') or obj.get('query')}
                f.write(json.dumps(q, ensure_ascii=False) + '\n')
                count += 1
    print(f'WROTE placeholder {count} queries to {out_path}')

print('Done')
