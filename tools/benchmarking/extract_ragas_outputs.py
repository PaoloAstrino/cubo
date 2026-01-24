"""
Extract 'Output:' JSON-like blocks from a run log and save them for inspection.
"""
import re
import json
from pathlib import Path

log_path = Path('results/ragas/scifact_openai_5_retry/run_full.log')
out_path = Path('results/ragas/scifact_openai_5_retry/raw_outputs.jsonl')

if not log_path.exists():
    print('Log not found:', log_path)
    raise SystemExit(1)

lines = log_path.read_text(encoding='utf-8', errors='replace').splitlines()
records = []
for i, line in enumerate(lines):
    if line.strip().startswith('Output:'):
        # capture following lines until a blank line or a For troubleshooting line
        block = []
        for j in range(i+1, min(i+60, len(lines))):
            l = lines[j]
            if l.strip().startswith('For troubleshooting') or l.strip() == '':
                break
            block.append(l)
        text = '\n'.join(block).strip()
        # try to clean and parse JSON-ish text
        # attempt to find the first '{' and last '}'
        s = text
        first = s.find('{')
        last = s.rfind('}')
        parsed = None
        if first != -1 and last != -1 and last > first:
            candidate = s[first:last+1]
            try:
                parsed = json.loads(candidate)
            except Exception:
                parsed = candidate
        else:
            parsed = s
        records.append({'context_before': lines[i-1] if i>0 else '', 'raw': parsed})

with open(out_path, 'w', encoding='utf-8') as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')

print('Wrote', len(records), 'raw outputs to', out_path)
