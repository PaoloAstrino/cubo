"""
Parse a run_full.log for RAGAS OutputParserException and TimeoutError messages and extract the offending outputs.
Writes results to results/ragas/<run>/failures_extracted.jsonl
"""
import re
import json
from pathlib import Path

log_path = Path("results/ragas/scifact_openai_5/run_full.log")
out_path = Path("results/ragas/scifact_openai_5/failures_extracted.jsonl")

if not log_path.exists():
    print("Log file not found:", log_path)
    raise SystemExit(1)

text = log_path.read_text(encoding='utf-8', errors='replace')

# Find exception blocks
records = []
lines = text.splitlines()
for i, line in enumerate(lines):
    if 'ERROR - ragas.executor - Exception raised in Job' in line:
        # Capture this and the next up to 40 lines or until new timestamp
        block = [line]
        for j in range(i+1, min(i+40, len(lines))):
            l = lines[j]
            if re.match(r'^\d{4}-\d{2}-\d{2}', l):
                break
            block.append(l)
        block_text = '\n'.join(block)
        # Extract job number and error type if possible
        m = re.search(r'Job\[(\d+)\]:\s*([^\(]+)', block_text)
        job = int(m.group(1)) if m else None
        etype = m.group(2).strip() if m else 'Unknown'
        excerpt = block_text.strip().replace('\n', ' ')[:2000]
        records.append({'job': job, 'error_type': etype, 'excerpt': excerpt})

# Also capture TimeoutError occurrences
for m in re.finditer(r"ERROR - ragas.executor - Exception raised in Job\[(?P<job>\d+)\]: TimeoutError\(\)", text):
    job = int(m.group('job'))
    records.append({'job': job, 'error_type': 'TimeoutError', 'excerpt': 'TimeoutError()'})

with open(out_path, 'w', encoding='utf-8') as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')

print('Wrote', len(records), 'failure records to', out_path)
