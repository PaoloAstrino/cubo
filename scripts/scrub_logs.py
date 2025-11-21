#!/usr/bin/env python3
"""
Rewrite existing log files to scrub queries (replace raw user queries with hash) when `scrub_queries` is enabled.

This script reads a JSONL or plaintext log file and writes a scrubbed version.
"""
import argparse
import json
import re
from pathlib import Path
from src.cubo.security.security import security_manager


def scrub_line_json(line: str) -> str:
    try:
        rec = json.loads(line)
    except Exception:
        return line
    msg = rec.get('message') or rec.get('msg')
    if not msg:
        return line
    # Find patterns that include query text
    # e.g. "Processed query: ..." or "Query: '...'", "Saved evaluation for query: ..."
    def replace_query(m):
        q = m.group(1)
        return m.group(0).replace(q, security_manager.hash_sensitive_data(q))

    patterns = [r"Processed query: (.+)$", r"Query: '(.+)'", r"Query: (.+)$", r"Saved evaluation for query: (.+?) with", r"Testing \[.*\]: (.+)$"]
    for pat in patterns:
        m = re.search(pat, msg)
        if m:
            new_msg = re.sub(pat, lambda mo: replace_query(mo), msg)
            if 'message' in rec:
                rec['message'] = new_msg
            elif 'msg' in rec:
                rec['msg'] = new_msg
            try:
                return json.dumps(rec) + "\n"
            except Exception:
                return line
    return line


def scrub_plain_text_line(line: str) -> str:
    patterns = [r"Processed query: (.+)$", r"Query: '(.+)'", r"Query: (.+)$", r"Saved evaluation for query: (.+?) with"]
    for pat in patterns:
        m = re.search(pat, line)
        if m:
            q = m.group(1)
            return line.replace(q, security_manager.hash_sensitive_data(q))
    return line


def scrub_file(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open('r', encoding='utf-8') as f_in, dst.open('w', encoding='utf-8') as f_out:
        for line in f_in:
            # try JSON first
            new_line = scrub_line_json(line)
            if new_line == line:
                new_line = scrub_plain_text_line(line)
            f_out.write(new_line)


def main():
    parser = argparse.ArgumentParser(description='Scrub sensitive queries from log file')
    parser.add_argument('--input', required=True, help='Input log file (JSONL or plaintext)')
    parser.add_argument('--output', required=False, help='Output scrubbed file (default: <input>.scrubbed)')
    args = parser.parse_args()
    src = Path(args.input)
    if not src.exists():
        print(f"Input log file not found: {src}")
        return
    dst = Path(args.output) if args.output else src.with_suffix(src.suffix + '.scrubbed')
    scrub_file(src, dst)
    print(f"Done. Scrubbed log written to: {dst}")


if __name__ == '__main__':
    main()
