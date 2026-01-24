#!/usr/bin/env python
"""Build BM25 index using Pyserini.

This script creates a Lucene-based BM25 index from a JSONL corpus.
Uses Pyserini for compatibility with standard BEIR benchmarks.

Usage:
    python tools/index_bm25_pyserini.py \\
        --corpus-file data/beir/scifact/corpus.jsonl \\
        --output-dir data/beir_index_bm25_scifact \\
        --memory-limit 14GB
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def convert_corpus_to_pyserini_format(corpus_file: Path, temp_dir: Path) -> Path:
    """Convert BEIR JSONL corpus to Pyserini format (one doc per line, id+text)."""
    output_file = temp_dir / "corpus_pyserini.jsonl"
    
    print(f"Converting corpus to Pyserini format...")
    with open(corpus_file) as fin, open(output_file, 'w') as fout:
        doc_count = 0
        for line in fin:
            doc = json.loads(line)
            
            # Handle different BEIR corpus formats
            doc_id = doc.get("_id") or doc.get("docid") or str(doc_count)
            title = doc.get("title", "")
            text = doc.get("text", "")
            
            # Combine title + text
            content = f"{title} {text}".strip()
            
            pyserini_doc = {
                "id": str(doc_id),
                "contents": content
            }
            
            fout.write(json.dumps(pyserini_doc) + "\n")
            doc_count += 1
            
            if (doc_count + 1) % 50000 == 0:
                print(f"  Converted {doc_count} documents...")
    
    print(f"[OK] Converted {doc_count} documents")
    return output_file, doc_count


def build_index_with_pyserini(corpus_file: Path, output_dir: Path, memory_limit: str = "14G") -> bool:
    """Build BM25 index using Pyserini CLI."""
    
    print(f"\n📑 Building BM25 index with Pyserini...")
    print(f"   Corpus: {corpus_file}")
    print(f"   Output: {output_dir}")
    print(f"   Memory: {memory_limit}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use Pyserini's index-built-in command
    cmd = [
        "python", "-m", "pyserini.index.lucene",
        "--collection", "JsonCollection",
        "--input", str(corpus_file),
        "--index", str(output_dir),
        "--generator", "DefaultLuceneDocumentGenerator",
        "--threads", "4",
        "--storePositions",
        "--storeDocvectors",
        "--storeRawDocs"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        
        if result.returncode == 0:
            print(f"[OK] Index built successfully")
            return True
        else:
            print(f"✗ Index build failed:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print(f"✗ Index build timed out (>1 hour)")
        return False
    except Exception as e:
        print(f"✗ Error during indexing: {e}")
        return False


def fallback_build_simple_bm25(corpus_file: Path, output_dir: Path) -> bool:
    """Fallback: Build simple BM25 index in pure Python (slower but no dependencies)."""
    
    print(f"\n[FALLBACK] Using fallback pure-Python BM25 indexing (no Pyserini)...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a simple inverted index in memory, then save
    index_data = {
        "doc_id_map": {},
        "inverted_index": {},
        "doc_term_counts": {},
        "doc_lengths": {}
    }
    
    print(f"Building inverted index from {corpus_file}...")
    
    with open(corpus_file) as f:
        doc_id = 0
        for line in f:
            doc = json.loads(line)
            
            # Extract text
            title = doc.get("title", "")
            text = doc.get("text", "")
            content = f"{title} {text}".lower()
            
            # Tokenize (basic: split on whitespace and punctuation)
            tokens = [t for t in content.split() if t.isalnum()]
            
            # Store mapping
            orig_id = doc.get("_id", str(doc_id))
            index_data["doc_id_map"][doc_id] = orig_id
            index_data["doc_lengths"][doc_id] = len(tokens)
            
            # Build inverted index
            term_counts = {}
            for token in tokens:
                if token not in index_data["inverted_index"]:
                    index_data["inverted_index"][token] = []
                if doc_id not in [d[0] for d in index_data["inverted_index"][token]]:
                    index_data["inverted_index"][token].append((doc_id, 0))
                
                # Count term frequency
                term_counts[token] = term_counts.get(token, 0) + 1
            
            index_data["doc_term_counts"][doc_id] = term_counts
            doc_id += 1
            
            if (doc_id + 1) % 50000 == 0:
                print(f"  Indexed {doc_id} documents...")
    
    # Save index (more compact)
    index_file = output_dir / "bm25_index.json"
    print(f"Saving index to {index_file}...")
    
    # Convert to compact JSON (no indentation to save space)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(index_file, 'w') as f:
        json.dump(index_data, f, separators=(',', ':'))
    
    size_mb = index_file.stat().st_size / (1024 ** 2)
    print(f"[OK] Index built: {doc_id} documents ({size_mb:.1f} MB)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Build BM25 index using Pyserini")
    parser.add_argument("--corpus-file", type=Path, required=True, help="Path to JSONL corpus")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for index")
    parser.add_argument("--memory-limit", type=str, default="14G", help="JVM memory limit")
    
    args = parser.parse_args()
    
    if not args.corpus_file.exists():
        print(f"✗ Corpus file not found: {args.corpus_file}")
        return 1
    
    print("=" * 70)
    print("BM25 INDEXING")
    print("=" * 70)
    
    # Check if Pyserini is available
    try:
        import pyserini
        print(f"✓ Pyserini found (v{pyserini.__version__})")
        
        # Try to use Pyserini
        with tempfile.TemporaryDirectory() as tmpdir:
            corpus_converted, doc_count = convert_corpus_to_pyserini_format(args.corpus_file, Path(tmpdir))
            success = build_index_with_pyserini(corpus_converted, args.output_dir, args.memory_limit)
            
            if success:
                print(f"\n[OK] Index ready at: {args.output_dir}")
                print(f"  Index size: {sum(f.stat().st_size for f in args.output_dir.glob('**/*')) / (1024**2):.0f} MB")
                return 0
            else:
                print(f"\n⚠ Pyserini indexing failed, trying fallback...")
                success = fallback_build_simple_bm25(args.corpus_file, args.output_dir)
                return 0 if success else 1
    
    except ImportError:
        print(f"[WARN] Pyserini not available, using fallback pure-Python indexing")
        success = fallback_build_simple_bm25(args.corpus_file, args.output_dir)
        return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
