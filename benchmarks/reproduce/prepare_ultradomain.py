"""
UltraDomain Dataset Preparation

Downloads and filters the UltraDomain dataset (TommyChien/UltraDomain) for benchmarking.
Extracts unique contexts from legal/medical domains matching LightRAG's methodology.
"""
import argparse
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Optional

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

from tqdm import tqdm


DEFAULT_DOMAINS = ["legal", "medical"]
DEFAULT_OUTPUT_DIR = Path("data/benchmark_corpus")
# CHUNK_SIZE/OVERLAP are managed by Cubo config now, but kept for legacy reference or fallback
CHUNK_SIZE = 1200  
OVERLAP = 100

# Initialize Cubo Document Loader to access centralized chunking logic
try:
    from cubo.ingestion.document_loader import DocumentLoader
    # Initialize without model (skip_model=True) since we only need the chunker, not embeddings/vision
    loader = DocumentLoader(skip_model=True)
    logger_msg = "Initialized Cubo DocumentLoader for benchmarking."
except ImportError:
    # Fallback only if running in context where cubo package isn't installed (unlikely)
    loader = None
    logger_msg = "Cubo not found. Using naive fallback (SHOULD NOT HAPPEN IN PROD)."


def simple_tokenize(text: str) -> List[str]:
    """Simple whitespace tokenizer for chunk estimation."""
    return text.split()


def cubo_chunk_text(text: str) -> List[str]:
    """
    Chunk text using the OFFICIAL Cubo pipeline.
    
    This ensures benchmarks test the exact same segmentation logic as the app.
    """
    if loader and hasattr(loader, "chunker"):
        # Use the HierarchicalChunker initialized in DocumentLoader
        # It returns dicts, we extract text for the benchmark corpus
        chunks = loader.chunker.chunk(text, format_type="auto")
        return [c["text"] for c in chunks]
    else:
        # Fallback (Should be deleted/warned against)
        print("WARNING: Using naive fallback chunking!")
        words = text.split()
        return [" ".join(words[i:i+CHUNK_SIZE]) for i in range(0, len(words), CHUNK_SIZE-OVERLAP)]

# Legacy wrapper for compatibility if needed, but we should switch to cubo_chunk_text
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> List[str]:
    return cubo_chunk_text(text)


def content_hash(text: str) -> str:
    """Generate a hash for deduplication."""
    return hashlib.md5(text.encode()).hexdigest()[:16]


def find_existing_data(domains: List[str], search_dirs: List[Path] = None) -> Dict[str, Path]:
    """
    Find existing JSONL files for requested domains.
    
    Searches in common locations:
    - data/ultradomain/
    - data/benchmark_corpus/
    - data/
    
    Returns:
        Dict mapping domain -> file path (if found)
    """
    if search_dirs is None:
        search_dirs = [
            Path("data/ultradomain"),
            Path("data/benchmark_corpus"),
            Path("data"),
        ]
    
    found = {}
    for domain in domains:
        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
            
            # Check various naming patterns
            patterns = [
                f"{domain}.jsonl",
                f"{domain}_corpus.jsonl",
                f"{domain.lower()}.jsonl",
            ]
            
            for pattern in patterns:
                candidate = search_dir / pattern
                if candidate.exists() and candidate.stat().st_size > 0:
                    found[domain] = candidate
                    break
            
            if domain in found:
                break
    
    return found


def prepare_ultradomain(
    domains: List[str] = DEFAULT_DOMAINS,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    max_docs_per_domain: Optional[int] = None,
    chunk: bool = True,
    force_download: bool = False,
) -> Dict[str, int]:
    """
    Download and prepare UltraDomain dataset.
    
    Automatically detects existing data and skips download if available.
    
    Args:
        domains: List of domain names to filter (e.g., ["legal", "medical"])
        output_dir: Directory to save processed JSONL files
        max_docs_per_domain: Limit documents per domain (for testing)
        chunk: Whether to chunk documents
        force_download: Force download even if data exists
        
    Returns:
        Dict with counts per domain
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for existing data
    existing = find_existing_data(domains) if not force_download else {}
    
    if existing:
        print(f"Found existing data for domains: {list(existing.keys())}")
        for domain, path in existing.items():
            print(f"  {domain}: {path}")
    
    # Determine which domains need download
    domains_to_download = [d for d in domains if d not in existing]
    
    stats = {}
    
    # Process existing data (just count/chunk if needed)
    for domain, source_path in existing.items():
        print(f"\n[EXISTING] Processing {domain} from {source_path}")
        output_file = output_dir / f"{domain}_corpus.jsonl"
        
        doc_count = 0
        chunk_count = 0
        seen_hashes = set()
        
        with open(output_file, "w", encoding="utf-8") as out_f:
            # Try loading as JSON list first
            try:
                with open(source_path, "r", encoding="utf-8") as in_f:
                    # Read first char to detect format
                    first_char = in_f.read(1)
                    in_f.seek(0)
                    
                    if first_char == '[':
                        # It's a JSON array
                        print(f"  Detected JSON array format in {source_path}")
                        data = json.load(in_f)
                    else:
                        # Assume JSONL
                        print(f"  Detected JSONL format in {source_path}")
                        data = []
                        for line in in_f:
                            line = line.strip()
                            if line:
                                try:
                                    data.append(json.loads(line))
                                except json.JSONDecodeError:
                                    pass
            except Exception as e:
                print(f"  Error reading {source_path}: {e}")
                data = []

            if max_docs_per_domain:
                data = data[:max_docs_per_domain]
            
            for doc in tqdm(data, desc=f"  {domain}"):
                text = doc.get("text", "") or doc.get("content", "") or doc.get("context", "")
                if not text:
                    continue
                
                doc_id = doc.get("id", doc.get("_id", content_hash(text)))
                
                if chunk:
                    chunks = chunk_text(text)
                    for i, chunk_text_item in enumerate(chunks):
                        chunk_hash = content_hash(chunk_text_item)
                        if chunk_hash in seen_hashes:
                            continue
                        seen_hashes.add(chunk_hash)
                        
                        record = {
                            "id": f"{doc_id}_chunk_{i}",
                            "doc_id": doc_id,
                            "domain": domain,
                            "text": chunk_text_item,
                            "chunk_index": i,
                        }
                        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        chunk_count += 1
                else:
                    record = {"id": doc_id, "domain": domain, "text": text}
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    chunk_count += 1
                
                doc_count += 1
        
        stats[domain] = {"docs": doc_count, "chunks": chunk_count}
        print(f"  Processed {chunk_count} chunks from {doc_count} docs")
    
    # Download missing domains
    if domains_to_download:
        if not DATASETS_AVAILABLE:
            raise ImportError(
                f"Need to download {domains_to_download} but datasets library not installed. "
                "Run: pip install cubo[benchmark]"
            )
        
        print(f"\n[DOWNLOAD] Fetching domains: {domains_to_download}")
        dataset = load_dataset("TommyChien/UltraDomain", split="train")
        
        seen_hashes = set()
        
        for domain in domains_to_download:
            print(f"\nProcessing domain: {domain}")
            output_file = output_dir / f"{domain}_corpus.jsonl"
            
            domain_docs = [
                doc for doc in dataset 
                if doc.get("domain", "").lower() == domain.lower()
            ]
            
            if max_docs_per_domain:
                domain_docs = domain_docs[:max_docs_per_domain]
            
            doc_count = 0
            chunk_count = 0
            
            with open(output_file, "w", encoding="utf-8") as f:
                for doc in tqdm(domain_docs, desc=f"  {domain}"):
                    text = doc.get("text", "") or doc.get("content", "")
                    if not text:
                        continue
                    
                    doc_id = doc.get("id", content_hash(text))
                    
                    if chunk:
                        chunks = chunk_text(text)
                        for i, chunk_text_item in enumerate(chunks):
                            chunk_hash = content_hash(chunk_text_item)
                            if chunk_hash in seen_hashes:
                                continue
                            seen_hashes.add(chunk_hash)
                            
                            record = {
                                "id": f"{doc_id}_chunk_{i}",
                                "doc_id": doc_id,
                                "domain": domain,
                                "text": chunk_text_item,
                                "chunk_index": i,
                            }
                            f.write(json.dumps(record, ensure_ascii=False) + "\n")
                            chunk_count += 1
                    else:
                        record = {"id": doc_id, "domain": domain, "text": text}
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    
                    doc_count += 1
            
            stats[domain] = {"docs": doc_count, "chunks": chunk_count}
            print(f"  Saved {chunk_count} records to {output_file}")
    
    # Save manifest
    manifest_file = output_dir / "manifest.json"
    with open(manifest_file, "w") as f:
        json.dump({
            "domains": domains,
            "chunk_size": CHUNK_SIZE,
            "overlap": OVERLAP,
            "stats": stats,
        }, f, indent=2)
    
    print(f"\nDataset prepared. Manifest: {manifest_file}")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Prepare UltraDomain benchmark dataset")
    parser.add_argument(
        "--domains", 
        nargs="+", 
        default=DEFAULT_DOMAINS,
        help="Domains to include (default: legal medical)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Max documents per domain (for testing)"
    )
    parser.add_argument(
        "--no-chunk",
        action="store_true",
        help="Don't chunk documents"
    )
    
    args = parser.parse_args()
    
    stats = prepare_ultradomain(
        domains=args.domains,
        output_dir=args.output_dir,
        max_docs_per_domain=args.max_docs,
        chunk=not args.no_chunk,
    )
    
    print("\nSummary:")
    for domain, counts in stats.items():
        print(f"  {domain}: {counts['docs']} docs, {counts['chunks']} chunks")


if __name__ == "__main__":
    main()
