"""
Synthetic Query Generation

Generates high-level questions from corpus chunks using LLM.
Matches LightRAG's query generation methodology.
"""
import argparse
import json
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional

from tqdm import tqdm

from cubo.config import config
from cubo.core import CuboCore


DEFAULT_CORPUS_DIR = Path("data/benchmark_corpus")
DEFAULT_OUTPUT_FILE = Path("data/benchmark_queries.jsonl")
QUERIES_PER_CHUNK = 2
DEFAULT_MODEL = "llama3.2:latest"

QUERY_GENERATION_PROMPT = """You are a legal/medical expert. Based on the following context, generate {n} diverse, high-level questions that a professional might ask. 

Questions should be:
- Specific to the content
- Require understanding, not just keyword matching
- Varied in complexity (some factual, some analytical)

Context:
{context}

Generate exactly {n} questions, one per line. Do not number them or add prefixes.
Questions:"""


def generate_queries_for_chunk(
    chunk_text: str,
    generator: Callable[[str], str],
    n_queries: int = QUERIES_PER_CHUNK,
) -> List[str]:
    """Generate queries for a single chunk using CuboCore's generator."""

    prompt = QUERY_GENERATION_PROMPT.format(context=chunk_text[:3000], n=n_queries)

    try:
        content = generator(prompt)

        queries = [
            line.strip().lstrip("0123456789.-) ")
            for line in content.split("\n")
            if line.strip() and len(line.strip()) > 10
        ]
        return queries[:n_queries]
    except Exception as e:
        print(f"  Warning: Query generation failed: {e}")
        return []


def generate_queries(
    corpus_dir: Path = DEFAULT_CORPUS_DIR,
    output_file: Path = DEFAULT_OUTPUT_FILE,
    domains: Optional[List[str]] = None,
    n_chunks: int = 50,
    queries_per_chunk: int = QUERIES_PER_CHUNK,
    model: str = DEFAULT_MODEL,
    seed: int = 42,
) -> Dict[str, int]:
    """
    Generate synthetic queries from corpus chunks.
    
    Args:
        corpus_dir: Directory containing *_corpus.jsonl files
        output_file: Output file for queries
        domains: Domains to process (None = all)
        n_chunks: Number of chunks to sample per domain
        queries_per_chunk: Queries to generate per chunk
        model: Ollama model to use
        seed: Random seed for reproducibility
        
    Returns:
        Stats dict with query counts per domain
    """
    random.seed(seed)
    corpus_dir = Path(corpus_dir)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if model:
        config.set("llm.model_name", model)
    core = CuboCore()
    core.initialize_components()

    def run_prompt(prompt: str) -> str:
        # Use the configured generator through CuboCore to ensure consistency.
        return core.generator.generate_text(prompt)
    
    # Find corpus files
    corpus_files = list(corpus_dir.glob("*_corpus.jsonl"))
    if domains:
        corpus_files = [f for f in corpus_files if any(d in f.stem for d in domains)]
    
    if not corpus_files:
        raise FileNotFoundError(f"No corpus files found in {corpus_dir}")
    
    stats = {}
    all_queries = []
    
    for corpus_file in corpus_files:
        domain = corpus_file.stem.replace("_corpus", "")
        print(f"\nProcessing domain: {domain}")
        
        # Load chunks
        chunks = []
        with open(corpus_file, "r", encoding="utf-8") as f:
            for line in f:
                chunks.append(json.loads(line))
        
        # Sample chunks
        sampled = random.sample(chunks, min(n_chunks, len(chunks)))
        query_count = 0
        
        for chunk in tqdm(sampled, desc=f"  Generating queries"):
            queries = generate_queries_for_chunk(
                chunk["text"],
                generator=run_prompt,
                n_queries=queries_per_chunk,
            )
            
            for q in queries:
                record = {
                    "query": q,
                    "domain": domain,
                    "source_chunk_id": chunk["id"],
                    "source_doc_id": chunk.get("doc_id", chunk["id"]),
                }
                all_queries.append(record)
                query_count += 1
        
        stats[domain] = query_count
        print(f"  Generated {query_count} queries")
    
    # Shuffle and save
    random.shuffle(all_queries)
    with open(output_file, "w", encoding="utf-8") as f:
        for q in all_queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    
    print(f"\nSaved {len(all_queries)} queries to {output_file}")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic benchmark queries")
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=DEFAULT_CORPUS_DIR,
        help="Directory with corpus JSONL files"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help="Output JSONL file for queries"
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=None,
        help="Domains to process (default: all)"
    )
    parser.add_argument(
        "--n-chunks",
        type=int,
        default=50,
        help="Chunks to sample per domain"
    )
    parser.add_argument(
        "--queries-per-chunk",
        type=int,
        default=QUERIES_PER_CHUNK,
        help="Queries per chunk"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Ollama model for generation"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    generate_queries(
        corpus_dir=args.corpus_dir,
        output_file=args.output,
        domains=args.domains,
        n_chunks=args.n_chunks,
        queries_per_chunk=args.queries_per_chunk,
        model=args.model,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
