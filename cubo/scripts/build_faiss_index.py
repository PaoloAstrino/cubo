"""CLI to build FAISS hot/cold indexes from chunk parquet data."""

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Set

import pandas as pd

from cubo.config import config
from cubo.embeddings.embedding_generator import EmbeddingGenerator
from cubo.indexing.faiss_index import FAISSIndexManager
from cubo.indexing.index_publisher import publish_version
from cubo.processing.enrichment import ChunkEnricher
from cubo.processing.generator import create_response_generator
from cubo.processing.scaffold import ScaffoldGenerator
from cubo.utils.logger import logger


def parse_args():
    parser = argparse.ArgumentParser(description="Build hot/cold FAISS indexes from chunk parquet")
    parser.add_argument(
        "--parquet", required=True, help="Parquet file containing chunk text and ids"
    )
    parser.add_argument(
        "--text-column", default="text", help="Name of the text column in the parquet file"
    )
    parser.add_argument(
        "--summary-column",
        default=None,
        help="Optional column to embed summaries instead of chunks",
    )
    parser.add_argument("--id-column", default="chunk_id", help="Column containing chunk ids")
    parser.add_argument("--index-dir", default=config.get("faiss_index_dir", "faiss_index"))
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Publish the index version by flipping pointer and updating metadata DB",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Cleanup old index versions according to retention policy",
    )
    parser.add_argument(
        "--retention",
        type=int,
        default=3,
        help="Number of index versions to keep when --cleanup is enabled",
    )
    parser.add_argument(
        "--index-root",
        default=config.get("faiss_index_root", None),
        help="Root dir where versioned faiss_v* dirs are stored",
    )
    parser.add_argument("--batch-size", type=int, default=config.get("embedding_batch_size", 32))
    parser.add_argument("--hot-fraction", type=float, default=0.25)
    parser.add_argument("--nlist", type=int, default=64)
    parser.add_argument("--hnsw-m", type=int, default=16)
    parser.add_argument(
        "--dry-run", action="store_true", help="Generate indexes without persisting them"
    )
    parser.add_argument("--verbose", action="store_true", help="Log progress details")
    parser.add_argument(
        "--enrich-chunks", action="store_true", help="Enrich chunks with summaries, keywords, etc."
    )
    parser.add_argument(
        "--output-parquet", help="Path to save the enriched data as a Parquet file."
    )
    parser.add_argument("--dedup-map", help="Path to the deduplication map file.")
    parser.add_argument(
        "--scaffold", action="store_true", help="Build scaffold index for fast retrieval"
    )
    parser.add_argument(
        "--scaffold-dir",
        default="scaffold_index",
        help="(Deprecated) Directory to save scaffold outputs",
    )
    parser.add_argument(
        "--scaffold-output-dir",
        default="data/scaffolds",
        help="Directory to save scaffold outputs (default: data/scaffolds)",
    )
    parser.add_argument(
        "--scaffold-size", type=int, default=5, help="Target number of chunks per scaffold"
    )
    parser.add_argument(
        "--use-opq",
        action="store_true",
        help="Use OPQ (Optimized Product Quantization) for cold index",
    )
    parser.add_argument("--opq-m", type=int, default=32, help="OPQ subspace dimension")
    return parser.parse_args()


def _load_and_prepare_data(args):
    """Load parquet data and apply deduplication if needed."""
    parquet_path = Path(args.parquet)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)

    if args.dedup_map:
        df = _filter_to_representatives(df, args.id_column, args.dedup_map)
        logger.info(f"Filtered down to {len(df)} canonical documents after dedup application.")
    
    return df


def _build_scaffold_index(df, args):
    """Build scaffold index from the dataframe."""
    logger.info("Building scaffold index...")
    generator = EmbeddingGenerator(batch_size=args.batch_size)

    # Create enricher if needed for scaffold summaries
    enricher = None
    if args.enrich_chunks or not args.summary_column:
        logger.info("Creating enricher for scaffold summaries...")
        enricher = ChunkEnricher(llm_provider=create_response_generator())

    # Initialize scaffold generator with enricher and embedding generator
    scaffold_gen = ScaffoldGenerator(
        enricher=enricher, embedding_generator=generator, scaffold_size=args.scaffold_size
    )

    # Generate scaffolds from the dataframe
    scaffolds_result = scaffold_gen.generate_scaffolds(
        df, text_column=args.text_column, id_column=args.id_column
    )

    return scaffolds_result


def _save_scaffold_run(scaffolds_result, df, args):
    """Save scaffold outputs and return paths."""
    from uuid import uuid4
    from cubo.processing.scaffold import save_scaffold_run

    run_id = f"scaffold_{int(time.time())}_{uuid4().hex[:8]}"
    scaffold_output = (
        Path(args.scaffold_output_dir)
        if getattr(args, "scaffold_output_dir", None)
        else Path(args.scaffold_dir)
    )
    
    res = save_scaffold_run(
        run_id,
        scaffolds_result,
        output_root=scaffold_output,
        model_version=config.get("llm_model"),
        input_chunks_df=df,
        id_column=args.id_column,
    )
    
    logger.info(f"Scaffold outputs saved to {res['run_dir']} (manifest: {res['manifest']})")
    return res


def _build_and_save_scaffold_faiss(scaffolds_result, scaffold_paths, args):
    """Build FAISS index on scaffolds and save if not dry-run."""
    scaffold_df = scaffolds_result["scaffolds_df"]
    texts = scaffold_df["summary"].fillna("").astype(str).tolist()
    ids = scaffold_df["scaffold_id"].astype(str).tolist()
    embeddings = scaffolds_result["scaffold_embeddings"]

    dimension = len(embeddings[0]) if embeddings else 0
    if dimension == 0:
        raise ValueError("Unable to determine embedding dimension for scaffolds")

    manager = FAISSIndexManager(
        dimension=dimension,
        index_dir=Path(scaffold_paths["run_dir"]) / "faiss",
        nlist=args.nlist,
        hnsw_m=args.hnsw_m,
        hot_fraction=args.hot_fraction,
        use_opq=args.use_opq,
        opq_m=args.opq_m,
    )
    manager.build_indexes(embeddings, ids)

    # Sample search to validate the index
    sample_query = embeddings[0]
    hits = manager.search(sample_query, k=min(5, len(ids)))
    logger.info(f"Scaffold sample search returned {len(hits)} hits")

    # Save the scaffold index if not a dry run
    if not args.dry_run:
        if args.index_root:
            _publish_versioned_index(manager, args)
        else:
            manager.save()
        logger.info(f"Scaffold FAISS index saved to {Path(args.scaffold_dir) / 'faiss'}")
    else:
        logger.info("Dry-run enabled; scaffold FAISS index was not saved")


def _enrich_chunks_if_needed(df, args):
    """Enrich chunks with summaries and keywords if requested."""
    if not args.enrich_chunks:
        return df
    
    logger.info("Enriching chunks...")
    enricher = ChunkEnricher(llm_provider=create_response_generator())
    enriched_data = enricher.enrich_chunks(df[args.text_column].tolist())
    enriched_df = pd.DataFrame(enriched_data)

    # Merge the enriched data with the original dataframe
    df = pd.concat([df.reset_index(drop=True), enriched_df.drop("text", axis=1)], axis=1)

    if args.output_parquet:
        df.to_parquet(args.output_parquet)
        logger.info(f"Enriched data saved to {args.output_parquet}")
    
    return df


def _generate_embeddings(df, args):
    """Generate embeddings for the text column."""
    embed_column = args.summary_column or ("summary" if args.enrich_chunks else args.text_column)
    texts = df[embed_column].fillna("").astype(str).tolist()
    ids = df[args.id_column].astype(str).tolist()

    if not texts:
        logger.warning("No chunks found in parquet file; nothing to index")
        return None, None, 0
    
    if len(texts) != len(ids):
        raise ValueError("Text and id columns must have the same length")

    generator = EmbeddingGenerator(batch_size=args.batch_size)
    embeddings = generator.encode(texts, batch_size=args.batch_size, prompt_name="document")
    dimension = len(embeddings[0]) if embeddings else 0
    
    if dimension == 0:
        raise ValueError("Unable to determine embedding dimension")
    
    return embeddings, ids, dimension


def _build_and_validate_index(embeddings, ids, dimension, args):
    """Build FAISS index and validate with sample search."""
    manager = FAISSIndexManager(
        dimension=dimension,
        index_dir=Path(args.index_dir),
        nlist=args.nlist,
        hnsw_m=args.hnsw_m,
        hot_fraction=args.hot_fraction,
        use_opq=args.use_opq,
        opq_m=args.opq_m,
    )
    manager.build_indexes(embeddings, ids)
    
    # Perform sample search to validate the index
    sample_query = embeddings[0]
    hits = manager.search(sample_query, k=min(5, len(ids)))
    logger.info(
        f"Sample search returned {len(hits)} hits (first id: {hits[0]['id'] if hits else 'none'})"
    )
    
    return manager


def _publish_versioned_index(manager, args):
    """Save index with versioning and optionally publish it."""
    ts = int(time.time())
    version_tmp = Path(args.index_root) / f"faiss_v{ts}.tmp"
    final_version = Path(args.index_root) / f"faiss_v{ts}"
    manager.save(path=version_tmp)
    
    # Atomic rename of dir: use os.replace; retry on transient errors
    try:
        os.replace(str(version_tmp), str(final_version))
    except Exception:
        # On some platforms, os.replace may fail for dir; fallback to rename
        os.rename(str(version_tmp), str(final_version))
    
    if args.publish:
        publish_version(final_version, Path(args.index_root))
        logger.info(f"Published version {final_version}")
    
    if args.cleanup:
        from cubo.indexing.index_publisher import cleanup
        cleanup(Path(args.index_root), keep_last_n=args.retention)


def main():
    """Main function to build FAISS indexes from parquet data."""
    args = parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Load and prepare data
    df = _load_and_prepare_data(args)

    # Handle scaffold generation if requested
    if args.scaffold:
        scaffolds_result = _build_scaffold_index(df, args)
        scaffold_paths = _save_scaffold_run(scaffolds_result, df, args)
        _build_and_save_scaffold_faiss(scaffolds_result, scaffold_paths, args)
        return  # Exit after scaffold processing

    # Enrich chunks if requested
    df = _enrich_chunks_if_needed(df, args)

    # Generate embeddings
    embeddings, ids, dimension = _generate_embeddings(df, args)
    if embeddings is None:
        return

    # Build and validate index
    manager = _build_and_validate_index(embeddings, ids, dimension, args)

    # Save index if not dry-run
    if not args.dry_run:
        if args.index_root:
            _publish_versioned_index(manager, args)
        else:
            manager.save()
    else:
        logger.info("Dry-run enabled; FAISS indexes were not saved")


def _filter_to_representatives(df: pd.DataFrame, id_column: str, map_path: str) -> pd.DataFrame:
    """Filter dataframe to only include representative chunks from deduplication map."""
    logger.info("Applying deduplication map from %s", map_path)
    with open(map_path, encoding="utf-8") as f:
        payload = json.load(f)

    canonical_ids = _extract_canonical_ids(payload)
    if not canonical_ids:
        logger.warning("Dedup map did not contain canonical ids; skipping filter")
        return df

    filtered = df[df[id_column].astype(str).isin(canonical_ids)].copy()
    cluster_lookup = _flatten_cluster_lookup(payload)
    if cluster_lookup:
        filtered["cluster_id"] = filtered[id_column].astype(str).map(cluster_lookup)
        filtered["is_representative"] = filtered[id_column].astype(str).isin(canonical_ids)
    return filtered


def _extract_canonical_ids(payload: Dict[str, Any]) -> Set[str]:
    """Extract canonical chunk IDs from deduplication payload."""
    representatives = payload.get("representatives")
    if isinstance(representatives, dict) and representatives:
        ids = {str(rep.get("chunk_id")) for rep in representatives.values() if rep.get("chunk_id")}
        if ids:
            return ids
    canonical_map = payload.get("canonical_map", {}) or {}
    return {str(value) for value in canonical_map.values()}


def _flatten_cluster_lookup(payload: Dict[str, Any]) -> Dict[str, int]:
    """Create a lookup dictionary mapping chunk IDs to their cluster IDs."""
    clusters = payload.get("clusters")
    if not isinstance(clusters, dict):
        return {}
    lookup: Dict[str, int] = {}
    for cluster_id, members in clusters.items():
        if not isinstance(members, list):
            continue
        for member in members:
            lookup[str(member)] = int(cluster_id)
    return lookup


if __name__ == "__main__":
    main()
