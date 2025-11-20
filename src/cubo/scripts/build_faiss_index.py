"""CLI to build FAISS hot/cold indexes from chunk parquet data."""
from pathlib import Path
import argparse
import logging
import json

import pandas as pd

from src.cubo.config import config
from src.cubo.embeddings.embedding_generator import EmbeddingGenerator
from src.cubo.indexing.faiss_index import FAISSIndexManager
from src.cubo.indexing.index_publisher import publish_version
import time
from src.cubo.utils.logger import logger
from src.cubo.processing.enrichment import ChunkEnricher
from src.cubo.processing.generator import create_response_generator
from src.cubo.processing.scaffold import ScaffoldGenerator


def parse_args():
    parser = argparse.ArgumentParser(description="Build hot/cold FAISS indexes from chunk parquet")
    parser.add_argument('--parquet', required=True, help='Parquet file containing chunk text and ids')
    parser.add_argument('--text-column', default='text', help='Name of the text column in the parquet file')
    parser.add_argument('--summary-column', default=None, help='Optional column to embed summaries instead of chunks')
    parser.add_argument('--id-column', default='chunk_id', help='Column containing chunk ids')
    parser.add_argument('--index-dir', default=config.get('faiss_index_dir', 'faiss_index'))
    parser.add_argument('--publish', action='store_true', help='Publish the index version by flipping pointer and updating metadata DB')
    parser.add_argument('--cleanup', action='store_true', help='Cleanup old index versions according to retention policy')
    parser.add_argument('--retention', type=int, default=3, help='Number of index versions to keep when --cleanup is enabled')
    parser.add_argument('--index-root', default=config.get('faiss_index_root', None), help='Root dir where versioned faiss_v* dirs are stored')
    parser.add_argument('--batch-size', type=int, default=config.get('embedding_batch_size', 32))
    parser.add_argument('--hot-fraction', type=float, default=0.25)
    parser.add_argument('--nlist', type=int, default=64)
    parser.add_argument('--hnsw-m', type=int, default=16)
    parser.add_argument('--dry-run', action='store_true', help='Generate indexes without persisting them')
    parser.add_argument('--verbose', action='store_true', help='Log progress details')
    parser.add_argument('--enrich-chunks', action='store_true', help='Enrich chunks with summaries, keywords, etc.')
    parser.add_argument('--output-parquet', help='Path to save the enriched data as a Parquet file.')
    parser.add_argument('--dedup-map', help='Path to the deduplication map file.')
    parser.add_argument('--scaffold', action='store_true', help='Build scaffold index for fast retrieval')
    parser.add_argument('--scaffold-dir', default='scaffold_index', help='(Deprecated) Directory to save scaffold outputs')
    parser.add_argument('--scaffold-output-dir', default='data/scaffolds', help='Directory to save scaffold outputs (default: data/scaffolds)')
    parser.add_argument('--scaffold-size', type=int, default=5, help='Target number of chunks per scaffold')
    parser.add_argument('--use-opq', action='store_true', help='Use OPQ (Optimized Product Quantization) for cold index')
    parser.add_argument('--opq-m', type=int, default=32, help='OPQ subspace dimension')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    parquet_path = Path(args.parquet)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    
    if args.dedup_map:
        logger.info(f"Applying deduplication map from {args.dedup_map}...")
        with open(args.dedup_map, 'r') as f:
            dedup_map = json.load(f)['canonical_map']
        
        canonical_ids = set(dedup_map.values())
        df = df[df[args.id_column].isin(canonical_ids)]
        logger.info(f"Filtered down to {len(df)} canonical documents.")

    # Handle scaffold generation if requested
    if args.scaffold:
        logger.info("Building scaffold index...")
        generator = EmbeddingGenerator(batch_size=args.batch_size)
        
        # Create enricher if needed
        enricher = None
        if args.enrich_chunks or not args.summary_column:
            logger.info("Creating enricher for scaffold summaries...")
            enricher = ChunkEnricher(llm_provider=create_response_generator())
        
        scaffold_gen = ScaffoldGenerator(
            enricher=enricher,
            embedding_generator=generator,
            scaffold_size=args.scaffold_size
        )
        
        # Generate scaffolds
        scaffolds_result = scaffold_gen.generate_scaffolds(
            df,
            text_column=args.text_column,
            id_column=args.id_column
        )
        
        # Save scaffolds into a run-specific directory and write a manifest
        from uuid import uuid4
        run_id = f"scaffold_{int(time.time())}_{uuid4().hex[:8]}"
        from src.cubo.processing.scaffold import save_scaffold_run
        scaffold_output = Path(args.scaffold_output_dir) if getattr(args, 'scaffold_output_dir', None) else Path(args.scaffold_dir)
        res = save_scaffold_run(run_id, scaffolds_result, output_root=scaffold_output, model_version=config.get('llm_model'), input_chunks_df=df, id_column=args.id_column)
        scaffold_paths = res
        logger.info(f"Scaffold outputs saved to {res['run_dir']} (manifest: {res['manifest']})")
        
        # Build FAISS index on scaffolds instead of chunks
        scaffold_df = scaffolds_result['scaffolds_df']
        texts = scaffold_df['summary'].fillna('').astype(str).tolist()
        ids = scaffold_df['scaffold_id'].astype(str).tolist()
        embeddings = scaffolds_result['scaffold_embeddings']
        
        # Build scaffold index
        dimension = len(embeddings[0]) if embeddings else 0
        if dimension == 0:
            raise ValueError("Unable to determine embedding dimension for scaffolds")
        
        manager = FAISSIndexManager(
            dimension=dimension,
            index_dir=Path(res['run_dir']) / 'faiss',
            nlist=args.nlist,
            hnsw_m=args.hnsw_m,
            hot_fraction=args.hot_fraction,
            use_opq=args.use_opq,
            opq_m=args.opq_m
        )
        manager.build_indexes(embeddings, ids)
        
        # Sample search
        sample_query = embeddings[0]
        hits = manager.search(sample_query, k=min(5, len(ids)))
        logger.info(f"Scaffold sample search returned {len(hits)} hits")
        
        if not args.dry_run:
                if args.index_root:
                    ts = int(time.time())
                    version_dir = Path(args.index_root) / f"faiss_v{ts}"
                    manager.save(path=version_dir)
                    if args.publish:
                        publish_version(version_dir, Path(args.index_root))
                        logger.info(f"Published version {version_dir}")
                    if args.cleanup:
                        from src.cubo.indexing.index_publisher import cleanup
                        cleanup(Path(args.index_root), keep_last_n=args.retention)
                else:
                    manager.save()
            logger.info(f"Scaffold FAISS index saved to {Path(args.scaffold_dir) / 'faiss'}")
        else:
            logger.info("Dry-run enabled; scaffold FAISS index was not saved")
        
        return  # Exit after scaffold processing

    if args.enrich_chunks:
        logger.info("Enriching chunks...")
        enricher = ChunkEnricher(llm_provider=create_response_generator())
        enriched_data = enricher.enrich_chunks(df[args.text_column].tolist())
        enriched_df = pd.DataFrame(enriched_data)
        
        # Merge the enriched data with the original dataframe
        df = pd.concat([df.reset_index(drop=True), enriched_df.drop('text', axis=1)], axis=1)

        if args.output_parquet:
            df.to_parquet(args.output_parquet)
            logger.info(f"Enriched data saved to {args.output_parquet}")

    # If a summary column is specified, build embeddings on that column instead of chunk text
    embed_column = args.summary_column or ('summary' if args.enrich_chunks else args.text_column)
    texts = df[embed_column].fillna('').astype(str).tolist()
    ids = df[args.id_column].astype(str).tolist()

    if not texts:
        logger.warning("No chunks found in parquet file; nothing to index")
        return
    if len(texts) != len(ids):
        raise ValueError("Text and id columns must have the same length")

    generator = EmbeddingGenerator(batch_size=args.batch_size)
    embeddings = generator.encode(texts, batch_size=args.batch_size)
    dimension = len(embeddings[0]) if embeddings else 0
    if dimension == 0:
        raise ValueError("Unable to determine embedding dimension")

    manager = FAISSIndexManager(
        dimension=dimension,
        index_dir=Path(args.index_dir),
        nlist=args.nlist,
        hnsw_m=args.hnsw_m,
        hot_fraction=args.hot_fraction,
        use_opq=args.use_opq,
        opq_m=args.opq_m
    )
    manager.build_indexes(embeddings, ids)
    sample_query = embeddings[0]
    hits = manager.search(sample_query, k=min(5, len(ids)))
    logger.info(f"Sample search returned {len(hits)} hits (first id: {hits[0]['id'] if hits else 'none'})")

    if not args.dry_run:
        manager.save()
    else:
        logger.info("Dry-run enabled; FAISS indexes were not saved")


if __name__ == '__main__':
    main()