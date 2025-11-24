"""
Scaffold Generator for semantic compression.

Creates compressed scaffolds from chunk summaries with persistent mappings
from scaffold_id -> [chunk_ids] for efficient retrieval and original chunk access.
"""
from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.cubo.embeddings.embedding_generator import EmbeddingGenerator
from src.cubo.processing.enrichment import ChunkEnricher
from src.cubo.utils.logger import logger


class ScaffoldGenerator:
    """Generates compressed semantic scaffolds from document chunks."""

    def __init__(
        self,
        enricher: ChunkEnricher,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        scaffold_size: int = 5,
        similarity_threshold: float = 0.75,
        use_clustering: bool = False,
        clustering_method: str = 'kmeans',
    ):
        if enricher is None:
            raise ValueError("Enricher is a mandatory component and cannot be None")
        self.enricher = enricher
        self.embedding_generator = embedding_generator
        self.scaffold_size = scaffold_size
        self.similarity_threshold = similarity_threshold
        self.use_clustering = use_clustering
        self.clustering_method = clustering_method
        
        # Initialize clusterer if needed
        if self.use_clustering:
            from src.cubo.processing.clustering import SemanticClusterer
            self.clusterer = SemanticClusterer(method=clustering_method, min_cluster_size=scaffold_size)
        else:
            self.clusterer = None

    def generate_scaffolds(
        self,
        chunks_df: pd.DataFrame,
        text_column: str = 'text',
        id_column: str = 'chunk_id',
        chunk_embeddings: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        if chunks_df.empty:
            logger.warning("Empty chunks DataFrame provided to scaffold generator")
            return {'scaffolds_df': pd.DataFrame(), 'mapping': {}, 'scaffold_embeddings': []}
        logger.info(f"Generating scaffolds from {len(chunks_df)} chunks (clustering={self.use_clustering})")
        enriched_chunks = self._enrich_chunks_if_needed(chunks_df, text_column)
        chunk_ids = chunks_df[id_column].tolist()
        scaffold_groups = self._group_chunks_into_scaffolds(enriched_chunks, chunk_ids, chunk_embeddings)
        scaffolds_data = self._create_scaffold_data(scaffold_groups, enriched_chunks, chunks_df, text_column)
        scaffold_embeddings = self._generate_scaffold_embeddings(scaffolds_data)
        scaffolds_df = pd.DataFrame.from_records(scaffolds_data)
        mapping = {row['scaffold_id']: row['chunk_ids'] for _, row in scaffolds_df.iterrows()}
        logger.info(f"Generated {len(scaffolds_df)} scaffolds from {len(chunks_df)} chunks")
        return {'scaffolds_df': scaffolds_df, 'mapping': mapping, 'scaffold_embeddings': scaffold_embeddings}

    def save_scaffolds(self, scaffolds_result: Dict[str, Any], output_dir: Path, model_version: Optional[str] = None) -> Dict[str, str]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = {}
        scaffolds_path = output_dir / 'scaffold_metadata.parquet'
        # Annotate model_version column if provided
        if model_version is not None and not scaffolds_result['scaffolds_df'].empty:
            scaffolds_result['scaffolds_df']['model_version'] = model_version
        scaffolds_result['scaffolds_df'].to_parquet(scaffolds_path, index=False)
        paths['scaffolds_parquet'] = str(scaffolds_path)
        logger.info(f"Saved scaffolds to {scaffolds_path}")
        mapping_path = output_dir / 'scaffold_mapping.json'
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump(scaffolds_result['mapping'], f, indent=2, ensure_ascii=False)
        paths['mapping_json'] = str(mapping_path)
        logger.info(f"Saved scaffold mapping to {mapping_path}")
        if scaffolds_result['scaffold_embeddings']:
            embeddings_path = output_dir / 'scaffold_embeddings.npy'
            np.save(embeddings_path, np.array(scaffolds_result['scaffold_embeddings']))
            paths['embeddings_npy'] = str(embeddings_path)
            logger.info(f"Saved scaffold embeddings to {embeddings_path}")
        return paths

    def load_scaffolds(self, input_dir: Path) -> Dict[str, Any]:
        input_dir = Path(input_dir)
        scaffolds_df = pd.read_parquet(input_dir / 'scaffold_metadata.parquet')
        with open(input_dir / 'scaffold_mapping.json', encoding='utf-8') as f:
            mapping = json.load(f)
        embeddings_path = input_dir / 'scaffold_embeddings.npy'
        scaffold_embeddings = []
        if embeddings_path.exists():
            scaffold_embeddings = np.load(embeddings_path).tolist()
        logger.info(f"Loaded {len(scaffolds_df)} scaffolds from {input_dir}")
        return {'scaffolds_df': scaffolds_df, 'mapping': mapping, 'scaffold_embeddings': scaffold_embeddings}

    def _enrich_chunks_if_needed(self, chunks_df: pd.DataFrame, text_column: str) -> List[Dict[str, Any]]:
        logger.info("Enriching chunks with summaries (mandatory step)")
        texts = chunks_df[text_column].fillna('').tolist()
        enriched = self.enricher.enrich_chunks(texts)
        return enriched

    def _group_chunks_into_scaffolds(
        self,
        enriched_chunks: List[Dict[str, Any]],
        chunk_ids: List[str],
        chunk_embeddings: Optional[np.ndarray] = None
    ) -> List[List[int]]:
        """Group chunks into scaffolds using clustering or sequential grouping."""
        
        # Use semantic clustering if enabled and embeddings are available
        if self.use_clustering and self.clusterer and chunk_embeddings is not None:
            try:
                labels, n_clusters = self.clusterer.cluster_chunks(chunk_embeddings)
                
                # Convert cluster labels to groups
                groups = [[] for _ in range(n_clusters)]
                for idx, label in enumerate(labels):
                    groups[label].append(idx)
                
                # Filter out empty groups
                groups = [g for g in groups if g]
                
                logger.info(f"Semantic clustering grouped {len(enriched_chunks)} chunks into {len(groups)} scaffolds")
                return groups
            except Exception as e:
                logger.warning(f"Clustering failed: {e}, falling back to sequential grouping")
        
        # Fallback to sequential grouping
        groups = []
        current_group = []
        for idx in range(len(enriched_chunks)):
            current_group.append(idx)
            if len(current_group) >= self.scaffold_size:
                groups.append(current_group)
                current_group = []
        if current_group:
            groups.append(current_group)
        logger.info(f"Sequential grouping created {len(groups)} scaffolds from {len(enriched_chunks)} chunks")
        return groups

    def _create_scaffold_data(
        self,
        scaffold_groups: List[List[int]],
        enriched_chunks: List[Dict[str, Any]],
        original_chunks_df: pd.DataFrame,
        text_column: str = 'text',
    ) -> List[Dict[str, Any]]:
        scaffolds_data = []
        for group_idx, group in enumerate(scaffold_groups):
            group_chunks = [enriched_chunks[i] for i in group]
            # Use original chunk ids from original_chunks_df by index
            chunk_ids = [original_chunks_df.iloc[i].get('chunk_id') for i in group]
            merged_summary = self._merge_summaries(group_chunks)
            all_keywords = []
            for chunk in group_chunks:
                all_keywords.extend(chunk.get('keywords', []))
            unique_keywords = list(set(all_keywords))[:10]
            categories = [chunk.get('category', 'general') for chunk in group_chunks]
            most_common_category = max(set(categories), key=categories.count)
            scaffold_id = self._generate_scaffold_id(group_idx, merged_summary)
            scaffold_data = {
                'scaffold_id': scaffold_id,
                'summary': merged_summary,
                'keywords': unique_keywords,
                'category': most_common_category,
                'chunk_ids': chunk_ids,
                'chunk_count': len(group),
                'group_index': group_idx,
                # compute sizes and token counts
                'original_size': sum(len(str(original_chunks_df.iloc[i].get(text_column, ''))) for i in group),
                'compressed_size': len(merged_summary),
                'original_token_count': sum(int(original_chunks_df.iloc[i].get('token_count', 0) or len(str(original_chunks_df.iloc[i].get(text_column, '')).split())) for i in group),
                'compressed_token_count': len(str(merged_summary).split()),
                'compression_ratio': (
                    sum(len(str(original_chunks_df.iloc[i].get(text_column, ''))) for i in group) / len(merged_summary)
                    if len(merged_summary) > 0 else 0
                ),
            }
            scaffolds_data.append(scaffold_data)
        return scaffolds_data

    def _merge_summaries(self, chunks: List[Dict[str, Any]]) -> str:
        summaries = [chunk.get('summary', '') for chunk in chunks if chunk.get('summary')]
        if not summaries:
            return "No summary available"
        merged = " | ".join(summaries[:3])
        return merged[:500]

    def _generate_scaffold_id(self, group_idx: int, summary: str) -> str:
        content = f"scaffold_{group_idx}_{summary[:50]}"
        hash_val = hashlib.md5(content.encode(), usedforsecurity=False).hexdigest()[:12]
        return f"scaffold_{hash_val}"

    def _generate_scaffold_embeddings(self, scaffolds_data: List[Dict[str, Any]]) -> List[List[float]]:
        if not self.embedding_generator:
            logger.warning("No embedding generator provided, returning empty embeddings")
            return []
        summaries = [scaffold['summary'] for scaffold in scaffolds_data]
        logger.info(f"Generating embeddings for {len(summaries)} scaffolds")
        embeddings = self.embedding_generator.encode(summaries)
        return embeddings


def _build_chunks_summary_from_df(scaffolds_df: pd.DataFrame, input_chunks_df: Optional[pd.DataFrame] = None, id_column: str = 'chunk_id') -> List[Dict[str, Any]]:
    """Build a chunk-level summary for the manifest, optionally enriching with source metadata.

    Each summary entry contains: scaffold_id, chunk_id, scaffold_group and optionally filename, file_hash, token_count
    """
    chunks_summary = []
    if not scaffolds_df.empty:
        for _, row in scaffolds_df.iterrows():
            for cid in row['chunk_ids']:
                entry = {'scaffold_id': row['scaffold_id'], 'chunk_id': cid, 'scaffold_group': row['group_index']}
                if input_chunks_df is not None and id_column in input_chunks_df.columns:
                    matched = input_chunks_df[input_chunks_df[id_column] == cid]
                    if not matched.empty:
                        m = matched.iloc[0]
                        entry['filename'] = m.get('filename', '')
                        entry['file_hash'] = m.get('file_hash', '')
                        entry['token_count'] = int(m.get('token_count', 0) or len(str(m.get('text', '')).split()))
                chunks_summary.append(entry)
    return chunks_summary


def save_scaffold_run(
    run_id: str,
    scaffolds_result: Dict[str, Any],
    output_root: Path,
    model_version: Optional[str] = None,
    manifests_dir: Optional[Path] = None,
    input_chunks_df: Optional[pd.DataFrame] = None,
    id_column: str = 'chunk_id',
):
    # Use a nested scaffolds subdirectory under run_id for the final storage layout
    run_dir = Path(output_root) / run_id / 'scaffolds'
    run_dir.mkdir(parents=True, exist_ok=True)
    # Create a minimal generator just for saving (no enricher needed for save operation)
    # We only need the save_scaffolds method, which doesn't use enricher/embedding_generator
    from src.cubo.processing.enrichment import ChunkEnricher
    
    # Create a dummy enricher for the generator (only used for save operation)
    class DummyLLM:
        def generate_response(self, prompt, context):
            return ""
    
    dummy_enricher = ChunkEnricher(llm_provider=DummyLLM())
    generator = ScaffoldGenerator(enricher=dummy_enricher)
    paths = generator.save_scaffolds(scaffolds_result, run_dir, model_version=model_version)
    manifest_dir = Path(manifests_dir or (Path(run_dir).parent.parent / 'manifests'))
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"{run_id}_scaffold_manifest.json"
    scaffolds_df = scaffolds_result.get('scaffolds_df', pd.DataFrame())
    scaffold_count = len(scaffolds_df) if hasattr(scaffolds_df, '__len__') else 0
    chunks_summary = _build_chunks_summary_from_df(scaffolds_df, input_chunks_df, id_column)
    manifest = {
        'run_id': run_id,
        'scaffold_dir': str(run_dir),
        'created_at': datetime.datetime.utcnow().isoformat(),
        'model_version': model_version or '',
        'scaffold_count': scaffold_count,
        'chunks_summary': chunks_summary,
    }
    with open(manifest_path, 'w', encoding='utf-8') as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    # Add model_version to parquet metadata if present and parquet exists
    try:
        sc_path = Path(paths.get('scaffolds_parquet', ''))
        if sc_path.exists() and model_version is not None:
            df = pd.read_parquet(sc_path)
            df['model_version'] = model_version
            df.to_parquet(sc_path, index=False)
    except Exception:
        logger.warning("Failed to annotate scaffold parquet with model_version; proceeding")
    try:
        from src.cubo.storage.metadata_manager import get_metadata_manager
        manager = get_metadata_manager()
        manager.record_scaffold_run(run_id, str(run_dir), model_version or '', int(manifest['scaffold_count']), str(manifest_path))
        for s in chunks_summary:
            manager.add_scaffold_mapping(run_id, s['scaffold_id'], s['chunk_id'], {'group_index': s.get('scaffold_group')})
    except Exception:
        logger.warning("Failed to write scaffold run or mappings to metadata DB; proceeding")
    return {'run_dir': str(run_dir), 'manifest': str(manifest_path)}


def create_scaffolds_from_parquet(
    parquet_path: str,
    output_dir: str,
    enricher: ChunkEnricher,
    embedding_generator: Optional[EmbeddingGenerator] = None,
    scaffold_size: int = 5,
    text_column: str = 'text',
    id_column: str = 'chunk_id',
    run_id: Optional[str] = None,
    manifests_dir: Optional[Path] = None,
) -> Dict[str, str]:
    chunks_df = pd.read_parquet(parquet_path)
    generator = ScaffoldGenerator(enricher=enricher, embedding_generator=embedding_generator, scaffold_size=scaffold_size)
    scaffolds_result = generator.generate_scaffolds(chunks_df, text_column=text_column, id_column=id_column)
    paths = generator.save_scaffolds(scaffolds_result, Path(output_dir))
    if run_id:
        # Persist run with enriched input metadata if available
        save_scaffold_run(run_id, scaffolds_result, Path(output_dir).parent, model_version=None, manifests_dir=manifests_dir, input_chunks_df=chunks_df, id_column=id_column)
    return paths
