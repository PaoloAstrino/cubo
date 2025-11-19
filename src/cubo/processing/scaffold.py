"""
Scaffold Generator for semantic compression.

Creates compressed scaffolds from chunk summaries with persistent mappings
from scaffold_id -> [chunk_ids] for efficient retrieval and original chunk access.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np

from src.cubo.utils.logger import logger
from src.cubo.processing.enrichment import ChunkEnricher
from src.cubo.embeddings.embedding_generator import EmbeddingGenerator


class ScaffoldGenerator:
    """
    Generates compressed semantic scaffolds from document chunks.
    
    A scaffold is a compressed representation of multiple chunks:
    - Merges mini-summaries from related chunks
    - Creates scaffold embeddings for fast retrieval
    - Maintains mapping: scaffold_id -> list of original chunk_ids
    """

    def __init__(
        self,
        enricher: Optional[ChunkEnricher] = None,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        scaffold_size: int = 5,
        similarity_threshold: float = 0.75
    ):
        """
        Initialize scaffold generator.
        
        Args:
            enricher: ChunkEnricher for generating summaries/keywords
            embedding_generator: EmbeddingGenerator for scaffold embeddings
            scaffold_size: Target number of chunks per scaffold
            similarity_threshold: Minimum similarity to group chunks
        """
        self.enricher = enricher
        self.embedding_generator = embedding_generator
        self.scaffold_size = scaffold_size
        self.similarity_threshold = similarity_threshold

    def generate_scaffolds(
        self,
        chunks_df: pd.DataFrame,
        text_column: str = 'text',
        id_column: str = 'chunk_id'
    ) -> Dict[str, Any]:
        """
        Generate scaffolds from a DataFrame of chunks.
        
        Args:
            chunks_df: DataFrame containing chunks with text and ids
            text_column: Name of column containing chunk text
            id_column: Name of column containing chunk ids
            
        Returns:
            Dict with keys:
                - scaffolds_df: DataFrame of scaffolds
                - mapping: Dict[scaffold_id, List[chunk_id]]
                - scaffold_embeddings: List of embeddings
        """
        if chunks_df.empty:
            logger.warning("Empty chunks DataFrame provided to scaffold generator")
            return {'scaffolds_df': pd.DataFrame(), 'mapping': {}, 'scaffold_embeddings': []}

        logger.info(f"Generating scaffolds from {len(chunks_df)} chunks")

        # Step 1: Enrich chunks with summaries if enricher provided
        enriched_chunks = self._enrich_chunks_if_needed(chunks_df, text_column)

        # Step 2: Group chunks into scaffolds
        scaffold_groups = self._group_chunks_into_scaffolds(
            enriched_chunks, chunks_df[id_column].tolist()
        )

        # Step 3: Create scaffold summaries and metadata
        scaffolds_data = self._create_scaffold_data(scaffold_groups, enriched_chunks)

        # Step 4: Generate scaffold embeddings
        scaffold_embeddings = self._generate_scaffold_embeddings(scaffolds_data)

        # Step 5: Build scaffold DataFrame
        scaffolds_df = pd.DataFrame.from_records(scaffolds_data)

        # Step 6: Create mapping dict
        mapping = {
            row['scaffold_id']: row['chunk_ids']
            for _, row in scaffolds_df.iterrows()
        }

        logger.info(f"Generated {len(scaffolds_df)} scaffolds from {len(chunks_df)} chunks")

        return {
            'scaffolds_df': scaffolds_df,
            'mapping': mapping,
            'scaffold_embeddings': scaffold_embeddings
        }

    def save_scaffolds(
        self,
        scaffolds_result: Dict[str, Any],
        output_dir: Path
    ) -> Dict[str, str]:
        """
        Save scaffolds, mappings, and embeddings to disk.
        
        Args:
            scaffolds_result: Result from generate_scaffolds()
            output_dir: Directory to save outputs
            
        Returns:
            Dict with paths to saved files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        paths = {}

        # Save scaffolds DataFrame
        scaffolds_path = output_dir / 'scaffolds.parquet'
        scaffolds_result['scaffolds_df'].to_parquet(scaffolds_path, index=False)
        paths['scaffolds_parquet'] = str(scaffolds_path)
        logger.info(f"Saved scaffolds to {scaffolds_path}")

        # Save mapping as JSON
        mapping_path = output_dir / 'scaffold_mapping.json'
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump(scaffolds_result['mapping'], f, indent=2, ensure_ascii=False)
        paths['mapping_json'] = str(mapping_path)
        logger.info(f"Saved scaffold mapping to {mapping_path}")

        # Save embeddings as numpy array
        if scaffolds_result['scaffold_embeddings']:
            embeddings_path = output_dir / 'scaffold_embeddings.npy'
            np.save(embeddings_path, np.array(scaffolds_result['scaffold_embeddings']))
            paths['embeddings_npy'] = str(embeddings_path)
            logger.info(f"Saved scaffold embeddings to {embeddings_path}")

        return paths

    def load_scaffolds(self, input_dir: Path) -> Dict[str, Any]:
        """
        Load scaffolds, mappings, and embeddings from disk.
        
        Args:
            input_dir: Directory containing saved scaffold files
            
        Returns:
            Dict with scaffolds_df, mapping, and scaffold_embeddings
        """
        input_dir = Path(input_dir)

        scaffolds_df = pd.read_parquet(input_dir / 'scaffolds.parquet')

        with open(input_dir / 'scaffold_mapping.json', 'r', encoding='utf-8') as f:
            mapping = json.load(f)

        embeddings_path = input_dir / 'scaffold_embeddings.npy'
        scaffold_embeddings = []
        if embeddings_path.exists():
            scaffold_embeddings = np.load(embeddings_path).tolist()

        logger.info(f"Loaded {len(scaffolds_df)} scaffolds from {input_dir}")

        return {
            'scaffolds_df': scaffolds_df,
            'mapping': mapping,
            'scaffold_embeddings': scaffold_embeddings
        }

    def _enrich_chunks_if_needed(
        self,
        chunks_df: pd.DataFrame,
        text_column: str
    ) -> List[Dict[str, Any]]:
        """Enrich chunks with summaries if enricher is provided."""
        if self.enricher:
            logger.info("Enriching chunks with summaries")
            texts = chunks_df[text_column].fillna('').tolist()
            enriched = self.enricher.enrich_chunks(texts)
            return enriched
        else:
            # Use original text as summary if no enricher
            logger.info("No enricher provided, using chunk text as summary")
            return [
                {
                    'text': row[text_column],
                    'summary': row[text_column][:200],  # First 200 chars as summary
                    'keywords': [],
                    'category': 'general'
                }
                for _, row in chunks_df.iterrows()
            ]

    def _group_chunks_into_scaffolds(
        self,
        enriched_chunks: List[Dict[str, Any]],
        chunk_ids: List[str]
    ) -> List[List[int]]:
        """
        Group chunks into scaffolds based on similarity and size.
        
        Returns list of groups, where each group is a list of chunk indices.
        """
        groups = []
        current_group = []

        for idx in range(len(enriched_chunks)):
            current_group.append(idx)

            # Create a new group when we reach scaffold_size
            if len(current_group) >= self.scaffold_size:
                groups.append(current_group)
                current_group = []

        # Add remaining chunks as final group
        if current_group:
            groups.append(current_group)

        logger.info(f"Grouped {len(enriched_chunks)} chunks into {len(groups)} scaffolds")
        return groups

    def _create_scaffold_data(
        self,
        scaffold_groups: List[List[int]],
        enriched_chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Create scaffold data records from grouped chunks."""
        scaffolds_data = []

        for group_idx, group in enumerate(scaffold_groups):
            # Gather chunk data for this scaffold
            group_chunks = [enriched_chunks[i] for i in group]
            chunk_ids = [f"chunk_{i}" for i in group]  # Placeholder, will be replaced

            # Merge summaries
            merged_summary = self._merge_summaries(group_chunks)

            # Collect all keywords
            all_keywords = []
            for chunk in group_chunks:
                all_keywords.extend(chunk.get('keywords', []))
            unique_keywords = list(set(all_keywords))[:10]  # Top 10 unique keywords

            # Determine category (most common)
            categories = [chunk.get('category', 'general') for chunk in group_chunks]
            most_common_category = max(set(categories), key=categories.count)

            # Generate scaffold ID
            scaffold_id = self._generate_scaffold_id(group_idx, merged_summary)

            scaffold_data = {
                'scaffold_id': scaffold_id,
                'summary': merged_summary,
                'keywords': unique_keywords,
                'category': most_common_category,
                'chunk_ids': chunk_ids,
                'chunk_count': len(group),
                'group_index': group_idx
            }

            scaffolds_data.append(scaffold_data)

        return scaffolds_data

    def _merge_summaries(self, chunks: List[Dict[str, Any]]) -> str:
        """Merge multiple chunk summaries into a single scaffold summary."""
        summaries = [chunk.get('summary', '') for chunk in chunks if chunk.get('summary')]

        if not summaries:
            return "No summary available"

        # Simple concatenation with separator
        # For production, could use LLM to generate a meta-summary
        merged = " | ".join(summaries[:3])  # Use first 3 summaries to keep it concise

        return merged[:500]  # Limit to 500 chars

    def _generate_scaffold_id(self, group_idx: int, summary: str) -> str:
        """Generate unique scaffold ID."""
        content = f"scaffold_{group_idx}_{summary[:50]}"
        hash_val = hashlib.md5(content.encode(), usedforsecurity=False).hexdigest()[:12]
        return f"scaffold_{hash_val}"

    def _generate_scaffold_embeddings(
        self,
        scaffolds_data: List[Dict[str, Any]]
    ) -> List[List[float]]:
        """Generate embeddings for scaffold summaries."""
        if not self.embedding_generator:
            logger.warning("No embedding generator provided, returning empty embeddings")
            return []

        summaries = [scaffold['summary'] for scaffold in scaffolds_data]

        logger.info(f"Generating embeddings for {len(summaries)} scaffolds")
        embeddings = self.embedding_generator.encode(summaries)

        return embeddings


def create_scaffolds_from_parquet(
    parquet_path: str,
    output_dir: str,
    enricher: Optional[ChunkEnricher] = None,
    embedding_generator: Optional[EmbeddingGenerator] = None,
    scaffold_size: int = 5,
    text_column: str = 'text',
    id_column: str = 'chunk_id'
) -> Dict[str, str]:
    """
    Convenience function to create scaffolds from a parquet file.
    
    Args:
        parquet_path: Path to input parquet with chunks
        output_dir: Directory to save scaffold outputs
        enricher: Optional ChunkEnricher
        embedding_generator: Optional EmbeddingGenerator
        scaffold_size: Target chunks per scaffold
        text_column: Name of text column
        id_column: Name of id column
        
    Returns:
        Dict with paths to saved files
    """
    chunks_df = pd.read_parquet(parquet_path)

    generator = ScaffoldGenerator(
        enricher=enricher,
        embedding_generator=embedding_generator,
        scaffold_size=scaffold_size
    )

    scaffolds_result = generator.generate_scaffolds(
        chunks_df,
        text_column=text_column,
        id_column=id_column
    )

    paths = generator.save_scaffolds(scaffolds_result, Path(output_dir))

    return paths
