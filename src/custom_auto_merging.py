"""
Custom Auto-Merging Retrieval for CUBO
Implements hierarchical chunking and intelligent merging without LlamaIndex.
"""

import hashlib
import json
from typing import List, Dict, Any
from pathlib import Path
from collections import defaultdict

from sentence_transformers import SentenceTransformer
import chromadb

from src.logger import logger
from src.config import config
from src.utils import Utils


class HierarchicalChunker:
    """Creates hierarchical chunks at multiple levels."""

    def __init__(self, chunk_sizes: List[int] = None):
        if chunk_sizes is None:
            # Use config values or defaults
            chunk_sizes = config.get("auto_merging_chunk_sizes", [2048, 512, 128])
        self.chunk_sizes = chunk_sizes

    def create_hierarchical_chunks(self, text: str, filename: str) -> List[Dict[str, Any]]:
        """Create hierarchical chunks from text."""
        chunks = []

        # Create chunks at each level
        for level, chunk_size in enumerate(self.chunk_sizes):
            level_chunks = self._create_chunks_at_level(text, chunk_size, level, filename)
            chunks.extend(level_chunks)

        return chunks

    def _create_chunks_at_level(self, text: str, chunk_size: int, level: int, filename: str) -> List[Dict[str, Any]]:
        """Create chunks at a specific hierarchical level."""
        chunks = []

        # Split text into chunks of approximately chunk_size tokens
        words = text.split()
        current_chunk = []
        current_tokens = 0

        for word in words:
            word_tokens = len(word.split())  # Approximate token count

            if current_tokens + word_tokens > chunk_size and current_chunk:
                # Create chunk
                chunk_text = ' '.join(current_chunk)
                chunk_id = self._generate_chunk_id(filename, level, len(chunks))

                chunk = {
                    'id': chunk_id,
                    'text': chunk_text,
                    'filename': filename,
                    'level': level,
                    'chunk_size': chunk_size,
                    'token_count': current_tokens,
                    'start_pos': len(' '.join(words[:len(current_chunk)])),
                    'end_pos': len(' '.join(words[:len(current_chunk) + len(current_chunk)])),
                    'parent_id': None,  # Will be set later
                    'child_ids': []
                }
                chunks.append(chunk)

                current_chunk = [word]
                current_tokens = word_tokens
            else:
                current_chunk.append(word)
                current_tokens += word_tokens

        # Add remaining chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunk_id = self._generate_chunk_id(filename, level, len(chunks))

            chunk = {
                'id': chunk_id,
                'text': chunk_text,
                'filename': filename,
                'level': level,
                'chunk_size': chunk_size,
                'token_count': current_tokens,
                'start_pos': len(' '.join(words[:len(current_chunk)])),
                'end_pos': len(text),
                'parent_id': None,
                'child_ids': []
            }
            chunks.append(chunk)

        return chunks

    def _generate_chunk_id(self, filename: str, level: int, index: int) -> str:
        """Generate unique chunk ID."""
        content = f"{filename}_{level}_{index}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def build_hierarchy(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build parent-child relationships between chunks."""
        # Group chunks by level
        level_groups = defaultdict(list)
        for chunk in chunks:
            level_groups[chunk['level']].append(chunk)

        # For each level, establish parent-child relationships
        for level in range(len(self.chunk_sizes) - 1):  # Don't process leaf level
            parent_level = level
            child_level = level + 1

            if parent_level in level_groups and child_level in level_groups:
                parents = level_groups[parent_level]
                children = level_groups[child_level]

                # Assign children to parents based on position overlap
                for parent in parents:
                    parent_children = []
                    parent_start = parent['start_pos']
                    parent_end = parent['end_pos']

                    for child in children:
                        child_start = child['start_pos']
                        child_end = child['end_pos']

                        # Check if child overlaps with parent
                        if (child_start >= parent_start and child_start <= parent_end) or \
                           (child_end >= parent_start and child_end <= parent_end) or \
                           (child_start <= parent_start and child_end >= parent_end):
                            parent_children.append(child['id'])
                            child['parent_id'] = parent['id']

                    parent['child_ids'] = parent_children

        return chunks


class AutoMergingRetriever:
    """Auto-merging retrieval system using ChromaDB."""

    def __init__(self, model: SentenceTransformer, chunk_sizes: List[int] = None):
        self.model = model
        self.chunker = HierarchicalChunker(chunk_sizes)

        # Use config for collection name and db path
        self.collection_name = config.get("auto_merging_collection_name", "cubo_auto_merging")

        # Configurable parameters
        self.candidate_multiplier = config.get("auto_merging_candidate_multiplier", 3)
        self.parent_similarity_threshold = config.get("auto_merging_parent_similarity_threshold", 0.1)

        # Initialize ChromaDB
        chroma_path = config.get("vector_db_path", "./chroma_db")
        self.client = chromadb.PersistentClient(
            path=chroma_path
        )
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name
        )

        # Track loaded documents
        self.loaded_documents = set()

    def add_document(self, filepath: str, force_reindex: bool = False) -> bool:
        """Add document with hierarchical chunking."""
        filename = Path(filepath).name

        if filename in self.loaded_documents and not force_reindex:
            logger.info(f"Document {filename} already loaded")
            return False

        # Read and process document
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
            text = Utils.clean_text(text)
        except Exception as e:
            logger.error(f"Failed to read {filepath}: {e}")
            return False

        # Create hierarchical chunks
        chunks = self.chunker.create_hierarchical_chunks(text, filename)
        chunks = self.chunker.build_hierarchy(chunks)

        # Prepare data for ChromaDB
        ids = []
        documents = []
        metadatas = []

        for chunk in chunks:
            ids.append(chunk['id'])
            documents.append(chunk['text'])

            metadata = {
                'filename': chunk['filename'],
                'level': chunk['level'],
                'chunk_size': chunk['chunk_size'],
                'token_count': chunk['token_count'],
                'start_pos': chunk['start_pos'],
                'end_pos': chunk['end_pos'],
                'child_ids': json.dumps(chunk['child_ids'])
            }

            # Only add parent_id if it's not None
            if chunk['parent_id'] is not None:
                metadata['parent_id'] = chunk['parent_id']
            metadatas.append(metadata)

        # Generate embeddings
        embeddings = self.model.encode(documents).tolist()

        # Add to ChromaDB
        try:
            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )

            self.loaded_documents.add(filename)
            logger.info(f"Added {len(chunks)} hierarchical chunks for {filename}")
            return True

        except Exception as e:
            logger.error(f"Failed to add chunks to ChromaDB: {e}")
            return False

    def retrieve(self, query: str, top_k: int = 6) -> List[Dict[str, Any]]:
        """Retrieve with auto-merging logic."""
        # Generate query embedding
        query_embedding = self.model.encode([query]).tolist()[0]

        # Get candidate chunks (more than needed for merging)
        candidates = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k * self.candidate_multiplier,  # Get more candidates
            include=['documents', 'metadatas', 'distances']
        )

        if not candidates['documents']:
            return []

        # Process candidates and apply merging
        merged_results = self._merge_chunks(candidates, top_k)

        return merged_results

    def _merge_chunks(self, candidates, top_k: int) -> List[Dict[str, Any]]:
        """Apply auto-merging logic to retrieved chunks."""
        merged_results = []

        documents = candidates['documents'][0]
        metadatas = candidates['metadatas'][0]
        distances = candidates['distances'][0]

        # Group chunks by document and level
        doc_chunks = defaultdict(lambda: defaultdict(list))

        for doc, metadata, distance in zip(documents, metadatas, distances):
            filename = metadata['filename']
            level = metadata['level']

            doc_chunks[filename][level].append({
                'text': doc,
                'metadata': metadata,
                'similarity': 1 - distance,
                'level': level
            })

        # For each document, apply merging logic
        for filename, level_chunks in doc_chunks.items():
            doc_results = self._merge_document_chunks(level_chunks, top_k)
            merged_results.extend(doc_results)

        # Sort by similarity and return top_k
        merged_results.sort(key=lambda x: x['similarity'], reverse=True)
        return merged_results[:top_k]

    def _merge_document_chunks(self, level_chunks: Dict[int, List], top_k: int) -> List[Dict[str, Any]]:
        """Merge chunks within a document using hierarchical logic."""
        results = []

        # Get leaf level chunks (smallest chunks)
        leaf_level = max(level_chunks.keys())
        leaf_chunks = level_chunks[leaf_level]

        # Sort by similarity
        leaf_chunks.sort(key=lambda x: x['similarity'], reverse=True)

        # For top chunks, try to merge with parents
        for chunk in leaf_chunks[:top_k]:
            merged_chunk = self._get_merged_content(chunk, level_chunks)
            results.append(merged_chunk)

        return results

    def _get_merged_content(self, chunk: Dict[str, Any], level_chunks: Dict[int, List]) -> Dict[str, Any]:
        """Get merged content for a chunk, preferring larger parent chunks when beneficial."""
        current_level = chunk['level']
        parent_id = chunk['metadata'].get('parent_id')

        # If this chunk has a parent and the parent would provide better context
        if parent_id and current_level > 0:
            parent_level = current_level - 1
            if parent_level in level_chunks:
                # Look for the parent chunk
                for parent_chunk in level_chunks[parent_level]:
                    if parent_chunk['metadata'].get('id') == parent_id:
                        # Check if parent has significantly higher similarity
                        parent_similarity = parent_chunk['similarity']
                        child_similarity = chunk['similarity']

                        # If parent is much better, use it
                        if parent_similarity > child_similarity + self.parent_similarity_threshold:
                            return {
                                'document': parent_chunk['text'],
                                'metadata': parent_chunk['metadata'],
                                'similarity': parent_similarity
                            }

        # Otherwise, return the original chunk
        return {
            'document': chunk['text'],
            'metadata': chunk['metadata'],
            'similarity': chunk['similarity']
        }

    def get_loaded_documents(self) -> List[str]:
        """Get list of loaded document filenames."""
        return list(self.loaded_documents)

    def clear_documents(self):
        """Clear all loaded documents."""
        try:
            # Delete collection and recreate
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name
            )
            self.loaded_documents.clear()
            logger.info("Cleared all auto-merging documents")
        except Exception as e:
            logger.error(f"Failed to clear documents: {e}")
