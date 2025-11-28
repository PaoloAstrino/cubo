"""
Retrieval data models.

This module defines Pydantic models for retrieval operations, providing
runtime validation and better IDE support. These replace the Dict[str, Any]
patterns throughout the retrieval code.
"""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class ScoreBreakdown(BaseModel):
    """Detailed breakdown of a retrieval score."""

    final_score: float = Field(description="Combined final score")
    semantic_score: float = Field(default=0.0, description="Semantic similarity score")
    bm25_score: float = Field(default=0.0, description="Raw BM25 score")
    bm25_normalized: float = Field(default=0.0, description="Normalized BM25 score")
    semantic_contribution: float = Field(default=0.0, description="Weighted semantic contribution")
    bm25_contribution: float = Field(default=0.0, description="Weighted BM25 contribution")
    tier_boost: float = Field(default=0.0, description="Boost from tiered retrieval")


class ChunkMetadata(BaseModel):
    """Metadata associated with a document chunk."""

    filename: str = Field(description="Source document filename")
    file_hash: str = Field(default="", description="Hash of source file content")
    filepath: str = Field(default="", description="Full path to source file")
    chunk_index: int = Field(default=0, description="Index of chunk in document")
    sentence_index: Optional[int] = Field(default=None, description="Sentence index for window chunks")
    window: str = Field(default="", description="Window context text")
    window_start: Optional[int] = Field(default=None, description="Start index of window")
    window_end: Optional[int] = Field(default=None, description="End index of window")
    sentence_token_count: int = Field(default=0, description="Token count of sentence")
    window_token_count: int = Field(default=0, description="Token count of window")
    token_count: int = Field(default=0, description="Token count (for character chunks)")
    total_chunks: Optional[int] = Field(default=None, description="Total chunks in document")
    score_breakdown: Optional[ScoreBreakdown] = Field(default=None, description="Score breakdown")
    dedup_cluster_id: Optional[int] = Field(default=None, description="Deduplication cluster ID")
    canonical_chunk_id: Optional[str] = Field(default=None, description="Canonical chunk ID for dedup")

    class Config:
        """Allow extra fields for backwards compatibility."""

        extra = "allow"


class RetrievalCandidate(BaseModel):
    """A candidate document chunk from retrieval."""

    id: Optional[str] = Field(default=None, description="Unique chunk identifier")
    document: str = Field(description="Text content of the chunk")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")
    similarity: float = Field(default=0.0, description="Similarity/relevance score")
    base_similarity: float = Field(default=0.0, description="Base semantic similarity")
    bm25_score: float = Field(default=0.0, description="BM25 score")
    bm25_normalized: float = Field(default=0.0, description="Normalized BM25 score")
    tier_boost: float = Field(default=0.0, description="Boost from tiered retrieval")
    canonical_chunk_id: Optional[str] = Field(default=None, description="Canonical chunk ID")

    class Config:
        """Allow extra fields for flexibility."""

        extra = "allow"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for backwards compatibility."""
        result = {
            "id": self.id,
            "document": self.document,
            "metadata": self.metadata,
            "similarity": self.similarity,
        }
        if self.base_similarity:
            result["base_similarity"] = self.base_similarity
        if self.bm25_normalized:
            result["bm25_normalized"] = self.bm25_normalized
        if self.tier_boost:
            result["tier_boost"] = self.tier_boost
        if self.canonical_chunk_id:
            result["canonical_chunk_id"] = self.canonical_chunk_id
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RetrievalCandidate":
        """Create from dictionary."""
        return cls(
            id=data.get("id"),
            document=data.get("document", data.get("content", "")),
            metadata=data.get("metadata", {}),
            similarity=data.get("similarity", data.get("score", 0.0)),
            base_similarity=data.get("base_similarity", 0.0),
            bm25_score=data.get("bm25_score", 0.0),
            bm25_normalized=data.get("bm25_normalized", 0.0),
            tier_boost=data.get("tier_boost", 0.0),
            canonical_chunk_id=data.get("canonical_chunk_id"),
        )


class RetrievalResult(BaseModel):
    """Result from a retrieval operation."""

    candidates: List[RetrievalCandidate] = Field(default_factory=list, description="Retrieved candidates")
    method: str = Field(default="hybrid", description="Retrieval method used")
    query: str = Field(default="", description="Original query")
    top_k: int = Field(default=0, description="Number of results requested")
    trace_id: Optional[str] = Field(default=None, description="Trace ID for debugging")

    def to_legacy_format(self) -> List[Dict[str, Any]]:
        """Convert to legacy list of dicts format for backwards compatibility."""
        return [c.to_dict() for c in self.candidates]


class FusedResult(BaseModel):
    """Result from fusing BM25 and semantic search."""

    doc_id: str = Field(description="Document identifier")
    score: float = Field(description="Fused relevance score")
    document: str = Field(default="", description="Document content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    semantic_score: float = Field(default=0.0, description="Semantic similarity score")
    bm25_score: float = Field(default=0.0, description="BM25 score")


class DocumentChunk(BaseModel):
    """A processed document chunk ready for indexing."""

    text: str = Field(description="Chunk text content")
    chunk_id: str = Field(description="Unique chunk identifier")
    metadata: ChunkMetadata = Field(description="Chunk metadata")
    embedding: Optional[List[float]] = Field(default=None, description="Embedding vector")


class ChunkData(BaseModel):
    """Prepared chunk data for batch operations."""

    texts: List[str] = Field(default_factory=list, description="List of chunk texts")
    metadatas: List[Dict[str, Any]] = Field(default_factory=list, description="List of metadata dicts")
    chunk_ids: List[str] = Field(default_factory=list, description="List of chunk IDs")
    embeddings: Optional[List[List[float]]] = Field(default=None, description="List of embeddings")


# Type aliases for backwards compatibility
CandidateDict = Dict[str, Any]
MetadataDict = Dict[str, Any]


def candidate_to_dict(candidate: Union[RetrievalCandidate, Dict[str, Any]]) -> Dict[str, Any]:
    """Convert a candidate to dictionary format."""
    if isinstance(candidate, RetrievalCandidate):
        return candidate.to_dict()
    return candidate


def dict_to_candidate(data: Dict[str, Any]) -> RetrievalCandidate:
    """Convert a dictionary to RetrievalCandidate."""
    return RetrievalCandidate.from_dict(data)
