"""
Configuration management using Pydantic.
Provides validation and type safety for application settings.
"""

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from cubo.config.prompt_defaults import DEFAULT_SYSTEM_PROMPT


class Paths(BaseSettings):
    """Path configuration."""

    data_folder: Path = Field(default=Path("./data"), description="Root data directory")
    deep_output_dir: Path = Field(
        default=Path("./data/deep"), description="Output for deep ingestion"
    )
    fast_pass_output_dir: Path = Field(
        default=Path("./data/fastpass"), description="Output for fast pass"
    )

    model_config = SettingsConfigDict(env_prefix="CUBO_PATH_")


class ChunkingSettings(BaseSettings):
    """Chunking strategy configuration."""

    # Standard chunking (Hierarchical)
    chunk_size: int = Field(default=1000, ge=100, description="Max characters per chunk")
    min_chunk_size: int = Field(default=100, ge=10, description="Min characters per chunk")
    chunk_overlap_sentences: int = Field(
        default=1, ge=0, description="Number of sentences to overlap"
    )

    # Legacy / Benchmarking
    benchmark_chunk_size: int = Field(
        default=1200, description="Token size for naive benchmarking if needed"
    )

    model_config = SettingsConfigDict(env_prefix="CUBO_CHUNKING_")


class DolphinSettings(BaseSettings):
    """Dolphin Vision/Enhanced processing settings."""

    enabled: bool = Field(default=False, description="Enable Dolphin vision processing")
    model_path: Optional[str] = Field(default=None, description="Path to custom Dolphin model")

    model_config = SettingsConfigDict(env_prefix="CUBO_DOLPHIN_")


class RetrievalSettings(BaseSettings):
    """Retrieval system configuration."""

    # BM25
    bm25_k1: float = Field(default=1.5, description="Term frequency saturation parameter")
    bm25_b: float = Field(default=0.75, description="Length normalization parameter")
    bm25_normalization_factor: float = Field(
        default=15.0, description="Empirical max score for normalization"
    )

    # RRF
    rrf_k: int = Field(default=60, description="Reciprocal Rank Fusion constant")

    # Retrieval Defaults
    default_top_k: int = Field(default=3, description="Default number of documents to retrieve")
    default_window_size: int = Field(default=3, description="Sentence window size")
    initial_retrieval_multiplier: int = Field(
        default=5, description="Multiplier for initial candidate retrieval"
    )
    min_candidate_pool_size: int = Field(
        default=50, description="Minimum number of candidates to retrieve for hybrid fusion"
    )
    complexity_length_threshold: int = Field(
        default=12, description="Query length threshold for complexity"
    )

    # Fusion Weights
    semantic_weight_default: float = Field(default=0.7, description="Default semantic weight")
    bm25_weight_default: float = Field(default=0.3, description="Default BM25 weight")
    semantic_weight_detailed: float = Field(
        default=0.1, description="Semantic weight for detailed queries"
    )
    bm25_weight_detailed: float = Field(default=0.9, description="BM25 weight for detailed queries")

    # Scoring Thresholds
    min_bm25_threshold: float = Field(
        default=0.05, description="Min normalized BM25 score for boosting"
    )
    keyword_boost_factor: float = Field(default=0.3, description="Boost factor for keyword matches")

    # Cache
    cache_size: int = Field(default=100, description="Number of entries in cache")
    cache_ttl: int = Field(default=3600, description="Cache TTL in seconds")

    model_config = SettingsConfigDict(env_prefix="CUBO_RETRIEVAL_")


class LLMSettings(BaseSettings):
    """LLM configuration."""

    provider: str = Field(default="ollama", description="LLM provider (ollama, local)")
    model_name: str = Field(default="llama3.2:latest", description="Model name to use")
    system_prompt: str = Field(default=DEFAULT_SYSTEM_PROMPT, description="Default system prompt")
    enable_streaming: bool = Field(
        default=False, description="Enable incremental token streaming (opt-in)"
    )

    model_config = SettingsConfigDict(env_prefix="CUBO_LLM_")


class Settings(BaseSettings):
    """Main application settings."""

    paths: Paths = Field(default_factory=Paths)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    dolphin: DolphinSettings = Field(default_factory=DolphinSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)

    # General
    log_level: str = Field(default="INFO")

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", env_nested_delimiter="__", extra="ignore"
    )


# Global settings instance
settings = Settings()
