"""
CUBO Document Retriever
Handles document embedding, storage, and retrieval with ChromaDB.
"""

from typing import List, Dict
import math
import os
import json
import re
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
if not hasattr(np, "float_"):
    np.float_ = np.float64

import chromadb
from sentence_transformers import SentenceTransformer
import hashlib
from src.cubo.utils.logger import logger
from src.cubo.config import config
from src.cubo.services.service_manager import get_service_manager
from src.cubo.embeddings.model_inference_threading import get_model_inference_threading
from src.cubo.utils.exceptions import (
    CUBOError, DatabaseError, DocumentAlreadyExistsError, EmbeddingGenerationError,
    ModelNotAvailableError, FileAccessError, RetrievalError
)

from src.cubo.rerank.reranker import LocalReranker


class DocumentRetriever:
    """Handles document retrieval using ChromaDB and sentence transformers."""

    def __init__(self, model: SentenceTransformer, use_sentence_window: bool = True,
                 use_auto_merging: bool = False, auto_merge_for_complex: bool = True,
                 window_size: int = 3, top_k: int = 3):
        self._set_basic_attributes(model, use_sentence_window, use_auto_merging,
                                   auto_merge_for_complex, window_size, top_k)
        self._initialize_auto_merging_retriever()
        self._setup_chromadb()
        self._setup_caching()
        self._initialize_postprocessors()
        self._log_initialization_status()

    # ... additional methods omitted for brevity (moved as-is from original module)
