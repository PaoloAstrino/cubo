"""FastAPI server for CUBO RAG system."""

# ruff: noqa: E402

import asyncio
import csv
import datetime
import hashlib
import io
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiofiles

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
    status,
)
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from cubo.config import config
from cubo.core import CuboCore
from cubo.security.security import security_manager
from cubo.services.service_manager import ServiceManager
from cubo.storage.metadata_manager import get_metadata_manager
from cubo.utils.exceptions import (
    CUBOError,
    DocumentAlreadyExistsError,
    DocumentNotFoundError,
    InvalidConfigurationError,
    ModelLoadError,
    ValidationError,
)
from cubo.utils.logger import logger
from cubo.utils.logging_context import generate_trace_id, trace_context
from cubo.utils.trace_collector import trace_collector

# Global app instance - uses CuboCore (no CLI side effects)
cubo_app: Optional[CuboCore] = None
service_manager: Optional[ServiceManager] = None
# Global lock for heavy compute operations (indexing, querying) to prevent OOM
compute_lock = asyncio.Lock()


class _ReadinessSnapshot:
    def __init__(self):
        self._lock = asyncio.Lock()
        self._components: Dict[str, bool] = {
            "api": True,
            "app": False,
            "service_manager": False,
            "retriever": False,
            "generator": False,
            "doc_loader": False,
            "vector_store": False,
        }
        self._updated_at: float = 0.0

    async def set_components(self, components: Dict[str, bool]) -> None:
        async with self._lock:
            # Make a shallow copy to avoid accidental mutation by callers.
            self._components = dict(components)
            self._updated_at = asyncio.get_running_loop().time()

    async def get_components(self) -> Dict[str, bool]:
        async with self._lock:
            return dict(self._components)


class _DocumentsCache:
    def __init__(self, data_dir: Path, ttl_seconds: float = 1.0):
        self._data_dir = data_dir
        self._ttl_seconds = ttl_seconds
        self._lock = asyncio.Lock()
        self._documents: List["DocumentResponse"] = []
        self._etag: str = 'W/"empty"'
        self._updated_at: float = 0.0
        self._resolved_dir: Optional[Path] = None

    def invalidate(self) -> None:
        self._updated_at = 0.0

    def _is_fresh(self) -> bool:
        now = asyncio.get_running_loop().time()
        return (now - self._updated_at) <= self._ttl_seconds

    @staticmethod
    def _compute_docs_etag(items: List[Tuple[str, int, int]]) -> str:
        # Weak ETag: stable across equivalent directory states.
        # items is a sorted list of (name, mtime_ns, size)
        # Use SHA-256 for ETag hashing (SHA-1 is considered weak for security-sensitive uses)
        h = hashlib.sha256()
        for name, mtime_ns, size in items:
            h.update(name.encode("utf-8", errors="ignore"))
            h.update(b"\0")
            h.update(str(mtime_ns).encode("ascii"))
            h.update(b"\0")
            h.update(str(size).encode("ascii"))
            h.update(b"\n")
        return f'W/"{h.hexdigest()}"'

    async def get(self) -> Tuple[List["DocumentResponse"], str]:
        async with self._lock:
            # If the underlying working directory changes (common in tests that chdir
            # into a tmp dir), a relative Path("data") points at a different folder.
            # Detect that and force refresh.
            try:
                resolved_dir = self._data_dir.resolve()
            except Exception:
                resolved_dir = None

            if resolved_dir is not None and self._resolved_dir != resolved_dir:
                self._resolved_dir = resolved_dir
                self._updated_at = 0.0

            if self._is_fresh():
                return self._documents, self._etag

            # Refresh in threadpool to avoid blocking the event loop.
            exists = await run_in_threadpool(self._data_dir.exists)
            if not exists:
                self._documents = []
                self._etag = 'W/"empty"'
                self._updated_at = asyncio.get_running_loop().time()
                return self._documents, self._etag

            def _scan_docs() -> Tuple[List["DocumentResponse"], str]:
                items: List[Tuple[str, int, int]] = []
                docs: List[DocumentResponse] = []

                for file_path in self._data_dir.glob("*"):
                    if file_path.is_file():
                        stats = file_path.stat()
                        items.append((file_path.name, stats.st_mtime_ns, stats.st_size))

                items.sort(key=lambda x: x[0])
                for name, mtime_ns, size in items:
                    docs.append(
                        DocumentResponse(
                            name=name,
                            size=f"{size / 1024 / 1024:.2f} MB",
                            uploadDate=datetime.datetime.fromtimestamp(mtime_ns / 1e9).strftime(
                                "%Y-%m-%d"
                            ),
                        )
                    )

                return docs, _DocumentsCache._compute_docs_etag(items)

            documents, etag = await run_in_threadpool(_scan_docs)
            self._documents = documents
            self._etag = etag
            self._updated_at = asyncio.get_running_loop().time()
            return self._documents, self._etag


class _CollectionsCache:
    def __init__(self, ttl_seconds: float = 2.0):
        self._ttl_seconds = ttl_seconds
        self._lock = asyncio.Lock()
        self._collections: List["CollectionResponse"] = []
        self._etag: str = 'W/"empty"'
        self._updated_at: float = 0.0
        self._store_id: Optional[int] = None

    def invalidate(self) -> None:
        self._updated_at = 0.0

    def _is_fresh(self) -> bool:
        now = asyncio.get_running_loop().time()
        return (now - self._updated_at) <= self._ttl_seconds

    @staticmethod
    def _compute_etag(collections: List[Dict[str, Any]]) -> str:
        # Use SHA-256 for collection ETags to avoid weak hash usage
        h = hashlib.sha256(json.dumps(collections, sort_keys=True).encode("utf-8"))
        return f'W/"{h.hexdigest()}"'

    async def get(self, vector_store: Any) -> Tuple[List["CollectionResponse"], str]:
        async with self._lock:
            store_id = id(vector_store)
            if self._store_id != store_id:
                self._store_id = store_id
                self._updated_at = 0.0

            if self._is_fresh():
                return self._collections, self._etag

            raw = await run_in_threadpool(vector_store.list_collections)
            etag = _CollectionsCache._compute_etag(raw)
            self._collections = [CollectionResponse(**c) for c in raw]
            self._etag = etag
            self._updated_at = asyncio.get_running_loop().time()
            return self._collections, self._etag


def _ensure_api_caches(app: FastAPI) -> None:
    # NOTE: Some test setups (e.g. httpx ASGITransport) may not run lifespan.
    # Lazily initialize caches so endpoints remain functional.
    if not hasattr(app.state, "documents_cache"):
        app.state.documents_cache = _DocumentsCache(data_dir=Path("data"), ttl_seconds=1.0)
    if not hasattr(app.state, "collections_cache"):
        app.state.collections_cache = _CollectionsCache(ttl_seconds=2.0)
    if not hasattr(app.state, "readiness"):
        app.state.readiness = _ReadinessSnapshot()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    global cubo_app, service_manager

    print(">>> LIFESPAN: Starting", flush=True)
    logger.info("Initializing CUBO application")
    try:
        # Preload Ollama model (fire-and-forget to warm up the LLM)
        try:
            import requests

            model_name = config.get("llm_model")
            if model_name:
                logger.info(f"Preloading Ollama model: {model_name}")
                requests.post(
                    "http://localhost:11434/api/generate",
                    json={"model": model_name, "prompt": "", "keep_alive": "60m"},
                    timeout=0.1,
                )
        except Exception:
            # Ignore errors here (e.g. timeout, connection refused, missing requests)
            # The app should still start even if preloading fails
            pass

        readiness = _ReadinessSnapshot()
        documents_cache = _DocumentsCache(data_dir=Path("data"), ttl_seconds=1.0)
        collections_cache = _CollectionsCache(ttl_seconds=2.0)
        app.state.readiness = readiness
        app.state.documents_cache = documents_cache
        app.state.collections_cache = collections_cache

        async def _refresh_readiness_forever():
            while True:
                components = {
                    "api": True,
                    "app": cubo_app is not None,
                    "service_manager": service_manager is not None,
                    "retriever": (
                        hasattr(cubo_app, "retriever") and cubo_app.retriever is not None
                        if cubo_app
                        else False
                    ),
                    "generator": (
                        hasattr(cubo_app, "generator") and cubo_app.generator is not None
                        if cubo_app
                        else False
                    ),
                    "doc_loader": (
                        hasattr(cubo_app, "doc_loader") and cubo_app.doc_loader is not None
                        if cubo_app
                        else False
                    ),
                    "vector_store": (
                        hasattr(cubo_app, "vector_store") and cubo_app.vector_store is not None
                        if cubo_app
                        else False
                    ),
                }
                try:
                    await readiness.set_components(components)
                except Exception:
                    pass
                await asyncio.sleep(0.5)

        readiness_task = None

        try:
            cubo_app = CuboCore()
            service_manager = ServiceManager()

            # Start readiness snapshot refresher (O(1) for /api/ready)
            readiness_task = asyncio.create_task(_refresh_readiness_forever())

            # Auto-initialize components in background to ensure readiness without blocking startup
            import threading

            def _auto_init():
                logger.info("Auto-initializing RAG components in background...")
                try:
                    cubo_app.initialize_components()
                    
                    # Warm-up inference if RAM > 16GB (prevents cold start delay)
                    try:
                        import psutil
                        mem = psutil.virtual_memory()
                        total_gb = mem.total / (1024**3)
                        
                        if total_gb > 16:
                            logger.info(f"System RAM ({total_gb:.1f}GB) > 16GB. Running warm-up inference...")
                            # Run dummy retrieval to load embedding model and FAISS indexes into hot RAM
                            cubo_app.query_retrieve(query="warmup", top_k=1)
                            logger.info("Warm-up inference complete. System ready.")
                        else:
                            logger.info(f"System RAM ({total_gb:.1f}GB) <= 16GB. Skipping warm-up to save resources.")
                    except Exception as e:
                        logger.warning(f"Warm-up inference failed: {e}")

                finally:
                    # Ensure caches reflect initialization as soon as possible
                    try:
                        documents_cache.invalidate()
                        collections_cache.invalidate()
                    except Exception:
                        pass
                logger.info("Auto-initialization complete.")

            threading.Thread(target=_auto_init, daemon=True).start()

            logger.info("CUBO application initialized successfully")
        except Exception as init_error:
            logger.warning(f"CUBOApp initialization failed: {init_error}")

        try:
            yield
        finally:
            if readiness_task is not None:
                readiness_task.cancel()
    except Exception as e:
        logger.error(f"Lifespan error: {e}", exc_info=True)
        raise
    finally:
        logger.info("Shutting down CUBO application")
        if service_manager:
            service_manager.shutdown()


app = FastAPI(
    title="CUBO RAG API",
    description="API for document ingestion, indexing, and retrieval",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def trace_id_middleware(request: Request, call_next):
    """Add trace_id to all requests and responses."""
    trace_id = request.headers.get("x-trace-id") or generate_trace_id()

    # Store trace_id in request state for easy access in handlers
    request.state.trace_id = trace_id

    try:
        trace_collector.record(
            trace_id,
            "api",
            "request.start",
            {"path": str(request.url.path), "method": request.method},
        )
    except Exception:
        pass

    with trace_context(trace_id):
        logger.info(
            "Incoming request",
            extra={
                "method": request.method,
                "url": str(request.url),
                "client": request.client.host if request.client else None,
                "trace_id": trace_id,
            },
        )

        response = await call_next(request)
        response.headers["x-trace-id"] = trace_id

        logger.info(
            "Response sent", extra={"status_code": response.status_code, "trace_id": trace_id}
        )

        return response


# =========================================================================
# Error Handling
# =========================================================================


class ErrorResponse(BaseModel):
    """Standardized error response."""

    error_code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    trace_id: str


@app.exception_handler(CUBOError)
async def cubo_exception_handler(request: Request, exc: CUBOError):
    """Handle custom CUBO exceptions and map to HTTP status codes."""
    trace_id = getattr(request.state, "trace_id", generate_trace_id())

    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_code = exc.error_code or "INTERNAL_ERROR"

    if isinstance(exc, DocumentNotFoundError):
        status_code = status.HTTP_404_NOT_FOUND
        error_code = "DOC_NOT_FOUND"
    elif isinstance(exc, DocumentAlreadyExistsError):
        status_code = status.HTTP_409_CONFLICT
        error_code = "DOC_EXISTS"
    elif isinstance(exc, (ValidationError, InvalidConfigurationError)):
        status_code = status.HTTP_400_BAD_REQUEST
        error_code = "VALIDATION_ERROR"
    elif isinstance(exc, ModelLoadError):
        status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        error_code = "MODEL_UNAVAILABLE"

    logger.error(
        f"{exc.__class__.__name__}: {exc.message}", extra={"trace_id": trace_id}, exc_info=False
    )

    return JSONResponse(
        status_code=status_code,
        content=ErrorResponse(
            error_code=error_code, message=exc.message, details=exc.details, trace_id=trace_id
        ).model_dump(),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle Pydantic validation errors."""
    trace_id = getattr(request.state, "trace_id", generate_trace_id())
    logger.warning(f"Validation error: {exc}", extra={"trace_id": trace_id})

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            error_code="INVALID_REQUEST",
            message="Request validation failed",
            details={"errors": exc.errors()},
            trace_id=trace_id,
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Catch-all for unhandled exceptions."""
    trace_id = getattr(request.state, "trace_id", generate_trace_id())

    # Log full traceback for unexpected errors
    logger.error(f"Unhandled exception: {exc}", extra={"trace_id": trace_id}, exc_info=True)

    # Prepare details; include stack trace only when verbose mode is enabled
    details = {"original_error": str(exc)}
    if os.environ.get("CUBO_VERBOSE") == "1":
        try:
            import traceback

            details["exception_type"] = exc.__class__.__name__
            details["stack"] = traceback.format_exc()
        except Exception:
            # If formatting the stack fails, don't block the error response
            pass

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error_code="INTERNAL_SERVER_ERROR",
            message="An unexpected error occurred. Please check logs for trace_id.",
            details=details,
            trace_id=trace_id,
        ).model_dump(),
    )


# =========================================================================
# Request/Response Models
# =========================================================================


class QueryRequest(BaseModel):
    """Query request model."""

    query: str = Field(..., min_length=1, max_length=1000, description="User query")
    top_k: int = Field(5, ge=1, le=20, description="Number of results to retrieve")
    use_reranker: bool = Field(True, description="Whether to use reranker")
    # Advanced retrieval parameters
    bm25_weight: float = Field(0.3, ge=0.0, le=1.0, description="Weight for BM25 sparse retrieval")
    dense_weight: float = Field(
        0.7, ge=0.0, le=1.0, description="Weight for dense vector retrieval"
    )
    retrieval_strategy: str = Field(
        "hybrid", description="Retrieval strategy: 'hybrid', 'dense', or 'sparse'"
    )
    # Collection filtering
    collection_id: Optional[str] = Field(None, description="Filter results to specific collection")
    # Streaming
    stream: bool = Field(False, description="Enable streaming response (opt-in)")


class Citation(BaseModel):
    """Citation model for GDPR-compliant source tracking."""

    source_file: str = Field(..., description="Original document filename")
    page: Optional[int] = Field(None, description="Page number if available")
    chunk_id: str = Field(..., description="Unique chunk identifier")
    chunk_index: int = Field(..., description="Position of chunk in source document")
    text_snippet: str = Field(..., description="Snippet of the cited text (max 200 chars)")
    relevance_score: float = Field(..., description="Relevance score 0-1")


class SourceWithCitation(BaseModel):
    """Source document with citation metadata."""

    content: str = Field(..., description="Chunk content (truncated)")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    score: float = Field(..., description="Relevance score")
    citation: Citation = Field(..., description="Formatted citation")


class QueryResponse(BaseModel):
    """Query response model."""

    answer: str
    sources: List[Dict[str, Any]]
    citations: List[Citation] = Field(
        default_factory=list, description="Formatted citations for GDPR compliance"
    )
    trace_id: str
    query_scrubbed: bool


class DeleteDocumentResponse(BaseModel):
    """Response model for document deletion."""

    doc_id: str
    deleted: bool
    chunks_removed: int
    trace_id: str
    message: str
    job_id: Optional[str] = None
    queued: bool = False


class UploadResponse(BaseModel):
    """Upload response model."""

    filename: str
    size: int
    trace_id: str
    message: str


class IngestRequest(BaseModel):
    """Ingest request model."""

    data_path: Optional[str] = None
    fast_pass: bool = Field(True, description="Use fast-pass mode")
    background: bool = Field(
        False, description="Run ingestion in background and return immediately"
    )
    background: bool = Field(False, description="Run ingestion in background")


class IngestResponse(BaseModel):
    """Ingest response model."""

    status: str
    documents_processed: int  # Actually chunks count
    run_id: Optional[str] = None
    trace_id: str
    message: str


class IngestRunStatus(BaseModel):
    """Run-level ingestion status."""

    run_id: str
    status: Optional[str] = None
    source_folder: Optional[str] = None
    chunks_count: Optional[int] = None
    output_parquet: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    file_status_counts: Dict[str, int] = Field(default_factory=dict)


class IngestFilesStatus(BaseModel):
    """Per-file ingestion status."""

    run_id: str
    files: List[Dict[str, Any]]


class BuildIndexRequest(BaseModel):
    """Build index request model."""

    force_rebuild: bool = Field(False, description="Force index rebuild")


class BuildIndexResponse(BaseModel):
    """Build index response model."""

    status: str
    trace_id: str
    message: str


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    components: Dict[str, str]


class DocumentResponse(BaseModel):
    """Document response model."""

    name: str
    size: str
    uploadDate: str


# Collection Models
class CollectionCreate(BaseModel):
    """Request model for creating a collection."""

    name: str = Field(..., min_length=1, max_length=100, description="Collection name")
    color: str = Field("#2563eb", description="Hex color for visual representation")
    emoji: Optional[str] = Field(None, description="Optional emoji to represent the collection")


class CollectionResponse(BaseModel):
    """Response model for a collection."""

    id: str
    name: str
    color: str
    emoji: Optional[str] = None
    created_at: str
    document_count: int


class AddDocumentsToCollectionRequest(BaseModel):
    """Request model for adding documents to a collection."""

    document_ids: List[str] = Field(..., description="List of document IDs to add")


class AddDocumentsResponse(BaseModel):
    """Response model for adding documents."""

    added_count: int
    already_in_collection: int


class LLMModel(BaseModel):
    """Model representing an available LLM."""

    name: str
    size: Optional[int] = None
    digest: Optional[str] = None
    family: Optional[str] = None


class SettingsResponse(BaseModel):
    """Response model for system settings."""

    llm_model: str
    llm_provider: str
    accent: Optional[str] = None
    # Add other settings as needed


class SettingsUpdate(BaseModel):
    """Request model for updating settings."""

    llm_model: Optional[str] = None
    llm_provider: Optional[str] = None
    accent: Optional[str] = None


# API Endpoints
@app.get("/api/llm/models", response_model=List[LLMModel])
async def list_llm_models(request: Request):
    """List available LLM models from Ollama."""

    def _get_models():
        try:
            import ollama

            models_response = ollama.list()
            # Support both object attribute and dict access
            models_data = getattr(models_response, "models", None) or models_response.get(
                "models", []
            )

            results = []
            for m in models_data:
                # Convert Pydantic models to dict if needed
                d = (
                    m.model_dump()
                    if hasattr(m, "model_dump")
                    else (m.dict() if hasattr(m, "dict") else m)
                )
                if not isinstance(d, dict):
                    continue

                results.append(
                    LLMModel(
                        name=d.get("model") or d.get("name"),
                        size=d.get("size"),
                        digest=d.get("digest"),
                        family=d.get("details", {}).get("family"),
                    )
                )
            return results
        except ImportError:
            logger.warning("Ollama python package not installed")
            return []
        except Exception as e:
            logger.error(f"Failed to list Ollama models: {e}")
            raise HTTPException(status_code=503, detail=f"Ollama service unavailable: {str(e)}")

    return await run_in_threadpool(_get_models)


@app.get("/api/settings", response_model=SettingsResponse)
async def get_settings(request: Request):
    """Get current system settings."""
    return SettingsResponse(
        llm_model=config.get("llm.model_name")
        or config.get("selected_llm_model")
        or config.get("llm_model")
        or "llama3",
        llm_provider=config.get("llm.provider", "ollama"),
        accent=config.get("ui.accent", "blue"),
    )


@app.put("/api/settings", response_model=SettingsResponse)
async def update_settings(settings: SettingsUpdate, request: Request):
    """Update system settings."""
    updates = {}
    if settings.llm_model:
        config.set("llm.model_name", settings.llm_model)
        config.set("llm_model", settings.llm_model)
        updates["llm.model_name"] = settings.llm_model

    if settings.llm_provider:
        config.set("llm.provider", settings.llm_provider)
        updates["llm.provider"] = settings.llm_provider

    if settings.accent:
        config.set("ui.accent", settings.accent)
        updates["ui.accent"] = settings.accent

    if updates:
        config.save()
        logger.info(f"Settings updated: {updates}")

    return await get_settings(request)


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    logger.info("Health check requested")

    components = {
        "api": "healthy",
        "app": "not_initialized" if cubo_app is None else "healthy",
        "service_manager": "not_initialized" if service_manager is None else "healthy",
    }

    if cubo_app is not None:
        try:
            retriever_ready = hasattr(cubo_app, "retriever") and cubo_app.retriever is not None
            components["retriever"] = "healthy" if retriever_ready else "not_ready"
        except Exception as e:
            components["retriever"] = f"error: {str(e)}"
            logger.error(f"Retriever health check failed: {e}")

    return HealthResponse(status="healthy", version="1.0.0", components=components)


@app.post("/api/initialize")
async def initialize_components(request: Request):
    """Explicitly initialize heavyweight components."""
    if not cubo_app:
        raise HTTPException(status_code=503, detail="CUBO app not created")

    logger.info("Initializing heavyweight CUBO components on demand")

    # Trigger initialize_components which loads the model and sets up retriever/generator
    success = await run_in_threadpool(cubo_app.initialize_components)
    if not success:
        raise HTTPException(status_code=500, detail="Initialization failed")

    # Nudge caches so subsequent calls reflect latest state.
    try:
        if hasattr(request.app.state, "documents_cache"):
            request.app.state.documents_cache.invalidate()
        if hasattr(request.app.state, "collections_cache"):
            request.app.state.collections_cache.invalidate()
    except Exception:
        pass

    return {"status": "initialized", "trace_id": request.state.trace_id}


@app.get("/api/ready")
async def readiness_check(request: Request):
    """Readiness endpoint."""
    _ensure_api_caches(request.app)
    readiness = getattr(request.app.state, "readiness", None)
    if readiness is None:
        # Fallback (should not happen in normal startup)
        components = {
            "api": True,
            "app": cubo_app is not None,
            "service_manager": service_manager is not None,
            "retriever": bool(cubo_app and getattr(cubo_app, "retriever", None)),
            "generator": bool(cubo_app and getattr(cubo_app, "generator", None)),
            "doc_loader": bool(cubo_app and getattr(cubo_app, "doc_loader", None)),
            "vector_store": bool(cubo_app and getattr(cubo_app, "vector_store", None)),
        }
        return {"components": components, "trace_id": request.state.trace_id}

    # If lifespan isn't running, the snapshot may never get refreshed.
    # Populate it once from in-memory state (still O(1), no I/O).
    try:
        if getattr(readiness, "_updated_at", 0.0) == 0.0:
            await readiness.set_components(
                {
                    "api": True,
                    "app": cubo_app is not None,
                    "service_manager": service_manager is not None,
                    "retriever": (
                        hasattr(cubo_app, "retriever") and cubo_app.retriever is not None
                        if cubo_app
                        else False
                    ),
                    "generator": (
                        hasattr(cubo_app, "generator") and cubo_app.generator is not None
                        if cubo_app
                        else False
                    ),
                    "doc_loader": (
                        hasattr(cubo_app, "doc_loader") and cubo_app.doc_loader is not None
                        if cubo_app
                        else False
                    ),
                    "vector_store": (
                        hasattr(cubo_app, "vector_store") and cubo_app.vector_store is not None
                        if cubo_app
                        else False
                    ),
                }
            )
    except Exception:
        pass

    components = await readiness.get_components()
    return {"components": components, "trace_id": request.state.trace_id}


@app.post("/api/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...), request: Request = None):
    """Upload a document file."""
    logger.info(
        "File upload started",
        extra={"uploaded_filename": file.filename, "content_type": file.content_type},
    )

    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    data_dir = Path("data")
    # Use run_in_threadpool for blocking mkdir
    await run_in_threadpool(data_dir.mkdir, exist_ok=True)

    # Sanitize filename
    safe_filename = Path(file.filename).name
    file_path = data_dir / safe_filename

    # Use a temporary file
    temp_path = file_path.with_suffix(file_path.suffix + ".tmp")

    try:
        size = 0
        async with aiofiles.open(temp_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):  # 1MB chunks
                await f.write(chunk)
                size += len(chunk)

        # Atomic rename (using run_in_threadpool for os.replace)
        await run_in_threadpool(os.replace, temp_path, file_path)

    except Exception as e:
        # Cleanup temp file if it exists
        if await run_in_threadpool(temp_path.exists):
            await run_in_threadpool(os.remove, temp_path)
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

    logger.info(
        "File uploaded successfully",
        extra={
            "uploaded_filename": file.filename,
            "size": size,
            "path": str(file_path),
        },
    )

    # Invalidate document listing cache (new file).
    try:
        if request is not None and hasattr(request.app.state, "documents_cache"):
            request.app.state.documents_cache.invalidate()
    except Exception:
        pass

    return UploadResponse(
        filename=file.filename,
        size=size,
        trace_id=request.state.trace_id,
        message=f"File {file.filename} uploaded successfully",
    )


@app.get("/api/documents", response_model=List[DocumentResponse])
async def list_documents(
    request: Request,
    response: Response,
    skip: int = Query(0, ge=0),
    limit: Optional[int] = Query(None, ge=1),
):
    """List uploaded documents (cached) with optional pagination.

    Supports conditional requests via ETag/If-None-Match.
    """
    _ensure_api_caches(request.app)
    documents_cache = getattr(request.app.state, "documents_cache", None)
    if documents_cache is None:
        return []

    documents, etag = await documents_cache.get()
    if_none_match = request.headers.get("if-none-match")
    if if_none_match and if_none_match == etag:
        return Response(status_code=status.HTTP_304_NOT_MODIFIED, headers={"ETag": etag})

    sliced = documents[skip : (skip + limit) if limit is not None else None]
    response.headers["ETag"] = etag
    return sliced


@app.delete("/api/documents")
async def delete_all_documents(request: Request, force: Optional[bool] = Query(False)):
    """Delete all documents by enqueuing deletion jobs for each document.

    This enqueues a compaction job per document via the vector store's
    enqueue_deletion method (preferred). If the vector store does not
    support enqueue_deletion, falls back to deleting immediately where
    supported.
    """
    if not cubo_app:
        raise HTTPException(status_code=503, detail="CUBO app not initialized")

    logger.info("Bulk document deletion requested", extra={"trace_id": request.state.trace_id, "force": bool(force)})

    # Prefer cubo_app.state.documents_cache when present (tests may attach a DummyCache there),
    # otherwise fall back to request.app.state cache.
    _ensure_api_caches(request.app)
    documents = []

    # First, try cubo_app.state cache if available
    try:
        doc_cache = getattr(cubo_app, "state", None)
        if doc_cache is not None:
            documents_cache2 = getattr(doc_cache, "documents_cache", None)
            if documents_cache2 is not None:
                try:
                    docs, _ = await documents_cache2.get()
                    documents = []
                    for d in docs:
                        if isinstance(d, dict):
                            name = d.get("name")
                        else:
                            name = getattr(d, "name", None) or getattr(d, "doc_id", None) or str(d)
                        if name:
                            documents.append(name)
                    logger.info("Enumerated from cubo_app.state cache", extra={"count": len(documents), "trace_id": request.state.trace_id})
                except Exception as e:
                    logger.warning("cubo_app.state.documents_cache.get failed", extra={"error": str(e), "trace_id": request.state.trace_id})
                    documents = []
    except Exception as e:
        logger.warning("Error checking cubo_app.state for documents_cache", extra={"error": str(e), "trace_id": request.state.trace_id})
        documents = []

    # If not found, try request.app.state cache
    if not documents:
        documents_cache = getattr(request.app.state, "documents_cache", None)
        if documents_cache is not None:
            try:
                docs, _ = await documents_cache.get()
                documents = []
                for d in docs:
                    if isinstance(d, dict):
                        name = d.get("name")
                    else:
                        name = getattr(d, "name", None) or getattr(d, "doc_id", None) or str(d)
                    if name:
                        documents.append(name)
            except Exception as e:
                logger.warning("documents_cache.get failed", extra={"error": str(e), "trace_id": request.state.trace_id})
                documents = []

    # If documents empty, try fallback: query vector store for doc ids
    if not documents:
        try:
            if hasattr(cubo_app, "vector_store") and cubo_app.vector_store:
                # vector_store may expose a method to list documents; use it if available
                getter = getattr(cubo_app.vector_store, "list_documents", None)
                if getter:
                    raw = getter()
                    # Expect an iterable of document dicts or ids
                    if raw and isinstance(raw, list):
                        # Try to normalize
                        if isinstance(raw[0], dict) and "id" in raw[0]:
                            documents = [r["id"] for r in raw]
                        else:
                            documents = [str(r) for r in raw]
        except Exception:
            documents = []

    if not documents:
        # Nothing to delete (or couldn't enumerate)
        logger.info("Bulk delete: no documents enumerated", extra={"trace_id": request.state.trace_id})
        return {"deleted_count": 0, "queued": [], "message": "No documents found or could not enumerate documents"}

    logger.info("Bulk delete: enumerated documents", extra={"count": len(documents), "trace_id": request.state.trace_id})

    queued_jobs = []
    deleted_count = 0
    errors = []

    for doc_id in documents:
        try:
            if hasattr(cubo_app, "vector_store") and cubo_app.vector_store:
                if hasattr(cubo_app.vector_store, "enqueue_deletion"):
                    job_id = cubo_app.vector_store.enqueue_deletion(doc_id, trace_id=request.state.trace_id, force=bool(force))
                    queued_jobs.append({"doc_id": doc_id, "job_id": job_id})
                    deleted_count += 1
                else:
                    # Fallback immediate delete
                    try:
                        cubo_app.vector_store.delete(ids=[doc_id])
                        deleted_count += 1
                    except Exception as e:
                        errors.append({"doc_id": doc_id, "error": str(e)})

            # Attempt to remove the physical file as part of bulk deletion
            try:
                data_path = Path("data") / doc_id
                if await run_in_threadpool(data_path.exists):
                    await run_in_threadpool(os.remove, data_path)
                    logger.info("Removed uploaded file from disk (bulk)", extra={"path": str(data_path), "trace_id": request.state.trace_id})
            except Exception as e:
                logger.warning(f"Failed to remove uploaded file {doc_id} during bulk delete: {e}")

            else:
                errors.append({"doc_id": doc_id, "error": "vector_store not available"})
        except Exception as e:
            errors.append({"doc_id": doc_id, "error": str(e)})

    # Invalidate documents cache
    try:
        if hasattr(request.app.state, "documents_cache"):
            request.app.state.documents_cache.invalidate()
    except Exception:
        pass

    # Record audit
    trace_collector.record(
        request.state.trace_id,
        "api",
        "document.delete.bulk",
        {"count": deleted_count, "queued_jobs": len(queued_jobs)},
    )

    return {"deleted_count": deleted_count, "queued": queued_jobs, "errors": errors}



# =========================================================================
# Collection Endpoints
# =========================================================================


@app.get("/api/collections", response_model=List[CollectionResponse])
async def list_collections(request: Request, response: Response):
    """List all document collections with their document counts."""
    if not cubo_app or not cubo_app.vector_store:
        return []

    _ensure_api_caches(request.app)
    collections_cache = getattr(request.app.state, "collections_cache", None)
    if collections_cache is None:
        raw = await run_in_threadpool(cubo_app.vector_store.list_collections)
        return [CollectionResponse(**c) for c in raw]

    collections, etag = await collections_cache.get(cubo_app.vector_store)
    if_none_match = request.headers.get("if-none-match")
    if if_none_match and if_none_match == etag:
        return Response(status_code=status.HTTP_304_NOT_MODIFIED, headers={"ETag": etag})

    response.headers["ETag"] = etag
    return collections


@app.post("/api/collections", response_model=CollectionResponse)
async def create_collection(collection_data: CollectionCreate, request: Request):
    """Create a new document collection."""
    if not cubo_app or not cubo_app.vector_store:
        raise HTTPException(status_code=503, detail="CUBO app not initialized")

    try:
        collection = cubo_app.vector_store.create_collection(
            name=collection_data.name,
            color=collection_data.color,
            emoji=collection_data.emoji if hasattr(collection_data, 'emoji') else None,
        )
    except ValueError as e:
        # Map ValueError from store to HTTP 409 for duplicate collections
        raise HTTPException(status_code=409, detail=str(e))
    logger.info(f"Created collection: {collection['name']}")
    try:
        if hasattr(request.app.state, "collections_cache"):
            request.app.state.collections_cache.invalidate()
    except Exception:
        pass
    return CollectionResponse(**collection)


@app.get("/api/collections/{collection_id}", response_model=CollectionResponse)
async def get_collection(collection_id: str, request: Request):
    """Get a specific collection by ID."""
    if not cubo_app or not cubo_app.vector_store:
        raise HTTPException(status_code=503, detail="CUBO app not initialized")

    collection = cubo_app.vector_store.get_collection(collection_id)
    if not collection:
        raise HTTPException(status_code=404, detail="Collection not found")

    return CollectionResponse(**collection)


@app.delete("/api/collections/{collection_id}")
async def delete_collection(collection_id: str, request: Request):
    """Delete a collection (documents remain in store, just unlinked)."""
    if not cubo_app or not cubo_app.vector_store:
        raise HTTPException(status_code=503, detail="CUBO app not initialized")

    deleted = cubo_app.vector_store.delete_collection(collection_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Collection not found")

    logger.info(f"Deleted collection: {collection_id}")
    try:
        if hasattr(request.app.state, "collections_cache"):
            request.app.state.collections_cache.invalidate()
    except Exception:
        pass
    return {"status": "deleted", "collection_id": collection_id}


@app.post("/api/collections/{collection_id}/documents", response_model=AddDocumentsResponse)
async def add_documents_to_collection(
    collection_id: str, request_data: AddDocumentsToCollectionRequest, request: Request
):
    """Add documents to a collection by their IDs."""
    if not cubo_app or not cubo_app.vector_store:
        raise HTTPException(status_code=503, detail="CUBO app not initialized")

    # Check collection exists
    collection = cubo_app.vector_store.get_collection(collection_id)
    if not collection:
        raise HTTPException(status_code=404, detail="Collection not found")

    result = cubo_app.vector_store.add_documents_to_collection(
        collection_id, request_data.document_ids
    )
    logger.info(f"Added {result['added_count']} documents to collection {collection_id}")
    return AddDocumentsResponse(**result)


@app.delete("/api/collections/{collection_id}/documents")
async def remove_documents_from_collection(
    collection_id: str, request_data: AddDocumentsToCollectionRequest, request: Request
):
    """Remove documents from a collection."""
    if not cubo_app or not cubo_app.vector_store:
        raise HTTPException(status_code=503, detail="CUBO app not initialized")

    # Check collection exists
    collection = cubo_app.vector_store.get_collection(collection_id)
    if not collection:
        raise HTTPException(status_code=404, detail="Collection not found")

    removed_count = cubo_app.vector_store.remove_documents_from_collection(
        collection_id, request_data.document_ids
    )
    logger.info(f"Removed {removed_count} documents from collection {collection_id}")
    return {"removed_count": removed_count}


@app.get("/api/collections/{collection_id}/documents")
async def get_collection_documents(collection_id: str, request: Request):
    """Get all document IDs in a collection."""
    if not cubo_app or not cubo_app.vector_store:
        raise HTTPException(status_code=503, detail="CUBO app not initialized")

    # Check collection exists
    collection = cubo_app.vector_store.get_collection(collection_id)
    if not collection:
        raise HTTPException(status_code=404, detail="Collection not found")

    doc_ids = cubo_app.vector_store.get_collection_documents(collection_id)
    return {"collection_id": collection_id, "document_ids": doc_ids, "count": len(doc_ids)}


# =========================================================================
# End Collection Endpoints
# =========================================================================


@app.post("/api/ingest", response_model=IngestResponse)
async def ingest_documents(
    request_data: IngestRequest, background_tasks: BackgroundTasks, request: Request
):
    """Ingest documents from data directory."""
    if not cubo_app:
        raise HTTPException(status_code=503, detail="CUBO app not initialized")

    logger.info("Document ingestion started", extra={"fast_pass": request_data.fast_pass})

    data_path = Path(request_data.data_path) if request_data.data_path else Path("data")

    if not data_path.exists():
        raise HTTPException(status_code=404, detail=f"Data path not found: {data_path}")

    # Load documents
    documents = cubo_app.doc_loader.load_documents_from_folder(str(data_path))

    if not documents:
        logger.warning("No documents found to ingest")
        return IngestResponse(
            status="completed",
            documents_processed=0,
            trace_id=request.state.trace_id,
            message="No documents found",
        )

    logger.info(f"Loaded {len(documents)} documents for ingestion")

    # Use DeepIngestor for proper document processing and run in threadpool
    from cubo.ingestion.deep_ingestor import DeepIngestor

    # If background mode is requested, start task and return immediately
    if request_data.background:
        import uuid

        run_id = f"deep_bg_{uuid.uuid4().hex[:8]}"

        # Record run immediately as pending
        manager = get_metadata_manager()
        manager.record_ingestion_run(run_id, str(data_path), 0, None, status="pending")

        def run_ingestor_bg(rid: str):
            try:
                ingestor = DeepIngestor(
                    input_folder=str(data_path),
                    output_dir=config.get("ingestion.deep.output_dir", "./data/deep"),
                    run_id=rid,
                )
                ingestor.ingest()
            except Exception as e:
                logger.error(f"Background ingestion failed: {e}")

        background_tasks.add_task(run_ingestor_bg, run_id)

        return IngestResponse(
            status="started",
            documents_processed=0,
            run_id=run_id,
            trace_id=request.state.trace_id,
            message="Ingestion started in background",
        )

    def run_ingestor():
        ingestor = DeepIngestor(
            input_folder=str(data_path),
            output_dir=config.get("ingestion.deep.output_dir", "./data/deep"),
        )
        return ingestor.ingest()

    # Acquire lock to prevent OOM from concurrent heavy tasks
    async with compute_lock:
        result = await run_in_threadpool(run_ingestor)

    if not result:
        logger.warning("Deep ingestion produced no results")
        return IngestResponse(
            status="completed",
            documents_processed=0,
            trace_id=request.state.trace_id,
            message="No chunks generated from documents",
        )

    chunks_count = result.get("chunks_count", 0)
    parquet_path = result.get("chunks_parquet", "")
    run_id = result.get("run_id")

    logger.info(
        "Document ingestion completed",
        extra={"chunks_processed": chunks_count, "parquet_path": parquet_path},
    )
    trace_collector.record(
        request.state.trace_id,
        "api",
        "ingest.completed",
        {"chunks_processed": chunks_count, "parquet_path": parquet_path},
    )

    return IngestResponse(
        status="completed",
        documents_processed=chunks_count,
        run_id=run_id,
        trace_id=request.state.trace_id,
        message="Ingestion completed successfully",
    )


@app.get("/api/ingest/{run_id}", response_model=IngestRunStatus)
async def get_ingest_run_status(run_id: str, request: Request):
    manager = get_metadata_manager()
    run = manager.get_ingestion_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Ingestion run not found")

    file_counts = manager.get_file_status_counts(run_id)
    return IngestRunStatus(
        run_id=run.get("id", run_id),
        status=run.get("status"),
        source_folder=run.get("source_folder"),
        chunks_count=run.get("chunks_count"),
        output_parquet=run.get("output_parquet"),
        started_at=run.get("started_at"),
        finished_at=run.get("finished_at"),
        file_status_counts=file_counts,
    )


@app.get("/api/ingest/{run_id}/files", response_model=IngestFilesStatus)
async def get_ingest_files_status(run_id: str, request: Request):
    manager = get_metadata_manager()
    files = manager.list_files_for_run(run_id)
    if not files and not manager.get_ingestion_run(run_id):
        raise HTTPException(status_code=404, detail="Ingestion run not found")
    return IngestFilesStatus(run_id=run_id, files=files)


@app.post("/api/build-index", response_model=BuildIndexResponse)
async def build_index(
    request_data: BuildIndexRequest, background_tasks: BackgroundTasks, request: Request
):
    """Build or rebuild search indexes."""
    if not cubo_app:
        raise HTTPException(status_code=503, detail="CUBO app not initialized")

    logger.info("Index build started", extra={"force_rebuild": request_data.force_rebuild})

    # Build index using CUBOApp's build_index method with explicit data folder
    # Acquire lock to prevent OOM
    async with compute_lock:
        doc_count = await run_in_threadpool(cubo_app.build_index, "data")

    logger.info(f"Index build completed with {doc_count} documents")
    trace_collector.record(
        request.state.trace_id, "api", "build_index.completed", {"documents_indexed": doc_count}
    )

    return BuildIndexResponse(
        status="completed",
        trace_id=request.state.trace_id,
        message=f"Index built successfully with {doc_count} documents",
    )


async def _query_stream_generator(request_data: QueryRequest, request: Request):
    """Generate streaming NDJSON events for query response."""
    try:
        # Scrub query for logging
        scrubbed_query = security_manager.scrub(request_data.query)

        logger.info(
            "Streaming query generator started",
            extra={"query": scrubbed_query, "query_scrubbed": scrubbed_query != request_data.query},
        )

        # Check retriever initialization
        if not hasattr(cubo_app, "retriever") or not cubo_app.retriever:
            logger.error("Retriever not initialized")
            yield (
                json.dumps({"type": "error", "message": "Retriever not initialized"}) + "\n"
            ).encode()
            return

        # Check collection count
        try:
            collection_count = cubo_app.retriever.collection.count()
        except Exception:
            collection_count = 0

        if collection_count == 0:
            logger.error("Vector index empty")
            yield (json.dumps({"type": "error", "message": "Vector index empty"}) + "\n").encode()
            return

        # Build where filter for collection if specified
        where_filter = None
        if request_data.collection_id and cubo_app.vector_store:
            filenames = cubo_app.vector_store.get_document_filenames_in_collection(
                request_data.collection_id
            )
            if filenames:
                where_filter = {"filename": {"$in": filenames}}
            else:
                logger.error(f"Collection {request_data.collection_id} has no documents")
                yield (
                    json.dumps({"type": "error", "message": "Collection has no documents"}) + "\n"
                ).encode()
                return

        # Retrieve documents
        retrieve_kwargs = {
            "query": request_data.query,
            "top_k": request_data.top_k,
            "trace_id": request.state.trace_id,
        }
        if where_filter:
            retrieve_kwargs["where"] = where_filter

        logger.info("Starting document retrieval")
        async with compute_lock:
            retrieved_docs = await run_in_threadpool(lambda: cubo_app.query_retrieve(**retrieve_kwargs))
            logger.info(f"Retrieved {len(retrieved_docs)} documents")

            # Emit source events
            for idx, doc in enumerate(retrieved_docs):
                metadata = doc.get("metadata", {})
                content = doc.get("document", doc.get("content", ""))
                score = doc.get("similarity", doc.get("score", 0.0))

                source_event = {
                    "type": "source",
                    "index": idx,
                    "content": content[:500],
                    "metadata": metadata,
                    "score": float(score) if score else 0.0,
                    "trace_id": request.state.trace_id,
                }
                logger.debug(f"Emitting source event {idx}")
                yield (json.dumps(source_event) + "\n").encode()

            # Build context and stream generation
            context = "\n\n".join(
                [doc.get("document", doc.get("content", "")) for doc in retrieved_docs]
            )

            logger.info("Starting LLM streaming generation")
            def _stream_generator():
                return cubo_app.generate_response_stream(
                    query=request_data.query,
                    context=context,
                    trace_id=request.state.trace_id,
                )

            # Stream tokens from generator
            event_count = 0
            for event in await run_in_threadpool(_stream_generator):
                event_count += 1
                event_type = event.get('type')
                logger.debug(f"Generator event {event_count}: {event_type}")
                if event_type == 'done':
                    logger.info(f"Done event from generator: type={event_type}, has_answer={'answer' in event}, answer_length={len(event.get('answer', ''))}")
                    logger.info(f"Full done event: {event}")
                yield (json.dumps(event) + "\n").encode()
            
            logger.info(f"Streaming completed with {event_count} events")
            
            # Safety: if no done event was sent, send one
            if event_count == 0:
                logger.error("No events generated from LLM")
                yield (json.dumps({
                    "type": "done",
                    "answer": "I apologize, but I was unable to generate a response. Please try again.",
                    "trace_id": request.state.trace_id
                }) + "\n").encode()

    except Exception as e:
        logger.error(f"Streaming query error: {e}", exc_info=True)
        yield (
            json.dumps({"type": "error", "message": f"Server error: {str(e)}", "trace_id": request.state.trace_id})
            + "\n"
        ).encode()
        yield (
            json.dumps({"type": "error", "message": str(e), "trace_id": request.state.trace_id})
            + "\n"
        ).encode()


@app.post("/api/query", response_model=QueryResponse)
async def query(request_data: QueryRequest, request: Request):
    """Query the RAG system."""
    if not cubo_app:
        raise HTTPException(status_code=503, detail="CUBO app not initialized")

    # Check if streaming is requested and enabled
    streaming_enabled = config.get("llm.enable_streaming", False)
    logger.info(f"Query endpoint: stream={request_data.stream}, streaming_enabled={streaming_enabled}")
    
    if request_data.stream and streaming_enabled:
        logger.info("Returning streaming response")
        # Return streaming response
        return StreamingResponse(
            _query_stream_generator(request_data, request),
            media_type="application/x-ndjson",
            headers={"X-Trace-ID": request.state.trace_id},
        )

    logger.info("Returning non-streaming response")
    # Non-streaming path (existing logic)
    # Scrub query for logging
    scrubbed_query = security_manager.scrub(request_data.query)
    query_scrubbed = scrubbed_query != request_data.query

    logger.info(
        "Query received",
        extra={
            "query": scrubbed_query,
            "top_k": request_data.top_k,
            "use_reranker": request_data.use_reranker,
            "scrubbed": query_scrubbed,
        },
    )
    try:
        trace_collector.record(
            request.state.trace_id,
            "api",
            "query.received",
            {
                "query": scrubbed_query,
                "top_k": request_data.top_k,
                "use_reranker": request_data.use_reranker,
            },
        )
    except Exception:
        pass

    # Check if retriever is initialized
    if not hasattr(cubo_app, "retriever") or not cubo_app.retriever:
        raise HTTPException(
            status_code=503,
            detail="Retriever not initialized. Please run ingestion and index building first.",
        )

    # Ensure vector store has documents (FAISS index built)
    try:
        collection_count = cubo_app.retriever.collection.count()
    except Exception as e:
        logger.warning(f"Could not check collection count: {e}")
        collection_count = 0

    if collection_count == 0:
        raise HTTPException(
            status_code=503,
            detail="Vector index empty. Please run /api/build-index to initialize the index before querying.",
        )

    # Build where filter for collection if specified
    where_filter = None
    if request_data.collection_id and cubo_app.vector_store:
        filenames = cubo_app.vector_store.get_document_filenames_in_collection(
            request_data.collection_id
        )
        if filenames:
            where_filter = {"filename": {"$in": filenames}}
            logger.info(
                f"Filtering query to collection {request_data.collection_id} ({len(filenames)} files)"
            )
        else:
            logger.warning(f"Collection {request_data.collection_id} has no documents")
            raise HTTPException(
                status_code=400,
                detail=f"Collection '{request_data.collection_id}' has no documents. Please add documents to this collection before querying.",
            )

    # Retrieve documents using the correct method
    retrieve_kwargs = {
        "query": request_data.query,
        "top_k": request_data.top_k,
        "trace_id": request.state.trace_id,
    }
    if where_filter:
        retrieve_kwargs["where"] = where_filter

    # Acquire lock for heavy retrieval and generation
    async with compute_lock:
        # Use CUBOApp query_retrieve wrapper which guards access to the retriever
        retrieved_docs = await run_in_threadpool(lambda: cubo_app.query_retrieve(**retrieve_kwargs))

        logger.info(f"Retrieved {len(retrieved_docs)} documents")

        # Generate answer using the correct generator method
        if hasattr(cubo_app, "generator") and cubo_app.generator:
            # Build context from retrieved documents
            context = "\n\n".join(
                [doc.get("document", doc.get("content", "")) for doc in retrieved_docs]
            )
            answer = await run_in_threadpool(
                lambda: cubo_app.generate_response_safe(
                    query=request_data.query, context=context, trace_id=request.state.trace_id
                )
            )
        else:
            # Fallback: return retrieved documents as context
            answer = "Retrieved documents (no generator available):\n\n"
            answer += "\n\n".join(
                [doc.get("document", doc.get("content", ""))[:200] for doc in retrieved_docs[:3]]
            )

    # Format sources - handle both 'document' and 'content' keys
    sources = []
    citations = []
    for idx, doc in enumerate(retrieved_docs):
        metadata = doc.get("metadata", {})
        content = doc.get("document", doc.get("content", ""))
        score = doc.get("similarity", doc.get("score", 0.0))

        # Build source entry
        sources.append(
            {
                "content": content[:500],
                "metadata": metadata,
                "score": score,
            }
        )

        # Build citation for GDPR compliance
        chunk_id = metadata.get("chunk_id", metadata.get("id", f"chunk_{idx}"))
        citations.append(
            Citation(
                source_file=metadata.get("filename", metadata.get("source", "unknown")),
                page=metadata.get("page", metadata.get("page_number")),
                chunk_id=str(chunk_id),
                chunk_index=metadata.get("chunk_index", idx),
                text_snippet=content[:200] if content else "",
                relevance_score=round(float(score), 4) if score else 0.0,
            )
        )

    logger.info(
        "Query processed successfully",
        extra={"query": scrubbed_query, "sources_count": len(sources)},
    )
    try:
        trace_collector.record(
            request.state.trace_id, "api", "query.completed", {"sources_count": len(sources)}
        )
    except Exception:
        pass

    return QueryResponse(
        answer=answer,
        sources=sources,
        citations=citations,
        trace_id=request.state.trace_id,
        query_scrubbed=query_scrubbed,
    )


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "CUBO RAG API", "version": "1.0.0", "docs": "/docs"}


@app.get("/api/traces/{trace_id}")
async def get_trace(trace_id: str):
    """Return recorded trace events for a given trace_id."""
    events = trace_collector.get_trace(trace_id)
    if events is None:
        raise HTTPException(status_code=404, detail="Trace not found")
    return {"trace_id": trace_id, "events": events}


@app.delete("/api/documents/{doc_id}", response_model=DeleteDocumentResponse)
async def delete_document(doc_id: str, request: Request):
    """Delete a document and its chunks from the index."""
    if not cubo_app:
        raise HTTPException(status_code=503, detail="CUBO app not initialized")

    logger.info(
        "Document deletion requested", extra={"doc_id": doc_id, "trace_id": request.state.trace_id}
    )

    # Record deletion request in trace for GDPR audit
    trace_collector.record(
        request.state.trace_id, "api", "document.delete.requested", {"doc_id": doc_id}
    )

    chunks_removed = 0
    deleted = False
    job_id = None

    force = bool(request.query_params.get("force", "false").lower() in ("1", "true", "yes"))

    # Enqueue deletion job in vector store (preferred)
    if hasattr(cubo_app, "vector_store") and cubo_app.vector_store:
        try:
            # Use enqueue_deletion which removes DB rows and schedules compaction
            if hasattr(cubo_app.vector_store, "enqueue_deletion"):
                try:
                    job_id = cubo_app.vector_store.enqueue_deletion(
                        doc_id, trace_id=request.state.trace_id, force=force
                    )
                    deleted = True
                    chunks_removed += 1
                except Exception as e:
                    # If enqueue fails, attempt to remove the physical file as a fallback
                    logger.warning(f"Vector store enqueue_deletion failed: {e}; attempting best-effort file removal", extra={"trace_id": request.state.trace_id})
                    data_path = Path("data") / doc_id
                    try:
                        if await run_in_threadpool(data_path.exists):
                            await run_in_threadpool(os.remove, data_path)
                            logger.info("Removed uploaded file from disk as fallback after enqueue failure", extra={"path": str(data_path), "trace_id": request.state.trace_id})
                            deleted = True
                        else:
                            # No file to remove and enqueue failed -> not found
                            deleted = False
                    except Exception as e2:
                        logger.warning(f"Failed to remove uploaded file during enqueue failure fallback: {e2}", extra={"trace_id": request.state.trace_id})
                        # Keep deleted as False in this case
            else:
                # Fallback to immediate delete (legacy behavior)
                cubo_app.vector_store.delete(ids=[doc_id])
                deleted = True
                chunks_removed += 1
        except Exception as e:
            logger.warning(f"Vector store delete/enqueue failed: {e}")

    # Optionally notify retriever memory caches to remove doc quickly
    if hasattr(cubo_app, "retriever") and cubo_app.retriever:
        try:
            if hasattr(cubo_app.retriever, "remove_document"):
                try:
                    result = cubo_app.retriever.remove_document(doc_id)
                    if result:
                        deleted = True
                except Exception:
                    # Best-effort: ignore retriever removal errors
                    pass
        except Exception as e:
            logger.warning(f"Retriever remove_document failed: {e}")

    # Log deletion for GDPR audit trail
    trace_collector.record(
        request.state.trace_id,
        "api",
        "document.delete.completed",
        {"doc_id": doc_id, "deleted": deleted, "chunks_removed": chunks_removed},
    )

    # Write to audit log
    audit_entry = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "action": "document_delete",
        "trace_id": request.state.trace_id,
        "doc_id": doc_id,
        "success": deleted,
        "chunks_removed": chunks_removed,
    }
    logger.info("GDPR Audit: Document deletion", extra=audit_entry)

    if not deleted:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")

    # Attempt to remove the physical file from the data directory
    try:
        data_path = Path("data") / doc_id
        if await run_in_threadpool(data_path.exists):
            await run_in_threadpool(os.remove, data_path)
            logger.info("Removed uploaded file from disk", extra={"path": str(data_path), "trace_id": request.state.trace_id})
    except Exception as e:
        logger.warning(f"Failed to remove uploaded file {doc_id}: {e}")

    # Invalidate documents cache so listings refresh promptly
    try:
        if request is not None and hasattr(request.app.state, "documents_cache"):
            request.app.state.documents_cache.invalidate()
    except Exception:
        pass

    return DeleteDocumentResponse(
        doc_id=doc_id,
        deleted=deleted,
        chunks_removed=chunks_removed,
        trace_id=request.state.trace_id,
        message=(
            f"Document {doc_id} deletion enqueued (job: {job_id})"
            if job_id
            else f"Document {doc_id} deleted"
        ),
        job_id=job_id,
        queued=bool(job_id),
    )


@app.get("/api/delete-status/{job_id}")
async def get_delete_status(job_id: str):
    """Get status for a deletion job."""
    if not cubo_app or not hasattr(cubo_app, "vector_store"):
        raise HTTPException(status_code=503, detail="CUBO app not initialized")

    if not hasattr(cubo_app.vector_store, "get_deletion_status"):
        raise HTTPException(status_code=404, detail="Deletion jobs not supported by current store")

    status = cubo_app.vector_store.get_deletion_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return status


@app.get("/api/export-audit")
async def export_audit(
    request: Request,
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    format: str = Query("csv", description="Export format: csv or json"),
):
    """Export GDPR audit log as CSV or JSON."""
    logger.info(
        "GDPR audit export requested",
        extra={
            "start_date": start_date,
            "end_date": end_date,
            "format": format,
            "trace_id": request.state.trace_id,
        },
    )

    # Parse dates if provided
    start_dt = None
    end_dt = None
    if start_date:
        try:
            start_dt = datetime.datetime.strptime(start_date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid start_date format. Use YYYY-MM-DD")
    if end_date:
        try:
            end_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d")
            end_dt = end_dt.replace(hour=23, minute=59, second=59)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid end_date format. Use YYYY-MM-DD")

    # Read JSONL log files
    log_dir = Path(config.get("log_dir", "./logs"))

    # List files in threadpool
    jsonl_files = await run_in_threadpool(lambda: sorted(list(log_dir.glob("*.jsonl*"))))

    async def audit_generator():
        # Yield header for CSV
        if format.lower() != "json":
            output = io.StringIO()
            writer = csv.DictWriter(
                output,
                fieldnames=["timestamp", "trace_id", "query_hash", "level", "component", "action"],
            )
            writer.writeheader()
            yield output.getvalue()

        if format.lower() == "json":
            yield '{"audit_entries": ['

        first_entry = True

        for log_file in jsonl_files:
            try:
                async with aiofiles.open(log_file, "r", encoding="utf-8") as f:
                    async for line in f:
                        line = line.strip()
                        if not line:
                            continue

                        try:
                            entry = json.loads(line)

                            # Extract timestamp
                            ts_str = entry.get("asctime", "")
                            if not ts_str:
                                continue

                            try:
                                ts = datetime.datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S,%f")
                            except ValueError:
                                continue

                            # Apply date filters
                            if start_dt and ts < start_dt:
                                continue
                            if end_dt and ts > end_dt:
                                continue

                            # Extract relevant info for audit
                            trace = entry.get("trace_id", "")
                            message = entry.get("message", "")

                            # Hash query for privacy (GDPR)
                            query_text = ""

                            # Try to extract query from structured message
                            if isinstance(message, str) and "query" in message.lower():
                                # Hash any query text found
                                query_hash = hashlib.sha256(message.encode()).hexdigest()[:16]
                                query_text = f"[hashed:{query_hash}]"

                            audit_entry = {
                                "timestamp": ts.isoformat(),
                                "trace_id": trace,
                                "query_hash": query_text,
                                "level": entry.get("levelname", "INFO"),
                                "component": entry.get("name", "unknown"),
                                "action": (
                                    message[:200]
                                    if isinstance(message, str)
                                    else str(message)[:200]
                                ),
                            }

                            if format.lower() == "json":
                                if not first_entry:
                                    yield ", "
                                yield json.dumps(audit_entry)
                                first_entry = False
                            else:
                                output = io.StringIO()
                                writer = csv.DictWriter(
                                    output,
                                    fieldnames=[
                                        "timestamp",
                                        "trace_id",
                                        "query_hash",
                                        "level",
                                        "component",
                                        "action",
                                    ],
                                )
                                writer.writerow(audit_entry)
                                yield output.getvalue()

                        except json.JSONDecodeError:
                            continue

            except Exception as e:
                logger.warning(f"Failed to read log file {log_file}: {e}")

        if format.lower() == "json":
            yield '], "count": null}'

    return StreamingResponse(
        audit_generator(),
        media_type="application/json" if format.lower() == "json" else "text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=cubo_audit_{datetime.date.today().isoformat()}.{'json' if format.lower() == 'json' else 'csv'}"
        },
    )
