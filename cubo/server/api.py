"""FastAPI server for CUBO RAG system."""

import csv
import datetime
import hashlib
import io
import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from cubo.config import config
from cubo.core import CuboCore
from cubo.security.security import security_manager
from cubo.services.service_manager import ServiceManager
from cubo.utils.logger import logger
from cubo.utils.logging_context import generate_trace_id, trace_context
from cubo.utils.trace_collector import trace_collector

# Global app instance - uses CuboCore (no CLI side effects)
cubo_app: Optional[CuboCore] = None
service_manager: Optional[ServiceManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle.

    Initialize CUBO application and service manager on startup,
    and ensure proper shutdown on application exit.

    Args:
        app: FastAPI application instance.

    Yields:
        None: Control is yielded to the application during runtime.

    Raises:
        Exception: If critical initialization or shutdown errors occur.
    """
    global cubo_app, service_manager

    print(">>> LIFESPAN: Starting (MINIMAL TEST)", flush=True)
    logger.info("Initializing CUBO application")
    try:
        # Initialize CUBO application (using CuboCore - no CLI side effects)
        try:
            cubo_app = CuboCore()
            service_manager = ServiceManager()
            logger.info("CUBO application initialized successfully")
        except Exception as init_error:
            logger.warning(f"CUBOApp initialization failed: {init_error}")

        print(">>> LIFESPAN: About to yield (no init)", flush=True)
        yield
        print(">>> LIFESPAN: After yield (shutting down)", flush=True)
    except Exception as e:
        print(f">>> LIFESPAN: Exception caught: {e}", flush=True)
        logger.error(f"Lifespan error: {e}", exc_info=True)
        raise
    finally:
        print(">>> LIFESPAN: In finally block", flush=True)
        logger.info("Shutting down CUBO application")
        if service_manager:
            print(">>> LIFESPAN: Calling service_manager.shutdown()", flush=True)
            service_manager.shutdown()
            print(">>> LIFESPAN: shutdown() completed", flush=True)


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
    """Add trace_id to all requests and responses.

    This middleware ensures every request has a unique trace ID for logging
    and debugging purposes. The trace ID can be provided in the request header
    or will be auto-generated.

    Args:
        request: Incoming HTTP request.
        call_next: Next middleware in the chain.

    Returns:
        Response with trace_id header added.
    """
    trace_id = request.headers.get("x-trace-id") or generate_trace_id()
    # Record a generic request start event; endpoint-specific events are recorded by handlers
    try:
        trace_collector.record(
            trace_id,
            "api",
            "request.start",
            {"path": str(request.url.path), "method": request.method},
        )
    except Exception:
        pass

    # Set trace context for this request
    with trace_context(trace_id):
        # Log incoming request
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

        # Log response
        logger.info(
            "Response sent", extra={"status_code": response.status_code, "trace_id": trace_id}
        )

        return response


# Request/Response Models
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
    citations: List[Citation] = Field(default_factory=list, description="Formatted citations for GDPR compliance")
    trace_id: str
    query_scrubbed: bool


class DeleteDocumentResponse(BaseModel):
    """Response model for document deletion."""

    doc_id: str
    deleted: bool
    chunks_removed: int
    trace_id: str
    message: str


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


class IngestResponse(BaseModel):
    """Ingest response model."""

    status: str
    documents_processed: int  # Actually chunks count
    trace_id: str
    message: str


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


class CollectionResponse(BaseModel):
    """Response model for a collection."""

    id: str
    name: str
    color: str
    created_at: str
    document_count: int


class AddDocumentsToCollectionRequest(BaseModel):
    """Request model for adding documents to a collection."""

    document_ids: List[str] = Field(..., description="List of document IDs to add")


class AddDocumentsResponse(BaseModel):
    """Response model for adding documents."""

    added_count: int
    already_in_collection: int


# API Endpoints
@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint.

    Returns the current health status of the API and its components,
    including the CUBO app, service manager, and retriever.

    Returns:
        HealthResponse: Health status of all system components.

    Example:
        >>> response = await health_check()
        >>> print(response.status)  # "healthy"
    """
    trace_id = generate_trace_id()

    with trace_context(trace_id):
        logger.info("Health check requested")

        components = {
            "api": "healthy",
            "app": "not_initialized" if cubo_app is None else "healthy",
            "service_manager": "not_initialized" if service_manager is None else "healthy",
        }

        # Check if retriever is ready (only if app is initialized)
        if cubo_app is not None:
            try:
                retriever_ready = hasattr(cubo_app, "retriever") and cubo_app.retriever is not None
                components["retriever"] = "healthy" if retriever_ready else "not_ready"
            except Exception as e:
                components["retriever"] = f"error: {str(e)}"
                logger.error(f"Retriever health check failed: {e}")

        # API is healthy even if components aren't initialized yet (lazy loading)
        overall_status = "healthy"

        return HealthResponse(status=overall_status, version="1.0.0", components=components)


@app.post("/api/initialize")
async def initialize_components(request: Request = None):
    """Explicitly initialize heavyweight components (model, retriever, generator).

    This endpoint allows programmatic control over when the model and
    retrieval components are loaded — useful for offline environments where
    heavy initialization should only happen on demand.
    """
    trace_id = request.headers.get("x-trace-id") or generate_trace_id()

    with trace_context(trace_id):
        if not cubo_app:
            raise HTTPException(status_code=503, detail="CUBO app not created")

        logger.info("Initializing heavyweight CUBO components on demand")

        try:
            # Trigger initialize_components which loads the model and sets up retriever/generator
            success = cubo_app.initialize_components()
            if not success:
                raise HTTPException(status_code=500, detail="Initialization failed")

            return {"status": "initialized", "trace_id": trace_id}
        except Exception as e:
            logger.error(f"Initialization endpoint failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Initialization failed: {str(e)}")


@app.get("/api/ready")
async def readiness_check():
    """Readiness endpoint returning detailed component readiness.
    This complements /api/health by returning boolean readiness for each subsystem.
    """
    trace_id = generate_trace_id()
    with trace_context(trace_id):
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
            "vector_store": False,
        }

        try:
            if cubo_app and hasattr(cubo_app, "retriever") and cubo_app.retriever:
                components["vector_store"] = (
                    getattr(cubo_app.retriever, "collection", None) is not None
                )
        except Exception:
            components["vector_store"] = False

        return {"components": components, "trace_id": trace_id}


@app.post("/api/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...), request: Request = None):
    """Upload a document file.

    Accepts a file upload and saves it to the data directory for
    subsequent ingestion and indexing.

    Args:
        file: File uploaded via multipart/form-data.
        request: FastAPI request object (auto-injected).

    Returns:
        UploadResponse: Upload confirmation with file metadata.

    Raises:
        HTTPException: 400 if no filename provided, 500 if upload fails.
    """
    trace_id = request.headers.get("x-trace-id") or generate_trace_id()

    with trace_context(trace_id):
        logger.info(
            "File upload started",
            extra={"uploaded_filename": file.filename, "content_type": file.content_type},
        )

        try:
            # Validate file
            if not file.filename:
                raise HTTPException(status_code=400, detail="No filename provided")

            # Ensure data directory exists
            data_dir = Path("data")
            data_dir.mkdir(exist_ok=True)

            # Save file
            file_path = data_dir / file.filename
            content = await file.read()

            with open(file_path, "wb") as f:
                f.write(content)

            logger.info(
                "File uploaded successfully",
                extra={
                    "uploaded_filename": file.filename,
                    "size": len(content),
                    "path": str(file_path),
                },
            )

            return UploadResponse(
                filename=file.filename,
                size=len(content),
                trace_id=trace_id,
                message=f"File {file.filename} uploaded successfully",
            )

        except Exception as e:
            logger.error(f"File upload failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.get("/api/documents", response_model=List[DocumentResponse])
async def list_documents(request: Request = None):
    """List uploaded documents.

    Returns a list of files in the data directory.
    """
    trace_id = request.headers.get("x-trace-id") or generate_trace_id()

    with trace_context(trace_id):
        try:
            data_dir = Path("data")
            if not data_dir.exists():
                return []

            documents = []
            for file_path in data_dir.glob("*"):
                if file_path.is_file():
                    stats = file_path.stat()
                    documents.append(
                        DocumentResponse(
                            name=file_path.name,
                            size=f"{stats.st_size / 1024 / 1024:.2f} MB",
                            uploadDate=datetime.datetime.fromtimestamp(stats.st_mtime).strftime(
                                "%Y-%m-%d"
                            ),
                        )
                    )

            return documents
        except Exception as e:
            logger.error(f"List documents failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")


# =========================================================================
# Collection Endpoints
# =========================================================================


@app.get("/api/collections", response_model=List[CollectionResponse])
async def list_collections(request: Request = None):
    """List all document collections with their document counts."""
    trace_id = request.headers.get("x-trace-id") or generate_trace_id()

    with trace_context(trace_id):
        try:
            if not cubo_app or not cubo_app.vector_store:
                return []
            
            collections = cubo_app.vector_store.list_collections()
            return [CollectionResponse(**c) for c in collections]
        except Exception as e:
            logger.error(f"List collections failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to list collections: {str(e)}")


@app.post("/api/collections", response_model=CollectionResponse)
async def create_collection(collection_data: CollectionCreate, request: Request = None):
    """Create a new document collection."""
    trace_id = request.headers.get("x-trace-id") or generate_trace_id()

    with trace_context(trace_id):
        try:
            if not cubo_app or not cubo_app.vector_store:
                raise HTTPException(status_code=503, detail="CUBO app not initialized")
            
            collection = cubo_app.vector_store.create_collection(
                name=collection_data.name,
                color=collection_data.color
            )
            logger.info(f"Created collection: {collection['name']}")
            return CollectionResponse(**collection)
        except ValueError as e:
            raise HTTPException(status_code=409, detail=str(e))
        except Exception as e:
            logger.error(f"Create collection failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to create collection: {str(e)}")


@app.get("/api/collections/{collection_id}", response_model=CollectionResponse)
async def get_collection(collection_id: str, request: Request = None):
    """Get a specific collection by ID."""
    trace_id = request.headers.get("x-trace-id") or generate_trace_id()

    with trace_context(trace_id):
        try:
            if not cubo_app or not cubo_app.vector_store:
                raise HTTPException(status_code=503, detail="CUBO app not initialized")
            
            collection = cubo_app.vector_store.get_collection(collection_id)
            if not collection:
                raise HTTPException(status_code=404, detail="Collection not found")
            
            return CollectionResponse(**collection)
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Get collection failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to get collection: {str(e)}")


@app.delete("/api/collections/{collection_id}")
async def delete_collection(collection_id: str, request: Request = None):
    """Delete a collection (documents remain in store, just unlinked)."""
    trace_id = request.headers.get("x-trace-id") or generate_trace_id()

    with trace_context(trace_id):
        try:
            if not cubo_app or not cubo_app.vector_store:
                raise HTTPException(status_code=503, detail="CUBO app not initialized")
            
            deleted = cubo_app.vector_store.delete_collection(collection_id)
            if not deleted:
                raise HTTPException(status_code=404, detail="Collection not found")
            
            logger.info(f"Deleted collection: {collection_id}")
            return {"status": "deleted", "collection_id": collection_id}
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Delete collection failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to delete collection: {str(e)}")


@app.post("/api/collections/{collection_id}/documents", response_model=AddDocumentsResponse)
async def add_documents_to_collection(
    collection_id: str, request_data: AddDocumentsToCollectionRequest, request: Request = None
):
    """Add documents to a collection by their IDs."""
    trace_id = request.headers.get("x-trace-id") or generate_trace_id()

    with trace_context(trace_id):
        try:
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
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Add documents to collection failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to add documents: {str(e)}")


@app.delete("/api/collections/{collection_id}/documents")
async def remove_documents_from_collection(
    collection_id: str, request_data: AddDocumentsToCollectionRequest, request: Request = None
):
    """Remove documents from a collection."""
    trace_id = request.headers.get("x-trace-id") or generate_trace_id()

    with trace_context(trace_id):
        try:
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
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Remove documents from collection failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to remove documents: {str(e)}")


@app.get("/api/collections/{collection_id}/documents")
async def get_collection_documents(collection_id: str, request: Request = None):
    """Get all document IDs in a collection."""
    trace_id = request.headers.get("x-trace-id") or generate_trace_id()

    with trace_context(trace_id):
        try:
            if not cubo_app or not cubo_app.vector_store:
                raise HTTPException(status_code=503, detail="CUBO app not initialized")
            
            # Check collection exists
            collection = cubo_app.vector_store.get_collection(collection_id)
            if not collection:
                raise HTTPException(status_code=404, detail="Collection not found")
            
            doc_ids = cubo_app.vector_store.get_collection_documents(collection_id)
            return {"collection_id": collection_id, "document_ids": doc_ids, "count": len(doc_ids)}
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Get collection documents failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to get documents: {str(e)}")


# =========================================================================
# End Collection Endpoints
# =========================================================================


@app.post("/api/ingest", response_model=IngestResponse)
async def ingest_documents(
    request_data: IngestRequest, background_tasks: BackgroundTasks, request: Request = None
):
    """Ingest documents from data directory.

    Loads documents from the specified data path and prepares them
    for indexing. Supports fast-pass mode for quick processing.

    Args:
        request_data: Ingestion configuration (data path, fast-pass mode).
        background_tasks: FastAPI background tasks manager (auto-injected).
        request: FastAPI request object (auto-injected).

    Returns:
        IngestResponse: Ingestion status with document count.

    Raises:
        HTTPException: 503 if CUBO app not initialized, 404 if path not found,
                      500 if ingestion fails.
    """
    trace_id = request.headers.get("x-trace-id") or generate_trace_id()

    with trace_context(trace_id):
        if not cubo_app:
            raise HTTPException(status_code=503, detail="CUBO app not initialized")

        logger.info("Document ingestion started", extra={"fast_pass": request_data.fast_pass})

        try:
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
                    trace_id=trace_id,
                    message="No documents found",
                )

            logger.info(f"Loaded {len(documents)} documents for ingestion")

            # Use DeepIngestor for proper document processing and run in threadpool
            from cubo.ingestion.deep_ingestor import DeepIngestor

            def run_ingestor():
                ingestor = DeepIngestor(
                    input_folder=str(data_path),
                    output_dir=config.get("ingestion.deep.output_dir", "./data/deep"),
                )
                return ingestor.ingest()

            result = await run_in_threadpool(run_ingestor)

            if not result:
                logger.warning("Deep ingestion produced no results")
                return IngestResponse(
                    status="completed",
                    documents_processed=0,
                    trace_id=trace_id,
                    message="No chunks generated from documents",
                )

            chunks_count = result.get("chunks_count", 0)
            parquet_path = result.get("chunks_parquet", "")

            logger.info(
                "Document ingestion completed",
                extra={"chunks_processed": chunks_count, "parquet_path": parquet_path},
            )
            trace_collector.record(
                trace_id,
                "api",
                "ingest.completed",
                {"chunks_processed": chunks_count, "parquet_path": parquet_path},
            )

            return IngestResponse(
                status="completed",
                documents_processed=chunks_count,
                trace_id=trace_id,
                message="Ingestion completed successfully",
            )

        except Exception as e:
            import traceback

            traceback.print_exc()
            logger.error(
                "Document ingestion failed",
                extra={"error": str(e), "trace_id": trace_id},
                exc_info=True,
            )
            raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.post("/api/build-index", response_model=BuildIndexResponse)
async def build_index(
    request_data: BuildIndexRequest, background_tasks: BackgroundTasks, request: Request = None
):
    """Build or rebuild search indexes.

    Creates vector embeddings and BM25 indexes for all ingested documents.
    Supports force rebuild to recreate indexes from scratch.

    Args:
        request_data: Index build configuration (force rebuild flag).
        background_tasks: FastAPI background tasks manager (auto-injected).
        request: FastAPI request object (auto-injected).

    Returns:
        BuildIndexResponse: Build status with indexed document count.

    Raises:
        HTTPException: 503 if CUBO app not initialized, 500 if build fails.
    """
    trace_id = request.headers.get("x-trace-id") or generate_trace_id()

    with trace_context(trace_id):
        if not cubo_app:
            raise HTTPException(status_code=503, detail="CUBO app not initialized")

        logger.info("Index build started", extra={"force_rebuild": request_data.force_rebuild})

        try:
            # Build index using CUBOApp's build_index method
            # This will:
            # 1. Initialize components if needed (model, retriever, generator)
            # 2. Load documents from data folder
            # 3. Add documents to vector DB (FAISS)
            # 4. Update BM25 indexes

            doc_count = await run_in_threadpool(cubo_app.build_index)

            logger.info(f"Index build completed with {doc_count} documents")
            trace_collector.record(
                trace_id, "api", "build_index.completed", {"documents_indexed": doc_count}
            )

            return BuildIndexResponse(
                status="completed",
                trace_id=trace_id,
                message=f"Index built successfully with {doc_count} documents",
            )

        except Exception as e:
            logger.error(f"Index build failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Index build failed: {str(e)}")


@app.post("/api/query", response_model=QueryResponse)
async def query(request_data: QueryRequest, request: Request = None):
    """Query the RAG system.

    Performs semantic search over indexed documents and generates
    an answer using the retriever and generator components.

    Args:
        request_data: Query parameters (query text, top_k, use_reranker).
        request: FastAPI request object (auto-injected).

    Returns:
        QueryResponse: Generated answer with source documents and metadata.

    Raises:
        HTTPException: 503 if CUBO app or retriever not initialized,
                      500 if query processing fails.

    Example:
        >>> response = await query(QueryRequest(query="What is RAG?", top_k=5))
        >>> print(response.answer)
    """
    trace_id = request.headers.get("x-trace-id") or generate_trace_id()

    with trace_context(trace_id):
        if not cubo_app:
            raise HTTPException(status_code=503, detail="CUBO app not initialized")

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
                trace_id,
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

        try:
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
                    logger.info(f"Filtering query to collection {request_data.collection_id} ({len(filenames)} files)")
                else:
                    logger.warning(f"Collection {request_data.collection_id} has no documents")

            # Retrieve documents using the correct method
            retrieve_kwargs = {
                "query": request_data.query,
                "top_k": request_data.top_k,
                "trace_id": trace_id
            }
            if where_filter:
                retrieve_kwargs["where"] = where_filter
            
            # Use CUBOApp query_retrieve wrapper which guards access to the retriever
            retrieved_docs = await run_in_threadpool(
                lambda: cubo_app.query_retrieve(**retrieve_kwargs)
            )

            logger.info(f"Retrieved {len(retrieved_docs)} documents")

            # Generate answer using the correct generator method
            if hasattr(cubo_app, "generator") and cubo_app.generator:
                # Build context from retrieved documents
                context = "\n\n".join(
                    [doc.get("document", doc.get("content", "")) for doc in retrieved_docs]
                )
                answer = await run_in_threadpool(
                    lambda: cubo_app.generate_response_safe(
                        query=request_data.query, context=context, trace_id=trace_id
                    )
                )
            else:
                # Fallback: return retrieved documents as context
                answer = "Retrieved documents (no generator available):\n\n"
                answer += "\n\n".join(
                    [
                        doc.get("document", doc.get("content", ""))[:200]
                        for doc in retrieved_docs[:3]
                    ]
                )

            # Format sources - handle both 'document' and 'content' keys
            sources = []
            citations = []
            for idx, doc in enumerate(retrieved_docs):
                metadata = doc.get("metadata", {})
                content = doc.get("document", doc.get("content", ""))
                score = doc.get("similarity", doc.get("score", 0.0))
                
                # Build source entry
                sources.append({
                    "content": content[:500],
                    "metadata": metadata,
                    "score": score,
                })
                
                # Build citation for GDPR compliance
                chunk_id = metadata.get("chunk_id", metadata.get("id", f"chunk_{idx}"))
                citations.append(Citation(
                    source_file=metadata.get("filename", metadata.get("source", "unknown")),
                    page=metadata.get("page", metadata.get("page_number")),
                    chunk_id=str(chunk_id),
                    chunk_index=metadata.get("chunk_index", idx),
                    text_snippet=content[:200] if content else "",
                    relevance_score=round(float(score), 4) if score else 0.0,
                ))

            logger.info(
                "Query processed successfully",
                extra={"query": scrubbed_query, "sources_count": len(sources)},
            )
            try:
                trace_collector.record(
                    trace_id, "api", "query.completed", {"sources_count": len(sources)}
                )
            except Exception:
                pass

            return QueryResponse(
                answer=answer, 
                sources=sources, 
                citations=citations,
                trace_id=trace_id, 
                query_scrubbed=query_scrubbed
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(
                f"Query processing failed: {e}", extra={"query": scrubbed_query}, exc_info=True
            )
            raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint.

    Provides basic API information and links to documentation.

    Returns:
        dict: API metadata including version and documentation URL.
    """
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
    """Delete a document and its chunks from the index.
    
    GDPR-compliant document deletion endpoint. Removes document chunks
    from FAISS index and metadata from SQLite store. Logs deletion for audit.
    
    Args:
        doc_id: Document ID or filename to delete.
        request: FastAPI request object.
    
    Returns:
        DeleteDocumentResponse: Deletion status and chunk count.
    
    Raises:
        HTTPException: 503 if CUBO app not initialized, 404 if doc not found.
    """
    trace_id = request.headers.get("x-trace-id") or generate_trace_id()
    
    with trace_context(trace_id):
        if not cubo_app:
            raise HTTPException(status_code=503, detail="CUBO app not initialized")
        
        logger.info(f"Document deletion requested", extra={"doc_id": doc_id, "trace_id": trace_id})
        
        try:
            # Record deletion request in trace for GDPR audit
            trace_collector.record(
                trace_id, "api", "document.delete.requested", {"doc_id": doc_id}
            )
            
            chunks_removed = 0
            deleted = False
            
            # Try to delete from vector store
            if hasattr(cubo_app, "vector_store") and cubo_app.vector_store:
                try:
                    # Try delete by document ID/filename
                    cubo_app.vector_store.delete(ids=[doc_id])
                    deleted = True
                    chunks_removed += 1
                except Exception as e:
                    logger.warning(f"Vector store delete failed: {e}")
            
            # Try to delete from retriever's document store
            if hasattr(cubo_app, "retriever") and cubo_app.retriever:
                try:
                    if hasattr(cubo_app.retriever, "remove_document"):
                        result = cubo_app.retriever.remove_document(doc_id)
                        if result:
                            deleted = True
                except Exception as e:
                    logger.warning(f"Retriever remove_document failed: {e}")
            
            # Log deletion for GDPR audit trail
            trace_collector.record(
                trace_id, "api", "document.delete.completed", 
                {"doc_id": doc_id, "deleted": deleted, "chunks_removed": chunks_removed}
            )
            
            # Write to audit log
            audit_entry = {
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "action": "document_delete",
                "trace_id": trace_id,
                "doc_id": doc_id,
                "success": deleted,
                "chunks_removed": chunks_removed
            }
            logger.info("GDPR Audit: Document deletion", extra=audit_entry)
            
            if not deleted:
                raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
            
            return DeleteDocumentResponse(
                doc_id=doc_id,
                deleted=deleted,
                chunks_removed=chunks_removed,
                trace_id=trace_id,
                message=f"Document {doc_id} deleted successfully"
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Document deletion failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")


@app.get("/api/export-audit")
async def export_audit(
    request: Request,
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    format: str = Query("csv", description="Export format: csv or json")
):
    """Export GDPR audit log as CSV or JSON.
    
    Parses JSONL log files to extract query traces with sources for compliance audits.
    
    Args:
        start_date: Filter from this date (inclusive).
        end_date: Filter to this date (inclusive).
        format: Output format - 'csv' or 'json'.
    
    Returns:
        StreamingResponse: CSV or JSON file with audit entries.
    
    Example:
        GET /api/export-audit?start_date=2024-11-01&format=csv
    """
    trace_id = request.headers.get("x-trace-id") or generate_trace_id()
    
    with trace_context(trace_id):
        logger.info("GDPR audit export requested", extra={
            "start_date": start_date, 
            "end_date": end_date,
            "format": format,
            "trace_id": trace_id
        })
        
        try:
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
            audit_entries = []
            
            # Look for JSONL logs
            jsonl_files = list(log_dir.glob("*.jsonl*"))
            for log_file in jsonl_files:
                try:
                    with open(log_file, "r", encoding="utf-8") as f:
                        for line in f:
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
                                sources_info = ""
                                
                                # Try to extract query from structured message
                                if isinstance(message, str) and "query" in message.lower():
                                    # Hash any query text found
                                    query_hash = hashlib.sha256(message.encode()).hexdigest()[:16]
                                    query_text = f"[hashed:{query_hash}]"
                                
                                audit_entries.append({
                                    "timestamp": ts.isoformat(),
                                    "trace_id": trace,
                                    "query_hash": query_text,
                                    "level": entry.get("levelname", "INFO"),
                                    "component": entry.get("name", "unknown"),
                                    "action": message[:200] if isinstance(message, str) else str(message)[:200]
                                })
                                
                            except json.JSONDecodeError:
                                continue
                except Exception as e:
                    logger.warning(f"Failed to read log file {log_file}: {e}")
            
            # Sort by timestamp
            audit_entries.sort(key=lambda x: x["timestamp"])
            
            if format.lower() == "json":
                # Return JSON
                content = json.dumps({"audit_entries": audit_entries, "count": len(audit_entries)}, indent=2)
                return StreamingResponse(
                    io.BytesIO(content.encode("utf-8")),
                    media_type="application/json",
                    headers={"Content-Disposition": f"attachment; filename=cubo_audit_{datetime.date.today().isoformat()}.json"}
                )
            else:
                # Return CSV
                output = io.StringIO()
                writer = csv.DictWriter(output, fieldnames=["timestamp", "trace_id", "query_hash", "level", "component", "action"])
                writer.writeheader()
                writer.writerows(audit_entries)
                
                return StreamingResponse(
                    io.BytesIO(output.getvalue().encode("utf-8")),
                    media_type="text/csv",
                    headers={"Content-Disposition": f"attachment; filename=cubo_audit_{datetime.date.today().isoformat()}.csv"}
                )
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Audit export failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")
