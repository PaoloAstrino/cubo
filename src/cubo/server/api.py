"""FastAPI server for CUBO RAG system."""
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional
import datetime
import os

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.cubo.main import CUBOApp
from src.cubo.security.security import security_manager
from src.cubo.services.service_manager import ServiceManager
from src.cubo.utils.logger import logger
from src.cubo.utils.logging_context import generate_trace_id, trace_context

# Global app instance
cubo_app: Optional[CUBOApp] = None
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
        # Initialize CUBO application
        try:
            cubo_app = CUBOApp()
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
    lifespan=lifespan
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

    # Set trace context for this request
    with trace_context(trace_id):
        # Log incoming request
        logger.info(
            "Incoming request",
            extra={
                "method": request.method,
                "url": str(request.url),
                "client": request.client.host if request.client else None,
                "trace_id": trace_id
            }
        )

        response = await call_next(request)
        response.headers["x-trace-id"] = trace_id

        # Log response
        logger.info(
            "Response sent",
            extra={
                "status_code": response.status_code,
                "trace_id": trace_id
            }
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
    dense_weight: float = Field(0.7, ge=0.0, le=1.0, description="Weight for dense vector retrieval")
    retrieval_strategy: str = Field('hybrid', description="Retrieval strategy: 'hybrid', 'dense', or 'sparse'")


class QueryResponse(BaseModel):
    """Query response model."""
    answer: str
    sources: List[Dict[str, Any]]
    trace_id: str
    query_scrubbed: bool


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
            "service_manager": "not_initialized" if service_manager is None else "healthy"
        }

        # Check if retriever is ready (only if app is initialized)
        if cubo_app is not None:
            try:
                retriever_ready = hasattr(cubo_app, 'retriever') and cubo_app.retriever is not None
                components["retriever"] = "healthy" if retriever_ready else "not_ready"
            except Exception as e:
                components["retriever"] = f"error: {str(e)}"
                logger.error(f"Retriever health check failed: {e}")

        # API is healthy even if components aren't initialized yet (lazy loading)
        overall_status = "healthy"

        return HealthResponse(
            status=overall_status,
            version="1.0.0",
            components=components
        )


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
            "retriever": hasattr(cubo_app, 'retriever') and cubo_app.retriever is not None if cubo_app else False,
            "generator": hasattr(cubo_app, 'generator') and cubo_app.generator is not None if cubo_app else False,
            "doc_loader": hasattr(cubo_app, 'doc_loader') and cubo_app.doc_loader is not None if cubo_app else False,
            "vector_store": False
        }

        try:
            if cubo_app and hasattr(cubo_app, 'retriever') and cubo_app.retriever:
                components['vector_store'] = getattr(cubo_app.retriever, 'collection', None) is not None
        except Exception:
            components['vector_store'] = False

        return {"components": components, "trace_id": trace_id}


@app.post("/api/upload", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    request: Request = None
):
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
            extra={"uploaded_filename": file.filename, "content_type": file.content_type}
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
                extra={"uploaded_filename": file.filename, "size": len(content), "path": str(file_path)}
            )

            return UploadResponse(
                filename=file.filename,
                size=len(content),
                trace_id=trace_id,
                message=f"File {file.filename} uploaded successfully"
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
                    documents.append(DocumentResponse(
                        name=file_path.name,
                        size=f"{stats.st_size / 1024 / 1024:.2f} MB",
                        uploadDate=datetime.datetime.fromtimestamp(stats.st_mtime).strftime('%Y-%m-%d')
                    ))
            
            return documents
        except Exception as e:
            logger.error(f"List documents failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")



@app.post("/api/ingest", response_model=IngestResponse)
async def ingest_documents(
    request_data: IngestRequest,
    background_tasks: BackgroundTasks,
    request: Request = None
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
                    message="No documents found"
                )

            logger.info(f"Loaded {len(documents)} documents for ingestion")

            # Use DeepIngestor for proper document processing
            from src.cubo.ingestion.deep_ingestor import DeepIngestor
            
            ingestor = DeepIngestor(
                input_folder=str(data_path),
                output_dir=config.get('ingestion.deep.output_dir', './data/deep')
            )
            
            # Run ingestion (creates parquet with chunks)
            result = ingestor.ingest()
            
            if not result:
                logger.warning("Deep ingestion produced no results")
                return IngestResponse(
                    status="completed",
                    documents_processed=0,
                    trace_id=trace_id,
                    message="No chunks generated from documents"
                )
            
            chunks_count = result.get('chunks_count', 0)
            parquet_path = result.get('chunks_parquet', '')
            
            logger.info(
                "Document ingestion completed",
                extra={
                    "chunks_processed": chunks_count,
                    "parquet_path": parquet_path
                }
            )
            
            return IngestResponse(
                status="completed",
                documents_processed=chunks_count,
                trace_id=trace_id,
                message="Ingestion completed successfully"
            )
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.error(
                "Document ingestion failed",
                extra={"error": str(e), "trace_id": trace_id},
                exc_info=True
            )
            raise HTTPException(
                status_code=500,
                detail=f"Ingestion failed: {str(e)}"
            )


@app.post("/api/build-index", response_model=BuildIndexResponse)
async def build_index(
    request_data: BuildIndexRequest,
    background_tasks: BackgroundTasks,
    request: Request = None
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

        logger.info(
            "Index build started",
            extra={"force_rebuild": request_data.force_rebuild}
        )

        try:
            # Build index using CUBOApp's build_index method
            # This will:
            # 1. Initialize components if needed (model, retriever, generator)
            # 2. Load documents from data folder
            # 3. Add documents to vector DB (FAISS)
            # 4. Update BM25 indexes

            doc_count = cubo_app.build_index()

            logger.info(f"Index build completed with {doc_count} documents")

            return BuildIndexResponse(
                status="completed",
                trace_id=trace_id,
                message=f"Index built successfully with {doc_count} documents"
            )

        except Exception as e:
            logger.error(f"Index build failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Index build failed: {str(e)}")


@app.post("/api/query", response_model=QueryResponse)
async def query(
    request_data: QueryRequest,
    request: Request = None
):
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
                "scrubbed": query_scrubbed
            }
        )

        try:
            # Check if retriever is initialized
            if not hasattr(cubo_app, 'retriever') or not cubo_app.retriever:
                raise HTTPException(
                    status_code=503,
                    detail="Retriever not initialized. Please run ingestion and index building first."
                )

            # Retrieve documents
            retrieved_docs = cubo_app.retriever.retrieve(
                query=request_data.query,
                top_k=request_data.top_k,
                use_reranker=request_data.use_reranker
            )

            logger.info(f"Retrieved {len(retrieved_docs)} documents")

            # Generate answer
            if hasattr(cubo_app, 'generator') and cubo_app.generator:
                answer = cubo_app.generator.generate(
                    query=request_data.query,
                    context_docs=retrieved_docs
                )
            else:
                # Fallback: return retrieved documents as context
                answer = "Retrieved documents (no generator available):\n\n"
                answer += "\n\n".join([doc.get('content', '')[:200] for doc in retrieved_docs[:3]])

            # Format sources
            sources = [
                {
                    "content": doc.get('content', '')[:500],
                    "metadata": doc.get('metadata', {}),
                    "score": doc.get('score', 0.0)
                }
                for doc in retrieved_docs
            ]

            logger.info(
                "Query processed successfully",
                extra={
                    "query": scrubbed_query,
                    "sources_count": len(sources)
                }
            )

            return QueryResponse(
                answer=answer,
                sources=sources,
                trace_id=trace_id,
                query_scrubbed=query_scrubbed
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(
                f"Query processing failed: {e}",
                extra={"query": scrubbed_query},
                exc_info=True
            )
            raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint.
    
    Provides basic API information and links to documentation.
    
    Returns:
        dict: API metadata including version and documentation URL.
    """
    return {
        "message": "CUBO RAG API",
        "version": "1.0.0",
        "docs": "/docs"
    }
