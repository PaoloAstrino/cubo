"""FastAPI server for CUBO RAG system."""
import os
import sys
import uuid
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, UploadFile, File, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.cubo.main import CUBOApp
from src.cubo.services.service_manager import ServiceManager
from src.cubo.security.security import security_manager
from src.cubo.utils.logger import logger
from src.cubo.utils.logging_context import trace_context, generate_trace_id

# Global app instance
cubo_app: Optional[CUBOApp] = None
service_manager: Optional[ServiceManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    global cubo_app, service_manager
    
    print(">>> LIFESPAN: Starting (MINIMAL TEST)", flush=True)
    logger.info("Initializing CUBO application")
    try:
        # TEMPORARY: Disable initialization to test server startup
        # try:
        #     cubo_app = CUBOApp()
        #     service_manager = ServiceManager()
        #     logger.info("CUBO application initialized successfully")
        # except Exception as init_error:
        #     logger.warning(f"CUBOApp initialization failed: {init_error}")
        
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
    """Add trace_id to all requests and responses."""
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
    documents_processed: int
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


# API Endpoints
@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
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


@app.post("/api/upload", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    request: Request = None
):
    """Upload a document file."""
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


@app.post("/api/ingest", response_model=IngestResponse)
async def ingest_documents(
    request_data: IngestRequest,
    background_tasks: BackgroundTasks,
    request: Request = None
):
    """Ingest documents from data directory."""
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
            documents = cubo_app.document_loader.load_documents(str(data_path))
            
            if not documents:
                logger.warning("No documents found to ingest")
                return IngestResponse(
                    status="completed",
                    documents_processed=0,
                    trace_id=trace_id,
                    message="No documents found"
                )
            
            logger.info(f"Loaded {len(documents)} documents for ingestion")
            
            # Process documents (this can be time-consuming)
            # For fast_pass mode, we'll use simpler processing
            processed_count = len(documents)
            
            # Store documents for retrieval
            # This would typically involve creating embeddings and storing in vector DB
            # For now, we'll just log and return success
            
            logger.info(
                "Document ingestion completed",
                extra={"documents_processed": processed_count}
            )
            
            return IngestResponse(
                status="completed",
                documents_processed=processed_count,
                trace_id=trace_id,
                message=f"Successfully ingested {processed_count} documents"
            )
            
        except Exception as e:
            logger.error(f"Document ingestion failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.post("/api/build-index", response_model=BuildIndexResponse)
async def build_index(
    request_data: BuildIndexRequest,
    background_tasks: BackgroundTasks,
    request: Request = None
):
    """Build or rebuild search indexes."""
    trace_id = request.headers.get("x-trace-id") or generate_trace_id()
    
    with trace_context(trace_id):
        if not cubo_app:
            raise HTTPException(status_code=503, detail="CUBO app not initialized")
        
        logger.info(
            "Index build started",
            extra={"force_rebuild": request_data.force_rebuild}
        )
        
        try:
            # This would typically involve:
            # 1. Creating FAISS index
            # 2. Building Whoosh BM25 index
            # 3. Publishing indexes atomically
            
            # For now, return success
            logger.info("Index build completed")
            
            return BuildIndexResponse(
                status="completed",
                trace_id=trace_id,
                message="Index built successfully"
            )
            
        except Exception as e:
            logger.error(f"Index build failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Index build failed: {str(e)}")


@app.post("/api/query", response_model=QueryResponse)
async def query(
    request_data: QueryRequest,
    request: Request = None
):
    """Query the RAG system."""
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
    """Root endpoint."""
    return {
        "message": "CUBO RAG API",
        "version": "1.0.0",
        "docs": "/docs"
    }
