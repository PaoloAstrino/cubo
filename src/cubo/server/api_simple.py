"""Simplified CUBO FastAPI server - no lifespan, lazy initialization."""
import sys
from pathlib import Path
from typing import Dict, List, Optional
from contextlib import contextmanager

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.cubo.main import CUBOApp
from src.cubo.services.service_manager import ServiceManager
from src.cubo.utils.logging_context import trace_context, generate_trace_id
from src.cubo.utils.logger import logger

# Global state - lazy initialized on first use
cubo_app: Optional[CUBOApp] = None
service_manager: Optional[ServiceManager] = None


def get_cubo_app() -> CUBOApp:
    """Get or initialize CUBOApp instance."""
    global cubo_app
    if cubo_app is None:
        logger.info("Initializing CUBOApp")
        cubo_app = CUBOApp()
    return cubo_app


def get_service_manager() -> ServiceManager:
    """Get or initialize ServiceManager instance."""
    global service_manager
    if service_manager is None:
        logger.info("Initializing ServiceManager")
        service_manager = ServiceManager()
    return service_manager


# Create FastAPI app
app = FastAPI(
    title="CUBO RAG API",
    description="REST API for CUBO RAG system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def trace_id_middleware(request: Request, call_next):
    """Add trace ID to all requests."""
    trace_id = request.headers.get("x-trace-id") or generate_trace_id()
    
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
    sources: List[Dict]
    trace_id: str


class UploadResponse(BaseModel):
    """Upload response model."""
    message: str
    uploaded_filename: str
    file_path: str
    trace_id: str


class IngestResponse(BaseModel):
    """Ingest response model."""
    message: str
    documents_loaded: int
    trace_id: str


class BuildIndexResponse(BaseModel):
    """Build index response model."""
    message: str
    trace_id: str


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
        
        return HealthResponse(
            status="healthy",
            version="1.0.0",
            components=components
        )


@app.post("/api/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...), request: Request = None):
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
            
            logger.info(f"File uploaded successfully: {file_path}")
            
            return UploadResponse(
                message="File uploaded successfully",
                uploaded_filename=file.filename,
                file_path=str(file_path),
                trace_id=trace_id
            )
            
        except Exception as e:
            logger.error(f"File upload failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.post("/api/ingest", response_model=IngestResponse)
async def ingest_documents(request: Request):
    """Ingest documents from the data directory."""
    trace_id = request.headers.get("x-trace-id") or generate_trace_id()
    
    with trace_context(trace_id):
        logger.info("Document ingestion requested")
        
        try:
            # Lazy initialize on first use
            app_instance = get_cubo_app()
            
            # Load documents
            app_instance.load_documents()
            
            doc_count = len(app_instance.documents) if hasattr(app_instance, 'documents') else 0
            
            logger.info(f"Documents ingested successfully: {doc_count} documents")
            
            return IngestResponse(
                message="Documents ingested successfully",
                documents_loaded=doc_count,
                trace_id=trace_id
            )
            
        except Exception as e:
            logger.error(f"Document ingestion failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.post("/api/build-index", response_model=BuildIndexResponse)
async def build_index(request: Request):
    """Build the vector index from ingested documents."""
    trace_id = request.headers.get("x-trace-id") or generate_trace_id()
    
    with trace_context(trace_id):
        logger.info("Index building requested")
        
        try:
            # Lazy initialize on first use
            app_instance = get_cubo_app()
            
            # Build index
            app_instance.build_index()
            
            logger.info("Index built successfully")
            
            return BuildIndexResponse(
                message="Index built successfully",
                trace_id=trace_id
            )
            
        except Exception as e:
            logger.error(f"Index building failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Index building failed: {str(e)}")


@app.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest, http_request: Request):
    """Query the RAG system."""
    trace_id = http_request.headers.get("x-trace-id") or generate_trace_id()
    
    with trace_context(trace_id):
        logger.info(f"Query request received: {request.query[:50]}...")
        
        try:
            # Lazy initialize on first use
            app_instance = get_cubo_app()
            
            # Retrieve documents
            retrieved_docs = app_instance.retrieve(
                query=request.query,
                top_k=request.top_k,
                use_reranker=request.use_reranker
            )
            
            # Generate response
            response_text = app_instance.generate(
                query=request.query,
                context=retrieved_docs
            )
            
            # Format sources
            sources = [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": doc.metadata.get("score", 0.0)
                }
                for doc in retrieved_docs
            ]
            
            logger.info(f"Query processed successfully, returned {len(sources)} sources")
            
            return QueryResponse(
                answer=response_text,
                sources=sources,
                trace_id=trace_id
            )
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "CUBO RAG API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
