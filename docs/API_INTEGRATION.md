# CUBO API Integration Guide

This guide explains how to use the CUBO RAG system with the FastAPI backend and Next.js frontend.

## Architecture

```
┌─────────────────┐         ┌──────────────────┐         ┌─────────────────┐
│  Next.js        │         │  FastAPI         │         │  CUBO RAG       │
│  Frontend       │────────>│  API Server      │────────>│  Backend        │
│  (Port 3000)    │         │  (Port 8000)     │         │  Engine         │
└─────────────────┘         └──────────────────┘         └─────────────────┘
```

## Quick Start

### 1. Install Dependencies

```bash
# Backend dependencies
pip install -r requirements.txt

# Frontend dependencies
cd frontend
pnpm install
cd ..
```

### 2. Start the Backend API Server

```bash
# Option 1: Using Python directly
python src/cubo/server/run.py

# Option 2: Using uvicorn directly
uvicorn cubo.server.api:app --host 0.0.0.0 --port 8000 --reload

# Option 3: Development mode with auto-reload
python src/cubo/server/run.py --reload
```

The API server will start on `http://localhost:8000`. Visit `http://localhost:8000/docs` for interactive API documentation.

### 3. Start the Frontend Dev Server

```bash
cd frontend
pnpm dev
```

The frontend will start on `http://localhost:3000`.

### 4. Upload and Query Documents

1. Visit `http://localhost:3000/upload`
2. Upload documents
3. Click "Ingest Documents"
4. Click "Build Index"
5. Go to `http://localhost:3000/chat` and ask questions

## API Endpoints

### Health Check
```bash
GET /api/health
```

### Upload File
```bash
POST /api/upload
Content-Type: multipart/form-data

{
  "file": <file>
}
```

### Ingest Documents
```bash
POST /api/ingest
Content-Type: application/json

{
  "data_path": "data",  # optional
  "fast_pass": true     # optional
}
```

### Build Index
```bash
POST /api/build-index
Content-Type: application/json

{
  "force_rebuild": false  # optional
}
```

### Query
```bash
POST /api/query
Content-Type: application/json

{
  "query": "What is this about?",
  "top_k": 5,              # optional, default 5
  "use_reranker": true     # optional, default true
}
```

## Trace ID Propagation

All API requests and responses include a `x-trace-id` header for request tracking:

```bash
# Request with custom trace ID
curl -H "x-trace-id: my-trace-123" http://localhost:8000/api/health

# Response includes trace ID
x-trace-id: my-trace-123
```

Trace IDs are automatically:
- Generated if not provided
- Logged with all operations
- Returned in responses
- Searchable in logs

## Query Scrubbing

Sensitive queries are automatically scrubbed in logs based on `config.json`:

```json
{
  "security": {
    "scrub_queries": true
  }
}
```

When enabled, queries are hashed before logging for privacy.

## Running Tests

### API Unit Tests
```bash
pytest tests/api/ -v
```

### E2E Smoke Test
```bash
# Start the API server first
python src/cubo/server/run.py

# In another terminal, run the E2E test
python scripts/e2e_smoke.py
```

### Frontend Tests
```bash
cd frontend
pnpm test
```

## CI/CD Integration

The E2E smoke test runs automatically in CI via `.github/workflows/e2e.yml`:

1. Starts the API server
2. Runs the complete flow: upload → ingest → build → query
3. Verifies trace_id presence in logs
4. Checks for errors in logs
5. Runs API unit tests

## Development Tips

### Hot Reload

Both backend and frontend support hot reload:

```bash
# Backend with auto-reload
python src/cubo/server/run.py --reload

# Frontend (auto-enabled in dev mode)
cd frontend && pnpm dev
```

### Debugging

Enable debug logging in `config.json`:

```json
{
  "logging": {
    "level": "DEBUG",
    "format": "json"
  }
}
```

### Log Search

Search logs by trace_id:

```bash
python scripts/logcli.py search --trace-id <trace-id>
```

### CORS Configuration

Update `src/cubo/server/api.py` to add more allowed origins:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://yourdomain.com"
    ],
    ...
)
```

## Troubleshooting

### API Server Won't Start

1. Check if port 8000 is already in use:
   ```bash
   netstat -ano | findstr :8000
   ```

2. Use a different port:
   ```bash
   python src/cubo/server/run.py --port 8001
   ```

3. Check logs in `logs/cubo_log.txt`

### Frontend Can't Connect to API

1. Verify API server is running: `curl http://localhost:8000/api/health`
2. Check Next.js rewrites in `frontend/next.config.mjs`
3. Ensure CORS is properly configured

### Queries Return 503

The retriever needs to be initialized first:
1. Upload documents via `/api/upload`
2. Run ingestion via `/api/ingest`
3. Build index via `/api/build-index`
4. Then queries will work

### No Trace IDs in Logs

1. Check logging configuration in `config.json`
2. Verify logs are being written to `logs/cubo_log.txt`
3. Ensure JSON format is enabled

## Production Deployment

### Backend

```bash
# Using gunicorn with uvicorn workers
gunicorn cubo.server.api:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

### Frontend

```bash
cd frontend
pnpm build
pnpm start
```

### Environment Variables

Create `.env.local` in frontend:

```bash
NEXT_PUBLIC_API_URL=https://api.yourdomain.com
```

## Security Considerations

1. **Query Scrubbing**: Enable in production to protect sensitive data
2. **CORS**: Restrict allowed origins in production
3. **Rate Limiting**: Add rate limiting middleware (not included)
4. **Authentication**: Add auth middleware (not included)
5. **HTTPS**: Use HTTPS in production with proper certificates

## Support

For issues, check:
- API docs: `http://localhost:8000/docs`
- Logs: `logs/cubo_log.txt`
- Test results: Run `python scripts/e2e_smoke.py`
