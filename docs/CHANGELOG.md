# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]
- Refactored file I/O in API to be fully async using `aiofiles` and threadpools for high concurrency.
- Added `simplemma` integration for multilingual lemmatization in BM25 retrieval (configurable via `bm25.use_lemmatization`).
- Introduced `AdvancedPDFParser` using PyMuPDF (fitz) and EasyOCR for handling complex layouts and scanned documents.
- Added `parser` configuration option (default: `basic`, set to `advanced` to enable PyMuPDF/EasyOCR).
- Updated `/api/upload` to support streaming uploads with atomic writes.
- Updated `/api/export-audit` to stream large logs efficiently.
- Removed all ChromaDB references and defaulted vector store to FAISS.
- Converted root-level script tests to proper pytest tests in tests/api/.
- Moved dev requirements to `requirements/requirements-dev.txt` and exposed a `dev` extra in `pyproject.toml` for easier `pip install -e .[dev]` installs.
- Fixed syntax error in src/cubo/server/api.py for proper error handling.
- Added Dockerfile for backend and docker-compose.yml to orchestrate backend, frontend, and vector store (FAISS).
- Introduced scripts to generate Whoosh index from sample data.
- Added sample dataset for quick demos.
- Updated .gitignore to exclude generated artifacts.

## [0.1.0] - 2025-11-22
- Initial release of CUBO AI Document Assistant.
- Implemented full backend API with FastAPI.
- Added ingestion, retrieval, and query endpoints.
- Provided Docker support for backend, frontend, and vector store.
- Added comprehensive logging and health checks.
