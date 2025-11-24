# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]
- Removed all ChromaDB references and defaulted vector store to FAISS.
- Converted root-level script tests to proper pytest tests in tests/api/.
- Updated requirements-dev.txt to include 'requests' for test dependencies.
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
