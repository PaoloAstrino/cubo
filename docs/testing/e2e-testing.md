# Running E2E Integration Tests

## Quick Start

```bash
# Install test dependencies
# Preferred: install dev extras
pip install -e '.[dev]'
# Or: pip install -r requirements/requirements-dev.txt

# Install Playwright browsers (for frontend tests)
playwright install chromium

# Run all E2E tests
pytest tests/e2e/ -v -s

# Run specific test suite
pytest tests/e2e/test_full_rag_pipeline.py -v
pytest tests/e2e/test_gdpr_compliance.py -v
pytest tests/e2e/test_multimodal_pdf.py -v
pytest tests/e2e/test_reranking_quality.py -v
pytest tests/e2e/test_laptop_mode_memory.py -v
pytest tests/e2e/test_frontend_integration.py -v
```

## Test Descriptions

### 1. Full RAG Pipeline (`test_full_rag_pipeline.py`)
**Validates**: Upload → Ingest → Index → Query → Generate Answer with Citations

**Tests**:
- `test_upload_ingest_query_with_citations` - Complete pipeline flow
- `test_multifile_citation_accuracy` - Source attribution from multiple files
- `test_pipeline_with_trace_id_propagation` - Trace ID system for debugging

**Time**: ~30 seconds

---

### 2. GDPR Compliance (`test_gdpr_compliance.py`)
**Validates**: Query scrubbing, audit logs, data erasure compliance

**Tests**:
- `test_query_scrubbing_enabled` - Sensitive queries are hashed
- `test_audit_log_contains_trace_ids` - Audit trail includes trace IDs
- `test_audit_export_with_date_range` - Export logs by date range
- `test_no_plaintext_queries_in_logs` - PII not stored in logs
- `test_right_to_erasure_simulation` - User data can be deleted

**Time**: ~10 seconds

---

### 3. Multimodal PDF Processing (`test_multimodal_pdf.py`)
**Validates**: PDF table extraction and structured data handling

**Tests**:
- `test_table_extraction_with_pdfplumber` - Tables are extracted from PDFs
- `test_pdf_table_ingestion_and_query` - Table data is queryable
- `test_structured_data_preservation` - Row relationships maintained
- `test_mixed_content_pdf_ingestion` - Text + tables processed correctly

**Requirements**: `reportlab` for test PDF generation

**Time**: ~20 seconds

---

### 4. Reranking Quality (`test_reranking_quality.py`)
**Validates**: Reranking improves precision and recall

**Tests**:
- `test_reranking_improves_top_result` - Most relevant doc ranks higher
- `test_reranking_filters_irrelevant_results` - Irrelevant docs demoted
- `test_recall_at_k_with_reranking` - Recall@K metric measurement

**Time**: ~25 seconds

---

### 5. Laptop Mode Memory (`test_laptop_mode_memory.py`)
**Validates**: RAM usage < 2GB during operation

**Tests**:
- `test_laptop_mode_ram_under_2gb` - Peak RAM < 2048 MB
- `test_lazy_model_unloading` - Models unload after idle
- `test_memory_mapped_embeddings` - Mmap config verified
- `test_batch_size_reduction` - Small batches in laptop mode

**Requirements**: `psutil` for memory monitoring

**Time**: ~60 seconds (ingests 1000 docs)

---

### 6. Frontend-Backend Integration (`test_frontend_integration.py`)
**Validates**: Full stack with Next.js + FastAPI

**Tests**:
- `test_upload_flow_via_ui` - File upload through browser
- `test_query_flow_via_ui` - Query submission through browser
- `test_health_endpoint` - API health check
- `test_upload_endpoint` - File upload API
- `test_query_endpoint` - Query API

**Requirements**: `playwright` for browser automation

**Time**: ~90 seconds (starts servers)

---

## Troubleshooting

### Playwright Install Fails
```bash
# Install system dependencies (Linux)
apt-get install -y libglib2.0-0 libnss3 libnspr4 libdbus-1-3

# macOS/Windows: Playwright handles dependencies automatically
playwright install chromium
```

### Frontend Tests Skip
- Ensure `frontend/` directory exists
- Ensure `npm` is installed
- Check `frontend/package.json` has `dev` script

### Memory Tests Fail
- Close other applications to free RAM
- Laptop mode requires at least 4GB total system RAM
- Reduce `large_document_set` size if needed

### PDF Tests Skip
- Install: `pip install reportlab pdfplumber`

---

## CI/CD Integration

Add to `.github/workflows/test.yml`:

```yaml
- name: Run E2E Tests
  run: |
    pip install -r requirements/requirements-dev.txt
    playwright install chromium --with-deps
    pytest tests/e2e/ -v --maxfail=1
```

---

## Test Coverage Summary

| Test Suite | Critical Paths Covered | Time | Dependencies |
|------------|------------------------|------|--------------|
| Full RAG Pipeline | ✅ Upload, Ingest, Query, Generate | 30s | None |
| GDPR Compliance | ✅ Query scrubbing, Audit logs | 10s | None |
| Multimodal PDF | ✅ Table extraction, Structured data | 20s | reportlab |
| Reranking Quality | ✅ Precision, Recall measurement | 25s | None |
| Laptop Mode  Memory | ✅ RAM < 2GB validation | 60s | psutil |
| Frontend Integration | ✅ UI upload/query flows | 90s | playwright, npm |

**Total E2E Coverage**: 6 test suites, 20+ test cases, ~4 minutes runtime
