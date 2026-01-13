# 🌸 Code Quality Analysis Report 🌸

## Overall Assessment

- **Quality Score**: 39.71/100
- **Quality Level**: 😷 Code reeks, mask up - Code is starting to stink, approach with caution and a mask.
- **Analyzed Files**: 304
- **Total Lines**: 49294

## Quality Metrics

| Metric | Score | Weight | Status |
|------|------|------|------|
| State Management | 14.42 | 0.20 | ✓✓ |
| Error Handling | 25.00 | 0.10 | ✓ |
| Code Structure | 30.00 | 0.15 | ✓ |
| Code Duplication | 35.00 | 0.15 | ○ |
| Comment Ratio | 44.99 | 0.15 | ○ |
| Cyclomatic Complexity | 66.05 | 0.30 | ⚠ |

## Problem Files (Top 50)

### 1. C:\Users\paolo\Desktop\cubo\cubo\scripts\debug_retrieval.py (Score: 61.48)
**Issue Categories**: 🔄 Complexity Issues:2, ⚠️ Other Issues:1

**Main Issues**:
- Function test_retrieval has very high cyclomatic complexity (23), consider refactoring
- 函数 'test_retrieval' () 极度过长 (126 行)，必须拆分
- 函数 'test_retrieval' () 复杂度严重过高 (23)，必须简化

### 2. C:\Users\paolo\Desktop\cubo\scripts\run_beir_batch.py (Score: 57.86)
**Issue Categories**: 🔄 Complexity Issues:6, ⚠️ Other Issues:3

**Main Issues**:
- Function run_benchmark has very high cyclomatic complexity (34), consider refactoring
- Function generate_summary_markdown has high cyclomatic complexity (13), consider simplifying
- Function main has high cyclomatic complexity (13), consider simplifying
- 函数 'run_benchmark' () 极度过长 (134 行)，必须拆分
- 函数 'run_benchmark' () 复杂度严重过高 (34)，必须简化
- 函数 'generate_summary_markdown' () 较长 (54 行)，可考虑重构
- 函数 'generate_summary_markdown' () 复杂度过高 (13)，建议简化
- 函数 'main' () 较长 (69 行)，可考虑重构
- 函数 'main' () 复杂度过高 (13)，建议简化

### 3. C:\Users\paolo\Desktop\cubo\scripts\measure_reranker_effect.py (Score: 57.65)
**Issue Categories**: 🔄 Complexity Issues:2, ⚠️ Other Issues:1

**Main Issues**:
- Function measure_reranker_effect has very high cyclomatic complexity (18), consider refactoring
- 函数 'measure_reranker_effect' () 过长 (111 行)，建议拆分
- 函数 'measure_reranker_effect' () 复杂度过高 (18)，建议简化

### 4. C:\Users\paolo\Desktop\cubo\cubo\scripts\build_faiss_index.py (Score: 56.41)
**Issue Categories**: 🔄 Complexity Issues:2, 📝 Comment Issues:1, ⚠️ Other Issues:2

**Main Issues**:
- Function main has very high cyclomatic complexity (48), consider refactoring
- 函数 'parse_args' () 过长 (76 行)，建议拆分
- 函数 'main' () 极度过长 (184 行)，必须拆分
- 函数 'main' () 复杂度严重过高 (48)，必须简化
- Code comment ratio is low (8.21%), consider adding more comments

### 5. C:\Users\paolo\Desktop\cubo\cubo\scripts\migrate_chunk_ids.py (Score: 56.09)
**Issue Categories**: 🔄 Complexity Issues:22, 📝 Comment Issues:1, ⚠️ Other Issues:8

**Main Issues**:
- Function _generate_new_id has high cyclomatic complexity (12), consider simplifying
- Function main has high cyclomatic complexity (15), consider simplifying
- Function _generate_new_id has high cyclomatic complexity (12), consider simplifying
- Function main has high cyclomatic complexity (15), consider simplifying
- Function _generate_new_id has high cyclomatic complexity (12), consider simplifying
- Function main has high cyclomatic complexity (15), consider simplifying
- Function main has very high cyclomatic complexity (38), consider refactoring
- Function _generate_new_id has high cyclomatic complexity (12), consider simplifying
- Function main has high cyclomatic complexity (14), consider simplifying
- Function _generate_new_id has high cyclomatic complexity (12), consider simplifying
- Function main has high cyclomatic complexity (14), consider simplifying
- Function _generate_new_id has high cyclomatic complexity (12), consider simplifying
- Function main has high cyclomatic complexity (14), consider simplifying
- Function main has very high cyclomatic complexity (38), consider refactoring
- 函数 'main' () 较长 (51 行)，可考虑重构
- 函数 'main' () 复杂度过高 (15)，建议简化
- 函数 'main' () 较长 (51 行)，可考虑重构
- 函数 'main' () 复杂度过高 (15)，建议简化
- 函数 'main' () 较长 (51 行)，可考虑重构
- 函数 'main' () 复杂度过高 (15)，建议简化
- 函数 'main' () 过长 (99 行)，建议拆分
- 函数 'main' () 复杂度严重过高 (38)，必须简化
- 函数 'main' () 较长 (46 行)，可考虑重构
- 函数 'main' () 复杂度过高 (14)，建议简化
- 函数 'main' () 较长 (46 行)，可考虑重构
- 函数 'main' () 复杂度过高 (14)，建议简化
- 函数 'main' () 较长 (46 行)，可考虑重构
- 函数 'main' () 复杂度过高 (14)，建议简化
- 函数 'main' () 过长 (120 行)，建议拆分
- 函数 'main' () 复杂度严重过高 (38)，必须简化
- Code comment ratio is extremely low (4.84%), almost no comments

### 6. C:\Users\paolo\Desktop\cubo\scripts\prepare_ultradomain_by_category.py (Score: 56.00)
**Issue Categories**: 🔄 Complexity Issues:2, ⚠️ Other Issues:1

**Main Issues**:
- Function prepare_category has very high cyclomatic complexity (24), consider refactoring
- 函数 'prepare_category' () 过长 (78 行)，建议拆分
- 函数 'prepare_category' () 复杂度严重过高 (24)，必须简化

### 7. C:\Users\paolo\Desktop\cubo\cubo\scripts\inspect_db.py (Score: 55.85)
**Issue Categories**: 📝 Comment Issues:1, ⚠️ Other Issues:1

**Main Issues**:
- 函数 'main' () 较长 (43 行)，可考虑重构
- Code comment ratio is extremely low (4.55%), almost no comments

### 8. C:\Users\paolo\Desktop\cubo\scripts\verify_recall_mismatch.py (Score: 55.58)
**Issue Categories**: 🔄 Complexity Issues:2, ⚠️ Other Issues:1

**Main Issues**:
- 函数 'check_coverage' () 较长 (66 行)，可考虑重构
- 函数 'check_coverage' () 复杂度严重过高 (21)，必须简化
- Function check_coverage has very high cyclomatic complexity (21), consider refactoring

### 9. C:\Users\paolo\Desktop\cubo\cubo\scripts\migrate_chunk_ids_clean.py (Score: 55.15)
**Issue Categories**: 🔄 Complexity Issues:3, 📝 Comment Issues:1, ⚠️ Other Issues:1

**Main Issues**:
- Function _generate_new_id has high cyclomatic complexity (12), consider simplifying
- Function main has high cyclomatic complexity (15), consider simplifying
- 函数 'main' () 较长 (51 行)，可考虑重构
- 函数 'main' () 复杂度过高 (15)，建议简化
- Code comment ratio is extremely low (4.63%), almost no comments

### 10. C:\Users\paolo\Desktop\cubo\cubo\scripts\query.py (Score: 54.80)
**Issue Categories**: 🔄 Complexity Issues:2, ⚠️ Other Issues:1

**Main Issues**:
- Function main has very high cyclomatic complexity (17), consider refactoring
- 函数 'main' () 过长 (75 行)，建议拆分
- 函数 'main' () 复杂度过高 (17)，建议简化

### 11. C:\Users\paolo\Desktop\cubo\scripts\start_fullstack.py (Score: 54.59)
**Issue Categories**: 🔄 Complexity Issues:2, ⚠️ Other Issues:1

**Main Issues**:
- Function main has very high cyclomatic complexity (25), consider refactoring
- 函数 'main' () 过长 (86 行)，建议拆分
- 函数 'main' () 复杂度严重过高 (25)，必须简化

### 12. C:\Users\paolo\Desktop\cubo\scripts\calc_metrics_from_run.py (Score: 54.36)
**Issue Categories**: 🔄 Complexity Issues:2, 📝 Comment Issues:1, ⚠️ Other Issues:1

**Main Issues**:
- Function compute_metrics has very high cyclomatic complexity (36), consider refactoring
- 函数 'compute_metrics' () 过长 (75 行)，建议拆分
- 函数 'compute_metrics' () 复杂度严重过高 (36)，必须简化
- Code comment ratio is low (9.35%), consider adding more comments

### 13. C:\Users\paolo\Desktop\cubo\scripts\run_beir_adapter.py (Score: 53.83)
**Issue Categories**: 🔄 Complexity Issues:4, ⚠️ Other Issues:2

**Main Issues**:
- Function load_queries has very high cyclomatic complexity (21), consider refactoring
- Function main has very high cyclomatic complexity (42), consider refactoring
- 函数 'load_queries' () 较长 (50 行)，可考虑重构
- 函数 'load_queries' () 复杂度严重过高 (21)，必须简化
- 函数 'main' () 极度过长 (177 行)，必须拆分
- 函数 'main' () 复杂度严重过高 (42)，必须简化

### 14. C:\Users\paolo\Desktop\cubo\cubo\scripts\reindex_parquet.py (Score: 53.59)
**Issue Categories**: 🔄 Complexity Issues:2, ⚠️ Other Issues:1

**Main Issues**:
- Function main has very high cyclomatic complexity (19), consider refactoring
- 函数 'main' () 较长 (62 行)，可考虑重构
- 函数 'main' () 复杂度严重过高 (19)，必须简化

### 15. C:\Users\paolo\Desktop\cubo\cubo\scripts\run_rag_tests.py (Score: 53.40)
**Issue Categories**: 🔄 Complexity Issues:6, 📝 Comment Issues:1, ⚠️ Other Issues:3

**Main Issues**:
- 函数 '_initialize_cubo_system' () 较长 (46 行)，可考虑重构
- 函数 '_initialize_cubo_system' () 复杂度过高 (13)，建议简化
- 函数 'run_single_test' () 较长 (52 行)，可考虑重构
- 函数 'calculate_statistics' () 过长 (79 行)，建议拆分
- 函数 'calculate_statistics' () 复杂度严重过高 (31)，必须简化
- 函数 'print_summary' () 复杂度过高 (15)，建议简化
- Code comment ratio is low (9.63%), consider adding more comments
- Function _initialize_cubo_system has high cyclomatic complexity (13), consider simplifying
- Function calculate_statistics has very high cyclomatic complexity (31), consider refactoring
- Function print_summary has high cyclomatic complexity (15), consider simplifying

### 16. C:\Users\paolo\Desktop\cubo\cubo\scripts\deduplicate.py (Score: 53.33)
**Issue Categories**: 📝 Comment Issues:1, ⚠️ Other Issues:1

**Main Issues**:
- 函数 'parse_args' () 较长 (47 行)，可考虑重构
- Code comment ratio is extremely low (0.61%), almost no comments

### 17. C:\Users\paolo\Desktop\cubo\cubo\scripts\fast_pass_ingest.py (Score: 53.10)
**Issue Categories**: 📝 Comment Issues:1

**Main Issues**:
- Code comment ratio is low (8.00%), consider adding more comments

### 18. C:\Users\paolo\Desktop\cubo\scripts\verify_frontend_clean.py (Score: 53.02)
**Issue Categories**: 📝 Comment Issues:1

**Main Issues**:
- Code comment ratio is low (8.11%), consider adding more comments

### 19. C:\Users\paolo\Desktop\cubo\cubo\routing\query_router.py (Score: 52.99)
**Issue Categories**: 🔄 Complexity Issues:2, 📝 Comment Issues:1, ⚠️ Other Issues:1

**Main Issues**:
- Function classify has very high cyclomatic complexity (26), consider refactoring
- 函数 'classify' () 较长 (60 行)，可考虑重构
- 函数 'classify' () 复杂度严重过高 (26)，必须简化
- Code comment ratio is low (9.55%), consider adding more comments

### 20. C:\Users\paolo\Desktop\cubo\scripts\prepare_ultradomain.py (Score: 52.76)
**Issue Categories**: 🔄 Complexity Issues:2, ⚠️ Other Issues:1

**Main Issues**:
- Function prepare_ultradomain has high cyclomatic complexity (13), consider simplifying
- 函数 'prepare_ultradomain' () 较长 (46 行)，可考虑重构
- 函数 'prepare_ultradomain' () 复杂度过高 (13)，建议简化

### 21. C:\Users\paolo\Desktop\cubo\scripts\download_beir_dataset.py (Score: 52.62)
**Issue Categories**: ⚠️ Other Issues:1

**Main Issues**:
- 函数 'main' () 较长 (47 行)，可考虑重构

### 22. C:\Users\paolo\Desktop\cubo\cubo\utils\cpu_tuner.py (Score: 52.62)
**Issue Categories**: 🔄 Complexity Issues:2, ⚠️ Other Issues:1

**Main Issues**:
- Function auto_tune_cpu has very high cyclomatic complexity (25), consider refactoring
- 函数 'auto_tune_cpu' () 过长 (71 行)，建议拆分
- 函数 'auto_tune_cpu' () 复杂度严重过高 (25)，必须简化

### 23. C:\Users\paolo\Desktop\cubo\scripts\prepare_ragbench.py (Score: 52.62)
**Issue Categories**: 🔄 Complexity Issues:2, ⚠️ Other Issues:1

**Main Issues**:
- Function prepare_ragbench has very high cyclomatic complexity (24), consider refactoring
- 函数 'prepare_ragbench' () 过长 (82 行)，建议拆分
- 函数 'prepare_ragbench' () 复杂度严重过高 (24)，必须简化

### 24. C:\Users\paolo\Desktop\cubo\cubo\server\run_hypercorn.py (Score: 52.22)
**Issue Categories**: 📝 Comment Issues:1

**Main Issues**:
- Code comment ratio is low (9.23%), consider adding more comments

### 25. C:\Users\paolo\Desktop\cubo\scripts\villain_baseline.py (Score: 52.17)
**Issue Categories**: 📝 Comment Issues:1

**Main Issues**:
- Code comment ratio is low (6.90%), consider adding more comments

### 26. C:\Users\paolo\Desktop\cubo\cubo\ingestion\deep_ingestor.py (Score: 51.72)
**Issue Categories**: 🔄 Complexity Issues:8, ⚠️ Other Issues:3

**Main Issues**:
- Function _merge_temp_parquets has very high cyclomatic complexity (17), consider refactoring
- Function ingest has very high cyclomatic complexity (47), consider refactoring
- Function _process_file has high cyclomatic complexity (11), consider simplifying
- Function _process_pdf has very high cyclomatic complexity (30), consider refactoring
- Function _make_chunk_id has high cyclomatic complexity (12), consider simplifying
- 函数 '_merge_temp_parquets' () 较长 (52 行)，可考虑重构
- 函数 '_merge_temp_parquets' () 复杂度过高 (17)，建议简化
- 函数 'ingest' () 极度过长 (194 行)，必须拆分
- 函数 'ingest' () 复杂度严重过高 (47)，必须简化
- 函数 '_process_pdf' () 过长 (90 行)，建议拆分
- 函数 '_process_pdf' () 复杂度严重过高 (30)，必须简化

### 27. C:\Users\paolo\Desktop\cubo\scripts\system_metrics.py (Score: 51.67)
**Issue Categories**: 📝 Comment Issues:1

**Main Issues**:
- Code comment ratio is low (5.19%), consider adding more comments

### 28. C:\Users\paolo\Desktop\cubo\scripts\worker_retrieve.py (Score: 51.63)
**Issue Categories**: ⚠️ Other Issues:1

**Main Issues**:
- 函数 'main' () 较长 (47 行)，可考虑重构

### 29. C:\Users\paolo\Desktop\cubo\cubo\deduplication\table_deduplicator.py (Score: 51.29)
**Issue Categories**: 📝 Comment Issues:1

**Main Issues**:
- Code comment ratio is extremely low (1.72%), almost no comments

### 30. C:\Users\paolo\Desktop\cubo\scripts\calculate_beir_metrics.py (Score: 51.29)
**Issue Categories**: 🔄 Complexity Issues:2, ⚠️ Other Issues:1

**Main Issues**:
- 函数 'calculate_metrics' () 极度过长 (128 行)，必须拆分
- 函数 'calculate_metrics' () 复杂度严重过高 (45)，必须简化
- Function calculate_metrics has very high cyclomatic complexity (45), consider refactoring

### 31. C:\Users\paolo\Desktop\cubo\cubo\utils\hardware.py (Score: 51.06)
**Issue Categories**: 🔄 Complexity Issues:1, ⚠️ Other Issues:1

**Main Issues**:
- Function detect_hardware has high cyclomatic complexity (11), consider simplifying
- 函数 'detect_hardware' () 较长 (56 行)，可考虑重构

### 32. C:\Users\paolo\Desktop\cubo\cubo\indexing\faiss_index.py (Score: 51.02)
**Issue Categories**: 🔄 Complexity Issues:8, ⚠️ Other Issues:4

**Main Issues**:
- Function add_to_hot has high cyclomatic complexity (12), consider simplifying
- Function _create_trained_cold_index has high cyclomatic complexity (14), consider simplifying
- Function _build_cold_index has very high cyclomatic complexity (19), consider refactoring
- Function search has very high cyclomatic complexity (21), consider refactoring
- Function save has high cyclomatic complexity (11), consider simplifying
- 函数 '_create_trained_cold_index' () 较长 (66 行)，可考虑重构
- 函数 '_create_trained_cold_index' () 复杂度过高 (14)，建议简化
- 函数 '_build_cold_index' () 过长 (89 行)，建议拆分
- 函数 '_build_cold_index' () 复杂度严重过高 (19)，必须简化
- 函数 'search' () 较长 (50 行)，可考虑重构
- 函数 'search' () 复杂度严重过高 (21)，必须简化
- 函数 'save' () 较长 (53 行)，可考虑重构

### 33. C:\Users\paolo\Desktop\cubo\cubo\ingestion\fast_pass_ingestor.py (Score: 50.75)
**Issue Categories**: 🔄 Complexity Issues:2, ⚠️ Other Issues:1

**Main Issues**:
- Function ingest_folder has very high cyclomatic complexity (22), consider refactoring
- 函数 'ingest_folder' () 过长 (103 行)，建议拆分
- 函数 'ingest_folder' () 复杂度严重过高 (22)，必须简化

### 34. C:\Users\paolo\Desktop\cubo\cubo\indexing\index_publisher.py (Score: 50.30)
**Issue Categories**: 🔄 Complexity Issues:4, ⚠️ Other Issues:2

**Main Issues**:
- Function _verify_index_dir has very high cyclomatic complexity (28), consider refactoring
- Function rollback_to_previous has very high cyclomatic complexity (22), consider refactoring
- 函数 '_verify_index_dir' () 较长 (70 行)，可考虑重构
- 函数 '_verify_index_dir' () 复杂度严重过高 (28)，必须简化
- 函数 'rollback_to_previous' () 较长 (69 行)，可考虑重构
- 函数 'rollback_to_previous' () 复杂度严重过高 (22)，必须简化

### 35. C:\Users\paolo\Desktop\cubo\scripts\run_reranker_eval.py (Score: 50.28)
**Issue Categories**: 📝 Comment Issues:1

**Main Issues**:
- Code comment ratio is low (5.94%), consider adding more comments

### 36. C:\Users\paolo\Desktop\cubo\scripts\audit_tests.py (Score: 50.23)
**Issue Categories**: 🔄 Complexity Issues:3, 📝 Comment Issues:1, ⚠️ Other Issues:2

**Main Issues**:
- Function visit_Call has high cyclomatic complexity (12), consider simplifying
- Function main has very high cyclomatic complexity (16), consider refactoring
- 函数 'visit_FunctionDef' () 较长 (42 行)，可考虑重构
- 函数 'main' () 过长 (79 行)，建议拆分
- 函数 'main' () 复杂度过高 (16)，建议简化
- Code comment ratio is low (8.05%), consider adding more comments

### 37. C:\Users\paolo\Desktop\cubo\scripts\extract_system_metrics.py (Score: 49.95)
**Issue Categories**: 🔄 Complexity Issues:2, ⚠️ Other Issues:1

**Main Issues**:
- 函数 'extract_system_metrics' () 较长 (59 行)，可考虑重构
- 函数 'extract_system_metrics' () 复杂度过高 (13)，建议简化
- Function extract_system_metrics has high cyclomatic complexity (13), consider simplifying

### 38. C:\Users\paolo\Desktop\cubo\cubo\ingestion\file_loader.py (Score: 49.94)
**Issue Categories**: 🔄 Complexity Issues:2, ⚠️ Other Issues:1

**Main Issues**:
- Function _load_csv has high cyclomatic complexity (15), consider simplifying
- 函数 '_load_csv' () 较长 (43 行)，可考虑重构
- 函数 '_load_csv' () 复杂度过高 (15)，建议简化

### 39. C:\Users\paolo\Desktop\cubo\cubo\utils\logger.py (Score: 49.83)
**Issue Categories**: 🔄 Complexity Issues:4, ⚠️ Other Issues:2

**Main Issues**:
- Function _get_formatter has high cyclomatic complexity (12), consider simplifying
- Function _setup_handlers has high cyclomatic complexity (11), consider simplifying
- Function _setup_logging has very high cyclomatic complexity (25), consider refactoring
- 函数 '_setup_handlers' () 较长 (41 行)，可考虑重构
- 函数 '_setup_logging' () 过长 (113 行)，建议拆分
- 函数 '_setup_logging' () 复杂度严重过高 (25)，必须简化

### 40. C:\Users\paolo\Desktop\cubo\scripts\rrf_sensitivity_sweep.py (Score: 49.53)
**Issue Categories**: 🔄 Complexity Issues:3, ⚠️ Other Issues:1

**Main Issues**:
- Function run_sweep_for_dataset has very high cyclomatic complexity (17), consider refactoring
- Function main has high cyclomatic complexity (11), consider simplifying
- 函数 'run_sweep_for_dataset' () 较长 (65 行)，可考虑重构
- 函数 'run_sweep_for_dataset' () 复杂度过高 (17)，建议简化

### 41. C:\Users\paolo\Desktop\cubo\scripts\validate_faiss_index.py (Score: 49.52)
**Issue Categories**: ⚠️ Other Issues:1

**Main Issues**:
- 函数 'main' () 过长 (83 行)，建议拆分

### 42. C:\Users\paolo\Desktop\cubo\scripts\scrub_logs.py (Score: 49.37)
**Issue Categories**: 🔄 Complexity Issues:1, 📝 Comment Issues:1

**Main Issues**:
- Function scrub_line_json has high cyclomatic complexity (12), consider simplifying
- Code comment ratio is low (9.38%), consider adding more comments

### 43. C:\Users\paolo\Desktop\cubo\tests\e2e\conftest.py (Score: 49.20)
**Issue Categories**: 🔄 Complexity Issues:2, ⚠️ Other Issues:1

**Main Issues**:
- Function manage_servers has very high cyclomatic complexity (22), consider refactoring
- 函数 'manage_servers' () 过长 (72 行)，建议拆分
- 函数 'manage_servers' () 复杂度严重过高 (22)，必须简化

### 44. C:\Users\paolo\Desktop\cubo\cubo\retrieval\router.py (Score: 47.77)
**Issue Categories**: ⚠️ Other Issues:1

**Main Issues**:
- 函数 'route_query' () 过长 (84 行)，建议拆分

### 45. C:\Users\paolo\Desktop\cubo\cubo\ingestion\hierarchical_chunker.py (Score: 47.58)
**Issue Categories**: 🔄 Complexity Issues:4, ⚠️ Other Issues:2

**Main Issues**:
- Function _simple_chunk has very high cyclomatic complexity (46), consider refactoring
- Function save_chunk has high cyclomatic complexity (14), consider simplifying
- 函数 '_simple_chunk' () 极度过长 (185 行)，必须拆分
- 函数 '_simple_chunk' () 复杂度严重过高 (46)，必须简化
- 函数 'save_chunk' () 较长 (56 行)，可考虑重构
- 函数 'save_chunk' () 复杂度过高 (14)，建议简化

### 46. C:\Users\paolo\Desktop\cubo\cubo\retrieval\bm25_sqlite_store.py (Score: 47.42)
**Issue Categories**: 🔄 Complexity Issues:2, ⚠️ Other Issues:3

**Main Issues**:
- Function search has very high cyclomatic complexity (43), consider refactoring
- 函数 'index_documents' () 较长 (52 行)，可考虑重构
- 函数 'add_documents' () 较长 (42 行)，可考虑重构
- 函数 'search' () 过长 (103 行)，建议拆分
- 函数 'search' () 复杂度严重过高 (43)，必须简化

### 47. C:\Users\paolo\Desktop\cubo\scripts\run_reranker_and_system_metrics_all.py (Score: 47.19)

### 48. C:\Users\paolo\Desktop\cubo\scripts\download_and_prepare.py (Score: 47.12)
**Issue Categories**: 🔄 Complexity Issues:7, ⚠️ Other Issues:4

**Main Issues**:
- 函数 '_download_file' () 较长 (47 行)，可考虑重构
- 函数 'download_dataset' () 过长 (120 行)，建议拆分
- 函数 'download_dataset' () 复杂度严重过高 (42)，必须简化
- 函数 '_handle_manual_download' () 较长 (49 行)，可考虑重构
- 函数 '_handle_manual_download' () 复杂度过高 (14)，建议简化
- 函数 'verify_model' () 较长 (64 行)，可考虑重构
- 函数 'verify_model' () 复杂度过高 (13)，建议简化
- Function _download_file has high cyclomatic complexity (12), consider simplifying
- Function download_dataset has very high cyclomatic complexity (42), consider refactoring
- Function _handle_manual_download has high cyclomatic complexity (14), consider simplifying
- Function verify_model has high cyclomatic complexity (13), consider simplifying

### 49. C:\Users\paolo\Desktop\cubo\cubo\retrieval\bm25_python_store.py (Score: 47.02)
**Issue Categories**: 🔄 Complexity Issues:6, ⚠️ Other Issues:2

**Main Issues**:
- Function add_documents has very high cyclomatic complexity (16), consider refactoring
- Function _tokenize has very high cyclomatic complexity (18), consider refactoring
- Function search has very high cyclomatic complexity (34), consider refactoring
- 函数 'add_documents' () 较长 (48 行)，可考虑重构
- 函数 'add_documents' () 复杂度过高 (16)，建议简化
- 函数 '_tokenize' () 复杂度过高 (18)，建议简化
- 函数 'search' () 过长 (118 行)，建议拆分
- 函数 'search' () 复杂度严重过高 (34)，必须简化

### 50. C:\Users\paolo\Desktop\cubo\cubo\utils\cpu_features.py (Score: 46.95)
**Issue Categories**: 🔄 Complexity Issues:3, ⚠️ Other Issues:1

**Main Issues**:
- Function get_topology has high cyclomatic complexity (11), consider simplifying
- Function detect_blas_backend has very high cyclomatic complexity (20), consider refactoring
- 函数 'detect_blas_backend' () 较长 (51 行)，可考虑重构
- 函数 'detect_blas_backend' () 复杂度严重过高 (20)，必须简化

## Improvement Suggestions

### High Priority
- Keep up the clean code standards, don't let the mess creep in

### Medium Priority
- Go further—optimize for performance and readability, just because you can
- Polish your docs and comments, make your team love you even more

