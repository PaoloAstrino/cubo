# Testing Guide for Processing Module

## Overview

This guide covers testing for the `src.cubo.processing` module, including unit tests, performance tests, and offline LLM validation.

## Test Structure

```
tests/
├── processing/
│   ├── test_enrichment.py              # ChunkEnricher tests
│   ├── test_postprocessor.py           # WindowReplacementPostProcessor tests
│   ├── test_scaffold_generator.py      # ScaffoldGenerator tests
│   ├── test_scaffold_persist.py        # Persistence tests
│   └── test_create_scaffolds_parquet.py # End-to-end tests
└── performance/
    └── test_scaffold_memory.py         # Memory/performance tests
```

## Running Tests

### All Processing Tests

```bash
pytest tests/processing/ -v
```

### Specific Component

```bash
# ChunkEnricher
pytest tests/processing/test_enrichment.py -v

# ScaffoldGenerator
pytest tests/processing/test_scaffold_generator.py -v

# WindowReplacementPostProcessor
pytest tests/processing/test_postprocessor.py -v
```

### Performance Tests

```bash
pytest tests/performance/test_scaffold_memory.py -v
```

### With Coverage

```bash
pytest tests/processing/ --cov=src.cubo.processing --cov-report=html
```

## Test Coverage

### ChunkEnricher (`test_enrichment.py`)

✅ **Covered:**
- Prompt generation for summaries, keywords, categories
- Output parsing and normalization
- Self-consistency scoring
- ROUGE score validation
- F1 score for keyword extraction
- Graceful failure handling

**Tests:**
- `test_enrich_chunks`: End-to-end enrichment with mock LLM

### ScaffoldGenerator (`test_scaffold_generator.py`)

✅ **Covered:**
- Basic scaffold generation
- Chunk grouping by similarity
- Summary merging logic
- Scaffold ID generation
- Empty DataFrame handling
- Save/load persistence

**Tests:**
- `test_generate_scaffolds_basic`
- `test_scaffold_grouping_logic`
- `test_merge_summaries`
- `test_generate_scaffold_id`
- `test_save_and_load_scaffolds`

### WindowReplacementPostProcessor (`test_postprocessor.py`)

✅ **Covered:**
- Window context replacement
- Missing window handling
- Empty window handling
- Multiple results processing
- Custom metadata keys

**Tests:**
- `test_replace_with_window_context`
- `test_no_window_in_metadata`
- `test_empty_window_context`
- `test_multiple_results`
- `test_custom_metadata_key`

### Performance Tests (`test_scaffold_memory.py`)

✅ **Covered:**
- Memory usage with 1K chunks (<50MB)
- Memory usage with 10K chunks (<200MB)
- Processing time scalability
- Batch processing for large datasets
- Enrichment failure handling

**Tests:**
- `test_memory_usage_small_dataset`
- `test_memory_usage_large_dataset`
- `test_processing_time_scalability`
- `test_batch_processing`
- `test_enrichment_failure_handling`

## Offline LLM Testing

### Setup

1. **Install llama-cpp-python:**
   ```bash
   pip install llama-cpp-python
   ```

2. **Download a test model:**
   ```bash
   # Small model for testing (~1.5GB)
   wget https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf -O test_model.gguf
   ```

3. **Configure for testing:**
   ```python
   # tests/processing/test_llm_local.py
   from src.cubo.processing.llm_local import LocalResponseGenerator
   
   llm = LocalResponseGenerator(model_path="test_model.gguf")
   response = llm.generate_response("Test query", "Test context")
   ```

### Manual Verification

```python
# Test offline enrichment
from src.cubo.processing.llm_local import LocalResponseGenerator
from src.cubo.processing.enrichment import ChunkEnricher

llm = LocalResponseGenerator(model_path="path/to/model.gguf")
enricher = ChunkEnricher(llm_provider=llm)

chunks = ["This is a test chunk about AI and machine learning."]
enriched = enricher.enrich_chunks(chunks)

print(enriched[0]['summary'])
print(enriched[0]['keywords'])
print(enriched[0]['category'])
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Processing Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -e .
          pip install pytest pytest-cov psutil
      
      - name: Run unit tests
        run: pytest tests/processing/ -v --cov=src.cubo.processing
      
      - name: Run performance tests
        run: pytest tests/performance/ -v
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

## Benchmarking

### Performance Baselines

**ChunkEnricher:**
- 1 chunk: ~1-2s (LLM dependent)
- 10 chunks: ~10-20s
- 100 chunks: ~100-200s

**ScaffoldGenerator:**
- 1K chunks: ~30-60s
- 10K chunks: ~5-10min
- Memory: ~15MB per 10K chunks

**WindowReplacementPostProcessor:**
- 1K results: <100ms
- 10K results: <500ms
- Memory: negligible

### Profiling

```bash
# Time profiling
python -m cProfile -o profile.stats scripts/test_processing.py
python -m pstats profile.stats

# Memory profiling
python -m memory_profiler scripts/test_processing.py
```

## Common Issues

### Issue: Tests fail with "no module named llama_cpp"

**Solution**: Mock the `llama_cpp` import:
```python
@unittest.mock.patch('src.cubo.processing.llm_local.Llama', create=True)
def test_something(self, mock_llama):
    # test code
```

### Issue: Performance tests are slow

**Solution**: Use smaller datasets or skip in CI:
```python
@unittest.skipIf(os.getenv('CI'), "Skip slow tests in CI")
def test_large_dataset(self):
    ...
```

### Issue: Memory tests fail in CI

**Solution**: Account for CI environment overhead:
```python
# Instead of strict limit:
self.assertLess(mem_used, 50.0)

# Use tolerance:
self.assertLess(mem_used, 50.0 * 1.5)  # 50% tolerance
```

## Best Practices

1. **Mock LLM calls**: Use `MagicMock` for unit tests
2. **Test edge cases**: Empty inputs, failures, malformed data
3. **Verify output format**: Check all expected fields exist
4. **Test persistence**: Save/load scaffolds in temp directories
5. **Monitor performance**: Use `psutil` to track memory
6. **Isolate tests**: Each test should be independent

## Extending Tests

### Adding New Enrichment Test

```python
def test_new_enrichment_feature(self):
    mock_llm = MagicMock()
    mock_llm.generate_response.return_value = "Expected output"
    
    enricher = ChunkEnricher(llm_provider=mock_llm)
    result = enricher.new_method(input_data)
    
    self.assertEqual(result, expected_output)
```

### Adding Performance Benchmark

```python
import time

def test_new_performance_metric(self):
    start = time.time()
    # operation to benchmark
    elapsed = time.time() - start
    
    self.assertLess(elapsed, threshold)
```

## Resources

- [pytest documentation](https://docs.pytest.org/)
- [unittest.mock guide](https://docs.python.org/3/library/unittest.mock.html)
-  [psutil documentation](https://psutil.readthedocs.io/)
- [Coverage.py](https://coverage.readthedocs.io/)
