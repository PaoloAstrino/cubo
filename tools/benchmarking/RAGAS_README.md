# RAGAS Evaluation Implementation Guide

## Overview
RAGAS evaluation in CUBO is designed for end-to-end RAG quality assessment with **local judge LLM** and **deterministic behavior**. This guide covers setup, usage, and implementation details.

---

## Critical Requirement: Ollama Must Be Running

**Ollama is MANDATORY** for RAGAS evaluation because:
1. It generates local LLM responses during the RAG pipeline (retrieval + generation)
2. It serves as the judge LLM for evaluating faithfulness, context precision, etc.
3. Evaluation cannot proceed without it

### Starting Ollama

#### Windows/PowerShell:
```powershell
# Open a new terminal and run:
ollama serve

# You should see:
# Listening on 127.0.0.1:11434
```

#### macOS/Linux:
```bash
ollama serve
# OR if using launchctl:
launchctl start ollama

# Verify:
curl http://127.0.0.1:11434/api/tags
```

**Do not proceed with RAGAS evaluation until Ollama is running.**

---

## Basic Usage

### Minimal Example (Local Judge)
```bash
python tools/run_generation_eval.py \
  --dataset politics \
  --num-samples 10
```

### Full Example (OpenAI Judge + Per-Sample Saving)
```bash
export OPENAI_API_KEY="sk-..."

python tools/run_generation_eval.py \
  --dataset scifact \
  --num-samples 50 \
  --judge openai \
  --judge-model gpt-4 \
  --judge-temperature 0.0 \
  --judge-retries 3 \
  --save-per-sample-raw \
  --output results/generation_eval/scifact
```

---

## CLI Flags (New in This Implementation)

### Judge Configuration
- `--judge {local|openai}` (default: local)
  - **local**: Uses Ollama (requires Ollama running)
  - **openai**: Uses OpenAI (requires `OPENAI_API_KEY` env var)

- `--judge-temperature FLOAT` (default: 0.0)
  - Controls judge LLM stochasticity
  - **0.0**: Deterministic (recommended for reproducible evaluations)
  - **>0.0**: Non-deterministic (higher = more creative)
  - Only applies to OpenAI judge

- `--judge-retries INT` (default: 2)
  - Number of retry attempts if judge returns non-JSON output
  - Higher values = more tolerance for parsing errors

- `--judge-request-timeout INT` (default: 120 seconds)
  - Timeout for judge LLM requests

### Output Control
- `--save-per-sample-raw` (flag, default: OFF)
  - Opt-in: saves per-sample RAGAS outputs to JSONL
  - Output file: `{output_dir}/{dataset}_ragas_raw.jsonl`
  - Includes: question, contexts, answer, ground_truth, retrieval_time, generation_time, per-sample metrics
  - Useful for auditing, debugging, and per-sample analysis

### Other Flags
- `--dataset {politics|legal|nfcorpus|fiqa|scifact}` (default: politics)
- `--num-samples INT` (default: 50)
- `--output PATH` (default: results/generation_eval)
- `--top-k INT` (default: 3) - Number of retrieved chunks for generation
- `--max-context-chars INT` (default: 4000) - Truncate long chunks
- `--max-total-context-chars INT` (default: 12000) - Total context budget
- `--ragas-max-workers INT` - Parallel evaluation workers (1 = serial, recommended for local judge)
- `--disable-judge-retry` - Skip JSON retry wrapper (not recommended)

---

## Output Files

After running RAGAS evaluation, three files are created:

### 1. `{dataset}_ragas_results.json` (Aggregate Metrics)
```json
{
  "dataset": "politics",
  "num_samples": 50,
  "ragas_scores": {
    "faithfulness": 0.78,
    "context_precision": 0.85,
    "context_recall": 0.92,
    "answer_relevancy": 0.81
  },
  "avg_rag_latency_s": 2.34,
  "ragas_eval_time_s": 142.56,
  "timestamp": "2026-01-21 11:30:45"
}
```

### 2. `{dataset}_ragas_manifest.json` (Provenance)
```json
{
  "dataset": "politics",
  "num_samples": 50,
  "ragas_scores": { ... },
  "judge": "openai",
  "judge_model": "gpt-4",
  "judge_temperature": 0.0,
  "llm_provider": "ollama",
  "llm_model": "llama2",
  "top_k": 3,
  "git_sha": "abc1234",
  "timestamp": "2026-01-21 11:30:45"
}
```

### 3. `{dataset}_ragas_raw.jsonl` (Per-Sample Details, Opt-In)
Only created if `--save-per-sample-raw` is passed.

```jsonl
{"sample_id": 0, "question": "What is GDPR Article 5?", "contexts": ["...", "..."], "ground_truth": "...", "answer": "...", "retrieval_time": 0.12, "generation_time": 0.45, "ragas_faithfulness": 0.8, "ragas_context_precision": 0.85}
{"sample_id": 1, "question": "...", ...}
```

---

## RAGAS Metrics Explained

### Faithfulness
- Measures how much the generated answer is grounded in the retrieved context
- Range: 0–1 (higher is better)
- Judge asks: "Does the answer follow from the context?"

### Context Precision
- Measures the relevance of retrieved contexts to the question
- Range: 0–1 (higher is better)
- Judge asks: "Are all contexts relevant to answering the question?"

### Context Recall
- Measures whether all necessary information to answer the question is in the context
- Range: 0–1 (higher is better)
- Judge asks: "Is enough information present in the context to answer?"

### Answer Relevancy
- Measures whether the generated answer directly addresses the question
- Range: 0–1 (higher is better)
- Judge asks: "Does the answer address the question?"

---

## Architecture & Components

### Local Judge (Ollama)
- Default behavior
- Uses `CuboRagasLLM` wrapper that calls Ollama via CUBO's generator
- Wrapped with `RetryingChatLLM` to handle non-JSON outputs
- Debug artifacts saved to `{output_dir}/judge_retry_debug/`

### OpenAI Judge
- Requires `OPENAI_API_KEY` environment variable
- Uses `ChatOpenAI` from LangChain
- Also wrapped with `RetryingChatLLM` for robustness
- Temperature control available via `--judge-temperature`

### Per-Sample Aggregation
- Serial mode (`--ragas-max-workers 1`): evaluates each sample individually and averages metrics
- Parallel mode (default): delegates to RAGAS library's built-in parallelism

### Latency Tracking
- Retrieval latency: time from query to returned chunks
- Generation latency: time from retrieved context to generated answer
- Both are per-sample and included in JSONL output

---

## Best Practices

### For Reproducible Evaluations
1. Always use `--judge-temperature 0.0` (deterministic)
2. Use OpenAI judge (gpt-4) for stable, calibrated evaluation
3. Save git commit in manifest: check `git_sha` field
4. Use `--ragas-max-workers 1` if local judge is slow or times out

### For Auditability
1. Always use `--save-per-sample-raw` for detailed inspection
2. Keep manifests with results for provenance
3. Review per-sample metrics to identify failure modes
4. Compare runs using git SHAs

### For Development & Debugging
1. Start with small sample sizes (`--num-samples 5`)
2. Check judge retry debug files if metrics seem off
3. Monitor latencies to identify bottlenecks
4. Use serial mode if parallel evaluation produces inconsistent results

---

## Testing

All RAGAS components are tested:
```bash
# Run unit tests (no Ollama required)
pytest tests/evaluation/test_ragas_evaluator.py -v
pytest tests/evaluation/test_ragas_raw_output_save.py -v
pytest tests/evaluation/test_openai_judge_creation.py -v
pytest tests/evaluation/test_retrying_chat_llm.py -v

# Run smoke CLI test (requires Ollama)
pytest tests/evaluation/test_ragas_smoke_cli.py -v
```

---

## Implementation Details

### Files Modified
- `evaluation/ragas_evaluator.py` — RAGAS wrappers, aggregation, per-sample saving
- `tools/run_generation_eval.py` — CLI, Ollama checks, latency tracking
- `.github/workflows/ragas.yml` — CI workflow for unit tests

### Key Classes
- `CuboRagasLLM`: Adapts CUBO generator to RAGAS/LangChain interface
- `RetryingChatLLM`: Retries judge requests on JSON parse failures
- `CuboRagasEmbeddings`: Adapts CUBO embeddings to RAGAS interface

### Ollama Checks
1. Initial check before loading queries (fails hard if unavailable)
2. Per-sample health check during RAG pipeline (ensures no mid-evaluation crashes)
3. Final check before RAGAS judge phase (prevents incomplete evaluations)

---

## Troubleshooting

### "Ollama is NOT running"
- Start Ollama in a separate terminal: `ollama serve`
- Verify: `curl http://127.0.0.1:11434/api/tags` (should list available models)
- Wait 5–10 seconds for startup before retrying

### "No successful RAG samples collected"
- Check retrieval: `--top-k 5` (increase if too few chunks)
- Check context: `--max-context-chars 5000` (increase if truncating too much)
- Check Ollama model availability: `ollama list`

### "RetryingChatLLM exhausted retries"
- Increase `--judge-retries` (default 2, try 3–5)
- Check judge response in `judge_retry_debug/` files
- Consider using OpenAI judge instead of local for stable JSON

### "Latencies seem too high"
- Monitor Ollama: `nvidia-smi` (if using GPU)
- Check disk I/O for mmap vector store
- Reduce `--max-context-chars` to lower per-sample load

---

## Roadmap

Future improvements:
- [ ] Judge calibration pipeline (alignment verification)
- [ ] Retrieval-only micro-metrics (recall@k, MRR)
- [ ] Multi-reference ground truth support
- [ ] Statistical significance testing (bootstrap CI)
- [ ] Per-sample judge reasoning saved to JSONL

---

**Status**: ✅ Production-ready for end-to-end RAG evaluation with local or OpenAI judges.

For questions or issues, see [the_final_analysis.md](../the_final_analysis.md) or [resolution_plan_critique_1.txt](../resolution_plan_critique_1.txt).
