"""
Generation Quality Evaluation Script using Local RAGAS.

This script evaluates end-to-end RAG quality (Faithfulness, Answer Relevancy, 
Context Precision) using CUBO's local LLM as the judge model.

Usage:
    python tools/run_generation_eval.py --dataset politics --num-samples 50
"""

import argparse
import json
import logging
import sys
import subprocess
import socket
from pathlib import Path
from typing import List, Dict, Any
import time

from cubo.core import CuboCore
from cubo.utils.logger import logger as app_logger
from evaluation.ragas_evaluator import run_ragas_evaluation

logger = app_logger


def load_test_queries(dataset_name: str, num_samples: int = 50) -> List[Dict[str, Any]]:
    """Load test queries from BEIR or UltraDomain datasets."""
    # Map dataset names to file paths
    dataset_paths = {
        "politics": "data/ultradomain/politics.jsonl",
        "legal": "data/ultradomain/legal.jsonl",
        "nfcorpus": "data/beir/nfcorpus/queries.jsonl",
        "fiqa": "data/beir/fiqa/queries.jsonl",
        "scifact": "data/beir/scifact/queries.jsonl",
    }
    
    if dataset_name not in dataset_paths:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(dataset_paths.keys())}")
    
    queries_path = Path(dataset_paths[dataset_name])
    if not queries_path.exists():
        logger.warning(f"Dataset file not found: {queries_path}. Using synthetic fallback.")
        return generate_synthetic_queries(dataset_name, num_samples)
    
    queries = []
    with open(queries_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            data = json.loads(line)
            queries.append({
                # Support multiple query keys: 'query', 'question', 'text'
                "question": data.get("query", data.get("question", data.get("text", ""))),
                "ground_truth": data.get("answer", data.get("ground_truth", "")),
                "query_id": data.get("id", data.get("_id", f"{dataset_name}_{i}"))
            })
    return queries


def generate_synthetic_queries(dataset_name: str, num_samples: int) -> List[Dict[str, Any]]:
    """Generate synthetic queries for testing when real data is unavailable."""
    logger.info(f"Generating {num_samples} synthetic queries for {dataset_name}")
    
    templates = {
        "politics": [
            ("What is the role of the European Parliament?", "The European Parliament is the legislative body of the EU."),
            ("How are EU laws created?", "EU laws are created through the ordinary legislative procedure."),
        ],
        "legal": [
            ("What is GDPR Article 5?", "GDPR Article 5 defines principles for processing personal data."),
            ("What are data subject rights?", "Data subjects have rights to access, rectification, and erasure."),
        ],
        "nfcorpus": [
            ("What causes diabetes?", "Diabetes is caused by insufficient insulin production or insulin resistance."),
            ("How is hypertension treated?", "Hypertension is treated with lifestyle changes and medication."),
        ],
    }
    
    base_templates = templates.get(dataset_name, templates["politics"])
    queries = []
    
    for i in range(num_samples):
        template_idx = i % len(base_templates)
        question, answer = base_templates[template_idx]
        queries.append({
            "question": f"{question} (variant {i})",
            "ground_truth": answer,
            "query_id": f"synthetic_{dataset_name}_{i}"
        })
    
    return queries


def run_rag_pipeline(cubo: CuboCore, question: str, args) -> Dict[str, Any]:
    """Run full RAG pipeline: retrieve + generate."""
    import time as _time
    
    # Use public CuboCore API: query_retrieve and generate_response_safe
    retrieval_start = _time.time()
    retrieval = cubo.query_retrieve(query=question, top_k=5)
    retrieval_time = _time.time() - retrieval_start
    # `query_retrieve` returns list of chunk dicts; inspect raw retrieval for debugging
    try:
        logger.info(f"Raw retrieval count: {len(retrieval)}")
        for idx, chunk in enumerate(retrieval[:3]):
            logger.info(f"Raw chunk[{idx}] keys={list(chunk.keys())}")
            # show if 'document' present and its length
            doc_val = chunk.get('document') or chunk.get('text') or chunk.get('content') or ''
            logger.info(f"Raw chunk[{idx}] document_len={len(doc_val) if isinstance(doc_val,str) else 'n/a'}")
    except Exception:
        pass

    contexts = []
    for chunk in retrieval[: args.top_k]:
        text = (
            chunk.get("text")
            or chunk.get("content")
            or chunk.get("document")
            or chunk.get("doc_text")
            or ""
        )
        text = text.strip() if isinstance(text, str) else ""
        if text:
            # Truncate per-chunk to avoid enormous prompts
            if args.max_context_chars and len(text) > args.max_context_chars:
                logger.info(f"Truncating context chunk from {len(text)} to {args.max_context_chars} chars")
                text = text[: args.max_context_chars]
            contexts.append(text)

    # Enforce total context budget (concatenate until max_total reached)
    if args.max_total_context_chars:
        total = 0
        truncated_contexts = []
        for t in contexts:
            if total + len(t) <= args.max_total_context_chars:
                truncated_contexts.append(t)
                total += len(t)
            else:
                remaining = args.max_total_context_chars - total
                if remaining > 0:
                    truncated_contexts.append(t[:remaining])
                    total += remaining
                break
        if len(truncated_contexts) != len(contexts):
            logger.info(f"Applied total context cap: {args.max_total_context_chars} chars (kept {len(truncated_contexts)} chunks)")
        contexts = truncated_contexts

    # Log retrieved context counts/sizes for diagnostics
    try:
        logger.info(f"Retrieved {len(contexts)} non-empty context(s) for query: {question[:60]}")
        for idx, c in enumerate(contexts[:3]):
            logger.info(f"Context[{idx}] length={len(c)} preview={c[:200]!r}")
    except Exception:
        pass

    # Generate answer using generator API (use stream to bypass ServiceManager timeouts)
    context_str = "\n\n".join(contexts)
    
    # Use streaming to align with main app behavior and avoid ServiceManager overhead
    answer = ""
    generation_start = _time.time()
    try:
        stream = cubo.generate_response_stream(query=question, context=context_str)
        for event in stream:
            if event.get("type") == "done":
                answer = event.get("answer", "")
            elif event.get("type") == "error":
                logger.error(f"Streaming error: {event.get('message')}")
        
        # Fallback if 'done' event missing but tokens arrived (though done usually carries full text)
        if not answer:
             logger.warning("No 'done' event in stream; using accumulated empty string.")
             
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        answer = ""
    
    generation_time = _time.time() - generation_start

    return {
        "contexts": contexts,
        "answer": answer,
        "retrieval_time": retrieval_time,
        "generation_time": generation_time,
    }


def main():
    parser = argparse.ArgumentParser(description="Run generation quality evaluation with local RAGAS")
    parser.add_argument("--dataset", type=str, default="politics", 
                       help="Dataset to evaluate (politics, legal, nfcorpus, fiqa)")
    parser.add_argument("--num-samples", type=int, default=50,
                       help="Number of queries to evaluate")
    parser.add_argument("--output", type=str, default="results/generation_eval",
                       help="Output directory for results")
    parser.add_argument("--judge", type=str, choices=['local','openai'], default='local',
                       help="Judge model to use for RAGAS evaluation (local or openai)")
    parser.add_argument("--judge-model", type=str, default='gpt-3.5-turbo',
                       help="OpenAI model name to use when --judge=openai")
    parser.add_argument("--judge-request-timeout", type=int, default=120,
                       help="Request timeout in seconds for judge model HTTP calls")
    parser.add_argument("--judge-retries", type=int, default=2,
                       help="Number of retry attempts to ask the judge to return valid JSON on parse failures")
    parser.add_argument("--judge-temperature", type=float, default=0.0,
                       help="Judge LLM temperature (0 for deterministic, higher for creative); default 0 for reproducibility")
    parser.add_argument("--save-per-sample-raw", action="store_true",
                       help="Save per-sample raw RAGAS outputs and latencies to JSONL (opt-in for auditability)")
    parser.add_argument("--laptop-mode", action="store_true",
                       help="Enable laptop mode (resource-constrained)")
    parser.add_argument("--llm-provider", type=str, choices=['ollama','local'], default=None,
                       help="Override LLM provider (ollama|local)")
    parser.add_argument("--llm-model", type=str, default=None,
                       help="Override LLM model name (e.g., llama3, llama3.2:latest)")

    parser.add_argument("--top-k", type=int, default=3, help="Number of contexts to include from retrieval (top_k)")
    parser.add_argument("--max-context-chars", type=int, default=4000, help="Max chars per context chunk (truncate long contexts)")
    parser.add_argument("--max-total-context-chars", type=int, default=12000, help="Max total chars for concatenated context")
    parser.add_argument("--index-dir", type=str, default=None, help="Path to existing index directory (e.g., data/beir_index_scifact)")
    parser.add_argument("--ragas-max-workers", type=int, default=None, help="Max parallel workers for RAGAS evaluation; set 1 to serialize judge calls and reduce timeouts")
    parser.add_argument("--disable-judge-retry", action="store_true", help="Disable RetryingChatLLM wrapper for judge (use base judge directly)")
    
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"🚀 Starting generation evaluation on {args.dataset} ({args.num_samples} samples)")
    
    # Initialize CUBO
    logger.info("Initializing CUBO core...")
    cubo = CuboCore()
    # Ensure components (model, retriever, generator) are ready
    if not cubo.initialize_components():
        logger.error("Failed to initialize CUBO components; aborting generation evaluation.")
        sys.exit(1)

    # Wait until retriever is available (some components may initialize asynchronously)
    import time as _time

    start_wait = _time.time()
    while True:
        if getattr(cubo, "retriever", None) and hasattr(cubo.retriever, "retrieve_top_documents"):
            break
        if _time.time() - start_wait > 60:
            logger.error("Timed out waiting for retriever to initialize; aborting.")
            sys.exit(1)
        _time.sleep(0.5)

    # Apply optional CLI overrides for LLM provider/model and index directory
    try:
        from cubo.config import config
        if args.index_dir:
            config.set("vector_store_path", args.index_dir)
            config.set("faiss_index_dir", args.index_dir)
            logger.info(f"Overriding index directory via CLI: {args.index_dir}")
        if args.llm_provider:
            config.set("llm.provider", args.llm_provider)
            logger.info(f"Overriding LLM provider via CLI: {args.llm_provider}")
        if args.llm_model:
            # Set both dotted key and legacy key for compatibility
            config.set("llm.model_name", args.llm_model)
            config.set("llm_model", args.llm_model)
            logger.info(f"Overriding LLM model via CLI: {args.llm_model}")
        provider = config.get("llm.provider", "ollama")
    except Exception:
        provider = "ollama"

    # CRITICAL: Ollama is REQUIRED for RAGAS evaluation (generates local LLM responses to be evaluated)
    logger.info("\n" + "=" * 80)
    logger.info("🔍 RAGAS Evaluation Requirement: Ollama LLM Service")
    logger.info("=" * 80)
    logger.info("Ollama is REQUIRED to generate local responses for RAGAS evaluation.")
    logger.info("Without Ollama, generation will fail and evaluation cannot proceed.\n")
    
    if provider == "ollama":
        def is_ollama_running(timeout=3, retries=2):
            """Check if Ollama is running with retry logic."""
            for attempt in range(retries):
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(timeout)
                    result = sock.connect_ex(("127.0.0.1", 11434))
                    sock.close()
                    if result == 0:
                        return True
                except Exception as e:
                    logger.debug(f"Ollama check attempt {attempt+1} failed: {e}")
                time.sleep(1)  # Brief pause before retry
            return False
        
        logger.info("Checking Ollama availability (with retries)...")
        if not is_ollama_running(retries=3):
            logger.error("\n" + "=" * 80)
            logger.error("❌ CRITICAL: Ollama is NOT running!")
            logger.error("=" * 80)
            logger.error("RAGAS evaluation CANNOT proceed without Ollama.")
            logger.error("\nPlease start Ollama in a separate terminal:")
            logger.error("")
            logger.error("  PowerShell/CMD:")
            logger.error("    1. Open a new terminal window")
            logger.error("    2. Run: ollama serve")
            logger.error("    3. Wait for 'Listening on 127.0.0.1:11434' message")
            logger.error("    4. Return to this window and run the evaluation again")
            logger.error("")
            logger.error("  macOS/Linux:")
            logger.error("    1. Run: ollama serve (or use: launchctl start ollama)")
            logger.error("    2. Verify: curl http://127.0.0.1:11434/api/tags")
            logger.error("")
            logger.error("Download Ollama: https://ollama.com/download")
            logger.error("=" * 80 + "\n")
            sys.exit(1)
        else:
            logger.info("✅ Ollama is RUNNING and READY for RAGAS evaluation!\n")

    # Load test queries
    logger.info(f"Loading test queries from {args.dataset}...")
    test_queries = load_test_queries(args.dataset, args.num_samples)
    logger.info(f"Loaded {len(test_queries)} queries")

    # Run RAG pipeline for each query
    logger.info("Running RAG pipeline...")
    questions = []
    contexts_list = []
    answers = []
    ground_truths = []
    retrieval_times = []
    generation_times = []

    start_time = time.time()
    for i, query_data in enumerate(test_queries):
        logger.info(f"Processing query {i+1}/{len(test_queries)}: {query_data['question'][:50]}...")

        try:
            # Health check: ensure Ollama is still available before each query
            if provider == "ollama":
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(2)
                    result = sock.connect_ex(("127.0.0.1", 11434))
                    sock.close()
                    if result != 0:
                        logger.error(f"⚠️  Ollama became unavailable during evaluation!")
                        logger.error("Please restart Ollama and re-run the evaluation.")
                        sys.exit(1)
                except Exception as e:
                    logger.error(f"⚠️  Ollama health check failed: {e}")
                    sys.exit(1)
            
            result = run_rag_pipeline(cubo, query_data["question"], args)

            # Validate result
            if not result or not result.get("answer") or not result.get("contexts"):
                logger.warning(f"Skipping query {i} due to empty retrieval/generation result")
                continue

            questions.append(query_data["question"])
            contexts_list.append(result["contexts"])
            answers.append(result["answer"])
            ground_truths.append(query_data["ground_truth"])
            retrieval_times.append(result.get("retrieval_time", 0.0))
            generation_times.append(result.get("generation_time", 0.0))

        except Exception as e:
            logger.error(f"Failed to process query {i}: {e}")
            continue

    total_time = time.time() - start_time
    logger.info(f"RAG pipeline completed in {total_time:.1f}s ({(total_time/len(test_queries)) if len(test_queries)>0 else 0:.2f}s per query)")

    # Ensure we have at least one successful sample before running RAGAS
    if len(questions) == 0:
        logger.error("No successful RAG samples collected; aborting RAGAS evaluation.")
        sys.exit(1)

    # CRITICAL: Final Ollama health check before RAGAS evaluation phase
    logger.info("\n" + "=" * 80)
    logger.info("📊 RAGAS Evaluation Phase: Final Ollama Verification")
    logger.info("=" * 80)
    
    if provider == "ollama":
        logger.info("Verifying Ollama is still available for RAGAS judge evaluation...")
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3)
            result = sock.connect_ex(("127.0.0.1", 11434))
            sock.close()
            if result != 0:
                logger.error("❌ CRITICAL: Ollama is no longer available!")
                logger.error("Ollama is required for the judge LLM during RAGAS evaluation.")
                logger.error("Please restart Ollama and re-run the evaluation.")
                sys.exit(1)
        except Exception as e:
            logger.error(f"❌ CRITICAL: Ollama verification failed: {e}")
            sys.exit(1)
        logger.info("✅ Ollama verified and ready for RAGAS judge evaluation!\n")

    # Auto-start and warm up Ollama before RAGAS evaluation
    logger.info("=" * 80)
    logger.info("RAGAS EVALUATION PHASE")
    logger.info("=" * 80)
    
    if provider == "ollama":
        logger.info("Verifying Ollama is still available for RAGAS evaluation...")
        # Define check function locally
        def check_ollama(timeout=3):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(timeout)
                result = sock.connect_ex(("127.0.0.1", 11434))
                sock.close()
                return result == 0
            except Exception as e:
                logger.debug(f"Ollama check failed: {e}")
                return False
        
        if not check_ollama():
            logger.error("❌ Ollama is not available!")
            logger.error("Please ensure 'ollama serve' is running in another terminal.")
            sys.exit(1)
        logger.info("✓ Ollama verified")

    # Optionally create an external judge LLM (OpenAI)
    judge_llm = None
    if args.judge == 'openai':
        # Lazy import to avoid hard dependency unless requested
        try:
            from langchain.chat_models import ChatOpenAI
        except Exception:
            logger.error("OpenAI judge requested but langchain.chat_models.ChatOpenAI unavailable. Install 'langchain' and dependencies.")
            raise
        import os
        if not os.environ.get('OPENAI_API_KEY'):
            logger.error("OPENAI_API_KEY not set in environment. Set the key to use OpenAI as judge or use --judge=local.")
            raise RuntimeError("OPENAI_API_KEY not set")
        # Create ChatOpenAI and wrap with a retrying wrapper that requests JSON-only if parsing fails
        base_judge = ChatOpenAI(
            model_name=args.judge_model, 
            temperature=args.judge_temperature, 
            request_timeout=args.judge_request_timeout
        )
        try:
            # Import the retry wrapper from our evaluator module, unless disabled
            if getattr(args, 'disable_judge_retry', False):
                logger.info("Judge retry wrapper disabled via CLI; using base judge directly")
                judge_llm = base_judge
            else:
                from evaluation.ragas_evaluator import RetryingChatLLM
                # Create debug directory for retry artifacts
                debug_dir = output_dir / "judge_retry_debug"
                judge_llm = RetryingChatLLM(
                    wrapped=base_judge, 
                    max_retries=args.judge_retries,
                    debug_dir=str(debug_dir)
                )
                logger.info(f"RetryingChatLLM enabled with max_retries={args.judge_retries}, temp={args.judge_temperature}, debug_dir={debug_dir}")
        except Exception as e:
            logger.warning(f"Failed to wrap judge with RetryingChatLLM: {e}; using base judge")
            judge_llm = base_judge

    try:
        ragas_start = time.time()
        
        # Prepare optional per-sample raw output path
        per_sample_path = None
        if args.save_per_sample_raw:
            per_sample_path = str(output_dir / f"{args.dataset}_ragas_raw.jsonl")
            logger.info(f"Per-sample raw outputs will be saved to {per_sample_path}")
        
        scores = run_ragas_evaluation(
            questions=questions,
            contexts=contexts_list,
            answers=answers,
            ground_truths=ground_truths,
            llm=judge_llm,
            max_workers=getattr(args, 'ragas_max_workers', None),
            save_per_sample_path=per_sample_path,
            retrieval_times=retrieval_times,
            generation_times=generation_times,
        )
        ragas_time = time.time() - ragas_start
        logger.info(f"RAGAS evaluation completed in {ragas_time:.1f}s")

        # Normalize scores to a mapping (RAGAS may return an EvaluationResult object)
        scores_map = None
        try:
            if hasattr(scores, "items"):
                scores_map = dict(scores)
            elif hasattr(scores, "to_dict"):
                scores_map = scores.to_dict()
            elif hasattr(scores, "dict"):
                scores_map = scores.dict()
            else:
                try:
                    scores_map = dict(scores)
                except Exception:
                    scores_map = {"raw": str(scores)}
        except Exception as e:
            logger.warning(f"Failed to normalize RAGAS scores: {e}; saving raw representation")
            scores_map = {"raw": str(scores)}

        # Print results
        print("\n" + "="*60)
        print(f"RAGAS Evaluation Results ({args.dataset})")
        print("="*60)
        for metric, value in scores_map.items():
            try:
                print(f"  {metric:25s}: {float(value):.4f}")
            except Exception:
                print(f"  {metric:25s}: {value}")
        print("="*60)

        # Save results
        results = {
            "dataset": args.dataset,
            "num_samples": len(questions),
            "ragas_scores": {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in scores_map.items()},
            "avg_rag_latency_s": total_time / len(test_queries),
            "ragas_eval_time_s": ragas_time,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        output_file = output_dir / f"{args.dataset}_ragas_results.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"✅ Results saved to {output_file}")

        # Save manifest/provenance for reproducibility
        try:
            import subprocess
            git_sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
        except Exception:
            git_sha = None
        manifest = {
            "dataset": args.dataset,
            "num_samples": len(questions),
            "ragas_scores": results.get("ragas_scores"),
            "judge": args.judge,
            "judge_model": args.judge_model if hasattr(args, 'judge_model') else None,
            "llm_provider": args.llm_provider,
            "llm_model": args.llm_model,
            "top_k": args.top_k,
            "max_context_chars": args.max_context_chars,
            "max_total_context_chars": args.max_total_context_chars,
            "git_sha": git_sha,
            "timestamp": results.get("timestamp"),
        }
        manifest_file = output_dir / f"{args.dataset}_ragas_manifest.json"
        with open(manifest_file, "w") as f:
            json.dump(manifest, f, indent=2)
        logger.info(f"Manifest saved to {manifest_file}")

    except Exception as e:
        logger.error(f"RAGAS evaluation failed: {e}")
        # Save exception to a file for debugging
        try:
            import traceback
            err_file = output_dir / f"{args.dataset}_ragas_error.txt"
            with open(err_file, "w", encoding="utf-8") as ef:
                ef.write(str(e) + "\n\n")
                traceback.print_exc(file=ef)
            logger.info(f"Saved RAGAS exception output to {err_file}")
        except Exception:
            logger.error("Failed to write RAGAS exception to file")
        sys.exit(1)


if __name__ == "__main__":
    main()
