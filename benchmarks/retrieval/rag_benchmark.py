#!/usr/bin/env python3
"""
CUBO RAG Testing Script
Runs systematic tests using the comprehensive question set and evaluation metrics.
"""

import asyncio
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List

# Add parent directory to path for imports
# Add parent directory to path for imports
# benchmarks/scripts/run_rag_tests.py -> benchmarks/scripts -> benchmarks -> root -> src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import subprocess
from cubo.config import config
import random

from tqdm import tqdm

from benchmarks.utils.metrics import AdvancedEvaluator, GroundTruthLoader, IRMetricsEvaluator
from benchmarks.utils.hardware import log_hardware_metadata, sample_latency, sample_memory
from cubo.main import CUBOApp
from benchmarks.utils.ragas_evaluator import get_ragas_evaluator

try:
    import pandas as pd
except Exception:
    pd = None
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import logging as _logging

_logging.getLogger("sentence_transformers").setLevel(_logging.ERROR)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("test_results.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

try:
    import ollama

    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logger.warning("Ollama client not available, LLM-based evaluations will be disabled")


class RAGTester:
    """Comprehensive RAG testing framework."""

    def __init__(
        self,
        questions_file: str = "test_questions.json",
        data_folder: str = "data",
        ground_truth_file: str = None,
        mode: str = "full",
        skip_index: bool = False,
        index_batch_size: int = 32,
        index_sample_size: int | None = None,
        index_commit_size: int = 4096,
    ):
        """Initialize the tester with question data and CUBO system."""
        self.questions_file = questions_file
        self.questions = self.load_questions()
        self.data_folder = data_folder
        self.mode = mode  # 'full', 'retrieval-only', 'ingestion-only'

        # Load ground truth for IR metrics if provided
        self.ground_truth = None
        if ground_truth_file:
            try:
                if ground_truth_file.endswith(".json"):
                    self.ground_truth = GroundTruthLoader.load_custom_format(ground_truth_file)
                else:
                    self.ground_truth = GroundTruthLoader.load_beir_format(ground_truth_file)
                logger.info(f"Loaded ground truth for {len(self.ground_truth)} questions")
            except Exception as e:
                logger.error(f"Failed to load ground truth: {e}")

        # Initialize evaluator
        if OLLAMA_AVAILABLE:
            self.evaluator = AdvancedEvaluator(ollama_client=ollama.Client())
            logger.info("AdvancedEvaluator initialized with Ollama client for LLM-based metrics")
        else:
            self.evaluator = AdvancedEvaluator()
            logger.info(
                "AdvancedEvaluator initialized without LLM client (LLM-based metrics disabled)"
            )

        # Initialize IR metrics evaluator
        self.ir_evaluator = IRMetricsEvaluator()

        # Initialize RAGAS evaluator (if available)
        # Use GLM model as requested
        self.ragas_evaluator = get_ragas_evaluator(llm_model="glm-4")
        if self.ragas_evaluator:
            logger.info("RAGAS Evaluator initialized (metrics: comprehensiveness, diversity, empowerment)")
        else:
            logger.warning("RAGAS Evaluator not available (skipping comprehensiveness metrics)")

        # Indexing and retrieval flags
        self.skip_index = skip_index
        self.index_batch_size = index_batch_size
        self.index_sample_size = index_sample_size
        self.index_commit_size = index_commit_size
        # Auto-populate DB from corpus: when skip_index is True and collection lacks doc rows
        self.auto_populate_db = False

        # Initialize CUBO system (skip for ingestion-only mode)
        self.cubo_app = None
        if mode != "ingestion-only":
            self._initialize_cubo_system()

        # Capture hardware metadata
        self.hardware_metadata = log_hardware_metadata()
        logger.info(
            f"Hardware: {self.hardware_metadata['cpu']['model']}, "
            f"{self.hardware_metadata['ram']['total_gb']:.1f}GB RAM"
        )

        self.results = {
            "metadata": {
                "test_run_timestamp": time.time(),
                "total_questions": 0,
                "questions_by_difficulty": {},
                "success_rate": 0.0,
                "mode": mode,
                "hardware": self.hardware_metadata,
            },
            "results": {"easy": [], "medium": [], "hard": []},
        }

    def _initialize_cubo_system(self):
        """Initialize the CUBO RAG system for testing."""
        try:
            logger.info("Initializing CUBO system for testing...")
            self.cubo_app = CUBOApp()

            # Skip the setup wizard - assume system is already configured
            if not self.cubo_app.initialize_components():
                # If initialization fails (most likely due to model path missing), provide a helpful hint
                logger.error(
                    "Failed to initialize CUBO components (model or dependencies likely missing). "
                    "If you have a prebuilt index and wish to skip indexing, ensure the embedding model is available at the configured model_path and the retriever collection is populated."
                )
                return

            # Load all documents from data folder
            if not os.path.exists(self.data_folder):
                logger.error(f"Data folder '{self.data_folder}' not found")
                return

            logger.info(f"Loading documents from {self.data_folder}...")
            # If corpus processed files exist, use dataset-specific loader (BEIR/RAGBench/UltraDomain)
            documents = None
            # Try BEIR corpus file
            beir_corpus = os.path.join(self.data_folder, "corpus.jsonl")
            if os.path.exists(beir_corpus):
                documents = self._load_beir_corpus(beir_corpus)
            elif os.path.exists(os.path.join(self.data_folder, "corpus_processed.json")):
                # Already preprocessed
                with open(
                    os.path.join(self.data_folder, "corpus_processed.json"), encoding="utf-8"
                ) as f:
                    documents = json.load(f)
            elif any(
                f.endswith(".parquet")
                for root, dirs, files in os.walk(self.data_folder)
                for f in files
            ):
                documents = self._load_ragbench_parquet(self.data_folder)
            elif any(f.endswith(".jsonl") for f in os.listdir(self.data_folder)):
                documents = self._load_ultradomain_corpus(self.data_folder)
            else:
                # Fall back to generic loader
                documents = self.cubo_app.doc_loader.load_documents_from_folder(self.data_folder)
            if not documents:
                logger.error("No documents loaded")
                return

            logger.info(f"Adding {len(documents)} document chunks to vector database...")
            # Extract text content from chunk dictionaries
            document_texts = []
            for chunk in documents:
                if isinstance(chunk, dict) and "text" in chunk:
                    document_texts.append(chunk["text"])
                elif isinstance(chunk, str):
                    document_texts.append(chunk)
                else:
                    logger.warning(f"Skipping invalid chunk format: {type(chunk)}")

            # If skip_index flag is set, verify that retriever is already populated.
            # If not, abort to avoid unexpectedly reindexing the entire corpus.
            if getattr(self, "skip_index", False):
                try:
                    logger.info("skip_index=True: verifying existing retriever index has documents...")
                    test_ctxs = None
                    try:
                        # Use the documented kwarg name 'top_k' (not 'k') to avoid TypeErrors
                        test_ctxs = self.cubo_app.retriever.retrieve_top_documents("test", top_k=1)
                    except Exception as e:
                        # Some retrievers may require valid content or raise exceptions; treat as failure
                        logger.warning(f"retriever.retrieve_top_documents failed during verification: {e}")
                        test_ctxs = None
                    if not test_ctxs:
                        logger.error("skip_index is True but retriever appears to be empty. Aborting tests to avoid reindexing.")
                        # Optionally, auto-populate DB from corpus if enabled and corpus is present
                        try:
                            beir_corpus = os.path.join(self.data_folder, "corpus.jsonl")
                            if getattr(self, "auto_populate_db", False) and os.path.exists(beir_corpus):
                                logger.info("Auto-populate enabled and BEIR corpus found; attempting to populate documents DB from corpus...")
                                index_dir = config.get("vector_store_path", "./faiss_index")
                                cmd = [
                                    sys.executable,
                                    "scripts/populate_documents_db_from_beir.py",
                                    "--index-dir",
                                    index_dir,
                                    "--corpus",
                                    beir_corpus,
                                    "--commit-size",
                                    str(getattr(self, "index_commit_size", 4096)),
                                ]
                                logger.info(f"Running populate script: {' '.join(cmd)}")
                                subprocess.run(cmd, check=True)
                                # Re-check retriever after running population script
                                try:
                                    test_ctxs = self.cubo_app.retriever.retrieve_top_documents(
                                        "test", top_k=1
                                    )
                                except Exception as e:
                                    logger.warning(f"retriever.retrieve_top_documents still failing after populate: {e}")
                            if not test_ctxs:
                                raise RuntimeError(
                                    "skip_index set but retriever returned no documents. Run ingestion or remove --skip-index to reindex."
                                )
                        except subprocess.CalledProcessError as se:
                            logger.error(f"Auto-populate script failed: {se}")
                            raise RuntimeError("Auto populate failed")
                    logger.info("Retriever verified: found existing documents; skipping indexing as requested.")
                except Exception:
                    # Re-raise so initialization stops and we don't proceed unintentionally
                    raise

            # Index using batch ingestion if collection supports adding meta/ids
            if (
                isinstance(documents, list)
                and documents
                and isinstance(documents[0], dict)
                and "id" in documents[0]
            ):
                # If skip_index flag set, do not reindex
                if getattr(self, "skip_index", False):
                    logger.info("Skipping indexing step (skip_index=True)")
                else:
                    self._index_documents_with_batching(
                        documents,
                        batch_size=getattr(self, "index_batch_size", 32),
                        sample_size=getattr(self, "index_sample_size", None),
                        seed=getattr(self, "index_sample_seed", 42),
                    )
            elif document_texts:
                self.cubo_app.retriever.add_documents(document_texts)
                logger.info("Documents added to vector database successfully")
            else:
                logger.error("No valid document texts found to add")
                return

            logger.info("CUBO system ready for testing!")

        except Exception as e:
            # Log full exception details for easier debugging when initialization fails
            logger.error(f"Failed to initialize CUBO system: {e}", exc_info=True)
            self.cubo_app = None

    def load_questions(self) -> Dict[str, Any]:
        """Load questions from JSON file."""
        try:
            with open(self.questions_file, encoding="utf-8") as f:
                data = json.load(f)
            logger.info(f"Loaded {data['metadata']['total_questions']} questions")
            # Store query IDs if available
            self.query_ids = data["metadata"].get("query_ids", None)
            return data["questions"]
        except Exception as e:
            logger.error(f"Failed to load questions: {e}")
            self.query_ids = None
            return {"easy": [], "medium": [], "hard": []}

    def run_single_test(
        self,
        question: str,
        difficulty: str,
        question_id: str = None,
        k_values: List[int] = [5, 10, 20],
    ) -> Dict[str, Any]:
        """Run a single question test with real RAG evaluation."""
        logger.info(f"Testing [{difficulty}]: {question[:50]}...")

        start_time = time.time()

        # MRR (Mean Reciprocal Rank) - reciprocal rank of first relevant doc
        mrr = 0.0
        relevant_docs = ground_truth.get(question_id, [])
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_docs:
                mrr = 1.0 / (i + 1)
                break
        metrics["mrr"] = mrr

        try:
            if not self.cubo_app:
                raise Exception("CUBO system not initialized")

            # Get actual retrieved contexts from CUBO with latency measurement
            retrieval_metrics = sample_latency(
                self.cubo_app.retriever.retrieve_top_documents, question, samples=1
            )
            contexts = self.cubo_app.retriever.retrieve_top_documents(question)

            # Record cache metrics if cache service is available
            cache_metrics = {}
            try:
                cache_metrics = self.cubo_app.retriever.cache_service.get_metrics()
            except Exception:
                cache_metrics = {}

            # Extract document IDs for IR metrics
            retrieved_ids = []
            for ctx in contexts:
                if isinstance(ctx, dict):
                    # Try direct keys first, then check metadata dict
                    doc_id = ctx.get("id") or ctx.get("doc_id") or ctx.get("chunk_id")
                    if not doc_id and "metadata" in ctx:
                        meta = ctx["metadata"]
                        if isinstance(meta, dict):
                            doc_id = meta.get("id") or meta.get("doc_id") or meta.get("chunk_id")
                    if doc_id:
                        retrieved_ids.append(str(doc_id))

            # Compute IR metrics if ground truth available
            ir_metrics = {}
            if self.ground_truth and question_id:
                ir_metrics = self.ir_evaluator.evaluate_retrieval(
                    question_id, retrieved_ids, self.ground_truth, k_values=k_values
                )

            # For retrieval-only mode, skip generation
            if self.mode == "retrieval-only":
                processing_time = time.time() - start_time

                result = {
                    "question": question,
                    "question_id": question_id,
                    "difficulty": difficulty,
                    "retrieved_ids": retrieved_ids,
                    "contexts": contexts,
                    "retrieval_latency": retrieval_metrics,
                    "cache_metrics": cache_metrics,
                    "ir_metrics": ir_metrics,
                    "processing_time": processing_time,
                    "success": True,
                    "timestamp": time.time(),
                }
                return result

            # Full RAG mode: generate response
            context_texts = [
                ctx.get("document", "") if isinstance(ctx, dict) else str(ctx) for ctx in contexts
            ]
            context_text = "\n".join(context_texts)

            # Measure generation latency
            generation_start = time.time()
            response = self.cubo_app.generator.generate_response(question, context_text)
            generation_time = time.time() - generation_start

            processing_time = time.time() - start_time

            # Sample memory during evaluation
            memory_metrics = sample_memory()

            # Evaluate the response using AdvancedEvaluator from metrics.py
            evaluation_results = asyncio.run(
                self.evaluate_response(question, response, context_texts, processing_time)
            )

            # Evaluate using RAGAS if available
            ragas_metrics = {}
            if self.ragas_evaluator:
                try:
                    # Use ground truth if available for this specific question
                    gt = None
                    if self.ground_truth and question_id and question_id in self.ground_truth:
                        # Ground truth might be a list of answers or single string
                        gt_data = self.ground_truth[question_id]
                        if isinstance(gt_data, list):
                            gt = gt_data[0]
                        elif isinstance(gt_data, str):
                            gt = gt_data
                        elif isinstance(gt_data, dict) and "text" in gt_data:
                            gt = gt_data["text"]

                    ragas_metrics = asyncio.run(self.ragas_evaluator.evaluate_single(
                        question=question,
                        answer=response,
                        contexts=context_texts,
                        ground_truth=gt
                    ))
                except Exception as e:
                    logger.error(f"RAGAS evaluation failed: {e}")
                    ragas_metrics = {"error": str(e)}

            result = {
                "question": question,
                "question_id": question_id,
                "difficulty": difficulty,
                "response": response,
                "retrieved_ids": retrieved_ids,
                "contexts": contexts,
                "retrieval_latency": retrieval_metrics,
                "generation_time": generation_time,
                "processing_time": processing_time,
                "memory": memory_metrics,
                "ir_metrics": ir_metrics,
                "ragas_metrics": ragas_metrics,
                "evaluation": evaluation_results,
                "success": True,
                "timestamp": time.time(),
            }

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Test failed for question: {question[:50]}... Error: {e}")
            result = {
                "question": question,
                "question_id": question_id,
                "difficulty": difficulty,
                "error": str(e),
                "processing_time": processing_time,
                "success": False,
                "timestamp": time.time(),
            }

        return result

    def _load_beir_corpus(self, corpus_path: str) -> List[Dict[str, Any]]:
        """Load BEIR corpus.jsonl into list of dicts with id/text/metadata."""
        docs = []
        try:
            with open(corpus_path, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        obj = json.loads(line.strip())
                        doc_id = str(obj.get("_id") or obj.get("id") or obj.get("doc_id"))
                        text = obj.get("text") or obj.get("excerpt") or obj.get("content") or ""
                        docs.append(
                            {
                                "id": doc_id,
                                "text": text,
                                "filename": os.path.basename(corpus_path),
                                "file_path": corpus_path,
                                "chunk_index": 0,
                            }
                        )
                    except Exception:
                        continue
        except Exception as e:
            logger.error(f"Failed to load BEIR corpus: {e}")
        return docs

    def _load_ultradomain_corpus(self, data_folder: str) -> List[Dict[str, Any]]:
        """Load UltraDomain .jsonl files (multiple domain files) into docs list."""
        docs = []
        try:
            for fname in os.listdir(data_folder):
                if fname.endswith(".jsonl"):
                    path = os.path.join(data_folder, fname)
                    with open(path, encoding="utf-8") as f:
                        for i, line in enumerate(f):
                            if not line.strip():
                                continue
                            try:
                                obj = json.loads(line.strip())
                                doc_id = obj.get("_id") or obj.get("id") or f"{fname}_{i}"
                                text = obj.get("text") or obj.get("content") or ""
                                docs.append(
                                    {
                                        "id": str(doc_id),
                                        "text": text,
                                        "filename": fname,
                                        "file_path": path,
                                        "chunk_index": i,
                                    }
                                )
                            except Exception:
                                continue
        except Exception as e:
            logger.error(f"Failed to load UltraDomain corpus: {e}")
        return docs

    def _load_ragbench_parquet(self, data_folder: str) -> List[Dict[str, Any]]:
        """Load RAGBench parquet files from subdirectories into docs list."""
        docs = []
        try:
            # Walk through data_folder for parquet files
            for root, dirs, files in os.walk(data_folder):
                for f in files:
                    if f.endswith(".parquet"):
                        path = os.path.join(root, f)
                        # Read using pandas if available
                        if pd is None:
                            logger.warning(
                                "Pandas not available; skipping RAGBench parquet loading"
                            )
                            continue
                        try:
                            df = pd.read_parquet(path)
                        except Exception as e:
                            logger.error(f"Failed to read parquet {path}: {e}")
                            continue
                        if "id" in df.columns:
                            id_col = "id"
                        elif "doc_id" in df.columns:
                            id_col = "doc_id"
                        else:
                            id_col = None
                        # Combine documents array into single text if present
                        for idx, row in df.iterrows():
                            doc_id = str(row[id_col]) if id_col else f"{f}_{idx}"
                            if "documents" in row and isinstance(row["documents"], (list,)):
                                # documents is an ndarray or array of strings
                                texts = [str(x) for x in row["documents"]]
                                text = "\n".join(texts)
                            elif "text" in row:
                                text = str(row["text"])
                            else:
                                # Try response or other fields
                                text = str(row["response"]) if "response" in row else ""
                            docs.append(
                                {
                                    "id": doc_id,
                                    "text": text,
                                    "filename": f,
                                    "file_path": path,
                                    "chunk_index": idx,
                                }
                            )
        except Exception as e:
            logger.error(f"Failed to load RAGBench parquet: {e}")
        return docs

    def _index_documents_with_batching(
        self,
        docs: List[Dict[str, Any]],
        batch_size: int = 32,
        sample_size: int = None,
        seed: int = 42,
    ):
        """Index documents into the retriever using batching and optional sampling."""
        if not docs:
            logger.info("No documents to index")
            return

        # Sample docs if requested
        if sample_size and sample_size < len(docs):
            random.seed(seed)
            docs = random.sample(docs, sample_size)
            logger.info(f"Sampled {len(docs)} docs for indexing (seed={seed})")

        total = len(docs)
        batch_size = int(batch_size) if batch_size and batch_size > 0 else 32
        logger.info(f"Indexing {total} documents with batch size {batch_size}")

        # Process in batches and add to vector store with metadata and preserved IDs
        commit_size = getattr(self, "index_commit_size", 4096) or 4096
        commit_size = min(commit_size, total)
        # For reproducibility and to avoid unintentionally appending to existing indexes,
        # reset the collection unless the user explicitly sets skip_index
        try:
            if hasattr(self.cubo_app.retriever, "collection") and hasattr(
                self.cubo_app.retriever.collection, "reset"
            ):
                logger.info(
                    "Resetting existing collection before indexing to avoid appending to old data"
                )
                self.cubo_app.retriever.collection.reset()
        except Exception:
            logger.warning(
                "Collection reset failed or not supported; proceeding with append behaviour"
            )
        all_embeddings = []
        all_ids = []
        all_docs = []
        all_metas = []
        with tqdm(total=total, desc="Indexing documents", unit="doc", leave=True) as pbar:
            for i in range(0, total, batch_size):
                batch = docs[i : i + batch_size]
                texts = [d["text"] for d in batch]
                ids = [d["id"] for d in batch]
                metadatas = [
                    {
                        "id": d["id"],
                        "filename": d.get("filename"),
                        "chunk_index": d.get("chunk_index"),
                    }
                    for d in batch
                ]

                # Generate embeddings in a threaded way (with fallback to sequential encode if threaded fails)
                try:
                    embeddings = (
                        self.cubo_app.retriever.inference_threading.generate_embeddings_threaded(
                            texts,
                            self.cubo_app.retriever.model,
                            batch_size=batch_size,
                            timeout_per_batch=120,
                        )
                    )
                except Exception as e:
                    logger.warning(
                        f"Threaded embedding generation failed: {e}. Falling back to sequential encode."
                    )
                    try:
                        # Use the model directly (may accept batch_size kwarg)
                        if hasattr(self.cubo_app.retriever.model, "encode"):
                            try:
                                embeddings = self.cubo_app.retriever.model.encode(
                                    texts, batch_size=batch_size
                                )
                            except TypeError:
                                embeddings = self.cubo_app.retriever.model.encode(texts)
                            if hasattr(embeddings, "tolist"):
                                embeddings = embeddings.tolist()
                        else:
                            embeddings = [[] for _ in texts]
                    except Exception as e2:
                        logger.error(f"Sequential embedding fallback failed: {e2}")
                        embeddings = [[] for _ in texts]

                # Buffer embeddings and metadata and perform commits in larger chunks
                all_embeddings.extend(embeddings)
                all_ids.extend(ids)
                all_docs.extend(texts)
                all_metas.extend(metadatas)
                try:
                    if len(all_embeddings) >= commit_size:
                        if hasattr(self.cubo_app.retriever, "collection") and hasattr(
                            self.cubo_app.retriever.collection, "add"
                        ):
                            self.cubo_app.retriever.collection.add(
                                embeddings=all_embeddings,
                                documents=all_docs,
                                metadatas=all_metas,
                                ids=all_ids,
                            )
                        else:
                            for j, text in enumerate(all_docs):
                                self.cubo_app.retriever.add_documents([text])
                        all_embeddings.clear()
                        all_ids.clear()
                        all_docs.clear()
                        all_metas.clear()
                except Exception as e:
                    logger.error(f"Failed to commit batch to collection: {e}")

                pbar.update(len(batch))
                # Update postfix with throughput
                try:
                    elapsed = time.time() - start_time
                    processed = min(i + len(batch), total)
                    throughput = processed / elapsed if elapsed > 0 else 0
                    pbar.set_postfix(
                        {
                            "batch": batch_size,
                            "commit_size": commit_size,
                            "qps": f"{throughput:.1f}/s",
                        }
                    )
                except Exception:
                    pass

        # Commit any remaining
        if all_embeddings:
            try:
                if hasattr(self.cubo_app.retriever, "collection") and hasattr(
                    self.cubo_app.retriever.collection, "add"
                ):
                    self.cubo_app.retriever.collection.add(
                        embeddings=all_embeddings,
                        documents=all_docs,
                        metadatas=all_metas,
                        ids=all_ids,
                    )
                else:
                    for j, text in enumerate(all_docs):
                        self.cubo_app.retriever.add_documents([text])
            except Exception as e:
                logger.error(f"Final commit failed: {e}")
        logger.info(f"Indexing completed: {total} documents")

    async def evaluate_response(
        self, question: str, answer: str, contexts: List[str], response_time: float
    ) -> Dict[str, Any]:
        """Evaluate the RAG response using advanced metrics."""
        try:
            # Run comprehensive evaluation
            evaluation = await self.evaluator.evaluate_comprehensive(
                question=question, answer=answer, contexts=contexts, response_time=response_time
            )

            # Extract key metrics for summary
            key_metrics = {
                "answer_relevance": evaluation.get("answer_relevance", 0),
                "context_relevance": evaluation.get("context_relevance", 0),
                "groundedness": evaluation.get("groundedness_score", 0),
                "answer_quality": evaluation.get("answer_quality", {}),
                "context_utilization": evaluation.get("context_utilization", {}),
                "response_efficiency": evaluation.get("response_efficiency", {}),
                "information_completeness": evaluation.get("information_completeness", {}),
                "llm_metrics": evaluation.get("llm_metrics", {}),
            }

            return key_metrics

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {
                "error": str(e),
                "answer_relevance": 0,
                "context_relevance": 0,
                "groundedness": 0,
            }

    def run_difficulty_tests(
        self, difficulty: str, limit: int = None, k_values: List[int] = [5, 10, 20]
    ) -> List[Dict[str, Any]]:
        """Run all tests for a specific difficulty level."""
        questions = self.questions.get(difficulty, [])
        if limit:
            questions = questions[:limit]

        logger.info(f"Running {len(questions)} {difficulty} tests")
        # Reset cache metrics for this difficulty set (to measure hit/miss rates across this set)
        try:
            if self.cubo_app and hasattr(self.cubo_app.retriever, "cache_service") and self.cubo_app.retriever.cache_service:
                self.cubo_app.retriever.cache_service.reset_metrics()
        except Exception:
            pass
        results = []
        with tqdm(total=len(questions), desc=f"Testing {difficulty}", unit="question") as pbar:
            for i, question in enumerate(questions, 1):
                # Use actual query ID from metadata if available, otherwise generate one
                if self.query_ids and i - 1 < len(self.query_ids):
                    question_id = self.query_ids[i - 1]
                else:
                    question_id = f"{difficulty}_{i}"
                result = self.run_single_test(
                    question, difficulty, question_id=question_id, k_values=k_values
                )
                results.append(result)
                pbar.update(1)

        return results

    def run_all_tests(
        self,
        easy_limit: int = None,
        medium_limit: int = None,
        hard_limit: int = None,
        k_values: List[int] = [5, 10, 20],
    ) -> Dict[str, Any]:
        """Run comprehensive test suite."""
        logger.info(f"Starting comprehensive RAG testing (mode: {self.mode})")

        # Run tests by difficulty
        self.results["results"]["easy"] = self.run_difficulty_tests(
            "easy", easy_limit, k_values=k_values
        )
        self.results["results"]["medium"] = self.run_difficulty_tests(
            "medium", medium_limit, k_values=k_values
        )
        self.results["results"]["hard"] = self.run_difficulty_tests(
            "hard", hard_limit, k_values=k_values
        )

        # Calculate statistics
        self.calculate_statistics()

        logger.info("Testing completed")
        return self.results

    def calculate_statistics(self):
        """Calculate test statistics including evaluation and IR metrics."""
        all_results = []
        evaluation_metrics = {"answer_relevance": [], "context_relevance": [], "groundedness": []}
        for difficulty in ["easy", "medium", "hard"]:
            results = self.results["results"][difficulty]
            all_results.extend(results)

            # Initialize ragas_metrics_agg for current difficulty
            ragas_metrics_agg_difficulty = {"comprehensiveness": [], "diversity": [], "empowerment": [], "overall": []}

            # Difficulty-specific stats
            total = len(results)
            successful = sum(1 for r in results if r.get("success", False))
            avg_time = sum(r.get("processing_time", 0) for r in results) / total if total > 0 else 0

            # Collect latency stats
            retrieval_latencies = []
            for r in results:
                if "retrieval_latency" in r and r["retrieval_latency"]:
                    retrieval_latencies.append(r["retrieval_latency"].get("p50_ms", 0))

            avg_retrieval_latency = (
                sum(retrieval_latencies) / len(retrieval_latencies) if retrieval_latencies else 0
            )

            # Collect IR metrics (including MRR)
            ir_stats = {}
            mrr_scores = []
            cache_stats = {}
            for r in results:
                if "ir_metrics" in r and r["ir_metrics"]:
                    # Collect MRR separately (it's a single value, not nested)
                    if "mrr" in r["ir_metrics"]:
                        mrr_scores.append(r["ir_metrics"]["mrr"])
                    
                    for metric_name, values in r["ir_metrics"].items():
                        if metric_name == "mrr":
                            continue  # Already handled above
                        if isinstance(values, dict):
                            for k, score in values.items():
                                key = f"{metric_name}_{k}"
                                if key not in ir_stats:
                                    ir_stats[key] = []
                                ir_stats[key].append(score)
                        elif isinstance(values, (int, float)):
                            # Handle flat metrics like recall_at_k_10
                            if metric_name not in ir_stats:
                                ir_stats[metric_name] = []
                            ir_stats[metric_name].append(values)

            # Average IR metrics
            avg_ir_metrics = {}
            for key, scores in ir_stats.items():
                avg_ir_metrics[f"avg_{key}"] = sum(scores) / len(scores) if scores else 0
            
            # Add MRR average
            avg_ir_metrics["avg_mrr"] = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0

            # Collect cache metrics
            cache_agg = {}
            for r in results:
                if "cache_metrics" in r and r["cache_metrics"]:
                    for ck, cv in r["cache_metrics"].items():
                        if isinstance(cv, (int, float)):
                            cache_agg.setdefault(ck, []).append(cv)

            # Compute average cache metrics (if any)
            avg_cache_metrics = {}
            for ck, values in cache_agg.items():
                avg_cache_metrics[f"avg_{ck}"] = sum(values) / len(values) if values else 0

            # Add RAGAS metrics to stats
            avg_ragas_metrics = {}
            for key, values in ragas_metrics_agg_difficulty.items():
                avg_ragas_metrics[f"avg_ragas_{key}"] = sum(values) / len(values) if values else 0

            # Collect evaluation metrics (for full RAG mode)
            relevance_scores = []
            context_scores = []
            groundedness_scores = []

            for r in results:
                if "evaluation" in r and r["evaluation"]:
                    eval_data = r["evaluation"]
                    if (
                        "answer_relevance" in eval_data
                        and eval_data["answer_relevance"] is not None
                    ):
                        relevance_scores.append(eval_data["answer_relevance"])
                        evaluation_metrics["answer_relevance"].append(eval_data["answer_relevance"])
                    if (
                        "context_relevance" in eval_data
                        and eval_data["context_relevance"] is not None
                    ):
                        context_scores.append(eval_data["context_relevance"])
                        evaluation_metrics["context_relevance"].append(
                            eval_data["context_relevance"]
                        )
                    if "groundedness" in eval_data and eval_data["groundedness"] is not None:
                        groundedness_scores.append(eval_data["groundedness"])
                        evaluation_metrics["groundedness"].append(eval_data["groundedness"])

            # Collect RAGAS metrics
            for r in results:
                if "ragas_metrics" in r and r["ragas_metrics"] and "error" not in r["ragas_metrics"]:
                    rm = r["ragas_metrics"]
                    for key in ["comprehensiveness", "diversity", "empowerment", "overall"]:
                        if key in rm:
                            ragas_metrics_agg_difficulty[key].append(rm[key])
                            # overall_ragas_metrics_agg[key].append(rm[key]) # Uncomment if overall stats needed
            
            avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
            avg_context = sum(context_scores) / len(context_scores) if context_scores else 0
            avg_groundedness = (
                sum(groundedness_scores) / len(groundedness_scores) if groundedness_scores else 0
            )

            avg_ragas_metrics = {}
            for key, values in ragas_metrics_agg_difficulty.items():
                avg_ragas_metrics[f"avg_ragas_{key}"] = sum(values) / len(values) if values else 0

            self.results["metadata"]["questions_by_difficulty"][difficulty] = {
                "total": total,
                "successful": successful,
                "success_rate": successful / total if total > 0 else 0,
                "avg_processing_time": avg_time,
                "avg_retrieval_latency_p50_ms": avg_retrieval_latency,
                "avg_answer_relevance": avg_relevance,
                "avg_context_relevance": avg_context,
                "avg_groundedness": avg_groundedness,
                **avg_ir_metrics,
                **avg_cache_metrics,
                **avg_ragas_metrics,
            }

        # Overall stats
        total_questions = len(all_results)
        successful_questions = sum(1 for r in all_results if r.get("success", False))
        overall_success_rate = successful_questions / total_questions if total_questions > 0 else 0

        # Overall evaluation metrics
        overall_metrics = {}
        for metric_name, scores in evaluation_metrics.items():
            if scores:
                overall_metrics[f"avg_{metric_name}"] = sum(scores) / len(scores)
            else:
                overall_metrics[f"avg_{metric_name}"] = 0

        # Overall IR metrics
        overall_ir_metrics = {}
        overall_mrr_scores = []
        for r in all_results:
            if "ir_metrics" in r and r["ir_metrics"]:
                for metric_name, values in r["ir_metrics"].items():
                    if isinstance(values, dict):
                        for k, score in values.items():
                            key = f"{metric_name}_{k}"
                            if key not in overall_ir_metrics:
                                overall_ir_metrics[key] = []
                            overall_ir_metrics[key].append(score)
                    elif metric_name == "mrr" and isinstance(values, (int, float)):
                        overall_mrr_scores.append(values)

        for key, scores in overall_ir_metrics.items():
            overall_metrics[f"avg_{key}"] = sum(scores) / len(scores) if scores else 0
        # Add overall MRR average
        overall_metrics["avg_mrr"] = sum(overall_mrr_scores) / len(overall_mrr_scores) if overall_mrr_scores else 0

        # Overall cache metrics
        overall_cache_agg = {}
        for r in all_results:
            if "cache_metrics" in r and r["cache_metrics"]:
                for ck, cv in r["cache_metrics"].items():
                    if isinstance(cv, (int, float)):
                        overall_cache_agg.setdefault(ck, []).append(cv)

        for ck, values in overall_cache_agg.items():
            overall_metrics[f"avg_{ck}"] = sum(values) / len(values) if values else 0

        self.results["metadata"].update(
            {
                "total_questions": total_questions,
                "successful_questions": successful_questions,
                "success_rate": overall_success_rate,
                "total_processing_time": sum(r.get("processing_time", 0) for r in all_results),
                **overall_metrics,
            }
        )

    def save_results(self, output_file: str = "test_results.json"):
        """Save test results to JSON file."""
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")

    def print_summary(self):
        """Print test summary with evaluation and IR metrics."""
        meta = self.results["metadata"]

        print("\n" + "=" * 60)
        print(f"CUBO RAG TESTING SUMMARY (Mode: {meta.get('mode', 'full')})")
        print("=" * 60)

        print(f"Total Questions Tested: {meta['total_questions']}")
        print(f"Success Rate: {meta['success_rate']*100:.1f}%")
        print(f"Total Processing Time: {meta['total_processing_time']:.2f}s")

        # Print overall IR metrics
        print("\nOverall IR Metrics:")
        for key, value in meta.items():
            if key.startswith("avg_recall_at_k"):
                k = key.split("_")[-1]
                print(f"  Recall@{k}: {value:.3f}")
        for key, value in meta.items():
            if key.startswith("avg_ndcg_at_k"):
                k = key.split("_")[-1]
                print(f"  nDCG@{k}: {value:.3f}")
        if "avg_mrr" in meta:
            print(f"  MRR: {meta['avg_mrr']:.3f}")

        # Print overall cache metrics if available
        if "avg_semantic_hit_rate_percent" in meta:
            print(f"  Avg Semantic Cache Hit Rate: {meta['avg_semantic_hit_rate_percent']:.1f}%")
        if "avg_semantic_hits" in meta:
            print(f"  Avg Semantic Cache Hits: {meta['avg_semantic_hits']:.1f}")

        # Print overall evaluation metrics (full RAG mode)
        if meta.get("mode") == "full":
            print("\nOverall RAG Metrics:")
            if "avg_answer_relevance" in meta and meta["avg_answer_relevance"] > 0:
                print(f"  Answer Relevance: {meta['avg_answer_relevance']:.3f}")
            if "avg_context_relevance" in meta and meta["avg_context_relevance"] > 0:
                print(f"  Context Relevance: {meta['avg_context_relevance']:.3f}")
            if "avg_groundedness" in meta and meta["avg_groundedness"] > 0:
                print(f"  Groundedness: {meta['avg_groundedness']:.3f}")

        print("\nBy Difficulty:")
        for difficulty, stats in meta["questions_by_difficulty"].items():
            print(f"  {difficulty.capitalize()}:")
            print(f"    Questions: {stats['total']}")
            print(f"    Success Rate: {stats['success_rate']*100:.1f}%")
            print(f"    Avg Processing Time: {stats['avg_processing_time']:.2f}s")

            if "avg_retrieval_latency_p50_ms" in stats:
                print(
                    f"    Avg Retrieval Latency (p50): {stats['avg_retrieval_latency_p50_ms']:.1f}ms"
                )

            # Show IR metrics per difficulty
            for key, value in stats.items():
                if key.startswith("avg_recall_at_k"):
                    k = key.split("_")[-1]
                    print(f"    Recall@{k}: {value:.3f}")

            # Show RAG metrics per difficulty (full mode)
            if meta.get("mode") == "full":
                if "avg_answer_relevance" in stats and stats["avg_answer_relevance"] > 0:
                    print(f"    Answer Relevance: {stats['avg_answer_relevance']:.3f}")
                if "avg_groundedness" in stats and stats["avg_groundedness"] > 0:
                    print(f"    Groundedness: {stats['avg_groundedness']:.3f}")

            # Show cache metrics per difficulty
            if "avg_semantic_hit_rate_percent" in stats:
                print(f"    Semantic Cache Hit Rate: {stats['avg_semantic_hit_rate_percent']:.1f}%")

        # Hardware summary
        if "hardware" in meta:
            hw = meta["hardware"]
            print("\nHardware Configuration:")
            print(f"  CPU: {hw['cpu']['model']}")
            print(f"  RAM: {hw['ram']['total_gb']:.1f} GB")
            if hw["gpu"].get("available"):
                print(
                    f"  GPU: {hw['gpu']['device_name']} ({hw['gpu']['vram_total_gb']:.1f} GB VRAM)"
                )

        print("\nDetailed results saved to test_results.json")


def main():
    """Main testing function."""
    parser = argparse.ArgumentParser(description="CUBO RAG Testing Framework")
    parser.add_argument(
        "--questions", default="test_questions.json", help="Path to questions JSON file"
    )
    parser.add_argument(
        "--data-folder", default="data", help="Path to data folder containing documents"
    )
    parser.add_argument(
        "--ground-truth",
        default=None,
        help="Path to ground truth file (BeIR format or custom JSON)",
    )
    parser.add_argument(
        "--mode",
        default="full",
        choices=["full", "retrieval-only", "ingestion-only"],
        help="Testing mode: full RAG, retrieval-only, or ingestion-only",
    )
    parser.add_argument(
        "--k-values",
        default="5,10,20",
        help="Comma-separated K values for IR metrics (default: 5,10,20)",
    )
    parser.add_argument("--easy-limit", type=int, help="Limit number of easy questions")
    parser.add_argument("--medium-limit", type=int, help="Limit number of medium questions")
    parser.add_argument("--hard-limit", type=int, help="Limit number of hard questions")
    parser.add_argument("--output", default="test_results.json", help="Output file for results")
    parser.add_argument(
        "--index-batch-size",
        type=int,
        default=32,
        help="Batch size to use for indexing/inference embedding generation",
    )
    parser.add_argument(
        "--index-sample-size",
        type=int,
        default=None,
        help="If set, randomly sample this many documents from corpus for indexing (performance vs accuracy)",
    )
    parser.add_argument(
        "--index-sample-seed", type=int, default=42, help="Seed for random sampling of corpus"
    )
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="Skip indexing corpus and assume vector store already populated",
    )
    parser.add_argument(
        "--save-processed-corpus",
        action="store_true",
        help="Save processed corpus into corpus_processed.json for faster reuse",
    )
    parser.add_argument(
        "--index-commit-size",
        type=int,
        default=4096,
        help="Number of documents to commit per vector store.add() call (avoids frequent FAISS re-train warnings)",
    )
    parser.add_argument(
        "--auto-populate-db",
        dest="auto_populate_db",
        action="store_true",
        default=None,
        help="If --skip-index and the vector store exists but documents DB is empty, automatically populate DB from BEIR corpus.jsonl",
    )
    parser.add_argument(
        "--no-auto-populate-db",
        dest="auto_populate_db",
        action="store_false",
        help="If supplied, do not auto-populate DB when --skip-index is used",
    )

    args = parser.parse_args()

    # Parse K values
    k_values = [int(k.strip()) for k in args.k_values.split(",")]

    # Initialize tester with data folder
    tester = RAGTester(
        args.questions,
        args.data_folder,
        ground_truth_file=args.ground_truth,
        mode=args.mode,
        skip_index=args.skip_index,
        index_batch_size=args.index_batch_size,
        index_sample_size=args.index_sample_size,
        index_commit_size=args.index_commit_size,
    )

    # Set indexing options
    tester.index_batch_size = args.index_batch_size
    tester.index_sample_size = args.index_sample_size
    tester.index_sample_seed = args.index_sample_seed
    tester.skip_index = args.skip_index
    tester.save_processed_corpus = args.save_processed_corpus
    tester.index_commit_size = args.index_commit_size
    # Default behavior: auto-populate if skip-index, unless user used --no-auto-populate-db
    if args.auto_populate_db is None:
        tester.auto_populate_db = bool(args.skip_index)
    else:
        tester.auto_populate_db = bool(args.auto_populate_db)

    # Run tests
    results = tester.run_all_tests(
        easy_limit=args.easy_limit,
        medium_limit=args.medium_limit,
        hard_limit=args.hard_limit,
        k_values=k_values,
    )

    # Save and display results
    tester.save_results(args.output)
    tester.print_summary()


if __name__ == "__main__":
    main()
