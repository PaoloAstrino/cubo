"""
RAGAS Evaluator for CUBO.

Wraps CUBO's local LLM and Embedding models to be used with the RAGAS framework
for evaluating generation quality (Faithfulness, Answer Relevance, Context Precision).
"""

import logging
from typing import List, Optional, Any, Dict

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from pydantic import Field

from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

# For retry wrapper
import json
import time
import os
from pathlib import Path
from langchain_core.messages import HumanMessage
from typing import Optional
from pydantic import BaseModel
from datasets import Dataset

from cubo.embeddings.embedding_generator import EmbeddingGenerator
from cubo.processing.generator import create_response_generator

logger = logging.getLogger(__name__)


class CuboRagasEmbeddings(Embeddings):
    """Wrapper for CUBO EmbeddingGenerator to work with RAGAS/LangChain."""

    def __init__(self):
        self.generator = EmbeddingGenerator()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        # Convert numpy arrays to lists for RAGAS compatibility
        embeddings = self.generator.encode(texts)
        return [e.tolist() if hasattr(e, "tolist") else e for e in embeddings]

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        embedding = self.generator.encode([text])[0]
        return embedding.tolist() if hasattr(embedding, "tolist") else embedding


class CuboRagasLLM(BaseChatModel):
    """Wrapper for CUBO Local/Ollama LLM to work with RAGAS/LangChain."""

    generator: Any = Field(default=None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Use factory to get the active generator (Ollama or Local)
        self.generator = create_response_generator()

    @property
    def _llm_type(self) -> str:
        return "cubo-local"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate response from messages."""
        # Convert LangChain messages to CUBO format
        cubo_messages = []
        for m in messages:
            if isinstance(m, HumanMessage):
                role = "user"
            elif isinstance(m, AIMessage):
                role = "assistant"
            elif isinstance(m, SystemMessage):
                role = "system"
            else:
                role = "user"
            cubo_messages.append({"role": role, "content": m.content})

        # Extract query from the last user message
        query = cubo_messages[-1]["content"]
        
        # Extract context from kwargs if provided by RAGAS (some metrics pass it)
        context = kwargs.get("context", "")
        
        # If no explicit context, build full prompt from all messages for judge compatibility
        if not context:
            full_prompt = "\n".join([m["content"] for m in cubo_messages])
            response_text = self.generator.generate_response(query=full_prompt, context="")
        else:
            # Use explicit context for faithful generation
            response_text = self.generator.generate_response(query=query, context=context)

        return ChatResult(
            generations=[
                ChatGeneration(message=AIMessage(content=response_text))
            ]
        )


class RetryingChatLLM(BaseChatModel):
    """Wrap a Chat LLM and retry on invalid (non-JSON) outputs.

    If the underlying LLM returns text that is not valid JSON, the wrapper will
    re-prompt the LLM up to `max_retries` times asking it to return ONLY a valid
    JSON object (no extra text). This is a defensive measure for RAGAS schema parsing.
    """

    wrapped: Any = Field(default=None, exclude=True)
    max_retries: int = Field(default=2)
    retry_pause: float = Field(default=0.5)
    debug_dir: Optional[str] = Field(default=None)
    
    class Config:
        arbitrary_types_allowed = True

    def __init__(self, wrapped: Any, max_retries: int = 2, debug_dir: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.wrapped = wrapped
        self.max_retries = max_retries
        self.debug_dir = debug_dir
        if self.debug_dir:
            Path(self.debug_dir).mkdir(parents=True, exist_ok=True)
            logger.info(f"RetryingChatLLM debug directory initialized: {self.debug_dir}")

    @property
    def _llm_type(self) -> str:
        return getattr(self.wrapped, '_llm_type', 'retrying-chat')

    def _normalize_result(self, res: Any) -> ChatResult:
        """Normalize various LLM return types to ChatResult."""
        # If it's already a ChatResult, return as-is
        if isinstance(res, ChatResult):
            return res
        # If it's a tuple (e.g., (ChatResult, list)), extract the ChatResult
        if isinstance(res, tuple) and len(res) > 0:
            if isinstance(res[0], ChatResult):
                return res[0]
            # Try to extract from nested tuple
            return self._normalize_result(res[0])
        # If it's a list, try first element
        if isinstance(res, list) and len(res) > 0:
            return self._normalize_result(res[0])
        # Fallback: return empty ChatResult
        logger.warning(f"Could not normalize LLM result of type {type(res).__name__}")
        return ChatResult(generations=[])

    def _extract_text_from_result(self, res: Any) -> str:
        """Extract text from various result formats."""
        try:
            # First normalize to ChatResult
            result = self._normalize_result(res)
            if not result.generations or len(result.generations) == 0:
                return ""
            gen = result.generations[0]
            # Support both message and text styles
            if hasattr(gen, "message") and getattr(gen.message, "content", None) is not None:
                return gen.message.content
            return getattr(gen, "text", "")
        except Exception as e:
            logger.warning(f"Error extracting text from result: {e}")
            return ""

    def _generate(self, *args, **kwargs) -> ChatResult:
        # Forward-compatible signature: accept positional args (messages, stop, run_manager, ...)
        # Extract messages and stop from args/kwargs for compatibility with different call styles
        messages = kwargs.get('messages') if 'messages' in kwargs else (args[0] if len(args) > 0 else [])
        stop = kwargs.get('stop') if 'stop' in kwargs else (args[1] if len(args) > 1 else None)

        # Use a simple counter tracking instead of Field
        if not hasattr(self, '_call_counter'):
            self._call_counter = 0
        self._call_counter += 1
        call_id = self._call_counter
        
        # First attempt
        try:
            # Call wrapped._generate() directly to get ChatResult
            # Pass all extra kwargs that _generate might need (e.g., context for RAGAS)
            res = self.wrapped._generate(messages=messages, stop=stop, **{k: v for k, v in kwargs.items() if k not in ('messages','stop')})
        except Exception as e:
            # If the underlying LLM fails outright, re-raise
            logger.error(f"[RetryingChatLLM call={call_id}] Underlying LLM failed: {e}")
            raise

        text = self._extract_text_from_result(res)
        
        # Save initial attempt if debug enabled
        if self.debug_dir:
            try:
                debug_file = Path(self.debug_dir) / f"call_{call_id:04d}_attempt_0.txt"
                with open(debug_file, "w", encoding="utf-8") as f:
                    f.write(f"Initial response (call {call_id}):\n\n{text}")
            except Exception as e:
                logger.warning(f"Failed to write debug file: {e}")

        # Quick JSON check
        try:
            json.loads(text)
            logger.debug(f"[RetryingChatLLM call={call_id}] Valid JSON on first attempt")
            return res
        except Exception as parse_err:
            # Not valid JSON; attempt retries
            logger.warning(f"[RetryingChatLLM call={call_id}] Initial response not valid JSON: {str(parse_err)[:100]}")
            last_text = text
            last_error = str(parse_err)
            
            for attempt in range(1, self.max_retries + 1):
                logger.info(f"[RetryingChatLLM call={call_id}] Retry attempt {attempt}/{self.max_retries}")
                
                # Craft a strict follow-up instruction asking for valid JSON only
                # Be very explicit about the format requirement
                followup = HumanMessage(content=(
                    "CRITICAL: Your previous response was not valid JSON and caused a parsing error.\n"
                    f"Error: {last_error[:200]}\n\n"
                    "You MUST respond with ONLY a valid JSON object. Requirements:\n"
                    "1. Start with { and end with }\n"
                    "2. Use double quotes for all strings\n"
                    "3. No surrounding text, explanations, or markdown\n"
                    "4. No trailing commas\n"
                    "5. Properly escape special characters\n\n"
                    "If you cannot provide the requested information in valid JSON format, "
                    "return exactly: {\"error\": \"UNPARSABLE\"}\n\n"
                    f"Previous invalid output was:\n{last_text[:500]}\n\n"
                    "Provide ONLY the corrected JSON now:"
                ))
                
                try:
                    raw_res2 = self.wrapped.generate(messages=messages + [followup], stop=stop)
                    res2 = self._normalize_result(raw_res2)
                except Exception as retry_err:
                    logger.warning(f"[RetryingChatLLM call={call_id}] Retry {attempt} LLM call failed: {retry_err}")
                    time.sleep(self.retry_pause)
                    continue
                    
                text2 = self._extract_text_from_result(res2)
                
                # Save retry attempt if debug enabled
                if self.debug_dir:
                    try:
                        debug_file = Path(self.debug_dir) / f"call_{call_id:04d}_attempt_{attempt}.txt"
                        with open(debug_file, "w", encoding="utf-8") as f:
                            f.write(f"Retry {attempt} response (call {call_id}):\n\n{text2}")
                    except Exception as e:
                        logger.warning(f"Failed to write debug file: {e}")
                
                try:
                    json.loads(text2)
                    logger.info(f"[RetryingChatLLM call={call_id}] Valid JSON on retry attempt {attempt}")
                    return res2
                except Exception as retry_parse_err:
                    logger.warning(f"[RetryingChatLLM call={call_id}] Retry {attempt} still not valid JSON: {str(retry_parse_err)[:100]}")
                    last_text = text2
                    last_error = str(retry_parse_err)
                    time.sleep(self.retry_pause)
                    continue
                    
            # If we exhaust retries, log final failure and return the last response (so upstream can log/handle)
            logger.error(f"[RetryingChatLLM call={call_id}] Exhausted all {self.max_retries} retries; returning last response")
            
            # Save final failure summary if debug enabled
            if self.debug_dir:
                try:
                    summary_file = Path(self.debug_dir) / f"call_{call_id:04d}_FAILED.json"
                    with open(summary_file, "w", encoding="utf-8") as f:
                        json.dump({
                            "call_id": call_id,
                            "max_retries": self.max_retries,
                            "final_text": last_text[:1000],
                            "final_error": last_error,
                            "status": "EXHAUSTED_RETRIES"
                        }, f, indent=2)
                except Exception as e:
                    logger.warning(f"Failed to write failure summary: {e}")
                    
            return res



def run_ragas_evaluation(
    questions: List[str],
    contexts: List[List[str]],
    ground_truths: List[str],
    answers: List[str],
    llm: Optional[Any] = None,
    max_workers: Optional[int] = None,
    save_per_sample_path: Optional[str] = None,
    retrieval_times: Optional[List[float]] = None,
    generation_times: Optional[List[float]] = None,
) -> Dict[str, float]:
    """
    Run RAGAS evaluation on a set of RAG results.

    Args:
        questions: List of queries
        contexts: List of lists of retrieved chunks (strings)
        ground_truths: List of expected answers (strings)
        answers: List of generated answers
        llm: Optional judge LLM; if None, uses local CUBO LLM with retry wrapper
        max_workers: If 1, run per-sample serial evaluation (helps avoid concurrent judge timeouts)
        save_per_sample_path: Optional JSONL path to save per-sample raw outputs
        retrieval_times: Optional list of retrieval latencies (seconds) per sample
        generation_times: Optional list of generation latencies (seconds) per sample

    Returns:
        Dictionary of metric scores.
    """
    # Initialize models
    if llm is not None:
        eval_llm = llm
    else:
        # Wrap local LLM with RetryingChatLLM to handle non-JSON outputs common in local models
        base_llm = CuboRagasLLM()
        eval_llm = RetryingChatLLM(wrapped=base_llm, max_retries=2)
        logger.info("Using CuboRagasLLM wrapped with RetryingChatLLM for evaluation")

    embeddings = CuboRagasEmbeddings()

    # Define metrics
    metrics = [
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
    ]

    # If user requests serial execution (max_workers==1), evaluate each sample individually
    if max_workers == 1:
        logger.info("Running RAGAS evaluation in serial mode (max_workers=1) to reduce judge concurrency")
        aggregate = {}
        valid_counts = {}
        n = len(questions)
        for i in range(n):
            single_data = {
                "question": [questions[i]],
                "contexts": [contexts[i]],
                "answer": [answers[i]],
                "ground_truth": [ground_truths[i]],
            }
            single_dataset = Dataset.from_dict(single_data)
            try:
                res = evaluate(dataset=single_dataset, metrics=metrics, llm=eval_llm, embeddings=embeddings)
            except Exception as e:
                logger.warning(f"RAGAS per-sample evaluation failed for index {i}: {e}")
                continue

            # Normalize result to a dict of numeric metrics
            try:
                if hasattr(res, "to_dict"):
                    rd = res.to_dict()
                elif hasattr(res, "items"):
                    rd = dict(res)
                else:
                    rd = {k: float(v) for k, v in res.items()}
            except Exception:
                rd = {}

            for k, v in rd.items():
                try:
                    fv = float(v)
                except Exception:
                    # Skip non-numeric entries
                    continue
                aggregate[k] = aggregate.get(k, 0.0) + fv
                valid_counts[k] = valid_counts.get(k, 0) + 1

        # Compute averages for metrics with at least one valid value
        averaged = {k: (aggregate[k] / valid_counts[k]) for k in aggregate.keys() if valid_counts.get(k, 0) > 0}
        return averaged

    # Default: run evaluation over the full dataset (may be parallel depending on RAGAS defaults)
    data = {
        "question": questions,
        "contexts": contexts,
        "answer": answers,
        "ground_truth": ground_truths,
    }
    dataset = Dataset.from_dict(data)

    results = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=eval_llm,
        embeddings=embeddings,
    )

    # Optionally save per-sample raw outputs
    if save_per_sample_path:
        try:
            from pathlib import Path
            Path(save_per_sample_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_per_sample_path, 'w', encoding='utf-8') as f:
                for i in range(len(questions)):
                    sample = {
                        "sample_id": i,
                        "question": questions[i],
                        "contexts": contexts[i],
                        "ground_truth": ground_truths[i],
                        "answer": answers[i],
                        "retrieval_time": retrieval_times[i] if retrieval_times else None,
                        "generation_time": generation_times[i] if generation_times else None,
                    }
                    # Try to extract per-sample metrics from results if available
                    if hasattr(results, "to_pandas"):
                        try:
                            df = results.to_pandas()
                            if i < len(df):
                                for col in df.columns:
                                    sample[f"ragas_{col}"] = float(df[col].iloc[i]) if not pd.isna(df[col].iloc[i]) else None
                        except Exception as e:
                            logger.warning(f"Could not extract per-sample metrics: {e}")
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            logger.info(f"Saved per-sample raw outputs to {save_per_sample_path}")
        except Exception as e:
            logger.error(f"Failed to save per-sample outputs: {e}")

    # Normalize results to dict if possible
    try:
        if hasattr(results, "to_dict"):
            return results.to_dict()
        if hasattr(results, "items"):
            return dict(results)
    except Exception:
        pass

    return {"raw": str(results)}
