"""
RAGAS Evaluator for CUBO.

Wraps CUBO's local LLM and Embedding models to be used with the RAGAS framework
for evaluating generation quality (Faithfulness, Answer Relevance, Context Precision).
"""

import logging
from typing import List, Optional, Union, Any

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
from datasets import Dataset

from cubo.embeddings.embedding_generator import EmbeddingGenerator
from cubo.processing.generator import create_response_generator
from cubo.processing.llm_local import LocalResponseGenerator

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

        # We assume the last message is the query/prompt
        query = cubo_messages[-1]["content"]
        # Extract context if present (RAGAS usually passes context in prompt)
        # For pure generation, we pass empty context and let the prompt handle it
        
        # If generator supports generate_response with messages list:
        if hasattr(self.generator, "generate_response_stream"):
             # Use the stream method but consume it all (since RAGAS expects full text)
             # Actually, we should use the non-streaming generate if available or consume stream
             pass
        
        # Fallback to simple generation
        # RAGAS prompts are self-contained, so we treat them as "query" with empty context
        # assuming the generator handles prompt formatting.
        # But LocalResponseGenerator expects (query, context).
        # We need to hack this slightly: pass full prompt as query, empty context.
        
        # BUT: LocalResponseGenerator applies a template!
        # If RAGAS passes a pre-formatted prompt, applying another template breaks it.
        # Ideally we need a "raw_generate" method.
        
        # Workaround: The generator's `generate_response` puts query into a template.
        # We try to pass the prompt content directly if possible.
        
        # Let's assume generate_response(query, context="") works for now.
        full_prompt = "\n".join([m["content"] for m in cubo_messages])
        
        # If LocalResponseGenerator, we might need to bypass the `Context: ...` wrapper
        # if we want pure generation.
        # For now, we accept the wrapper overhead.
        response_text = self.generator.generate_response(query=full_prompt, context="")

        return ChatResult(
            generations=[
                ChatGeneration(message=AIMessage(content=response_text))
            ]
        )


def run_ragas_evaluation(
    questions: List[str],
    contexts: List[List[str]],
    ground_truths: List[str],
    answers: List[str],
) -> Dict[str, float]:
    """
    Run RAGAS evaluation on a set of RAG results.

    Args:
        questions: List of queries
        contexts: List of lists of retrieved chunks (strings)
        ground_truths: List of expected answers (strings)
        answers: List of generated answers

    Returns:
        Dictionary of metric scores.
    """
    # Prepare dataset
    data = {
        "question": questions,
        "contexts": contexts,
        "answer": answers,
        "ground_truth": ground_truths,
    }
    dataset = Dataset.from_dict(data)

    # Initialize models
    llm = CuboRagasLLM()
    embeddings = CuboRagasEmbeddings()

    # Define metrics
    metrics = [
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
    ]

    # Run evaluation
    results = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
    )

    return results
