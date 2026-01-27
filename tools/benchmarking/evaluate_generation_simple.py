"""
Simple Generation Quality Evaluation using Local LLM as Judge.

Evaluates end-to-end RAG quality (Faithfulness, Answer Relevancy)
by prompting the local LLM to act as a judge.
"""

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from cubo.core import CuboCore
from cubo.utils.logger import setup_logger

logger = setup_logger(__name__)

JUDGE_PROMPT = """You are an expert impartial judge evaluating the quality of an AI-generated answer based on a retrieved context.
Rate the answer on a scale of 1 to 5 for the following metrics.

METRICS:
1. FAITHFULNESS (1-5): Is the answer factually consistent with the provided context? (5 = perfect, 1 = major hallucinations)
2. RELEVANCY (1-5): Does the answer directly address the user's question? (5 = perfect, 1 = irrelevant)

INPUT:
Question: {question}
Context: {context}
Answer: {answer}

Provide your response in JSON format like this:
{{
  "faithfulness": 5,
  "relevancy": 4,
  "explanation": "..."
}}
"""


def load_test_queries(dataset_name: str, num_samples: int = 10) -> List[Dict[str, Any]]:
    """Small set of representative queries for each domain."""
    queries = {
        "politics": [
            {
                "question": "What are the three main institutions of the European Union?",
                "answer": "The European Parliament, the Council of the European Union, and the European Commission.",
            },
            {
                "question": "How often are European Parliament elections held?",
                "answer": "Every five years.",
            },
            {
                "question": "What is the role of the European Commission?",
                "answer": "It proposes new laws, manages EU policies, and allocates EU funding.",
            },
            {
                "question": "Which body represents the governments of EU member states?",
                "answer": "The Council of the European Union.",
            },
            {
                "question": "What is the European Single Market?",
                "answer": "An area where goods, services, capital, and people can move freely.",
            },
        ],
        "legal": [
            {
                "question": "What is the purpose of GDPR Article 5?",
                "answer": "It outlines the principles relating to the processing of personal data.",
            },
            {
                "question": "What are the common legal grounds for processing personal data under GDPR?",
                "answer": "Consent, contract, legal obligation, vital interests, public task, and legitimate interests.",
            },
            {
                "question": "What is a Data Processing Agreement (DPA)?",
                "answer": "A contract between a data controller and a data processor.",
            },
            {
                "question": "What constitutes a personal data breach under GDPR?",
                "answer": "A breach of security leading to accidental or unlawful destruction, loss, alteration, or unauthorized disclosure of personal data.",
            },
            {
                "question": "What is the 'Right to be Forgotten'?",
                "answer": "The right of data subjects to have their personal data erased under certain conditions.",
            },
        ],
    }

    return queries.get(dataset_name, queries["politics"])[:num_samples]


def parse_judge_response(text: str) -> Dict[str, Any]:
    """Extract JSON from LLM response."""
    try:
        # Find JSON block
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return {"faithfulness": 0, "relevancy": 0, "error": "No JSON found"}
    except Exception as e:
        return {"faithfulness": 0, "relevancy": 0, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Run generation evaluation with local LLM judge")
    parser.add_argument("--dataset", type=str, default="politics")
    parser.add_argument("--num-samples", type=int, default=5)
    args = parser.parse_args()

    cubo = CuboCore(laptop_mode=True)
    queries = load_test_queries(args.dataset, args.num_samples)

    results = []
    logger.info(f"Starting evaluation on {args.dataset}...")

    for i, q in enumerate(queries):
        logger.info(f"Query {i+1}/{len(queries)}: {q['question']}")

        # 1. Retrieve
        retrieval = cubo.retrieve(query=q["question"], top_k=3)
        context = "\n\n".join([c["text"] for c in retrieval["chunks"]])

        # 2. Generate
        answer = cubo.generate(query=q["question"], context=context)

        # 3. Judge
        prompt = JUDGE_PROMPT.format(question=q["question"], context=context, answer=answer)
        judge_raw = cubo.generate(query=prompt, context="")
        scores = parse_judge_response(judge_raw)

        results.append({"question": q["question"], "answer": answer, "scores": scores})
        logger.info(
            f"  Scores: Faithfulness={scores.get('faithfulness')}, Relevancy={scores.get('relevancy')}"
        )

    # Aggregates
    avg_f = sum(r["scores"].get("faithfulness", 0) for r in results) / len(results)
    avg_r = sum(r["scores"].get("relevancy", 0) for r in results) / len(results)

    print("\n" + "=" * 40)
    print(f"📊 Evaluation Results: {args.dataset}")
    print("=" * 40)
    print(f"Avg Faithfulness (1-5): {avg_f:.2f}")
    print(f"Avg Relevancy (1-5):    {avg_r:.2f}")
    print("=" * 40)

    # Save results
    output_path = Path(f"results/eval_{args.dataset}.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(
            {"avg_faithfulness": avg_f, "avg_relevancy": avg_r, "details": results}, f, indent=2
        )
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
