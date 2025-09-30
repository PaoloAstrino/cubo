#!/usr/bin/env python3
"""
Story Evaluation Script for CUBO Architecture
Tests 20 diverse questions across frog_story.txt and horse_story.txt
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any
import logging

# Add src to path - handle both relative and absolute paths
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from src.model_loader import ModelManager
from src.retriever import DocumentRetriever
from src.generator import ResponseGenerator
from src.document_loader import DocumentLoader
from evaluation.metrics import AdvancedEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StoryEvaluator:
    """Evaluates CUBO performance on story-based questions."""

    def __init__(self):
        self.model_loader = ModelManager()
        self.document_loader = DocumentLoader()
        self.generator = ResponseGenerator()
        self.evaluator = AdvancedEvaluator()

        # Load model and initialize retriever
        self.model = self.model_loader.load_model()
        self.retriever = DocumentRetriever(self.model, use_sentence_window=True)

        # Load story documents
        self._load_story_documents()

    def _load_story_documents(self):
        """Load the two story documents."""
        data_dir = Path(__file__).parent.parent / "data"
        story_files = ["frog_story.txt", "horse_story.txt"]

        for story_file in story_files:
            filepath = data_dir / story_file
            if filepath.exists():
                logger.info(f"Loading {story_file}...")
                documents = self.document_loader.load_single_document(str(filepath))
                if documents:
                    self.retriever.add_document(str(filepath), documents)
                    logger.info(f"Loaded {len(documents)} chunks from {story_file}")
                else:
                    logger.warning(f"Failed to load content from {story_file}")
            else:
                logger.error(f"Story file not found: {filepath}")

    def get_evaluation_questions(self) -> List[str]:
        """Return 20 diverse questions testing different aspects of the stories."""
        return [
            # Basic factual recall
            "What is the main character's name in the frog story?",
            "What animal is the main character in the horse story?",
            "Where does the frog live?",
            "What is the horse's name in the story?",

            # Character relationships and interactions
            "Who helps the frog in his adventure?",
            "Does the horse have any friends mentioned in the story?",
            "What does the frog learn from his experiences?",
            "How does the horse feel about his life?",

            # Plot and sequence questions
            "What happens when the frog jumps into the pond?",
            "Where does the horse go during his adventure?",
            "What is the most exciting part of the frog's story?",
            "What challenges does the horse face?",

            # Comparative questions (across both stories)
            "Which animal has a more adventurous story, the frog or the horse?",
            "Do both animals learn something from their experiences?",
            "Which story involves more water-related activities?",
            "Are there any similarities between the frog and horse's personalities?",

            # Inference and deeper understanding
            "Why do you think the frog is always jumping around?",
            "What might the horse be dreaming about at the end of the story?",
            "What lesson could children learn from the frog's story?",
            "How might the horse's story end if it continued?",

            # Specific details and themes
            "What colors are mentioned in the frog story?",
            "Does the horse story mention any other animals?",
            "What makes the frog happy?",
            "What is the horse's favorite activity?"
        ]

    def evaluate_question(self, question: str) -> Dict[str, Any]:
        """Evaluate a single question through the full CUBO pipeline."""
        start_time = time.time()

        try:
            # Step 1: Retrieve relevant documents
            retrieval_start = time.time()
            relevant_docs_data = self.retriever.retrieve_top_documents(question, top_k=6)
            retrieval_time = time.time() - retrieval_start

            # Extract context and sources
            relevant_docs = [doc_data['document'] for doc_data in relevant_docs_data]
            context = "\n\n".join(relevant_docs) if relevant_docs else ""

            sources = []
            for doc_data in relevant_docs_data:
                filename = doc_data['metadata'].get('filename', 'Unknown')
                if filename not in sources:
                    sources.append(filename)

            # Step 2: Generate response
            generation_start = time.time()
            response = self.generator.generate_response(question, context=context)
            generation_time = time.time() - generation_start

            # Step 3: Evaluate quality metrics
            evaluation_start = time.time()
            if context and response:
                # Use basic evaluation for now (can be enhanced)
                answer_relevance = self._compute_answer_relevance(question, response)
                context_relevance = self._compute_context_relevance(question, relevant_docs)
                groundedness = self._compute_groundedness(relevant_docs, response)
            else:
                answer_relevance = context_relevance = groundedness = 0
            evaluation_time = time.time() - evaluation_start

            total_time = time.time() - start_time

            return {
                'question': question,
                'response': response,
                'context': context[:500] + "..." if len(context) > 500 else context,
                'sources': sources,
                'metrics': {
                    'answer_relevance': answer_relevance,
                    'context_relevance': context_relevance,
                    'groundedness': groundedness
                },
                'timing': {
                    'retrieval': retrieval_time,
                    'generation': generation_time,
                    'evaluation': evaluation_time,
                    'total': total_time
                },
                'success': True
            }

        except Exception as e:
            logger.error(f"Failed to evaluate question '{question}': {e}")
            return {
                'question': question,
                'error': str(e),
                'success': False,
                'timing': {'total': time.time() - start_time}
            }

    def _compute_answer_relevance(self, question: str, answer: str) -> float:
        """Basic answer relevance computation."""
        if not answer or not question:
            return 0.0

        # Simple keyword overlap (can be enhanced with embeddings)
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())

        overlap = len(question_words.intersection(answer_words))
        union = len(question_words.union(answer_words))

        return overlap / union if union > 0 else 0.0

    def _compute_context_relevance(self, question: str, contexts: List[str]) -> float:
        """Basic context relevance computation."""
        if not contexts or not question:
            return 0.0

        question_words = set(question.lower().split())
        total_relevance = 0.0

        for context in contexts:
            context_words = set(context.lower().split())
            overlap = len(question_words.intersection(context_words))
            union = len(question_words.union(context_words))
            total_relevance += overlap / union if union > 0 else 0.0

        return total_relevance / len(contexts) if contexts else 0.0

    def _compute_groundedness(self, contexts: List[str], answer: str) -> float:
        """Basic groundedness computation."""
        if not contexts or not answer:
            return 0.0

        answer_words = set(answer.lower().split())
        total_groundedness = 0.0

        for context in contexts:
            context_words = set(context.lower().split())
            overlap = len(answer_words.intersection(context_words))
            total_groundedness += overlap / len(answer_words) if answer_words else 0.0

        return total_groundedness / len(contexts) if contexts else 0.0

    def run_full_evaluation(self) -> Dict[str, Any]:
        """Run evaluation on all 20 questions."""
        questions = self.get_evaluation_questions()
        results = []

        logger.info(f"Starting evaluation of {len(questions)} questions...")

        for i, question in enumerate(questions, 1):
            logger.info(f"Evaluating question {i}/{len(questions)}: {question[:50]}...")
            result = self.evaluate_question(question)
            results.append(result)

            # Show progress
            if result['success']:
                metrics = result['metrics']
                logger.info(".2f")
            else:
                logger.warning(f"Question {i} failed: {result.get('error', 'Unknown error')}")

        # Calculate summary statistics
        successful_results = [r for r in results if r['success']]

        if successful_results:
            avg_answer_relevance = sum(r['metrics']['answer_relevance'] for r in successful_results) / len(successful_results)
            avg_context_relevance = sum(r['metrics']['context_relevance'] for r in successful_results) / len(successful_results)
            avg_groundedness = sum(r['metrics']['groundedness'] for r in successful_results) / len(successful_results)
            avg_total_time = sum(r['timing']['total'] for r in successful_results) / len(successful_results)
        else:
            avg_answer_relevance = avg_context_relevance = avg_groundedness = avg_total_time = 0

        summary = {
            'total_questions': len(questions),
            'successful_questions': len(successful_results),
            'success_rate': len(successful_results) / len(questions),
            'average_metrics': {
                'answer_relevance': avg_answer_relevance,
                'context_relevance': avg_context_relevance,
                'groundedness': avg_groundedness
            },
            'average_timing': {
                'total_time': avg_total_time
            },
            'results': results
        }

        return summary

    def save_results(self, results: Dict[str, Any], output_file: str = "story_evaluation_results.json"):
        """Save evaluation results to JSON file."""
        output_path = Path(__file__).parent / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {output_path}")

def main():
    """Main evaluation function."""
    print("🐸🐎 CUBO Story Evaluation System")
    print("=" * 50)

    try:
        evaluator = StoryEvaluator()
        results = evaluator.run_full_evaluation()

        # Print summary
        print("\n📊 EVALUATION SUMMARY")
        print("-" * 30)
        print(f"Total Questions: {results['total_questions']}")
        print(f"Successful: {results['successful_questions']}")
        print(".1%")

        if results['successful_questions'] > 0:
            print("\n📈 AVERAGE METRICS")
            print("-" * 20)
            metrics = results['average_metrics']
            print(".3f")
            print(".3f")
            print(".3f")
            print(".2f")

        # Save detailed results
        evaluator.save_results(results)
        print(f"\n💾 Detailed results saved to story_evaluation_results.json")

        # Show sample results
        print("\n🔍 SAMPLE RESULTS")
        print("-" * 20)
        successful_results = [r for r in results['results'] if r['success']]
        for i, result in enumerate(successful_results[:3], 1):
            print(f"\n{i}. {result['question']}")
            print(f"   Answer: {result['response'][:100]}...")
            print(".2f")
            print(f"   Sources: {result['sources']}")

        print("\n✅ Evaluation completed successfully!")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        print(f"\n❌ Evaluation failed: {e}")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())