#!/usr/bin/env python3
"""
CUBO Evaluation Script
Run evaluations on saved query data that hasn't been evaluated yet.
"""

import sys
import os
import logging
import argparse
from typing import List, Optional
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.database import EvaluationDatabase
from evaluation.integration import get_evaluation_integrator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def evaluate_saved_queries(limit: Optional[int] = None, session_id: Optional[str] = None):
    """
    Evaluate saved queries that don't have evaluation metrics yet.

    Args:
        limit: Maximum number of queries to evaluate (None for all)
        session_id: Only evaluate queries from this session (None for all sessions)
    """
    try:
        # Initialize database and evaluator
        db = EvaluationDatabase()
        integrator = get_evaluation_integrator()

        # Get queries that need evaluation (where metrics are None/null)
        queries_to_evaluate = db.get_queries_needing_evaluation(session_id=session_id, limit=limit)

        if not queries_to_evaluate:
            logger.info("No queries found that need evaluation.")
            return

        logger.info(f"Found {len(queries_to_evaluate)} queries to evaluate.")

        evaluated_count = 0
        failed_count = 0

        for query_data in queries_to_evaluate:
            try:
                logger.info(f"Evaluating query: {query_data['question'][:50]}...")

                # Run evaluation with timeout to prevent hanging
                import asyncio
                try:
                    evaluation_result = await asyncio.wait_for(
                        integrator.evaluate_query(
                            question=query_data['question'],
                            answer=query_data['answer'],
                            contexts=query_data['contexts'],
                            response_time=query_data['response_time'],
                            model_used=query_data['model_used']
                        ),
                        timeout=60.0  # 60 second timeout per evaluation
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"⚠ Evaluation timed out for query: {query_data['question'][:50]}... - will retry later")
                    # Don't increment failed_count, leave record for retry
                    continue

                if evaluation_result:
                    # Update the existing record with evaluation metrics
                    db.update_evaluation_metrics(
                        evaluation_id=query_data['id'],
                        answer_relevance=evaluation_result.answer_relevance_score,
                        context_relevance=evaluation_result.context_relevance_score,
                        groundedness=evaluation_result.groundedness_score,
                        llm_metrics=evaluation_result.llm_metrics
                    )

                    evaluated_count += 1
                    logger.info(f"✓ Evaluation completed: AR={evaluation_result.answer_relevance_score:.2f}, "
                                f"CR={evaluation_result.context_relevance_score:.2f}, "
                                f"G={evaluation_result.groundedness_score:.2f}")
                else:
                    # LLM evaluation failed - leave record unchanged so it will be retried later
                    logger.warning(f"⚠ LLM evaluation failed for query: {query_data['question'][:50]}... - will retry later")
                    # Don't increment failed_count, don't update database

                # Small delay to avoid overwhelming the API
                time.sleep(0.5)

            except KeyboardInterrupt:
                logger.warning("Evaluation interrupted by user. Saving progress...")
                break
            except Exception as e:
                # For any other error, leave the record unchanged for retry
                logger.warning(f"⚠ Evaluation error for query {query_data['id']}: {e} - will retry later")
                # Don't increment failed_count, leave record for retry

        logger.info(f"Evaluation complete. Evaluated: {evaluated_count}, Failed: {failed_count}")
        if failed_count == 0:
            logger.info("All queries were either successfully evaluated or deferred for retry.")
        else:
            logger.warning(f"{failed_count} queries had permanent failures and were not processed.")

    except KeyboardInterrupt:
        logger.warning("Evaluation interrupted by user")
        logger.info(f"Partial results - Evaluated: {evaluated_count}, Failed: {failed_count}")
        logger.info("Interrupted queries will be retried in the next run.")
    except Exception as e:
        logger.error(f"Failed to run evaluations: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Evaluate saved query data")
    parser.add_argument('--limit', type=int, help='Maximum number of queries to evaluate')
    parser.add_argument('--session', type=str, help='Only evaluate queries from this session ID')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Run the async evaluation function
    import asyncio
    asyncio.run(evaluate_saved_queries(limit=args.limit, session_id=args.session))

if __name__ == "__main__":
    main()