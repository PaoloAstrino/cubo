"""
Postprocessors for enhancing retrieval results in sentence window retrieval.
"""

from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class WindowReplacementPostProcessor:
    """Replaces single sentence text with full window context."""

    def __init__(self, target_metadata_key: str = "window"):
        self.target_metadata_key = target_metadata_key

    def postprocess_results(self, retrieval_results: List[Dict]) -> List[Dict]:
        """
        Replace document text with window context from metadata.
        """
        processed_results = []

        for result in retrieval_results:
            processed_result = result.copy()
            metadata = result.get('metadata', {})

            # Replace text with window if available
            if self.target_metadata_key in metadata:
                window_text = metadata[self.target_metadata_key]
                if window_text and len(window_text.strip()) > 0:
                    processed_result['document'] = window_text
                    logger.debug(
                        f"Replaced sentence with window context "
                        f"({len(window_text)} chars)"
                    )

            processed_results.append(processed_result)

        return processed_results
