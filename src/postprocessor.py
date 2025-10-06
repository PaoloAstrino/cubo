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
        return [self._process_single_result(result) for result in retrieval_results]

    def _process_single_result(self, result: Dict) -> Dict:
        """Process a single retrieval result."""
        processed_result = result.copy()
        
        if self._should_replace_with_window(result):
            self._replace_with_window_context(processed_result, result)
        
        return processed_result

    def _should_replace_with_window(self, result: Dict) -> bool:
        """Check if result should be replaced with window context."""
        metadata = result.get('metadata', {})
        if self.target_metadata_key not in metadata:
            return False
        
        window_text = metadata[self.target_metadata_key]
        return window_text and len(window_text.strip()) > 0

    def _replace_with_window_context(self, processed_result: Dict, original_result: Dict):
        """Replace document text with window context."""
        metadata = original_result.get('metadata', {})
        window_text = metadata[self.target_metadata_key]
        
        processed_result['document'] = window_text
        logger.debug(
            f"Replaced sentence with window context "
            f"({len(window_text)} chars)"
        )
