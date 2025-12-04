import threading
import time
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


def test_generate_response_wrapper_called():
    """Verify that the API calls CUBOApp.generate_response_safe wrapper once.

    We mock `cubo.server.api.cubo_app` to a MagicMock and ensure generate_response_safe
    is invoked exactly once when hitting /api/query.
    """
    mock_app = MagicMock()
    # Ensure retriever exists and has at least one document (so query doesn't return 503)
    mock_app.retriever = MagicMock()
    mock_app.retriever.collection = MagicMock()
    mock_app.retriever.collection.count.return_value = 10
    mock_app.retriever.retrieve_top_documents.return_value = [
        {"document": "Example doc", "metadata": {"filename": "f.txt"}, "similarity": 0.9}
    ]
    # CUBOApp wrapper to be invoked
    mock_app.generate_response_safe.return_value = "Generated answer"

    with patch("cubo.server.api.cubo_app", mock_app):
        with patch("cubo.server.api.security_manager") as mock_security:
            mock_security.scrub.side_effect = lambda x: x
            from cubo.server.api import app
            client = TestClient(app)

            response = client.post("/api/query", json={"query": "What is this?", "top_k": 1})
            assert response.status_code == 200
            # Ensure wrapper called once
            assert mock_app.generate_response_safe.call_count == 1
            # Also ensure underlying generator wasn't called directly (good to check both)
            assert mock_app.generator.generate_response.call_count == 0
