from cubo.llm.session_manager import get_session_manager
from unittest.mock import patch


def test_clear_session_publishes_event():
    sm = get_session_manager()
    with patch("cubo.llm.session_manager.publish_event") as mock_publish:
        sm.clear_session("conv-1")
        mock_publish.assert_called_with("conversation.invalidated", {"id": "conv-1"})


def test_clear_sessions_by_collection_publishes_event():
    sm = get_session_manager()
    with patch("cubo.llm.session_manager.publish_event") as mock_publish:
        sm.clear_sessions_by_collection("col-1")
        mock_publish.assert_called_with("collection.invalidated", {"collection_id": "col-1"})
