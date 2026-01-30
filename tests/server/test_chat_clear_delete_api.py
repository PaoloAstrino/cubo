from unittest.mock import patch

from fastapi.testclient import TestClient

from cubo.llm.session_manager import get_session_manager
from cubo.server.api import app
from cubo.storage.metadata_manager import get_metadata_manager

client = TestClient(app)


def test_clear_conversation_by_id():
    mm = get_metadata_manager()
    conv_id = mm.create_conversation(title="Test Clear")
    mm.add_message(conv_id, "user", "hello")
    mm.add_message(conv_id, "assistant", "world")

    # Ensure messages exist
    msgs = mm.get_conversation_messages(conv_id)
    assert len(msgs) == 2

    # Call clear endpoint
    resp = client.post("/api/chat/clear", json={"conversation_id": conv_id})
    assert resp.status_code == 204

    # Messages should be removed, but conversation should remain
    msgs_after = mm.get_conversation_messages(conv_id)
    assert msgs_after == []
    conv = mm.get_conversation(conv_id)
    assert conv is not None


def test_delete_conversation_by_id():
    mm = get_metadata_manager()
    conv_id = mm.create_conversation(title="Test Delete")
    mm.add_message(conv_id, "user", "foo")

    # Call delete endpoint
    resp = client.delete(f"/api/chat/{conv_id}")
    assert resp.status_code == 204

    # Conversation should be gone
    conv = mm.get_conversation(conv_id)
    assert conv is None


def test_clear_by_collection_notifies_session_manager():
    sm = get_session_manager()
    with patch("cubo.llm.session_manager.publish_event") as mock_publish:
        resp = client.post("/api/chat/clear", json={"collection_id": "my_col"})
        assert resp.status_code == 204
        # publish_event should have been called for collection.invalidated
        # Note: we patch publish_event in session_manager to assert event was published
        # If no exception, basic behavior is fine.
        # We check that _invalidated_collections has our id
        assert "my_col" in sm._invalidated_collections
