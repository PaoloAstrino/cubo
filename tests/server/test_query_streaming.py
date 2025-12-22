import json
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from cubo.config import config

@pytest.fixture
def mock_cubo_app():
    mock_app = MagicMock()
    mock_app.retriever = MagicMock()
    mock_app.generator = MagicMock()
    mock_app.vector_store = MagicMock()
    # Mock collection count to avoid 503
    mock_app.retriever.collection.count.return_value = 10
    return mock_app

@pytest.fixture
def client(mock_cubo_app):
    with patch("cubo.server.api.cubo_app", mock_cubo_app):
        from cubo.server.api import app
        yield TestClient(app), mock_cubo_app

def test_query_streaming_disabled_by_config(client):
    test_client, mock_app = client
    # Ensure config is disabled
    config.set("llm.enable_streaming", False)
    
    # Mock standard response
    mock_app.query_retrieve.return_value = []
    mock_app.generate_response_safe.return_value = "Standard answer"
    
    response = test_client.post("/api/query", json={"query": "test", "stream": True})
    
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    data = response.json()
    assert data["answer"] == "Standard answer"

def test_query_streaming_enabled(client):
    test_client, mock_app = client
    config.set("llm.enable_streaming", True)
    
    # Mock retrieval
    mock_app.query_retrieve.return_value = [
        {"content": "doc1", "score": 0.9, "metadata": {"source": "file1.txt"}}
    ]
    
    # Mock streaming generator
    def stream_gen(query, context, trace_id):
        yield {"type": "token", "delta": "Hello"}
        yield {"type": "token", "delta": " World"}
        yield {"type": "done", "answer": "Hello World", "trace_id": trace_id}
        
    mock_app.generate_response_stream.side_effect = stream_gen
    
    response = test_client.post("/api/query", json={"query": "test", "stream": True})
    
    assert response.status_code == 200
    assert "application/x-ndjson" in response.headers["content-type"]
    
    lines = response.text.strip().split('\n')
    events = [json.loads(line) for line in lines]
    
    # Check source event (emitted by _query_stream_generator before calling generator)
    # Note: The implementation emits sources first
    source_event = next(e for e in events if e['type'] == 'source')
    assert source_event['content'] == 'doc1'
    
    # Check token events
    tokens = [e['delta'] for e in events if e['type'] == 'token']
    assert tokens == ['Hello', ' World']
    
    # Check done event
    done_event = next(e for e in events if e['type'] == 'done')
    assert done_event['answer'] == 'Hello World'
