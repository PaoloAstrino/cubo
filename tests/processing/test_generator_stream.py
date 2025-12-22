import pytest
from unittest.mock import MagicMock, patch
from cubo.processing.generator import ResponseGenerator

@pytest.fixture
def mock_ollama():
    with patch('cubo.processing.generator.ollama') as mock:
        with patch('cubo.processing.generator.OLLAMA_AVAILABLE', True):
            yield mock

def test_generate_response_stream_success(mock_ollama):
    generator = ResponseGenerator()
    
    # Mock stream chunks
    mock_chunks = [
        {'message': {'content': 'Hello'}},
        {'message': {'content': ' world'}},
    ]
    mock_ollama.chat.return_value = iter(mock_chunks)
    
    events = list(generator.generate_response_stream("query", "context", [], "trace-123"))
    
    assert len(events) == 3
    assert events[0]['type'] == 'token'
    assert events[0]['delta'] == 'Hello'
    assert events[1]['type'] == 'token'
    assert events[1]['delta'] == ' world'
    assert events[2]['type'] == 'done'
    assert events[2]['answer'] == 'Hello world'

def test_generate_response_stream_error(mock_ollama):
    generator = ResponseGenerator()
    
    mock_ollama.chat.side_effect = Exception("Ollama error")
    
    events = list(generator.generate_response_stream("query", "context", [], "trace-123"))
    
    assert len(events) == 1
    assert events[0]['type'] == 'error'
    assert "Ollama error" in events[0]['message']
