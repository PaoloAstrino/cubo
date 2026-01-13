import sys
from unittest.mock import MagicMock, patch, ANY
from cubo.processing.llm_local import LocalResponseGenerator

def test_local_streaming_yields_tokens():
    """
    Verify that LocalResponseGenerator yields individual tokens correctly.
    """
    # Mock llama_cpp module if not present
    mock_llama_cpp = MagicMock()
    sys.modules["llama_cpp"] = mock_llama_cpp
    
    # Mock the Llama class instance
    mock_llm = MagicMock()
    mock_llama_cpp.Llama.return_value = mock_llm
    
    # Simulate a stream of tokens
    mock_stream = [
        {"choices": [{"text": "Hello"}]},
        {"choices": [{"text": " world"}]},
        {"choices": [{"text": "!"}]}
    ]
    
    # When called with stream=True, return the list (iterable)
    # Configure both __call__ (fallback) and create_completion (primary)
    mock_llm.return_value = mock_stream
    mock_llm.create_completion.return_value = mock_stream
    
    # Mock service manager to execute sync
    with patch("cubo.processing.llm_local.get_service_manager") as mock_sm:
        mock_sm.return_value.execute_sync.side_effect = lambda name, func: func()
        
        # Initialize (LocalResponseGenerator tries to import llama_cpp inside)
        # We need to make sure the import inside the class finds our mock
        with patch.dict(sys.modules, {"llama_cpp": mock_llama_cpp}):
            generator = LocalResponseGenerator(model_path="fake/path")
            # Inject our mock instance directly since __init__ might have created a new one
            generator._llm = mock_llm
            
            # Execute
            stream = generator.generate_response_stream("query", "context")
            
            # Collect results
            events = list(stream)
            print(f"DEBUG: Events collected: {events}")
            print(f"DEBUG: Mock LLM called: {mock_llm.call_count} times")
            
            # Verify structure
            tokens = [e["content"] for e in events if e["type"] == "token"]
            assert tokens == ["Hello", " world", "!"], f"Expected tokens mismatch: {tokens}"
            
            # Verify final done event
            assert events[-1]["type"] == "done"
            
            # Verify LLM was called with stream=True
            # Use ANY for the prompt because it goes through ChatTemplateManager
            mock_llm.create_completion.assert_called_with(
                prompt=ANY, 
                stream=True, 
                max_tokens=512
            )
            print("Local Streaming Unit Test Passed")