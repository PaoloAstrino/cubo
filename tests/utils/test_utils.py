import pytest
import os
import tempfile
from src.cubo.utils.utils import Utils

def test_sanitize_path_valid():
    """Test path sanitization with valid paths."""
    # Use platform-appropriate paths
    if os.name == 'nt':  # Windows
        base_dir = "C:\\base"
        path = "C:\\base\\subdir\\file.txt"
    else:  # Unix/Linux
        base_dir = "/base"
        path = "/base/subdir/file.txt"

    result = Utils.sanitize_path(path, base_dir)
    # The result should be the absolute path
    expected = os.path.abspath(path)
    assert result == expected

def test_sanitize_path_traversal():
    """Test path sanitization blocks directory traversal."""
    base_dir = "/base"
    path = "/base/../outside/file.txt"
    with pytest.raises(ValueError, match="Path traversal detected"):
        Utils.sanitize_path(path, base_dir)

def test_validate_file_size_valid():
    """Test file size validation for valid size."""
    with tempfile.NamedTemporaryFile() as f:
        f.write(b"x" * 1000)  # 1KB
        f.flush()
        Utils.validate_file_size(f.name, 10)  # Should not raise

def test_validate_file_size_too_large():
    """Test file size validation for oversized file."""
    with tempfile.NamedTemporaryFile() as f:
        f.write(b"x" * (2 * 1024 * 1024))  # 2MB
        f.flush()
        with pytest.raises(ValueError, match="File size .* exceeds limit"):
            Utils.validate_file_size(f.name, 1)  # 1MB limit

def test_validate_file_type_valid():
    """Test file type validation for allowed extension."""
    allowed = [".txt", ".pdf"]
    Utils.validate_file_type("file.txt", allowed)  # Should not raise

def test_validate_file_type_invalid():
    """Test file type validation for disallowed extension."""
    allowed = [".txt", ".pdf"]
    with pytest.raises(ValueError, match="File type .* not allowed"):
        Utils.validate_file_type("file.exe", allowed)

def test_clean_text():
    """Test text cleaning."""
    text = "  Hello   World!  "
    result = Utils.clean_text(text)
    assert result == "Hello World!"

def test_preprocess_text():
    """Test text preprocessing."""
    text = "Hello, WORLD!"
    result = Utils.preprocess_text(text, lowercase=True, remove_punct=True)
    assert result == "hello world"

def test_chunk_text():
    """Test text chunking."""
    text = "This is a test document with multiple sentences."
    chunks = Utils.chunk_text(text, chunk_size=10, overlap=5)
    assert len(chunks) > 1
    assert all(len(chunk) <= 15 for chunk in chunks)  # Allow some flexibility