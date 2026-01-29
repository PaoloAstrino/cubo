import pytest

pytest.importorskip("torch")

from cubo.config import config
from cubo.retrieval.retriever import DocumentRetriever


def test_chunk_ids_use_file_hash_when_config_enabled():
    config.set("deep_chunk_id_use_file_hash", True)
    dr = DocumentRetriever(model=None)
    filename = "file.txt"
    metadatas = [
        {"file_hash": "abc123", "sentence_index": 0},
        {"file_hash": "abc123", "sentence_index": 1},
    ]

    chunk_ids = dr._create_chunk_ids(metadatas, filename)
    assert chunk_ids == ["abc123_s0", "abc123_s1"]


def test_chunk_ids_use_filename_when_config_disabled():
    config.set("deep_chunk_id_use_file_hash", False)
    dr = DocumentRetriever(model=None)
    filename = "file.txt"
    metadatas = [
        {"file_hash": "abc123", "sentence_index": 0},
        {"file_hash": "abc123", "sentence_index": 1},
    ]

    chunk_ids = dr._create_chunk_ids(metadatas, filename)
    assert chunk_ids == ["file.txt_s0", "file.txt_s1"]


def test_chunk_ids_sanitize_spaces_in_filename():
    """Test that filenames with spaces are sanitized to underscores in chunk IDs."""
    config.set("deep_chunk_id_use_file_hash", False)
    dr = DocumentRetriever(model=None)
    filename = "Ricevuta occasionale Paolo Astrino.pdf"
    metadatas = [
        {"sentence_index": 0},
        {"sentence_index": 1},
    ]

    chunk_ids = dr._create_chunk_ids(metadatas, filename)
    # Spaces should be replaced with underscores
    assert chunk_ids == ["Ricevuta_occasionale_Paolo_Astrino.pdf_s0", "Ricevuta_occasionale_Paolo_Astrino.pdf_s1"]


def test_chunk_ids_sanitize_special_chars_in_filename():
    """Test that special characters in filenames are sanitized in chunk IDs."""
    config.set("deep_chunk_id_use_file_hash", False)
    dr = DocumentRetriever(model=None)
    filename = "document (v1).txt"
    metadatas = [
        {"chunk_index": 0},
        {"chunk_index": 1},
    ]

    chunk_ids = dr._create_chunk_ids(metadatas, filename)
    # Parentheses and spaces should be replaced with underscores
    assert chunk_ids == ["document__v1_.txt_chunk_0", "document__v1_.txt_chunk_1"]

