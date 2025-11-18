from src.retriever import DocumentRetriever
from src.config import config


def test_chunk_ids_use_file_hash_when_config_enabled():
    config.set('deep_chunk_id_use_file_hash', True)
    dr = DocumentRetriever(model=None)
    filename = 'file.txt'
    metadatas = [
        {'file_hash': 'abc123', 'sentence_index': 0},
        {'file_hash': 'abc123', 'sentence_index': 1},
    ]

    chunk_ids = dr._create_chunk_ids(metadatas, filename)
    assert chunk_ids == ['abc123_s0', 'abc123_s1']


def test_chunk_ids_use_filename_when_config_disabled():
    config.set('deep_chunk_id_use_file_hash', False)
    dr = DocumentRetriever(model=None)
    filename = 'file.txt'
    metadatas = [
        {'file_hash': 'abc123', 'sentence_index': 0},
        {'file_hash': 'abc123', 'sentence_index': 1},
    ]

    chunk_ids = dr._create_chunk_ids(metadatas, filename)
    assert chunk_ids == ['file.txt_s0', 'file.txt_s1']
