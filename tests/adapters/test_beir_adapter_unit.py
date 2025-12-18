import unittest
from unittest.mock import MagicMock, patch, mock_open
import os
import json
import numpy as np
from pathlib import Path
import shutil
import tempfile

from cubo.adapters.beir_adapter import CuboBeirAdapter

class TestCuboBeirAdapter(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.index_dir = os.path.join(self.test_dir, "index")
        self.corpus_path = os.path.join(self.test_dir, "corpus.jsonl")
        
        # Create dummy corpus
        with open(self.corpus_path, 'w') as f:
            f.write(json.dumps({"_id": "doc1", "title": "Title 1", "text": "Text 1"}) + "\n")
            f.write(json.dumps({"_id": "doc2", "title": "Title 2", "text": "Text 2"}) + "\n")

        # Mock EmbeddingGenerator
        self.mock_embedder = MagicMock()
        self.mock_embedder.dimension = 768
        self.mock_embedder.encode.return_value = np.random.rand(2, 768).astype('float32')

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @patch("cubo.adapters.beir_adapter.DocumentRetriever")
    @patch("cubo.adapters.beir_adapter.FAISSIndexManager")
    @patch("cubo.adapters.beir_adapter.sqlite3")
    def test_index_corpus(self, mock_sqlite, mock_faiss_cls, mock_retriever_cls):
        # Setup mocks
        mock_faiss_instance = MagicMock()
        mock_faiss_cls.return_value = mock_faiss_instance
        
        # Setup mock retriever
        mock_retriever_instance = MagicMock()
        mock_retriever_cls.return_value = mock_retriever_instance
        
        adapter = CuboBeirAdapter(embedding_generator=self.mock_embedder)
        
        # Run indexing
        count = adapter.index_corpus(self.corpus_path, self.index_dir, batch_size=2)
        
        # Verify
        self.assertEqual(count, 2)
        mock_faiss_cls.assert_called()
        mock_faiss_instance.build_indexes.assert_called()
        # save() is called with a Path object, not a string
        mock_faiss_instance.save.assert_called()
        args, _ = mock_faiss_instance.save.call_args
        # Compare as Path objects to handle Windows/Unix differences
        from pathlib import Path as PathType
        self.assertEqual(PathType(str(args[0])).resolve(), PathType(self.index_dir).resolve())
        
        # Verify DB interaction
        mock_sqlite.connect.assert_called()
        
        # Verify retriever reload was called
        mock_retriever_instance.collection.load.assert_called_with(self.index_dir)

    @patch("cubo.adapters.beir_adapter.DocumentRetriever")
    @patch("cubo.adapters.beir_adapter.FAISSIndexManager")
    def test_retrieve(self, mock_faiss_cls, mock_retriever_cls):
        # Setup mock retriever
        mock_retriever_instance = MagicMock()
        mock_retriever_cls.return_value = mock_retriever_instance
        
        # Mock retrieve_top_documents return
        mock_retriever_instance.retrieve_top_documents.return_value = [
            {"id": "doc1", "similarity": 0.9, "document": "text1"},
            {"id": "doc2", "similarity": 0.8, "document": "text2"}
        ]
        
        adapter = CuboBeirAdapter(embedding_generator=self.mock_embedder, lightweight=False)
        
        results = adapter.retrieve("query", top_k=10)
        
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][0], "doc1")
        self.assertEqual(results[0][1], 0.9)
        mock_retriever_instance.retrieve_top_documents.assert_called_with("query", top_k=10)

    @patch("cubo.adapters.beir_adapter.DocumentRetriever")
    @patch("cubo.adapters.beir_adapter.FAISSIndexManager")
    def test_retrieve_bulk(self, mock_faiss_cls, mock_retriever_cls):
        mock_retriever_instance = MagicMock()
        mock_retriever_cls.return_value = mock_retriever_instance
        mock_retriever_instance.retrieve_top_documents.return_value = [
            {"id": "doc1", "similarity": 0.9, "document": "text1"}
        ]
        
        adapter = CuboBeirAdapter(embedding_generator=self.mock_embedder, lightweight=False)
        
        queries = {"q1": "query 1", "q2": "query 2"}
        results = adapter.retrieve_bulk(queries, top_k=5)
        
        self.assertIn("q1", results)
        self.assertIn("q2", results)
        self.assertEqual(results["q1"]["doc1"], 0.9)
        # Should be called twice (once per query)
        self.assertEqual(mock_retriever_instance.retrieve_top_documents.call_count, 2)

if __name__ == '__main__':
    unittest.main()
