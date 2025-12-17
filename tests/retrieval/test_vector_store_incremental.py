
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from cubo.retrieval.vector_store import FaissStore
from cubo.indexing.faiss_index import FAISSIndexManager

@pytest.fixture
def vector_store(tmp_path):
    store = FaissStore(dimension=4, index_dir=tmp_path)
    yield store
    store.close()

def test_incremental_promotion_logic(vector_store):
    # 1. Add some data (initially goes to hot, then we force it to cold for testing)
    vectors = [[0.1]*4, [0.2]*4, [0.3]*4, [0.4]*4]
    ids = ["doc1", "doc2", "doc3", "doc4"]
    documents = ["c1", "c2", "c3", "c4"]
    metadatas = [{"m": 1}, {"m": 2}, {"m": 3}, {"m": 4}]
    
    # Force all to hot initially
    vector_store._index.hot_fraction = 1.0
    vector_store.add(embeddings=vectors, documents=documents, metadatas=metadatas, ids=ids)
    
    # Verify initial state
    assert len(vector_store._index.hot_ids) == 4
    
    # Force move to cold index manually to simulate a state where we need promotion
    # We rebuild index putting everything in cold (except 1 because of max(1, ...))
    vector_store._index.hot_fraction = 0.0 
    vector_store._index.build_indexes(vectors, ids)
    
    # With hot_fraction=0.0, logic is max(1, 0) = 1 item in hot.
    assert len(vector_store._index.hot_ids) == 1
    assert len(vector_store._index.cold_ids) == 3
    
    # Identify a doc that is in cold
    cold_doc_id = vector_store._index.cold_ids[0]
    assert cold_doc_id not in vector_store._index.hot_ids
    
    # Now promote this cold doc back to hot
    # We want to verify it uses the incremental path
    
    # Mock build_indexes to ensure it's NOT called (which would mean full rebuild)
    with patch.object(vector_store._index, 'build_indexes', wraps=vector_store._index.build_indexes) as mock_build:
        vector_store._promote_batch_to_hot([cold_doc_id])
        
        # Should NOT call build_indexes
        mock_build.assert_not_called()
        
    # Verify doc is now in hot index
    assert cold_doc_id in vector_store._index.hot_ids
    # Size should increase by 1
    assert len(vector_store._index.hot_ids) == 2
    
    # Verify doc is STILL in cold index (because we didn't rebuild)
    assert cold_doc_id in vector_store._index.cold_ids
    
    # Verify query deduplication
    # Query for this doc
    query_vec = [vectors[ids.index(cold_doc_id)]]
    results = vector_store.query(query_embeddings=query_vec, n_results=5)
    
    # Should only get doc ONCE
    found_ids = results["ids"][0]
    assert found_ids.count(cold_doc_id) == 1

def test_reconstruct_vectors(vector_store):
    vectors = [[0.5, 0.5, 0.5, 0.5]]
    ids = ["test_doc"]
    vector_store.add(embeddings=vectors, documents=["c"], metadatas=[{}], ids=ids)
    
    # Clear vector cache to force reconstruction
    vector_store._vector_cache.clear()
    
    # Get vector - should come from index reconstruction, not DB
    # We can verify this by mocking the DB call or checking cache after
    
    # Mock DB connection to ensure we don't read from it
    with patch('sqlite3.connect') as mock_connect:
        vec = vector_store.get_vector("test_doc")
        
        assert vec is not None
        assert np.allclose(vec, vectors[0])
        # Should not have called DB because reconstruction succeeded
        mock_connect.assert_not_called()

