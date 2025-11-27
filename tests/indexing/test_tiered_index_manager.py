"""
Tests for the TieredIndexManager module.

Tests the unified FAISS/BM25 index lifecycle management.
"""

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from src.cubo.indexing.tiered_index_manager import (
    IndexHealthChecker,
    IndexSnapshot,
    IndexState,
    IndexStats,
    IndexType,
    TieredIndexManager,
    get_tiered_index_manager,
)


# Fixture to reset singleton between tests
@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset TieredIndexManager singleton before each test."""
    TieredIndexManager._instance = None
    import src.cubo.indexing.tiered_index_manager as tim_module

    tim_module._tiered_index_manager = None
    yield
    # Cleanup after test
    TieredIndexManager._instance = None
    tim_module._tiered_index_manager = None


# ============================================================================
# IndexStats Tests
# ============================================================================


class TestIndexStats:
    """Tests for IndexStats dataclass."""

    def test_default_stats(self):
        """Test default IndexStats values."""
        stats = IndexStats(
            index_type=IndexType.FAISS_HOT,
            state=IndexState.UNINITIALIZED,
        )
        assert stats.index_type == IndexType.FAISS_HOT
        assert stats.state == IndexState.UNINITIALIZED
        assert stats.document_count == 0
        assert stats.vector_count == 0
        assert stats.query_count == 0
        assert stats.avg_query_time_ms == 0.0
        assert stats.last_updated is None

    def test_custom_stats(self):
        """Test IndexStats with custom values."""
        now = datetime.now()
        stats = IndexStats(
            index_type=IndexType.FAISS_HOT,
            state=IndexState.READY,
            document_count=1000,
            vector_count=1000,
            last_updated=now,
            size_bytes=1024000,
            build_time_ms=1500.0,
            query_count=5000,
            avg_query_time_ms=15.5,
            metadata={"version": "1.0"},
        )
        assert stats.document_count == 1000
        assert stats.vector_count == 1000
        assert stats.query_count == 5000
        assert stats.avg_query_time_ms == 15.5
        assert stats.last_updated == now
        assert stats.metadata == {"version": "1.0"}

    def test_stats_to_dict(self):
        """Test IndexStats serialization."""
        now = datetime.now()
        stats = IndexStats(
            index_type=IndexType.BM25,
            state=IndexState.READY,
            document_count=500,
            last_updated=now,
        )
        result = stats.to_dict()

        assert result["index_type"] == "bm25"
        assert result["state"] == "ready"
        assert result["document_count"] == 500
        assert result["last_updated"] == now.isoformat()


class TestIndexSnapshot:
    """Tests for IndexSnapshot dataclass."""

    def test_snapshot_creation(self):
        """Test IndexSnapshot creation."""
        now = datetime.now()
        snapshot = IndexSnapshot(
            snapshot_id="snap_001",
            created_at=now,
            indices=[IndexType.FAISS_HOT, IndexType.BM25],
            path=Path("/tmp/snapshots/snap_001"),
            metadata={"reason": "manual"},
        )
        assert snapshot.snapshot_id == "snap_001"
        assert len(snapshot.indices) == 2
        assert snapshot.metadata["reason"] == "manual"

    def test_snapshot_to_dict(self):
        """Test IndexSnapshot serialization."""
        now = datetime.now()
        snapshot = IndexSnapshot(
            snapshot_id="snap_002",
            created_at=now,
            indices=[IndexType.FAISS_COLD],
            path=Path("/tmp/snap_002"),
        )
        result = snapshot.to_dict()

        assert result["snapshot_id"] == "snap_002"
        assert result["created_at"] == now.isoformat()
        assert "faiss_cold" in result["indices"]


# ============================================================================
# IndexHealthChecker Tests
# ============================================================================


class TestIndexHealthChecker:
    """Tests for IndexHealthChecker."""

    def test_health_checker_creation(self):
        """Test creating health checker."""
        manager = TieredIndexManager()
        checker = IndexHealthChecker(manager)
        assert checker._manager is manager

    def test_check_faiss_health_uninitialized(self):
        """Test checking uninitialized FAISS index."""
        manager = TieredIndexManager()
        result = manager.health_checker.check_faiss_health()

        assert result["healthy"] is False
        assert any("uninitialized" in issue for issue in result["issues"])

    def test_check_bm25_health_uninitialized(self):
        """Test checking uninitialized BM25 index."""
        manager = TieredIndexManager()
        result = manager.health_checker.check_bm25_health()

        assert result["healthy"] is False
        assert "details" in result

    def test_check_all_returns_dict(self):
        """Test check_all returns all index health."""
        manager = TieredIndexManager()
        result = manager.health_checker.check_all()

        assert isinstance(result, dict)
        assert "overall_healthy" in result
        assert "faiss" in result
        assert "bm25" in result


# ============================================================================
# TieredIndexManager Initialization Tests
# ============================================================================


class TestTieredIndexManagerInit:
    """Tests for TieredIndexManager initialization."""

    def test_init_default_values(self):
        """Test initialization with default values."""
        manager = TieredIndexManager()
        assert manager._dimension == 384
        assert manager._hot_fraction == 0.2
        assert manager._enable_snapshots is True
        assert manager._max_snapshots == 3
        assert manager._initialized is True

    def test_init_custom_dimension(self):
        """Test initialization with custom dimension."""
        manager = TieredIndexManager(dimension=768)
        assert manager._dimension == 768

    def test_init_custom_hot_fraction(self):
        """Test initialization with custom hot fraction."""
        manager = TieredIndexManager(hot_fraction=0.3)
        assert manager._hot_fraction == 0.3

    def test_singleton_pattern(self):
        """Test that TieredIndexManager is a singleton."""
        manager1 = TieredIndexManager()
        manager2 = TieredIndexManager()
        assert manager1 is manager2

    def test_stats_initialized(self):
        """Test that stats are properly initialized."""
        manager = TieredIndexManager()

        assert IndexType.FAISS_HOT in manager._stats
        assert IndexType.FAISS_COLD in manager._stats
        assert IndexType.BM25 in manager._stats

        for stats in manager._stats.values():
            assert stats.state == IndexState.UNINITIALIZED


# ============================================================================
# TieredIndexManager Stats Tests
# ============================================================================


class TestTieredIndexManagerStats:
    """Tests for TieredIndexManager statistics."""

    def test_get_stats_returns_correct_type(self):
        """Test getting stats for index type."""
        manager = TieredIndexManager()
        stats = manager.get_stats(IndexType.FAISS_HOT)

        assert isinstance(stats, IndexStats)
        assert stats.index_type == IndexType.FAISS_HOT

    def test_get_stats_nonexistent(self):
        """Test getting stats for untracked index type."""
        manager = TieredIndexManager()
        # Create a fake index type by using a string that matches the enum value pattern
        stats = manager.get_stats(IndexType.SCAFFOLD)

        # Should return stats with UNINITIALIZED state
        assert stats.state == IndexState.UNINITIALIZED

    def test_get_all_stats(self):
        """Test getting all stats."""
        manager = TieredIndexManager()
        all_stats = manager.get_all_stats()

        assert isinstance(all_stats, dict)
        assert "faiss_hot" in all_stats
        assert "faiss_cold" in all_stats
        assert "bm25" in all_stats

    def test_get_total_document_count_empty(self):
        """Test total document count when empty."""
        manager = TieredIndexManager()
        count = manager.get_total_document_count()
        assert count == 0


# ============================================================================
# TieredIndexManager Property Tests
# ============================================================================


class TestTieredIndexManagerProperties:
    """Tests for TieredIndexManager lazy-loaded properties."""

    def test_faiss_manager_initially_none(self):
        """Test FAISS manager is None initially."""
        manager = TieredIndexManager()
        assert manager._faiss_manager is None

    def test_bm25_store_initially_none(self):
        """Test BM25 store is None initially."""
        manager = TieredIndexManager()
        assert manager._bm25_store is None

    def test_faiss_manager_lazy_load(self):
        """Test FAISS manager is lazy loaded."""
        manager = TieredIndexManager()

        # Access should trigger lazy load
        with patch("src.cubo.indexing.faiss_index.FAISSIndexManager") as MockFAISS:
            MockFAISS.return_value = MagicMock()
            _ = manager.faiss_manager
            # Just verify we accessed the property without error
            assert manager._faiss_manager is not None or MockFAISS.called

    def test_bm25_store_lazy_load(self):
        """Test BM25 store is lazy loaded."""
        manager = TieredIndexManager()

        # Access should trigger lazy load
        with patch("src.cubo.retrieval.bm25_store_factory.get_bm25_store") as mock_get:
            mock_store = MagicMock()
            mock_get.return_value = mock_store
            _ = manager.bm25_store
            # Just verify we accessed the property
            assert manager._bm25_store is not None or mock_get.called


# ============================================================================
# TieredIndexManager Build Tests
# ============================================================================


class TestTieredIndexManagerBuild:
    """Tests for TieredIndexManager build operations."""

    def test_build_faiss_index_updates_state(self):
        """Test that building FAISS index updates state."""
        manager = TieredIndexManager()

        # Mock the FAISS manager
        mock_faiss = MagicMock()
        mock_faiss.hot_ids = ["id1", "id2"]
        mock_faiss.cold_ids = []

        with patch.object(
            TieredIndexManager, "faiss_manager", new_callable=PropertyMock
        ) as mock_prop:
            mock_prop.return_value = mock_faiss

            vectors = [[0.1, 0.2, 0.3] * 128] * 2  # 2 vectors of dimension 384
            ids = ["id1", "id2"]

            manager.build_faiss_index(vectors, ids)

            mock_faiss.build_indexes.assert_called_once()
            assert manager._stats[IndexType.FAISS_HOT].state == IndexState.READY

    def test_build_bm25_index_updates_state(self):
        """Test that building BM25 index updates state."""
        manager = TieredIndexManager()

        # Mock the BM25 store
        mock_bm25 = MagicMock()
        mock_bm25.docs = ["doc1", "doc2"]

        with patch.object(TieredIndexManager, "bm25_store", new_callable=PropertyMock) as mock_prop:
            mock_prop.return_value = mock_bm25

            documents = [
                {"doc_id": "doc1", "text": "First document"},
                {"doc_id": "doc2", "text": "Second document"},
            ]

            manager.build_bm25_index(documents)

            mock_bm25.index_documents.assert_called_once()
            assert manager._stats[IndexType.BM25].state == IndexState.READY

    def test_build_bm25_append_mode(self):
        """Test BM25 build in append mode."""
        manager = TieredIndexManager()

        mock_bm25 = MagicMock()
        mock_bm25.docs = ["doc1"]

        with patch.object(TieredIndexManager, "bm25_store", new_callable=PropertyMock) as mock_prop:
            mock_prop.return_value = mock_bm25

            documents = [{"doc_id": "doc2", "text": "Second document"}]
            manager.build_bm25_index(documents, append=True)

            # Should call add_documents, not index_documents
            mock_bm25.add_documents.assert_called_once()


# ============================================================================
# TieredIndexManager Search Tests
# ============================================================================


class TestTieredIndexManagerSearch:
    """Tests for TieredIndexManager search operations."""

    def test_search_faiss_returns_results(self):
        """Test FAISS search returns results."""
        manager = TieredIndexManager()

        mock_faiss = MagicMock()
        mock_faiss.search.return_value = [
            {"id": "doc1", "distance": 0.1},
            {"id": "doc2", "distance": 0.2},
        ]

        with patch.object(
            TieredIndexManager, "faiss_manager", new_callable=PropertyMock
        ) as mock_prop:
            mock_prop.return_value = mock_faiss

            query_vector = [0.1] * 384
            results = manager.search_faiss(query_vector, top_k=2)

            assert len(results) == 2
            assert results[0]["id"] == "doc1"

    def test_search_bm25_returns_results(self):
        """Test BM25 search returns results."""
        manager = TieredIndexManager()

        mock_bm25 = MagicMock()
        mock_bm25.search.return_value = [
            {"doc_id": "doc1", "similarity": 0.8, "text": "Test"},
        ]

        with patch.object(TieredIndexManager, "bm25_store", new_callable=PropertyMock) as mock_prop:
            mock_prop.return_value = mock_bm25

            results = manager.search_bm25("test query", top_k=1)

            assert len(results) == 1
            assert results[0]["doc_id"] == "doc1"

    def test_search_records_query_time(self):
        """Test that search records query times."""
        manager = TieredIndexManager()

        mock_faiss = MagicMock()
        mock_faiss.search.return_value = []

        with patch.object(
            TieredIndexManager, "faiss_manager", new_callable=PropertyMock
        ) as mock_prop:
            mock_prop.return_value = mock_faiss

            manager.search_faiss([0.1] * 384, top_k=5)

            # Should have recorded query time
            assert manager._stats[IndexType.FAISS_HOT].query_count == 1
            assert len(manager._query_times[IndexType.FAISS_HOT]) == 1


# ============================================================================
# TieredIndexManager Save/Load Tests
# ============================================================================


class TestTieredIndexManagerPersistence:
    """Tests for TieredIndexManager persistence operations."""

    def test_save_indices_creates_directory(self):
        """Test that save creates directory if needed."""
        manager = TieredIndexManager()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "new_dir"

            # Mock FAISS and BM25 stores
            manager._faiss_manager = MagicMock()
            manager._bm25_store = MagicMock()

            manager.save_indices(save_path)

            assert save_path.exists()

    def test_load_indices_returns_false_if_not_exists(self):
        """Test loading from nonexistent path returns False."""
        manager = TieredIndexManager()

        result = manager.load_indices(Path("/nonexistent/path"))
        assert result is False


# ============================================================================
# TieredIndexManager Snapshot Tests
# ============================================================================


class TestTieredIndexManagerSnapshots:
    """Tests for TieredIndexManager snapshot operations."""

    def test_create_snapshot_disabled(self):
        """Test snapshot creation when disabled."""
        manager = TieredIndexManager(enable_snapshots=False)

        result = manager.create_snapshot()
        assert result is None

    def test_list_snapshots_empty(self):
        """Test listing snapshots when none exist."""
        manager = TieredIndexManager()

        snapshots = manager.list_snapshots()
        assert snapshots == []

    def test_restore_snapshot_not_found(self):
        """Test restoring nonexistent snapshot."""
        manager = TieredIndexManager()

        result = manager.restore_snapshot("nonexistent")
        assert result is False


# ============================================================================
# TieredIndexManager Reset Tests
# ============================================================================


class TestTieredIndexManagerReset:
    """Tests for TieredIndexManager reset operation."""

    def test_reset_clears_indices(self):
        """Test that reset clears all index references."""
        manager = TieredIndexManager()

        # Set some fake state
        manager._faiss_manager = MagicMock()
        manager._bm25_store = MagicMock()
        manager._stats[IndexType.FAISS_HOT].state = IndexState.READY

        manager.reset()

        assert manager._faiss_manager is None
        assert manager._bm25_store is None
        assert manager._stats[IndexType.FAISS_HOT].state == IndexState.UNINITIALIZED


# ============================================================================
# Singleton Accessor Tests
# ============================================================================


class TestSingletonAccessor:
    """Tests for get_tiered_index_manager function."""

    def test_get_returns_singleton(self):
        """Test that accessor returns singleton."""
        manager1 = get_tiered_index_manager()
        manager2 = get_tiered_index_manager()

        assert manager1 is manager2

    def test_get_with_dimension(self):
        """Test accessor with custom dimension."""
        manager = get_tiered_index_manager(dimension=768)

        assert manager._dimension == 768


# ============================================================================
# Build All Tests
# ============================================================================


class TestTieredIndexManagerBuildAll:
    """Tests for build_all operation."""

    def test_build_all_calls_both_builds(self):
        """Test that build_all builds both FAISS and BM25."""
        manager = TieredIndexManager()

        mock_faiss = MagicMock()
        mock_faiss.hot_ids = ["id1"]
        mock_faiss.cold_ids = []

        mock_bm25 = MagicMock()
        mock_bm25.docs = ["doc1"]

        with patch.object(
            TieredIndexManager, "faiss_manager", new_callable=PropertyMock
        ) as mock_faiss_prop:
            with patch.object(
                TieredIndexManager, "bm25_store", new_callable=PropertyMock
            ) as mock_bm25_prop:
                mock_faiss_prop.return_value = mock_faiss
                mock_bm25_prop.return_value = mock_bm25

                vectors = [[0.1] * 384]
                texts = ["Document text"]
                ids = ["id1"]

                manager.build_all(vectors, texts, ids)

                mock_faiss.build_indexes.assert_called_once()
                mock_bm25.index_documents.assert_called_once()

    def test_build_all_with_progress_callback(self):
        """Test build_all with progress callback."""
        manager = TieredIndexManager()

        mock_faiss = MagicMock()
        mock_faiss.hot_ids = []
        mock_faiss.cold_ids = []

        mock_bm25 = MagicMock()
        mock_bm25.docs = []

        progress_calls = []

        def progress_callback(phase, current, total):
            progress_calls.append((phase, current, total))

        with patch.object(
            TieredIndexManager, "faiss_manager", new_callable=PropertyMock
        ) as mock_faiss_prop:
            with patch.object(
                TieredIndexManager, "bm25_store", new_callable=PropertyMock
            ) as mock_bm25_prop:
                mock_faiss_prop.return_value = mock_faiss
                mock_bm25_prop.return_value = mock_bm25

                vectors = [[0.1] * 384]
                texts = ["Text"]
                ids = ["id1"]

                manager.build_all(vectors, texts, ids, progress_callback=progress_callback)

                # Should have 4 progress calls (start/end for faiss and bm25)
                assert len(progress_calls) == 4
                assert ("faiss", 0, 1) in progress_calls
                assert ("faiss", 1, 1) in progress_calls
                assert ("bm25", 0, 1) in progress_calls
                assert ("bm25", 1, 1) in progress_calls
