import asyncio
import time
import pytest
from unittest.mock import MagicMock, patch
from httpx import AsyncClient, ASGITransport

@pytest.mark.asyncio
async def test_async_lock_serialization():
    """
    Verify that `compute_lock` enforces strict serialization of heavy tasks
    while allowing light tasks (health checks) to pass through.
    """
    # 1. Setup Mock App
    mock_app = MagicMock()
    mock_app.retriever = MagicMock()
    mock_app.retriever.collection.count.return_value = 10
    
    # We use asyncio.Event for coordination
    build_started = asyncio.Event()
    build_finish_permission = asyncio.Event()
    
    # Needs to be a sync function because it's called via run_in_threadpool
    # But we want to coordinate with async test. 
    # Use a threading Event for the sync code running in threadpool
    import threading
    sync_build_started = threading.Event()
    sync_build_finish_permission = threading.Event()

    def slow_build_index(*args, **kwargs):
        sync_build_started.set()
        # Wait until test allows us to finish
        if not sync_build_finish_permission.wait(timeout=5):
            raise RuntimeError("Test timed out waiting for permission")
        return 10

    mock_app.build_index.side_effect = slow_build_index
    
    query_execution_ts = 0.0
    def tracked_query_retrieve(*args, **kwargs):
        nonlocal query_execution_ts
        query_execution_ts = time.time()
        return [{"document": "doc", "metadata": {}, "similarity": 0.9}]

    mock_app.query_retrieve.side_effect = tracked_query_retrieve
    mock_app.generate_response_safe.return_value = "Answer"

    # 2. Patch and Run
    with patch("cubo.server.api.cubo_app", mock_app):
        from cubo.server.api import app
        
        # Use ASGITransport to bypass TCP/Network overhead/issues
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            
            # Task A: Heavy Build (Background)
            # We can't simply await it, so we wrap it in a task
            build_task = asyncio.create_task(
                client.post("/api/build-index", json={"force_rebuild": True})
            )

            # Wait for build to enter the critical section
            # We poll the threading event because it's set from a thread
            for _ in range(20):
                if sync_build_started.is_set():
                    break
                await asyncio.sleep(0.1)
            else:
                pytest.fail("Build did not start in time")

            # Task B: Health Check (Should be fast)
            t0 = time.time()
            resp_health = await client.get("/api/health")
            t1 = time.time()
            
            assert resp_health.status_code == 200
            assert (t1 - t0) < 0.5, "Health check was blocked!"

            # Task C: Query (Should be blocked)
            query_task = asyncio.create_task(
                client.post("/api/query", json={"query": "test", "top_k": 1})
            )
            
            # Give the query task a moment to hit the lock
            await asyncio.sleep(0.2)
            
            # Verify query logic hasn't run yet
            assert query_execution_ts == 0.0, "Query ran despite lock!"

            # Release the Build
            sync_build_finish_permission.set()
            
            # Wait for both
            await build_task
            await query_task
            
            # Verify order
            assert query_execution_ts > 0.0
            print("Async Lock logic verified.")