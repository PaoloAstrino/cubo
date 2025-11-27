import uuid

from fastapi.testclient import TestClient


def test_trace_full_pipeline(client: TestClient, cubo_app, mini_data):
    """Integration test: run ingest -> build-index -> query and assert traces."""
    from src.cubo.server import api as server_api

    # Use lightweight cubo_app fixture for server
    server_api.cubo_app = cubo_app

    trace_id = f"e2e-{uuid.uuid4().hex[:8]}"
    headers = {"x-trace-id": trace_id}

    # Ingest
    r = client.post(
        "/api/ingest", json={"data_path": mini_data, "fast_pass": True}, headers=headers
    )
    assert r.status_code == 200

    # Build index
    r2 = client.post("/api/build-index", json={"force_rebuild": True}, headers=headers)
    assert r2.status_code == 200

    # Query
    r3 = client.post(
        "/api/query",
        json={"query": "What is CUBO?", "top_k": 3, "use_reranker": False},
        headers=headers,
    )
    # Query may return 503 if retriever not ready; accept both 200 and 503
    assert r3.status_code in [200, 503]

    # Retrieve trace events
    r4 = client.get(f"/api/traces/{trace_id}")
    assert r4.status_code == 200
    body = r4.json()
    assert "events" in body
    events = [e["event"] for e in body["events"]]

    # Basic expectations: ingest completed, build_index completed, query received
    assert any("ingest.completed" in ev for ev in events)
    assert any("build_index.completed" in ev for ev in events)
    assert any("query.received" in ev for ev in events)
