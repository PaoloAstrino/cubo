from fastapi.testclient import TestClient

from src.cubo.server.api import app
from src.cubo.utils.trace_collector import trace_collector


def test_trace_not_found():
    client = TestClient(app)
    r = client.get("/api/traces/notfound")
    assert r.status_code == 404


def test_trace_record_and_get():
    client = TestClient(app)
    trace_id = "integration-test"
    trace_collector.record(trace_id, "api", "test.start", {"x": 1})
    trace_collector.record(trace_id, "api", "test.end", {"x": 2})
    r = client.get(f"/api/traces/{trace_id}")
    assert r.status_code == 200
    body = r.json()
    assert "events" in body
    assert len(body["events"]) >= 2
