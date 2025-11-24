"""Unit test: Test TestClient + app import using TestClient.

This test uses FastAPI's TestClient to run a lightweight validation of the
application endpoints without needing a separately-running server.
"""
import pytest
from fastapi.testclient import TestClient

from src.cubo.server.api import app


def test_testclient_health_root():
    client = TestClient(app)
    # Test root endpoint
    r = client.get('/')
    assert r.status_code == 200
    assert isinstance(r.json(), dict)

    # Test health endpoint
    r = client.get('/api/health')
    assert r.status_code == 200
    body = r.json()
    assert 'status' in body
    assert body.get('status') == 'ok' or body.get('status') == 'healthy'
