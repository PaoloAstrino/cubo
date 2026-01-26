"""Integration: Live API server tests.

These tests contact a running server at `http://localhost:8000` and are
intended to be run manually or in CI that provisions the backend first.

Enable by setting the environment variable `CUBO_RUN_LIVE_TESTS=1` (e.g., in a separate terminal run the server first).
"""

import os
import time

import pytest
import requests

API_BASE = os.getenv("CUBO_API_BASE", "http://localhost:8000")


def _skip_live_tests():
    return os.getenv("CUBO_RUN_LIVE_TESTS") != "1"


@pytest.mark.integration
@pytest.mark.skipif(
    _skip_live_tests(), reason="Live API tests are disabled - set CUBO_RUN_LIVE_TESTS=1 to enable"
)
def test_root_live():
    # Wait briefly for server readiness
    for _ in range(10):
        try:
            r = requests.get(f"{API_BASE}/", timeout=2)
            if r.status_code == 200:
                break
        except Exception:
            time.sleep(0.8)
    else:
        pytest.skip("Server not available for live test")

    r = requests.get(f"{API_BASE}/")
    assert r.status_code == 200
    assert isinstance(r.json(), dict)


@pytest.mark.integration
@pytest.mark.skipif(
    _skip_live_tests(), reason="Live API tests are disabled - set CUBO_RUN_LIVE_TESTS=1 to enable"
)
def test_health_live():
    r = requests.get(f"{API_BASE}/api/health", timeout=5)
    assert r.status_code == 200
    body = r.json()
    assert "status" in body
    assert body.get("status") == "ok" or body.get("status") == "healthy"
