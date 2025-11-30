"""Test fixtures for API tests."""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from cubo.server.api import app


@pytest.fixture
def client():
    """Create test client."""
    with TestClient(app) as test_client:
        yield test_client
