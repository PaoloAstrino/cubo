import asyncio
import os
import pytest
from httpx import AsyncClient, ASGITransport
from pathlib import Path
from cubo.server.api import app

@pytest.fixture
def mock_data_dir(tmp_path):
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    yield tmp_path
    os.chdir(original_cwd)

@pytest.mark.asyncio
async def test_list_documents_concurrency(mock_data_dir):
    """Test concurrent listing of documents."""
    data_dir = mock_data_dir / "data"
    # Create 100 files
    for i in range(100):
        (data_dir / f"doc_{i}.txt").write_text("content")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        # Fire 10 concurrent list requests
        tasks = [ac.get("/api/documents") for _ in range(10)]
        responses = await asyncio.gather(*tasks)

    for r in responses:
        assert r.status_code == 200
        docs = r.json()
        assert len(docs) == 100
