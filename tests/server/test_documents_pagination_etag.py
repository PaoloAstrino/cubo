import os

import pytest
from httpx import ASGITransport, AsyncClient

from cubo.server.api import app


@pytest.fixture
def mock_data_dir(tmp_path):
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    yield tmp_path
    os.chdir(original_cwd)


@pytest.mark.asyncio
async def test_documents_pagination(mock_data_dir):
    data_dir = mock_data_dir / "data"
    for i in range(10):
        (data_dir / f"doc_{i}.txt").write_text("content")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        r1 = await ac.get("/api/documents", params={"skip": 0, "limit": 5})
        assert r1.status_code == 200
        docs1 = r1.json()
        assert len(docs1) == 5

        r2 = await ac.get("/api/documents", params={"skip": 5, "limit": 10})
        assert r2.status_code == 200
        docs2 = r2.json()
        assert len(docs2) == 5


@pytest.mark.asyncio
async def test_documents_etag_304(mock_data_dir):
    data_dir = mock_data_dir / "data"
    (data_dir / "a.txt").write_text("content")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        r1 = await ac.get("/api/documents")
        assert r1.status_code == 200
        etag = r1.headers.get("etag")
        assert etag

        r2 = await ac.get("/api/documents", headers={"If-None-Match": etag})
        assert r2.status_code == 304
        # The server should echo the current ETag on 304.
        assert r2.headers.get("etag") == etag
