import asyncio
import os

import pytest
from httpx import ASGITransport, AsyncClient

# Import app after setting up environment if needed, but here we just need the app object
from cubo.server.api import app


@pytest.fixture
def mock_data_dir(tmp_path):
    """Change CWD to tmp_path so 'data' dir is created there."""
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    yield tmp_path
    os.chdir(original_cwd)


@pytest.mark.asyncio
async def test_concurrent_uploads(mock_data_dir):
    """Test multiple concurrent uploads to ensure async I/O works."""
    # Create dummy content
    files_content = {}
    for i in range(10):
        files_content[f"test_{i}.txt"] = f"content {i}" * 1000

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        tasks = []
        for filename, content in files_content.items():
            # httpx files param: (filename, content, content_type)
            files_param = {"file": (filename, content.encode(), "text/plain")}
            tasks.append(ac.post("/api/upload", files=files_param))

        responses = await asyncio.gather(*tasks)

    for r in responses:
        assert r.status_code == 200
        data = r.json()
        assert data["filename"] in files_content
        assert "uploaded successfully" in data["message"]

    # Verify files exist in data dir
    data_dir = mock_data_dir / "data"
    assert data_dir.exists()
    for filename in files_content:
        assert (data_dir / filename).exists()
        assert (data_dir / filename).read_text() == files_content[filename]


@pytest.mark.asyncio
async def test_upload_large_file_chunking(mock_data_dir):
    """Test uploading a larger file to verify chunking logic."""
    # 5MB file
    large_content = "x" * (5 * 1024 * 1024)
    filename = "large_test.txt"

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        files_param = {"file": (filename, large_content.encode(), "text/plain")}
        response = await ac.post("/api/upload", files=files_param)

    assert response.status_code == 200
    assert response.json()["size"] == len(large_content)

    data_dir = mock_data_dir / "data"
    assert (data_dir / filename).stat().st_size == len(large_content)
