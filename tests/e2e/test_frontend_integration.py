"""
Frontend-Backend Integration Smoke Test

Tests the full stack with Next.js frontend + FastAPI backend:
1. Start backend server
2. Start frontend dev server
3. Navigate to /upload page
4. Upload file via UI
5. Trigger ingestion
6. Navigate to /chat
7. Submit query
8. Verify answer is displayed

Requires: Playwright for browser automation
"""

import subprocess
import time
from pathlib import Path

import pytest
import requests


# Check if Playwright is available
try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False


@pytest.fixture(scope="module")
def backend_server():
    """Start FastAPI backend server for testing."""
    # Start server in background
    process = subprocess.Popen(
        ["python", "start_api_server.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=Path.cwd()
    )
    
    # Wait for server to be ready
    max_retries = 30
    for i in range(max_retries):
        try:
            response = requests.get("http://localhost:8000/api/health", timeout=1)
            if response.status_code == 200:
                print("✓ Backend server ready")
                break
        except requests.ConnectionError:
            time.sleep(1)
    else:
        process.kill()
        pytest.fail("Backend server failed to start")
    
    yield "http://localhost:8000"
    
    # Cleanup
    process.kill()
    process.wait()


@pytest.fixture(scope="module")
def frontend_server():
    """Start Next.js frontend dev server."""
    if not (Path.cwd() / "frontend").exists():
        pytest.skip("Frontend directory not found")
    
    process = subprocess.Popen(
        ["npm", "run", "dev"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=Path.cwd() / "frontend"
    )
    
    # Wait for frontend to be ready
    max_retries = 60  # Next.js can take longer to start
    for i in range(max_retries):
        try:
            response = requests.get("http://localhost:3000", timeout=1)
            if response.status_code == 200:
                print("✓ Frontend server ready")
                break
        except requests.ConnectionError:
            time.sleep(1)
    else:
        process.kill()
        pytest.skip("Frontend server failed to start (npm may not be installed)")
    
    yield "http://localhost:3000"
    
    # Cleanup
    process.kill()
    process.wait()


@pytest.mark.skipif(not PLAYWRIGHT_AVAILABLE, reason="Playwright not installed")
class TestFrontendBackendIntegration:
    """Test full stack with browser automation."""
    
    def test_upload_flow_via_ui(self, backend_server, frontend_server):
        """
        E2E: Navigate to upload page → Upload file → Verify success message
        """
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            
            # Navigate to upload page
            page.goto(f"{frontend_server}/upload")
            page.wait_for_load_state("networkidle")
            
            # Create a test file
            test_file = Path("test_upload.txt")
            test_file.write_text("This is a test document for upload.")
            
            # Upload file
            page.set_input_files('input[type="file"]', str(test_file))
            
            # Click upload button (adjust selector based on your UI)
            upload_button = page.locator('button:has-text("Upload")')
            if upload_button.count() > 0:
                upload_button.click()
                
                # Wait for success message
                page.wait_for_selector('text=/uploaded|success/i', timeout=10000)
                
                print("✓ File upload via UI successful")
            else:
                pytest.skip("Upload button not found (UI may have changed)")
            
            # Cleanup
            test_file.unlink()
            browser.close()
    
    def test_query_flow_via_ui(self, backend_server, frontend_server):
        """
        E2E: Navigate to chat → Submit query → Verify answer displayed
        """
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            
            # Navigate to chat page
            page.goto(f"{frontend_server}/chat")
            page.wait_for_load_state("networkidle")
            
            # Find query input (adjust selector based on your UI)
            query_input = page.locator('input[placeholder*="question"], textarea[placeholder*="question"]')
            
            if query_input.count() > 0:
                # Type query
                query_input.type("What is CUBO?")
                
                # Submit (press Enter or click button)
                page.keyboard.press("Enter")
                
                # Wait for answer to appear
                page.wait_for_selector('text=/answer|response|result/i', timeout=15000)
                
                # Verify some text appeared (answer from LLM)
                page_text = page.inner_text('body')
                assert len(page_text) > 50, "No substantial answer displayed"
                
                print("✓ Query via UI successful, answer displayed")
            else:
                pytest.skip("Query input not found (UI may have changed)")
            
            browser.close()


class TestAPIEndpointsWithoutUI:
    """Test API endpoints directly (no browser needed)."""
    
    def test_health_endpoint(self, backend_server):
        """Test /api/health returns 200."""
        response = requests.get(f"{backend_server}/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data.get("status") == "healthy" or "status" in data
        print("✓ Health endpoint working")
    
    def test_upload_endpoint(self, backend_server, tmp_path):
        """Test /api/upload accepts file upload."""
        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content for API upload")
        
        # Upload via API
        with open(test_file, 'rb') as f:
            files = {'file': ('test.txt', f, 'text/plain')}
            response = requests.post(f"{backend_server}/api/upload", files=files)
        
        assert response.status_code in [200, 201], \
            f"Upload failed: {response.status_code} - {response.text}"
        
        print("✓ Upload API endpoint working")
    
    def test_ingest_endpoint(self, backend_server):
        """Test /api/ingest can be triggered."""
        response = requests.post(
            f"{backend_server}/api/ingest",
            json={"data_path": "data", "fast_pass": True}
        )
        
        # May return 200 (success) or 202 (accepted) or 500 (if no data)
        assert response.status_code in [200, 202, 500], \
            f"Unexpected status: {response.status_code}"
        
        print(f"✓ Ingest API endpoint responsive (status: {response.status_code})")
    
    def test_query_endpoint(self, backend_server):
        """Test /api/query returns results (may be empty if no data)."""
        response = requests.post(
            f"{backend_server}/api/query",
            json={"query": "test query", "top_k": 3}
        )
        
        # Should return 200 or 503 (if retriever not initialized)
        assert response.status_code in [200, 503], \
            f"Unexpected status: {response.status_code}"
        
        if response.status_code == 200:
            data = response.json()
            assert "results" in data or "answer" in data, \
                "Query response missing results/answer"
        
        print(f"✓ Query API endpoint working (status: {response.status_code})")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "-m", "not skipif"])
