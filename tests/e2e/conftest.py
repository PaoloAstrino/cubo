import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import requests

# Constants
BACKEND_PORT = 8000
FRONTEND_PORT = 3000
BACKEND_URL = f"http://localhost:{BACKEND_PORT}"
FRONTEND_URL = f"http://localhost:{FRONTEND_PORT}"


def is_service_ready(url: str, timeout: int = 5) -> bool:
    """Check if a service is responding."""
    try:
        requests.get(url, timeout=timeout)
        return True
    except requests.RequestException:
        return False


def wait_for_service(url: str, name: str, timeout: int = 60) -> bool:
    """Wait for a service to become available."""
    print(f"Waiting for {name} at {url}...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        if is_service_ready(url, timeout=1):
            print(f"✓ {name} is ready!")
            return True
        time.sleep(1)
    print(f"✗ {name} failed to start within {timeout}s")
    return False


@pytest.fixture(scope="session")
def base_url():
    return FRONTEND_URL


@pytest.fixture(scope="session", autouse=True)
def manage_servers():
    """
    Ensure Backend and Frontend are running.
    Strategy:
    1. Check if they are already running. If so, use them.
    2. If not, start them in subprocesses and kill them after tests.
    """
    backend_proc = None
    frontend_proc = None

    # 1. Manage Backend
    if is_service_ready(f"{BACKEND_URL}/api/health"):
        print("Backend already running. Using existing instance.")
    else:
        print("Starting Backend...")
        # Assuming start_api_server.py is in root
        root_dir = Path(__file__).parent.parent.parent
        backend_proc = subprocess.Popen(
            [sys.executable, "start_api_server.py"],
            cwd=root_dir,
            # Remove pipes to let output show in the new window
            creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == "nt" else 0,
        )
        if not wait_for_service(f"{BACKEND_URL}/api/health", "Backend"):
            if backend_proc:
                backend_proc.kill()
            pytest.fail("Failed to start Backend")

    # 2. Manage Frontend
    if is_service_ready(FRONTEND_URL):
        print("Frontend already running. Using existing instance.")
    else:
        print("Starting Frontend...")
        root_dir = Path(__file__).parent.parent.parent
        frontend_dir = root_dir / "frontend"

        # Check for npm
        npm_cmd = "npm.cmd" if os.name == "nt" else "npm"

        env = os.environ.copy()
        env["NEXT_TELEMETRY_DISABLED"] = "1"

        frontend_proc = subprocess.Popen(
            [npm_cmd, "run", "dev"],
            cwd=frontend_dir,
            env=env,
            # Remove pipes to let output show in the new window
            creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == "nt" else 0,
        )
        # Next.js startup can be slow
        if not wait_for_service(FRONTEND_URL, "Frontend", timeout=300):
            if frontend_proc:
                frontend_proc.terminate()
            pytest.fail("Failed to start Frontend - check the popped up terminal window for errors")

    yield

    # Teardown
    print("\nTeardown: Stopping services...")
    if backend_proc:
        print("Stopping Backend process...")
        backend_proc.terminate()
        backend_proc.wait()

    if frontend_proc:
        print("Stopping Frontend process...")
        # Frontends are notoriously hard to kill cleanly on Windows due to node chlidren
        # terminate() usually usually enough for the parent, but we might need taskkill if stuck
        frontend_proc.terminate()
        frontend_proc.wait()


@pytest.fixture(scope="session")
def browser_context_args(browser_context_args):
    return {
        **browser_context_args,
        "viewport": {"width": 1280, "height": 720},
        "record_video_dir": "test-results/videos",  # Record failure videos
    }
