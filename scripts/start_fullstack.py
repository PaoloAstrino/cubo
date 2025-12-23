#!/usr/bin/env python
"""
Minimal startup script for CUBO Full Stack.
Starts Ollama (if needed), Backend, and Frontend.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

# Project root
ROOT = Path(__file__).parent.parent


def main():
    print(">>> Starting CUBO Full Stack...")

    # 1. Start Ollama (best effort)
    print(">>> Ensuring Ollama is running...")
    try:
        subprocess.Popen(
            ["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True
        )
    except Exception:
        pass  # Assume it's running or user will handle it

    # 2. Start Backend
    print(">>> Starting Backend (http://localhost:8000)...")
    backend_env = os.environ.copy()
    backend_env["PYTHONPATH"] = str(ROOT)
    # Enable reload by default for dev experience
    backend_cmd = [sys.executable, "cubo/server/run.py", "--reload"]

    backend = subprocess.Popen(backend_cmd, cwd=ROOT, env=backend_env)

    # 3. Start Frontend
    print(">>> Starting Frontend (http://localhost:3000)...")
    frontend_dir = ROOT / "frontend"
    frontend_cmd = ["npm", "run", "dev"]

    # Windows needs shell=True for npm
    is_windows = sys.platform.startswith("win")
    frontend = subprocess.Popen(
        frontend_cmd, cwd=frontend_dir, shell=is_windows, env=os.environ.copy()
    )

    print("\n>>> Full Stack Running. Press Ctrl+C to stop.\n")

    try:
        while True:
            time.sleep(1)
            if backend.poll() is not None:
                print("!!! Backend exited unexpectedly")
                break
            if frontend.poll() is not None:
                print("!!! Frontend exited unexpectedly")
                break
    except KeyboardInterrupt:
        print("\n>>> Stopping...")
    finally:
        backend.terminate()
        frontend.terminate()
        print(">>> Stopped.")


if __name__ == "__main__":
    main()
