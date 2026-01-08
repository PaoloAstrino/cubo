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
    verbose = os.environ.get("CUBO_VERBOSE", "0") == "1"

    def p(msg):
        if verbose:
            print(msg)

    p(">>> Starting CUBO Full Stack...")

    # 1. Start Ollama (best effort)
    p(">>> Ensuring Ollama is running...")
    try:
        subprocess.Popen([
            "ollama",
            "serve",
        ], stdout=(None if verbose else subprocess.DEVNULL), stderr=(None if verbose else subprocess.DEVNULL), shell=True)
    except Exception:
        pass  # Assume it's running or user will handle it

    # 2. Start Backend
    p(">>> Starting Backend (http://localhost:8000)...")
    backend_env = os.environ.copy()
    backend_env["PYTHONPATH"] = str(ROOT)
    backend_cmd = [sys.executable, "cubo/server/run.py", "--reload"]

    backend = subprocess.Popen(backend_cmd, cwd=ROOT, env=backend_env, stdout=(None if verbose else subprocess.DEVNULL), stderr=(None if verbose else subprocess.DEVNULL))

    # 3. Start Frontend
    p(">>> Starting Frontend (http://localhost:3000)...")
    frontend_dir = ROOT / "frontend"
    frontend_cmd = ["npm", "run", "dev"]

    is_windows = sys.platform.startswith("win")
    frontend = subprocess.Popen(frontend_cmd, cwd=frontend_dir, shell=is_windows, env=os.environ.copy(), stdout=(None if verbose else subprocess.DEVNULL), stderr=(None if verbose else subprocess.DEVNULL))

    if verbose:
        print("\n>>> Full Stack Running. Press Ctrl+C to stop.\n")

    try:
        while True:
            time.sleep(1)
            if backend.poll() is not None:
                if verbose:
                    print("!!! Backend exited unexpectedly")
                break
            if frontend.poll() is not None:
                if verbose:
                    print("!!! Frontend exited unexpectedly")
                break
    except KeyboardInterrupt:
        if verbose:
            print("\n>>> Stopping...")
    finally:
        backend.terminate()
        frontend.terminate()
        if verbose:
            print(">>> Stopped.")


if __name__ == "__main__":
    main()
