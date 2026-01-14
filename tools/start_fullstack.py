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


def _setup_logging(verbose):
    """Setup logging directory and return printer function."""
    def p(msg):
        if verbose:
            print(msg)
    
    LOG_DIR = ROOT / "logs"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    return p, LOG_DIR


def _start_ollama(verbose):
    """Start Ollama service (best effort)."""
    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=(None if verbose else subprocess.DEVNULL),
            stderr=(None if verbose else subprocess.DEVNULL),
            shell=True
        )
    except Exception:
        pass


def _start_backend(verbose, LOG_DIR):
    """Start backend server and return process and log file handles."""
    backend_env = os.environ.copy()
    backend_env["PYTHONPATH"] = str(ROOT)
    backend_cmd = [sys.executable, "start_api_server.py"]

    if verbose:
        return subprocess.Popen(backend_cmd, cwd=ROOT, env=backend_env), None, None
    
    backend_out = open(LOG_DIR / "backend.log", "a", encoding="utf-8")
    backend_err = open(LOG_DIR / "backend.err", "a", encoding="utf-8")
    backend = subprocess.Popen(
        backend_cmd, cwd=ROOT, env=backend_env,
        stdout=backend_out, stderr=backend_err
    )
    return backend, backend_out, backend_err


def _start_frontend(verbose, LOG_DIR):
    """Start frontend server and return process and log file handles."""
    frontend_dir = ROOT / "frontend"
    frontend_cmd = ["npm", "run", "dev"]
    is_windows = sys.platform.startswith("win")

    if verbose:
        return subprocess.Popen(
            frontend_cmd, cwd=frontend_dir,
            shell=is_windows, env=os.environ.copy()
        ), None, None
    
    frontend_out = open(LOG_DIR / "frontend.log", "a", encoding="utf-8")
    frontend_err = open(LOG_DIR / "frontend.err", "a", encoding="utf-8")
    frontend = subprocess.Popen(
        frontend_cmd, cwd=frontend_dir, shell=is_windows,
        env=os.environ.copy(), stdout=frontend_out, stderr=frontend_err
    )
    return frontend, frontend_out, frontend_err


def _monitor_processes(backend, frontend, verbose):
    """Monitor backend and frontend processes."""
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


def _cleanup(backend, frontend, log_handles, verbose):
    """Cleanup processes and log file handles."""
    backend.terminate()
    frontend.terminate()
    
    backend_out, backend_err, frontend_out, frontend_err = log_handles
    try:
        for handle in [backend_out, backend_err, frontend_out, frontend_err]:
            if handle:
                handle.close()
    except Exception:
        pass
    
    if verbose:
        print(">>> Stopped.")


def main():
    verbose = os.environ.get("CUBO_VERBOSE", "0") == "1"
    p, LOG_DIR = _setup_logging(verbose)

    p(">>> Starting CUBO Full Stack...")
    p(">>> Ensuring Ollama is running...")
    _start_ollama(verbose)

    p(">>> Starting Backend (http://localhost:8000)...")
    backend, backend_out, backend_err = _start_backend(verbose, LOG_DIR)

    p(">>> Starting Frontend (http://localhost:3000)...")
    frontend, frontend_out, frontend_err = _start_frontend(verbose, LOG_DIR)

    if verbose:
        print("\n>>> Full Stack Running. Press Ctrl+C to stop.\n")

    try:
        _monitor_processes(backend, frontend, verbose)
    except KeyboardInterrupt:
        if verbose:
            print("\n>>> Stopping...")
    finally:
        _cleanup(backend, frontend, (backend_out, backend_err, frontend_out, frontend_err), verbose)


if __name__ == "__main__":
    main()
