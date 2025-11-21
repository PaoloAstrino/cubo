#!/usr/bin/env python
"""
Start both backend API server and frontend dev server.
"""
import os
import sys
import subprocess
import time
import signal
from pathlib import Path

project_root = Path(__file__).parent.parent

def start_backend():
    """Start the FastAPI backend server."""
    print("Starting backend API server on port 8000...")
    
    backend_process = subprocess.Popen(
        [sys.executable, "src/cubo/server/run.py", "--reload"],
        cwd=project_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    return backend_process


def start_frontend():
    """Start the Next.js frontend dev server."""
    print("Starting frontend dev server on port 3000...")
    
    frontend_dir = project_root / "frontend"
    
    # Check if node_modules exists
    if not (frontend_dir / "node_modules").exists():
        print("Installing frontend dependencies...")
        subprocess.run(["pnpm", "install"], cwd=frontend_dir, check=True)
    
    frontend_process = subprocess.Popen(
        ["pnpm", "dev"],
        cwd=frontend_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    return frontend_process


def wait_for_backend():
    """Wait for backend to be ready."""
    import requests
    
    print("Waiting for backend to be ready...")
    for i in range(30):
        try:
            response = requests.get("http://localhost:8000/api/health", timeout=1)
            if response.status_code == 200:
                print("✓ Backend is ready")
                return True
        except Exception:
            pass
        
        time.sleep(1)
    
    print("✗ Backend failed to start")
    return False


def main():
    """Main entry point."""
    print("=" * 60)
    print("Starting CUBO Full Stack")
    print("=" * 60)
    
    backend_process = None
    frontend_process = None
    
    try:
        # Start backend
        backend_process = start_backend()
        
        # Wait for backend to be ready
        if not wait_for_backend():
            return 1
        
        # Start frontend
        frontend_process = start_frontend()
        
        print("\n" + "=" * 60)
        print("✓ CUBO Full Stack Started")
        print("=" * 60)
        print("Backend API:  http://localhost:8000")
        print("API Docs:     http://localhost:8000/docs")
        print("Frontend:     http://localhost:3000")
        print("=" * 60)
        print("\nPress Ctrl+C to stop\n")
        
        # Monitor processes
        while True:
            # Check if processes are still running
            if backend_process.poll() is not None:
                print("\n✗ Backend process died")
                break
            
            if frontend_process.poll() is not None:
                print("\n✗ Frontend process died")
                break
            
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    
    finally:
        # Cleanup
        if backend_process:
            print("Stopping backend...")
            backend_process.terminate()
            try:
                backend_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                backend_process.kill()
        
        if frontend_process:
            print("Stopping frontend...")
            frontend_process.terminate()
            try:
                frontend_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                frontend_process.kill()
        
        print("✓ Shutdown complete")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
