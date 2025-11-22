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
    print("=" * 60)
    print("BACKEND OUTPUT:")
    print("=" * 60)
    
    backend_process = subprocess.Popen(
        [sys.executable, "src/cubo/server/run.py", "--reload"],
        cwd=project_root
        # No stdout/stderr capture - show output directly in terminal
    )
    
    return backend_process


def start_frontend():
    """Start the Next.js frontend dev server."""
    print("\n" + "=" * 60)
    print("FRONTEND OUTPUT:")
    print("=" * 60)
    print("Starting frontend dev server on port 3000...")
    
    frontend_dir = project_root / "frontend"
    
    # Check if node_modules exists
    if not (frontend_dir / "node_modules").exists():
        print("Installing frontend dependencies...")
        subprocess.run(["npm", "install"], cwd=frontend_dir, check=True, shell=True)
    
    frontend_process = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=frontend_dir,
        shell=True  # Required for Windows to find npm in PATH
        # No stdout/stderr capture - show output directly in terminal
    )
    
    return frontend_process


def wait_for_backend():
    """Wait for backend to be ready."""
    import requests
    
    print("\n" + "=" * 60)
    print("Waiting for backend to be ready...")
    print("=" * 60)
    for i in range(60):  # Wait up to 60 seconds
        try:
            response = requests.get("http://localhost:8000/api/health", timeout=2)
            if response.status_code == 200:
                print(f"\n✓ Backend is ready! (took {i+1} seconds)")
                print("=" * 60)
                return True
        except requests.exceptions.ConnectionError:
            # Backend not started yet, keep waiting
            pass
        except Exception as e:
            print(f"Health check attempt {i+1}/60 - Error: {e}")
        
        if i % 5 == 0 and i > 0:
            print(f"Still waiting... ({i} seconds elapsed)")
        time.sleep(1)
    
    print("\n✗ Backend failed to start within 60 seconds")
    print("Check the backend output above for errors")
    print("=" * 60)
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
