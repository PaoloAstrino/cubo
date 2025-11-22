#!/usr/bin/env python
"""Start both backend API server and frontend dev server.

Provides CLI arguments for configuring ports, paths, and logging.
"""
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import click

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.cubo.config import setup_logging

logger = logging.getLogger('cubo.fullstack')


def start_backend(backend_path: Path, port: int, reload: bool = True) -> subprocess.Popen:
    """Start the FastAPI backend server.
    
    Args:
        backend_path: Path to backend run.py script.
        port: Port number for backend server.
        reload: Enable auto-reload on code changes.
        
    Returns:
        Backend process handle.
    """
    logger.info(f"Starting backend API server on port {port}...")
    logger.info("=" * 60)
    logger.info("BACKEND OUTPUT:")
    logger.info("=" * 60)
    
    cmd = [sys.executable, str(backend_path)]
    if reload:
        cmd.append("--reload")
    
    env = os.environ.copy()
    env['PORT'] = str(port)
    
    backend_process = subprocess.Popen(cmd, cwd=project_root, env=env)
    return backend_process


def start_frontend(frontend_dir: Path, port: int) -> subprocess.Popen:
    """Start the Next.js frontend dev server.
    
    Args:
        frontend_dir: Path to frontend directory.
        port: Port number for frontend server.
        
    Returns:
        Frontend process handle.
    """
    logger.info("\n" + "=" * 60)
    logger.info("FRONTEND OUTPUT:")
    logger.info("=" * 60)
    logger.info(f"Starting frontend dev server on port {port}...")
    
    # Check if node_modules exists
    if not (frontend_dir / "node_modules").exists():
        logger.info("Installing frontend dependencies...")
        subprocess.run(["npm", "install"], cwd=frontend_dir, check=True, shell=True)
    
    env = os.environ.copy()
    env['PORT'] = str(port)
    
    frontend_process = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=frontend_dir,
        shell=True,  # Required for Windows to find npm in PATH
        env=env
    )
    
    return frontend_process


def wait_for_backend(backend_url: str, timeout: int = 60) -> bool:
    """Wait for backend to be ready.
    
    Args:
        backend_url: Backend health check URL.
        timeout: Maximum seconds to wait.
        
    Returns:
        True if backend is ready, False otherwise.
    """
    import requests
    
    logger.info("\n" + "=" * 60)
    logger.info("Waiting for backend to be ready...")
    logger.info("=" * 60)
    
    for i in range(timeout):
        try:
            response = requests.get(f"{backend_url}/api/health", timeout=2)
            if response.status_code == 200:
                logger.info(f"\n✓ Backend is ready! (took {i+1} seconds)")
                logger.info("=" * 60)
                return True
        except requests.exceptions.ConnectionError:
            pass
        except Exception as e:
            if i % 10 == 0:
                logger.debug(f"Health check attempt {i+1}/{timeout} - Error: {e}")
        
        if i % 5 == 0 and i > 0:
            logger.info(f"Still waiting... ({i} seconds elapsed)")
        time.sleep(1)
    
    logger.error(f"\n✗ Backend failed to start within {timeout} seconds")
    logger.error("Check the backend output above for errors")
    logger.info("=" * 60)
    return False


@click.command()
@click.option(
    '--backend-port',
    default=8000,
    help='Port for backend API server.',
    show_default=True
)
@click.option(
    '--frontend-port',
    default=3000,
    help='Port for frontend dev server.',
    show_default=True
)
@click.option(
    '--backend-path',
    default='src/cubo/server/run.py',
    help='Path to backend run.py script.',
    show_default=True
)
@click.option(
    '--frontend-dir',
    default='frontend',
    help='Path to frontend directory.',
    show_default=True
)
@click.option(
    '--no-reload',
    is_flag=True,
    help='Disable auto-reload for backend.'
)
@click.option(
    '--log-level',
    default='INFO',
    type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR'], case_sensitive=False),
    help='Logging level.',
    show_default=True
)
@click.option(
    '--timeout',
    default=60,
    help='Timeout in seconds for backend startup.',
    show_default=True
)
def main(
    backend_port: int,
    frontend_port: int,
    backend_path: str,
    frontend_dir: str,
    no_reload: bool,
    log_level: str,
    timeout: int
):
    """Start CUBO full stack (backend + frontend)."""
    # Setup logging
    setup_logging(level=log_level)
    
    logger.info("=" * 60)
    logger.info("Starting CUBO Full Stack")
    logger.info("=" * 60)
    
    # Resolve paths
    backend_script = project_root / backend_path
    frontend_directory = project_root / frontend_dir
    
    if not backend_script.exists():
        logger.error(f"Backend script not found: {backend_script}")
        return 1
    
    if not frontend_directory.exists():
        logger.error(f"Frontend directory not found: {frontend_directory}")
        return 1
    
    backend_process = None
    frontend_process = None
    
    try:
        # Start backend
        backend_process = start_backend(
            backend_script,
            backend_port,
            reload=not no_reload
        )
        
        # Wait for backend to be ready
        backend_url = f"http://localhost:{backend_port}"
        if not wait_for_backend(backend_url, timeout):
            return 1
        
        # Start frontend
        frontend_process = start_frontend(frontend_directory, frontend_port)
        
        logger.info("\n" + "=" * 60)
        logger.info("✓ CUBO Full Stack Started")
        logger.info("=" * 60)
        logger.info(f"Backend API:  http://localhost:{backend_port}")
        logger.info(f"API Docs:     http://localhost:{backend_port}/docs")
        logger.info(f"Frontend:     http://localhost:{frontend_port}")
        logger.info("=" * 60)
        logger.info("\nPress Ctrl+C to stop\n")
        
        # Monitor processes
        while True:
            if backend_process.poll() is not None:
                logger.error("\n✗ Backend process died")
                break
            
            if frontend_process.poll() is not None:
                logger.error("\n✗ Frontend process died")
                break
            
            time.sleep(1)
    
    except KeyboardInterrupt:
        logger.info("\n\nShutting down...")
    
    finally:
        # Cleanup
        if backend_process:
            logger.info("Stopping backend...")
            backend_process.terminate()
            try:
                backend_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                backend_process.kill()
        
        if frontend_process:
            logger.info("Stopping frontend...")
            frontend_process.terminate()
            try:
                frontend_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                frontend_process.kill()
        
        logger.info("✓ Shutdown complete")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
