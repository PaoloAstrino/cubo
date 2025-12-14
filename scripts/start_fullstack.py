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

try:
    import psutil
except Exception:
    psutil = None

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# from cubo.config import setup_logging

logger = logging.getLogger("cubo.fullstack")


def start_backend(
    backend_path: Path, port: int, reload: bool = True, mode: str = None, config_path: str = None
) -> subprocess.Popen:
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
    env["PORT"] = str(port)
    if mode == "laptop":
        env["CUBO_LAPTOP_MODE"] = "1"
    elif mode == "default":
        env["CUBO_LAPTOP_MODE"] = "0"
    if config_path:
        env["CUBO_CONFIG_PATH"] = config_path

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
    env["PORT"] = str(port)

    frontend_process = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=frontend_dir,
        shell=True,  # Required for Windows to find npm in PATH
        env=env,
    )

    return frontend_process


def check_ollama() -> bool:
    """Check if Ollama is running and accessible.

    Returns:
        True if Ollama is ready, False otherwise.
    """
    import requests

    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            models = response.json().get("models", [])
            if models:
                logger.info(f"[OK] Ollama is running with {len(models)} model(s)")
                return True
            else:
                logger.warning("[WARN] Ollama is running but no models found")
                logger.warning("  Run: ollama pull llama3.2")
                return False
    except requests.exceptions.ConnectionError:
        logger.warning("[WARN] Ollama is not running")
        logger.warning("  Please start Ollama and pull a model:")
        logger.warning("  1. Start Ollama from Start Menu or run 'ollama serve'")
        logger.warning("  2. Run: ollama pull llama3.2")
        return False
    except Exception as e:
        logger.debug(f"Ollama check failed: {e}")
        return False


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
                logger.info(f"\n[OK] Backend is ready! (took {i+1} seconds)")
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

    logger.error(f"\n[FAIL] Backend failed to start within {timeout} seconds")
    logger.error("Check the backend output above for errors")
    logger.info("=" * 60)
    return False


def initialize_rag_system(backend_url: str) -> bool:
    """Initialize the RAG system (load embeddings, vector store, etc.).

    Args:
        backend_url: Backend base URL.

    Returns:
        True if initialization succeeded, False otherwise.
    """
    import requests

    logger.info("\n" + "=" * 60)
    logger.info("Initializing RAG System...")
    logger.info("=" * 60)
    logger.info("Loading embedding models, vector store, and retriever...")

    try:
        response = requests.post(f"{backend_url}/api/initialize", timeout=120)
        if response.status_code == 200:
            logger.info("[OK] RAG system initialized successfully!")
            logger.info("=" * 60)
            return True
        else:
            logger.error(f"[FAIL] Initialization failed: {response.status_code}")
            logger.error(f"  Response: {response.text}")
            return False
    except requests.exceptions.Timeout:
        logger.error("[FAIL] Initialization timed out (>120s)")
        logger.error("  This may happen on first run while downloading models")
        return False
    except Exception as e:
        logger.error(f"[FAIL] Initialization error: {e}")
        return False


@click.command()
@click.option(
    "--backend-port", default=8000, help="Port for backend API server.", show_default=True
)
@click.option(
    "--mode",
    default=None,
    type=click.Choice(["laptop", "default"], case_sensitive=False),
    help="Start backend in laptop or default mode (sets CUBO_LAPTOP_MODE env var).",
)
@click.option(
    "--config-path",
    default=None,
    help="Path to a JSON config file to use for the backend (sets CUBO_CONFIG_PATH).",
)
@click.option(
    "--frontend-port", default=3000, help="Port for frontend dev server.", show_default=True
)
@click.option(
    "--backend-path",
    default="cubo/server/run.py",
    help="Path to backend run.py script.",
    show_default=True,
)
@click.option(
    "--frontend-dir", default="frontend", help="Path to frontend directory.", show_default=True
)
@click.option("--no-reload", is_flag=True, help="Disable auto-reload for backend.")
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    help="Logging level.",
    show_default=True,
)
@click.option(
    "--force-kill",
    is_flag=True,
    default=False,
    help="If passed, attempts to terminate processes occupying configured ports before starting.",
)
@click.option(
    "--timeout", default=60, help="Timeout in seconds for backend startup.", show_default=True
)
def main(
    backend_port: int,
    frontend_port: int,
    backend_path: str,
    frontend_dir: str,
    no_reload: bool,
    log_level: str,
    timeout: int,
    mode: str = None,
    config_path: str = None,
    force_kill: bool = False,
):
    """Start CUBO full stack (backend + frontend)."""
    # Setup logging
    logging.basicConfig(level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

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
        # Check Ollama first
        check_ollama()

        # Check ports and friendly messaging
        def find_pid_using_port(port: int) -> int | None:
            if psutil is None:
                return None
            try:
                for conn in psutil.net_connections(kind="inet"):
                    if (
                        conn.laddr
                        and conn.laddr.port == int(port)
                        and conn.status == psutil.CONN_LISTEN
                    ):
                        return conn.pid
            except Exception:
                pass
            return None

        conflicts = []
        backend_pid = find_pid_using_port(backend_port)
        frontend_pid = find_pid_using_port(frontend_port)
        if backend_pid:
            conflicts.append((backend_port, backend_pid, "backend"))
        if frontend_pid:
            conflicts.append((frontend_port, frontend_pid, "frontend"))
        if conflicts:
            msg_lines = ["Port conflict detected:"]
            for port, pid, name in conflicts:
                try:
                    proc_name = psutil.Process(pid).name() if psutil else "unknown"
                except Exception:
                    proc_name = "unknown"
                msg_lines.append(f" - {name} port {port} is in use by PID {pid} ({proc_name})")
            msg_lines.append("")
            msg_lines.append("Options to resolve:")
            msg_lines.append(
                " * Kill the process using the port (Windows: Stop-Process -Id <PID>, Linux/macOS: kill -9 <PID>)"
            )
            msg_lines.append(" * Select different ports: pass --backend-port and --frontend-port")
            msg_lines.append(
                " * Let this script try to terminate the process with --force-kill (requires appropriate permissions)"
            )
            logger.error("\n".join(msg_lines))
            if not force_kill:
                logger.error(
                    "Exiting due to port conflicts. To force termination, re-run with --force-kill"
                )
                return 1
            # Attempt to kill conflicting processes (best-effort)
            for port, pid, name in conflicts:
                try:
                    logger.info(f"Attempting to terminate PID {pid} occupying port {port}...")
                    p = psutil.Process(pid)
                    p.terminate()
                    p.wait(timeout=5)
                    logger.info(f"Terminated PID {pid}.")
                except Exception as e:
                    logger.warning(f"Failed to terminate PID {pid}: {e}")

        # Start backend
        backend_process = start_backend(
            backend_script, backend_port, reload=not no_reload, mode=mode, config_path=config_path
        )

        # Wait for backend to be ready
        backend_url = f"http://localhost:{backend_port}"
        if not wait_for_backend(backend_url, timeout):
            return 1

        # Initialize RAG system (load models, etc.)
        initialize_rag_system(backend_url)

        # Start frontend
        frontend_process = start_frontend(frontend_directory, frontend_port)

        logger.info("\n" + "=" * 60)
        logger.info("[OK] CUBO Full Stack Started")
        logger.info("=" * 60)
        logger.info(f"Backend API:  http://localhost:{backend_port}")
        logger.info(f"API Docs:     http://localhost:{backend_port}/docs")
        logger.info(f"Frontend:     http://localhost:{frontend_port}")
        logger.info("=" * 60)
        logger.info("\nPress Ctrl+C to stop\n")

        # Monitor processes
        while True:
            if backend_process.poll() is not None:
                logger.error("\n[FAIL] Backend process died")
                break

            if frontend_process.poll() is not None:
                logger.error("\n[FAIL] Frontend process died")
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

        logger.info("[OK] Shutdown complete")

    return 0


if __name__ == "__main__":
    sys.exit(main())
