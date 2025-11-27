"""Run the CUBO API server with hypercorn (better Windows support)."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import asyncio

from hypercorn.asyncio import serve
from hypercorn.config import Config

from src.cubo.utils.logger import logger


async def run_server_async(host: str = "0.0.0.0", port: int = 8000):
    """Run the FastAPI server with hypercorn."""
    config = Config()
    config.bind = [f"{host}:{port}"]
    config.accesslog = "-"  # Log to stdout
    config.errorlog = "-"

    logger.info(f"Starting CUBO API server on {host}:{port} with hypercorn")

    # Import the app
    from src.cubo.server.api import app

    await serve(app, config)


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Synchronous wrapper for async server."""
    try:
        asyncio.run(run_server_async(host, port))
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run CUBO API server with hypercorn")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")

    args = parser.parse_args()

    run_server(host=args.host, port=args.port)
