"""Run the CUBO API server with uvicorn."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import uvicorn

from cubo.utils.logger import logger


def run_server(
    host: str = "127.0.0.1", port: int = 8000, reload: bool = False, log_level: str = "info"
):
    if host == "0.0.0.0":  # nosec B104
        logger.warning(
            "Binding the server to 0.0.0.0 exposes it to all network interfaces. "
            "Prefer using 127.0.0.1 for local development or explicitly opt-in when containerizing."
        )
    """Run the FastAPI server with uvicorn.

    Args:
        host: Host to bind to
        port: Port to bind to
        reload: Enable auto-reload on code changes
        log_level: Logging level
    """
    logger.info(f"Starting CUBO API server on {host}:{port}")

    # Use loop="asyncio" and limit_max_requests to avoid Windows signal issues
    config = uvicorn.Config(
        "cubo.server.api:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level,
        access_log=True,
        loop="asyncio",  # Explicitly use asyncio loop
        use_colors=True,
    )
    server = uvicorn.Server(config)

    # Run with explicit exception handling
    try:
        server.run()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        raise


def main():
    """Console script entry point for CUBO API server."""
    import argparse

    parser = argparse.ArgumentParser(description="Run CUBO API server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to (default: localhost)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--log-level", default="info", help="Log level")

    args = parser.parse_args()

    # Uvicorn expects lower-case log levels (e.g. 'debug', 'info') - normalize
    run_server(
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level.lower() if isinstance(args.log_level, str) else args.log_level,
    )


if __name__ == "__main__":
    main()
