#!/usr/bin/env python
"""Start the CUBO API server - run this directly in a terminal."""
import os
import sys
from pathlib import Path

# Set up Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
os.environ["PYTHONPATH"] = str(project_root)

print("Starting CUBO API server...")
print(f"Project root: {project_root}")
print(f"Python: {sys.executable}")
print(f"Python version: {sys.version}")
print("=" * 60)

import argparse

import uvicorn

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start CUBO API server")
    parser.add_argument(
        "--mode",
        choices=["laptop", "default"],
        default=None,
        help="Start server with 'laptop' or 'default' settings. If omitted, auto-detection is used.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a config JSON file. Overrides packaged config if provided.",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--port", default=8000, type=int, help="Port to bind the server to")
    parser.add_argument("--log-level", default="info", help="Log level for Uvicorn")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print effective environment variables and exit (useful for testing).",
    )
    args = parser.parse_args()

    # Set environment variables so `src.cubo.config` picks them up on import
    if args.mode == "laptop":
        os.environ["CUBO_LAPTOP_MODE"] = "1"
    elif args.mode == "default":
        os.environ["CUBO_LAPTOP_MODE"] = "0"

    if args.config:
        os.environ["CUBO_CONFIG_PATH"] = args.config

    # For quick validation in tests, support a dry-run which prints env and exits
    if args.dry_run:
        print("CUBO_LAPTOP_MODE=", os.environ.get("CUBO_LAPTOP_MODE"))
        print("CUBO_CONFIG_PATH=", os.environ.get("CUBO_CONFIG_PATH"))
        print("HOST=", args.host)
        print("PORT=", args.port)
        print("LOG_LEVEL=", args.log_level)
        sys.exit(0)

    # Run server - this will block until Ctrl+C
    uvicorn.run(
        "cubo.server.api:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        access_log=True,
    )
