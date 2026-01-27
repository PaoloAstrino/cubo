#!/usr/bin/env python
"""Start the CUBO API server - run this directly in a terminal."""
import os
import socket
import sys
from pathlib import Path

# Set up Python path
# Assuming this script is in <root>/scripts/start_api_server.py
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.environ["PYTHONPATH"] = str(project_root)

print("Starting CUBO API server...")
print(f"Project root: {project_root}")
print(f"Python: {sys.executable}")
print(f"Python version: {sys.version}")
print("=" * 60)

# ruff: noqa: E402
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

    # Check if port is available, find alternative if not
    port = args.port
    max_attempts = 10
    for attempt in range(max_attempts):
        try:
            # Try to bind to the port to check availability
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind((args.host if args.host != "0.0.0.0" else "127.0.0.1", port))
                # Port is available
                break
        except OSError:
            # Port is in use, try next one
            if attempt == 0:
                print(f"⚠ Port {port} is already in use, trying alternative ports...")
            port += 1
            if attempt == max_attempts - 1:
                print(f"✗ Could not find an available port after {max_attempts} attempts")
                sys.exit(1)

    if port != args.port:
        print(f"⚠ Using port {port} instead of {args.port}")

    # Run server - this will block until Ctrl+C
    uvicorn.run(
        "cubo.server.api:app",
        host=args.host,
        port=port,
        log_level=args.log_level,
        access_log=True,
    )
