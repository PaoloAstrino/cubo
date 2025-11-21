#!/usr/bin/env python
"""Start the CUBO API server - run this directly in a terminal."""
import sys
import os
from pathlib import Path

# Set up Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.environ['PYTHONPATH'] = str(project_root)

print(f"Starting CUBO API server...")
print(f"Project root: {project_root}")
print(f"Python: {sys.executable}")
print(f"Python version: {sys.version}")
print("=" * 60)

import uvicorn

if __name__ == "__main__":
    # Run server - this will block until Ctrl+C
    uvicorn.run(
        "src.cubo.server.api_simple:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True
    )
