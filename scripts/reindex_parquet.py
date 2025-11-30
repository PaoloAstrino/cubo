#!/usr/bin/env python3
"""Thin shim script for reindex_parquet in `src/cubo/scripts`.
This keeps the top-level `scripts/` folder compatible with tests and existing workflows.
"""
import sys
from pathlib import Path

# Ensure `src` is on the path so imports resolve when the script is executed directly
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cubo.scripts.reindex_parquet import main

if __name__ == "__main__":
    main()
