"""Profile retrieval WITHOUT reranking to get clean breakdown."""

import json

# Disable reranking before importing CUBO
import os
import sys
from pathlib import Path

os.environ["CUBO_RERANKER_ENABLED"] = "false"

# Now run profiling
sys.path.insert(0, str(Path(__file__).parent.parent))

from cubo.config import config

config.set("retrieval.reranker_enabled", False)

# Import and run the profiler
from tools.profile_retrieval_breakdown_real import main

if __name__ == "__main__":
    print("\nRERANKER DISABLED FOR CLEAN BASELINE\n")
    main()
