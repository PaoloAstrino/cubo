import sys
from pathlib import Path

# Ensure the repository root is on sys.path when running integration tests directly.
ROOT = Path(__file__).parent.parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import pytest  # noqa: F401
except Exception:
    pass
