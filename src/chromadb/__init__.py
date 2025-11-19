"""Minimal chromadb shim for unit testing environment.
This module provides a small in-memory replacement for the chromadb
package used by tests to avoid heavy native dependencies like onnxruntime.

Note: This shim is used ONLY for local unit tests. For production use,
install the real chromadb package.
"""

from .api.client import PersistentClient  # noqa: E402

__all__ = ["PersistentClient"]
