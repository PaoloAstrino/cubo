# Compatibility shim so tests importing `src.cubo` can access the `cubo` package.
from importlib import import_module

try:
    import_module("cubo")
except Exception:
    # If the top-level `cubo` package is not importable, do nothing; tests will fail
    pass
