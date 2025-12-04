# Re-export the top-level `cubo` package under `src.cubo` so tests can import either name.
# We proxy attribute access to the real `cubo` package.
import importlib as _importlib
import sys as _sys

_real = _importlib.import_module("cubo")

# Copy attributes from real cubo module into this namespace
for _name in dir(_real):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_real, _name)

# Ensure modules referenced as src.cubo.submodule are resolved to the real cubo module
_sys.modules[__name__] = _real
