import importlib

import ruff

print("ruff module", ruff)
print(
    "dir includes:",
    [n for n in dir(ruff) if any(k in n for k in ("run", "cli", "api", "_main"))][:200],
)
try:
    m = importlib.import_module("ruff._main")
    print("_main:", dir(m)[:100])
except Exception as e:
    print("_main not importable", e)
