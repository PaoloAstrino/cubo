# Requirements directory

This folder contains copies of specialized requirements files used for documentation and traceability.

Files:
- `requirements-core.txt` — core (laptop) runtime
- `requirements-minimal.txt` — minimal runtime (Ollama-only)
- `requirements-lint.txt` — lint-only dependencies
- `requirements-dev.txt` — development dependencies (also available at project root)

Recommendation
--------------
- For most users who want a compact install that works out of the box, use the `minimal` extra:

  ```bash
  pip install -e .[minimal]
  ```

- For development and running tests, use the dev extra:

  ```bash
  pip install -e .[dev]
  ```

- CI pipelines that need a pinned, minimal set can continue to use `requirements-ci.txt` located at the repo root.

Notes
-----
We keep the full `requirements.txt` at the repo root for backward compatibility and reproducible installs, and we expose convenient extras in `pyproject.toml` for easier `pip install` usage.
