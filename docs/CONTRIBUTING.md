# Contributing to Cubo

We welcome contributions! Please follow these steps:

1. **Fork the repository** and create a new branch for your feature or bug fix.
2. **Set up the development environment**:
   ```bash
   # Backend (Python)
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   pip install -r requirements.txt
   # For development and tests, prefer the dev extra:
   pip install -e '.[dev]'
   # Or: pip install -r requirements/requirements-dev.txt

   # Install pre-commit hooks (automatic code checks)
   pre-commit install

   # Frontend (Node.js)
   npm run install:frontend
   ```

   ### Quickstart (one‑line)
   For new contributors who want a simple, one-line start, use the provided quickstart scripts (recommended):

   - Windows (PowerShell):
     ```powershell
     .\run_local.ps1
     ```
   - macOS / Linux (bash):
     ```bash
     ./run_local.sh
     ```

   These scripts create/activate `.venv`, install required dependencies (backend + frontend) and launch the full stack.

   Alternatively, to install the package in editable mode and run only the API:
   ```bash
   python -m pip install -e . && python -m cubo.server.run --reload
   ```

   (Optional) To start the frontend UI in another terminal:
   ```bash
   npm run dev --prefix frontend
   ```
3. **Make your changes** adhering to the existing code style:
   - Python: follow `ruff` and `black` formatting.
   - JavaScript/TypeScript: run `npm run lint` and `npm run format` (if configured).
4. **Write tests** for new functionality. Backend tests are in `tests/`, frontend tests can be added under `frontend/__tests__`.
5. **Run the full test suite**:
   ```bash
   # Backend tests
   pytest

   # Frontend lint
   npm run lint --prefix frontend
   ```
6. **Commit with a clear message** and push to your fork.
7. **Open a Pull Request** targeting the `main` branch. Include a description of the changes and any relevant issue numbers.

### Pre-commit Hooks

Pre-commit hooks automatically check code quality before each commit:
- **Black**: Code formatting (100 character line length)
- **Ruff**: Fast Python linting
- **isort**: Import sorting
- **mypy**: Type checking
- **Bandit**: Security scanning

To run checks manually on all files:
```bash
pre-commit run --all-files
```

To skip hooks (not recommended):
```bash
git commit --no-verify
```

### Code of Conduct
By contributing, you agree to abide by the project's [Code of Conduct](CODE_OF_CONDUCT.md).
