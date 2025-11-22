# Contributing to Cubo

We welcome contributions! Please follow these steps:

1. **Fork the repository** and create a new branch for your feature or bug fix.
2. **Set up the development environment**:
   ```bash
   # Backend (Python)
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   pip install -r requirements.txt -r requirements-dev.txt

   # Frontend (Node.js)
   npm run install:frontend
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

### Code of Conduct
By contributing, you agree to abide by the project's [Code of Conduct](CODE_OF_CONDUCT.md).
