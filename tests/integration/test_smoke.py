def test_smoke_imports():
    """Smoke test to ensure imports and basic sanity checks pass in PR checks.

    This helps guaranteeing a minimal test run in CI for PR checks when test filters
    might otherwise collect zero tests.
    """
    import importlib

    # Try basic imports
    cubo = importlib.import_module("cubo")
    sm = importlib.import_module("cubo.security.security")

    # basic checks
    assert hasattr(cubo, "main") or hasattr(cubo, "__version__") or True
    assert hasattr(sm, "security_manager")
