#!/usr/bin/env python
"""
Simple smoke test to verify CUBO works end-to-end.
Run this AFTER starting the server with: python scripts/start_fullstack.py
"""
import sys

import requests

API_URL = "http://localhost:8000"


def test_component(name, fn):
    """Helper to run and report test results."""
    try:
        print(f"  Testing {name}...", end=" ")
        fn()
        print("✅")
        return True
    except Exception as e:
        print(f"❌ {str(e)}")
        return False


def main():
    print("\n" + "=" * 60)
    print("CUBO SMOKE TEST")
    print("=" * 60 + "\n")

    results = []

    # 1. Health check
    def health_check():
        r = requests.get(f"{API_URL}/api/health", timeout=5)
        assert r.status_code == 200, f"Expected 200, got {r.status_code}"
        assert r.json().get("status") in ["healthy", "degraded"], "Invalid health status"

    results.append(test_component("Health endpoint", health_check))

    # 2. Readiness check
    def readiness_check():
        r = requests.get(f"{API_URL}/api/ready", timeout=5)
        assert r.status_code == 200, f"Expected 200, got {r.status_code}"
        components = r.json().get("components", {})
        assert components.get("api") == True, "API not ready"
        assert components.get("app") == True, "App not initialized"

    results.append(test_component("Readiness endpoint", readiness_check))

    # 3. Initialize components (if not already done)
    def initialize_components():
        r = requests.post(f"{API_URL}/api/initialize", timeout=30)
        assert r.status_code in [200, 400], f"Unexpected status {r.status_code}"
        # 400 is OK - means already initialized

    results.append(test_component("Initialize components", initialize_components))

    # 4. API documentation
    def api_docs():
        r = requests.get(f"{API_URL}/docs", timeout=5)
        assert r.status_code == 200, f"Expected 200, got {r.status_code}"

    results.append(test_component("API documentation", api_docs))

    # 5. Root endpoint
    def root_endpoint():
        r = requests.get(f"{API_URL}/", timeout=5)
        assert r.status_code == 200, f"Expected 200, got {r.status_code}"
        data = r.json()
        assert "message" in data, "Missing message in root response"

    results.append(test_component("Root endpoint", root_endpoint))

    # Summary
    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"RESULTS: {passed}/{total} tests passed")
    print("=" * 60 + "\n")

    if passed == total:
        print("✅ All smoke tests passed!")
        print("\nYour CUBO system is working correctly.")
        print("\nNext steps:")
        print("  • Upload documents: POST /api/upload")
        print("  • Ingest: POST /api/ingest")
        print("  • Build index: POST /api/build-index")
        print("  • Query: POST /api/query")
        print(f"\n  Full API docs: {API_URL}/docs")
        return 0
    else:
        print(f"❌ {total - passed} test(s) failed")
        print("\nCheck that the server is running:")
        print("  python scripts/start_fullstack.py")
        return 1


if __name__ == "__main__":
    sys.exit(main())
