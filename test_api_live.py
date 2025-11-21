"""Test the API server - run this in a SEPARATE terminal while server is running."""
import requests
import json
import time

API_BASE = "http://localhost:8000"

def test_health():
    """Test health endpoint."""
    print("\n" + "=" * 60)
    print("Testing /api/health")
    print("=" * 60)
    
    try:
        response = requests.get(f"{API_BASE}/api/health", timeout=5)
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_root():
    """Test root endpoint."""
    print("\n" + "=" * 60)
    print("Testing /")
    print("=" * 60)
    
    try:
        response = requests.get(f"{API_BASE}/", timeout=5)
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    """Run all tests."""
    print("\n🚀 Starting API Tests")
    print(f"API Base URL: {API_BASE}")
    
    # Wait a moment for server to be ready
    print("\nWaiting for server...")
    time.sleep(2)
    
    results = []
    
    # Test root
    results.append(("Root endpoint", test_root()))
    
    # Test health
    results.append(("Health endpoint", test_health()))
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {name}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit(main())
