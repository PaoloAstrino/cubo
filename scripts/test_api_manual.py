"""Quick manual test of the API server."""

import time

import requests

API_URL = "http://localhost:8000"

print("Testing CUBO API Server")
print("=" * 60)

# Wait for server to be ready
print("\n1. Waiting for server...")
for i in range(10):
    try:
        response = requests.get(f"{API_URL}/api/health", timeout=2)
        if response.status_code == 200:
            print("✓ Server is ready!")
            break
    except Exception:
        if i < 9:
            time.sleep(1)
        else:
            print("✗ Server not responding")
            exit(1)

# Health check
print("\n2. Health Check:")
response = requests.get(f"{API_URL}/api/health")
print(f"Status: {response.status_code}")
print(f"Response: {response.json()}")
print(f"Trace ID: {response.headers.get('x-trace-id')}")

# Upload a file
print("\n3. Upload File:")
with open("data/frog_story.txt", "rb") as f:
    files = {"file": ("frog_story.txt", f, "text/plain")}
    response = requests.post(f"{API_URL}/api/upload", files=files)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"✓ Uploaded: {data['filename']} ({data['size']} bytes)")
        print(f"Trace ID: {data['trace_id']}")
    else:
        print(f"✗ Failed: {response.text}")

# Query (will likely fail without retriever initialized)
print("\n4. Query:")
response = requests.post(f"{API_URL}/api/query", json={"query": "What is this about?", "top_k": 3})
print(f"Status: {response.status_code}")
if response.status_code == 200:
    data = response.json()
    print(f"✓ Answer: {data['answer'][:100]}...")
    print(f"Sources: {len(data['sources'])}")
    print(f"Trace ID: {data['trace_id']}")
elif response.status_code == 503:
    print("⚠ Expected - retriever not initialized")
    print("  (Need to run ingest and build-index first)")
else:
    print(f"✗ Failed: {response.text}")

print("\n" + "=" * 60)
print("✓ Manual API test complete!")
print("\nNext steps:")
print("  - Visit http://localhost:8000/docs for API documentation")
print("  - Start frontend: cd frontend && pnpm dev")
print("  - Run E2E test: python scripts/e2e_smoke.py")
