#!/usr/bin/env python
"""Debug script to test TestClient creation."""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("=" * 60)
print("Testing TestClient creation...")
print("=" * 60)

try:
    print("\n1. Importing FastAPI...")
    from fastapi import FastAPI
    print("   ✓ FastAPI imported")
    
    print("\n2. Importing TestClient...")
    from fastapi.testclient import TestClient
    print("   ✓ TestClient imported")
    
    print("\n3. Importing app from api module...")
    from src.cubo.server.api import app
    print(f"   ✓ App imported: {type(app)}")
    
    print("\n4. Creating TestClient with positional arg...")
    client = TestClient(app)
    print(f"   ✓ TestClient created: {type(client)}")
    
    print("\n5. Testing GET request to /api/health...")
    response = client.get("/api/health")
    print(f"   ✓ Response received: {response.status_code}")
    print(f"   Response body: {response.json()}")
    
    print("\n" + "=" * 60)
    print("SUCCESS: All tests passed!")
    print("=" * 60)
    
except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
