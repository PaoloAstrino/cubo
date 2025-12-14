import sys
import os
from unittest.mock import MagicMock

# Add project root to path to ensure we import from local source
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

try:
    from cubo import CuboCore
    print("[OK] Successfully imported CuboCore from cubo package")
except ImportError as e:
    print(f"[FAIL] Failed to import CuboCore: {e}")
    sys.exit(1)

def test_dev_flow():
    print("\n--- Testing Developer Library Flow ---")
    
    # 1. Initialize
    try:
        rag = CuboCore()
        print("[OK] CuboCore initialized")
    except Exception as e:
        print(f"[FAIL] CuboCore initialization failed: {e}")
        return

    # 2. Mock heavy components (Retriever, Generator, Model)
    # We do this to verify the API surface without waiting for models to load
    print("[INFO] Mocking heavy components for quick verification...")
    
    rag.retriever = MagicMock()
    rag.retriever.retrieve_top_documents.return_value = [
        {"content": "The invoice total is 500 EUR.", "document": "The invoice total is 500 EUR."}
    ]
    
    rag.generator = MagicMock()
    rag.generator.generate_response.return_value = "The total is 500 EUR."
    
    rag.model = MagicMock() 
    
    # 3. Test query() method (the one we added)
    try:
        query = "What is the total?"
        print(f"[ACTION] Running rag.query('{query}')")
        response = rag.query(query)
        print(f"[RESULT] Response: {response}")
        
        if "500 EUR" in response:
            print("[PASS] Developer Library Flow works as expected.")
        else:
            print("[FAIL] Response did not match expected output.")
            
    except Exception as e:
        print(f"[FAIL] Query execution failed: {e}")

if __name__ == "__main__":
    test_dev_flow()
