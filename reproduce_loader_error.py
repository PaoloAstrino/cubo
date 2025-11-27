import logging

from src.cubo.ingestion.document_loader import DocumentLoader

# Setup logging
logging.basicConfig(level=logging.DEBUG)

try:
    print("Initializing DocumentLoader...")
    loader = DocumentLoader()

    print("Calling load_documents('data')...")
    # Simulate the API call passing a string instead of a list
    result = loader.load_documents("data")

    print(f"Result: {result}")

except Exception:
    print("CAUGHT EXCEPTION:")
    import traceback

    traceback.print_exc()
