import logging

from src.cubo.ingestion.deep_ingestor import DeepIngestor

# Setup logging to see everything
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("cubo")
logger.setLevel(logging.DEBUG)

try:
    print("Starting ingestion...")
    ingestor = DeepIngestor(input_folder="data/uploads")
    result = ingestor.ingest()
    print("Ingestion result:", result)
except Exception:
    print("CAUGHT EXCEPTION:")
    import traceback

    traceback.print_exc()
