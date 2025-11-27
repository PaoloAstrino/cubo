"""
E2E smoke test for CUBO RAG system.

Tests the complete flow:
1. Upload document
2. Ingest documents
3. Build index
4. Query with trace_id validation
5. Verify logs contain trace_id
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.cubo.utils.logger import logger

API_BASE_URL = "http://localhost:8000"
TEST_DOCUMENT_PATH = project_root / "data" / "frog_story.txt"
TEST_QUERY = "What is the story about?"


class E2ESmokeTest:
    """E2E smoke test runner."""

    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        self.trace_ids = []

    def check_health(self) -> bool:
        """Check API health."""
        logger.info("Checking API health...")
        try:
            response = requests.get(f"{self.base_url}/api/health", timeout=5)
            response.raise_for_status()

            data = response.json()
            logger.info(f"Health check: {data}")

            if data["status"] != "healthy":
                logger.warning(f"API is degraded: {data}")
                return False

            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    def upload_document(self) -> Optional[Dict[str, Any]]:
        """Upload a test document."""
        logger.info(f"Uploading document: {TEST_DOCUMENT_PATH}")

        if not TEST_DOCUMENT_PATH.exists():
            logger.error(f"Test document not found: {TEST_DOCUMENT_PATH}")
            return None

        try:
            with open(TEST_DOCUMENT_PATH, "rb") as f:
                files = {"file": (TEST_DOCUMENT_PATH.name, f, "text/plain")}
                response = requests.post(f"{self.base_url}/api/upload", files=files, timeout=30)
                response.raise_for_status()

            data = response.json()
            trace_id = data.get("trace_id") or response.headers.get("x-trace-id")

            if trace_id:
                self.trace_ids.append(trace_id)
                logger.info(f"Upload successful with trace_id: {trace_id}")
            else:
                logger.warning("No trace_id in upload response")

            return data
        except Exception as e:
            logger.error(f"Upload failed: {e}", exc_info=True)
            return None

    def ingest_documents(self) -> Optional[Dict[str, Any]]:
        """Ingest documents."""
        logger.info("Ingesting documents...")

        try:
            response = requests.post(
                f"{self.base_url}/api/ingest", json={"fast_pass": True}, timeout=60
            )
            response.raise_for_status()

            data = response.json()
            trace_id = data.get("trace_id") or response.headers.get("x-trace-id")

            if trace_id:
                self.trace_ids.append(trace_id)
                logger.info(f"Ingest successful with trace_id: {trace_id}")
            else:
                logger.warning("No trace_id in ingest response")

            logger.info(f"Processed {data.get('documents_processed', 0)} documents")
            return data
        except Exception as e:
            logger.error(f"Ingest failed: {e}", exc_info=True)
            return None

    def build_index(self) -> Optional[Dict[str, Any]]:
        """Build search index."""
        logger.info("Building index...")

        try:
            response = requests.post(
                f"{self.base_url}/api/build-index", json={"force_rebuild": False}, timeout=120
            )
            response.raise_for_status()

            data = response.json()
            trace_id = data.get("trace_id") or response.headers.get("x-trace-id")

            if trace_id:
                self.trace_ids.append(trace_id)
                logger.info(f"Index build successful with trace_id: {trace_id}")
            else:
                logger.warning("No trace_id in build response")

            return data
        except Exception as e:
            logger.error(f"Index build failed: {e}", exc_info=True)
            return None

    def query(self, query_text: str = TEST_QUERY) -> Optional[Dict[str, Any]]:
        """Query the RAG system."""
        logger.info(f"Querying: {query_text}")

        try:
            response = requests.post(
                f"{self.base_url}/api/query",
                json={"query": query_text, "top_k": 5, "use_reranker": True},
                timeout=60,
            )
            response.raise_for_status()

            data = response.json()
            trace_id = data.get("trace_id") or response.headers.get("x-trace-id")

            if trace_id:
                self.trace_ids.append(trace_id)
                logger.info(f"Query successful with trace_id: {trace_id}")
            else:
                logger.warning("No trace_id in query response")

            logger.info(f"Answer: {data.get('answer', '')[:100]}...")
            logger.info(f"Sources: {len(data.get('sources', []))}")

            return data
        except Exception as e:
            logger.error(f"Query failed: {e}", exc_info=True)
            return None

    def verify_logs(self) -> bool:
        """Verify that logs contain trace_ids."""
        logger.info("Verifying logs for trace_ids...")

        log_file = project_root / "logs" / "cubo_log.txt"

        if not log_file.exists():
            logger.warning(f"Log file not found: {log_file}")
            return False

        found_trace_ids = []
        errors_in_logs = []

        try:
            with open(log_file, encoding="utf-8") as f:
                # Read last 100 lines
                lines = f.readlines()[-100:]

                for line in lines:
                    try:
                        log_entry = json.loads(line.strip())

                        # Check for errors
                        if log_entry.get("level") == "error":
                            errors_in_logs.append(log_entry)

                        # Check for trace_ids
                        trace_id = log_entry.get("trace_id")
                        if trace_id and trace_id in self.trace_ids:
                            found_trace_ids.append(trace_id)
                    except json.JSONDecodeError:
                        # Skip non-JSON lines
                        continue

            logger.info(f"Found {len(set(found_trace_ids))} trace_ids in logs")
            logger.info(f"Found {len(errors_in_logs)} errors in logs")

            if errors_in_logs:
                logger.error("Errors found in logs:")
                for error in errors_in_logs[-5:]:  # Show last 5 errors
                    logger.error(f"  {error}")

            # Verify all trace_ids are in logs
            missing_trace_ids = set(self.trace_ids) - set(found_trace_ids)
            if missing_trace_ids:
                logger.warning(f"Missing trace_ids in logs: {missing_trace_ids}")
                return False

            return True
        except Exception as e:
            logger.error(f"Log verification failed: {e}", exc_info=True)
            return False

    def run(self) -> bool:
        """Run the complete E2E smoke test."""
        logger.info("=" * 60)
        logger.info("Starting E2E smoke test")
        logger.info("=" * 60)

        start_time = time.time()

        # Step 1: Health check
        if not self.check_health():
            logger.error("Health check failed - aborting test")
            return False

        # Step 1.5: Ensure heavy components are initialized (model, retriever)
        logger.info("Ensuring backend heavy components are initialized...")
        try:
            init_resp = requests.post(f"{self.base_url}/api/initialize", timeout=600)
            if init_resp.status_code != 200:
                logger.warning(
                    f"Initialization failed or returned non-200: {init_resp.status_code} - {init_resp.text}"
                )
            else:
                init_data = init_resp.json()
                trace_id = init_data.get("trace_id") or init_resp.headers.get("x-trace-id")
                if trace_id:
                    self.trace_ids.append(trace_id)
                    logger.info(f"Initialization completed with trace_id: {trace_id}")
        except Exception as e:
            logger.error(f"Initialization request failed: {e}")
            # Continue even if initialization fails: ingestion or build may trigger initialization on demand

        # Step 2: Upload document
        upload_result = self.upload_document()
        if not upload_result:
            logger.error("Upload failed - aborting test")
            return False

        # Step 3: Ingest documents
        ingest_result = self.ingest_documents()
        if not ingest_result:
            logger.error("Ingest failed - aborting test")
            return False

        # Step 4: Build index
        build_result = self.build_index()
        if not build_result:
            logger.error("Index build failed - aborting test")
            return False

        # Step 5: Query
        query_result = self.query()
        if not query_result:
            logger.error("Query failed - aborting test")
            return False

        # Step 6: Verify logs
        time.sleep(2)  # Give logs time to flush
        logs_ok = self.verify_logs()

        elapsed = time.time() - start_time

        logger.info("=" * 60)
        if logs_ok:
            logger.info(f"✓ E2E smoke test PASSED in {elapsed:.2f}s")
        else:
            logger.warning(f"⚠ E2E smoke test completed with warnings in {elapsed:.2f}s")
        logger.info(f"Trace IDs: {self.trace_ids}")
        logger.info("=" * 60)

        return logs_ok


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run E2E smoke test")
    parser.add_argument("--url", default=API_BASE_URL, help="API base URL")
    parser.add_argument("--query", default=TEST_QUERY, help="Test query")

    args = parser.parse_args()

    test = E2ESmokeTest(base_url=args.url)
    success = test.run()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
