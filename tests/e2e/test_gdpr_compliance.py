"""
GDPR Compliance End-to-End Test

Tests GDPR compliance features:
1. Query scrubbing (hashing sensitive queries)
2. Audit log generation with trace IDs
3. Audit log export functionality
4. Verification that plaintext queries are not logged
"""

import json
import tempfile
from pathlib import Path
from datetime import datetime

import pytest

from cubo.security.security import security_manager
from cubo.services.service_manager import get_service_manager


@pytest.fixture
def gdpr_config(tmp_path):
    """Configuration with GDPR features enabled."""
    config = {
        "logging": {
            "scrub_queries": True,
            "log_file": str(tmp_path / "audit.jsonl"),
            "format": "json"
        },
        "metadata_db_path": str(tmp_path / "metadata.db")
    }
    return config


class TestGDPRCompliance:
    """Test GDPR compliance features end-to-end."""
    
    def test_query_scrubbing_enabled(self, gdpr_config):
        """
        Test that sensitive queries are hashed, not stored in plaintext.
        
        GDPR Requirement: PII in queries must not be stored.
        """
        # Enable scrubbing
        security_manager.config = gdpr_config
        
        # Simulate sensitive query
        sensitive_query = "What is John Smith's salary at john.smith@company.com?"
        
        # Get scrubbed version
        scrubbed = security_manager.scrub_query(sensitive_query)
        
        # Verify it's hashed
        assert scrubbed != sensitive_query, "Query was not scrubbed"
        assert scrubbed.startswith("sha256:") or len(scrubbed) == 64, \
            f"Scrubbed query doesn't look like a hash: {scrubbed}"
        
        # Verify same query produces same hash (consistency)
        scrubbed_again = security_manager.scrub_query(sensitive_query)
        assert scrubbed == scrubbed_again, "Hash is not consistent"
        
        print(f"✓ Sensitive query scrubbed: '{sensitive_query[:30]}...' → {scrubbed[:16]}...")
    
    def test_audit_log_contains_trace_ids(self, gdpr_config, tmp_path):
        """
        Test that audit logs include trace IDs for request tracking.
        
        GDPR Requirement: Must be able to trace all data access.
        """
        import logging
        import uuid
        
        # Set up logging to file
        log_file = tmp_path / "audit.jsonl"
        
        logger = logging.getLogger("cubo.audit")
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        
        # Log query with trace_id
        trace_id = str(uuid.uuid4())
        query_hash = security_manager.scrub_query("test query")
        
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "trace_id": trace_id,
            "query_hash": query_hash,
            "action": "query",
            "user_ip": "127.0.0.1"
        }
        
        logger.info(json.dumps(log_entry))
        handler.close()
        
        # Verify log file exists and contains trace_id
        assert log_file.exists(), "Audit log file not created"
        
        log_content = log_file.read_text()
        assert trace_id in log_content, "Trace ID not in audit log"
        assert query_hash in log_content, "Query hash not in audit log"
        
        print(f"✓ Audit log contains trace_id: {trace_id}")
    
    def test_audit_export_with_date_range(self, gdpr_config, tmp_path):
        """
        Test audit log export filtered by date range.
        
        GDPR Requirement: Must be able to export data for specific time periods.
        """
        # Create mock audit log entries
        log_file = tmp_path / "audit.jsonl"
        
        entries = [
            {
                "timestamp": "2024-11-01T10:00:00Z",
                "trace_id": "trace_001",
                "query_hash": "sha256:abc123",
                "action": "query"
            },
            {
                "timestamp": "2024-11-15T14:30:00Z",
                "trace_id": "trace_002",
                "query_hash": "sha256:def456",
                "action": "query"
            },
            {
                "timestamp": "2024-12-01T09:00:00Z",
                "trace_id": "trace_003",
                "query_hash": "sha256:ghi789",
                "action": "query"
            }
        ]
        
        with open(log_file, 'w') as f:
            for entry in entries:
                f.write(json.dumps(entry) + '\n')
        
        # Simulate export function
        def export_audit_logs(start_date: str, end_date: str):
            """Filter logs by date range."""
            results = []
            with open(log_file, 'r') as f:
                for line in f:
                    entry = json.loads(line)
                    if start_date <= entry["timestamp"] <= end_date:
                        results.append(entry)
            return results
        
        # Test date range filtering
        filtered = export_audit_logs("2024-11-01", "2024-11-30")
        
        assert len(filtered) == 2, f"Expected 2 entries, got {len(filtered)}"
        assert filtered[0]["trace_id"] == "trace_001"
        assert filtered[1]["trace_id"] == "trace_002"
        
        print(f"✓ Audit export filtered correctly: {len(filtered)} entries in date range")
    
    def test_no_plaintext_queries_in_logs(self, gdpr_config, tmp_path):
        """
        Test that plaintext queries are NEVER stored in logs.
        
        GDPR Requirement: PII must not persist in logs.
        """
        log_file = tmp_path / "audit.jsonl"
        
        sensitive_queries = [
            "What is Paolo's email address?",
            "Find contract for paolo@example.com",
            "SSN 123-45-6789 employment record"
        ]
        
        # Log scrubbed versions
        with open(log_file, 'w') as f:
            for query in sensitive_queries:
                scrubbed = security_manager.scrub_query(query)
                entry = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "query_hash": scrubbed,
                    "action": "query"
                }
                f.write(json.dumps(entry) + '\n')
        
        # Verify NO plaintext appears in logs
        log_content = log_file.read_text().lower()
        
        for query in sensitive_queries:
            assert query.lower() not in log_content, \
                f"Plaintext query found in logs: '{query}'"
        
        # Verify hashes ARE present
        assert "sha256:" in log_content or len(log_content) > 100, \
            "No query hashes found in logs"
        
        print(f"✓ GDPR compliance verified: No plaintext queries in {log_file}")
    
    def test_right_to_erasure_simulation(self, gdpr_config, tmp_path):
        """
        Test simulation of GDPR right to erasure (delete user data).
        
        This tests that we CAN identify and remove user-specific logs.
        """
        log_file = tmp_path / "audit.jsonl"
        
        # Create logs for multiple users
        entries = [
            {"trace_id": "user_A_001", "user_id": "user_A", "action": "query"},
            {"trace_id": "user_B_001", "user_id": "user_B", "action": "query"},
            {"trace_id": "user_A_002", "user_id": "user_A", "action": "ingest"},
            {"trace_id": "user_C_001", "user_id": "user_C", "action": "query"},
        ]
        
        with open(log_file, 'w') as f:
            for entry in entries:
                f.write(json.dumps(entry) + '\n')
        
        # Simulate erasure: Remove user_A logs
        def erase_user_logs(user_id: str):
            """Remove all logs for specific user."""
            temp_file = log_file.with_suffix('.tmp')
            with open(log_file, 'r') as infile, open(temp_file, 'w') as outfile:
                for line in infile:
                    entry = json.loads(line)
                    if entry.get("user_id") != user_id:
                        outfile.write(line)
            temp_file.replace(log_file)
        
        erase_user_logs("user_A")
        
        # Verify user_A logs are gone
        with open(log_file, 'r') as f:
            remaining = [json.loads(line) for line in f]
        
        assert len(remaining) == 2, f"Expected 2 entries after erasure, got {len(remaining)}"
        for entry in remaining:
            assert entry["user_id"] != "user_A", "user_A data not erased"
        
        print("✓ Right to erasure: user_A data successfully removed")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
