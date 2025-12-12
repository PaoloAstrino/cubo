"""Tests for GET /api/export-audit endpoint (GDPR compliance)."""

import csv
import io
import json
import pytest
pytest.importorskip("fastapi")
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


@pytest.fixture
def temp_log_dir(tmp_path):
    """Create temporary log directory with sample JSONL logs."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    
    # Create sample JSONL log file
    log_file = log_dir / "cubo_log.jsonl"
    
    sample_logs = [
        {
            "asctime": "2024-11-28 10:30:00,123",
            "levelname": "INFO",
            "name": "cubo.api",
            "message": "Query received: what is the vacation policy?",
            "trace_id": "tr_001"
        },
        {
            "asctime": "2024-11-28 10:30:01,456",
            "levelname": "INFO",
            "name": "cubo.retriever",
            "message": "Retrieved 5 documents",
            "trace_id": "tr_001"
        },
        {
            "asctime": "2024-11-29 14:00:00,000",
            "levelname": "INFO",
            "name": "cubo.api",
            "message": "Document deletion requested: contract.pdf",
            "trace_id": "tr_002"
        },
        {
            "asctime": "2024-11-30 09:00:00,000",
            "levelname": "WARNING",
            "name": "cubo.security",
            "message": "Query scrubbed for PII",
            "trace_id": "tr_003"
        },
    ]
    
    with open(log_file, "w", encoding="utf-8") as f:
        for log in sample_logs:
            f.write(json.dumps(log) + "\n")
    
    return log_dir


@pytest.fixture
def client(temp_log_dir):
    """Create test client with mocked log directory."""
    with patch("cubo.server.api.cubo_app", MagicMock()):
        with patch("cubo.server.api.config") as mock_config:
            mock_config.get.side_effect = lambda key, default=None: {
                "log_dir": str(temp_log_dir)
            }.get(key, default)
            
            from cubo.server.api import app
            yield TestClient(app)


class TestGDPRAuditExport:
    """Test cases for GDPR audit export endpoint."""

    def test_export_audit_returns_json(self, client):
        """Test JSON format export."""
        response = client.get("/api/export-audit?format=json")
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        
        data = response.json()
        assert "audit_entries" in data
        assert "count" in data
        assert isinstance(data["audit_entries"], list)

    def test_export_audit_returns_csv(self, client):
        """Test CSV format export."""
        response = client.get("/api/export-audit?format=csv")
        
        assert response.status_code == 200
        assert "text/csv" in response.headers["content-type"]
        
        # Parse CSV content
        content = response.content.decode("utf-8")
        reader = csv.DictReader(io.StringIO(content))
        rows = list(reader)
        
        # Verify CSV headers
        expected_headers = ["timestamp", "trace_id", "query_hash", "level", "component", "action"]
        assert reader.fieldnames == expected_headers

    def test_export_audit_default_format_is_csv(self, client):
        """Test that default format is CSV."""
        response = client.get("/api/export-audit")
        
        assert response.status_code == 200
        assert "text/csv" in response.headers["content-type"]

    def test_export_audit_date_filter_start(self, client):
        """Test filtering by start date."""
        response = client.get("/api/export-audit?start_date=2024-11-29&format=json")
        
        assert response.status_code == 200
        data = response.json()
        
        # All entries should be on or after 2024-11-29
        for entry in data["audit_entries"]:
            entry_date = datetime.fromisoformat(entry["timestamp"]).date()
            assert entry_date >= datetime(2024, 11, 29).date()

    def test_export_audit_date_filter_end(self, client):
        """Test filtering by end date."""
        response = client.get("/api/export-audit?end_date=2024-11-28&format=json")
        
        assert response.status_code == 200
        data = response.json()
        
        # All entries should be on or before 2024-11-28
        for entry in data["audit_entries"]:
            entry_date = datetime.fromisoformat(entry["timestamp"]).date()
            assert entry_date <= datetime(2024, 11, 28).date()

    def test_export_audit_date_filter_range(self, client):
        """Test filtering by date range."""
        response = client.get(
            "/api/export-audit?start_date=2024-11-28&end_date=2024-11-29&format=json"
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # All entries should be within range
        for entry in data["audit_entries"]:
            entry_date = datetime.fromisoformat(entry["timestamp"]).date()
            assert datetime(2024, 11, 28).date() <= entry_date <= datetime(2024, 11, 29).date()

    def test_export_audit_invalid_start_date(self, client):
        """Test error on invalid start date format."""
        response = client.get("/api/export-audit?start_date=invalid-date")
        
        assert response.status_code == 400
        assert "invalid" in response.json()["detail"].lower()

    def test_export_audit_invalid_end_date(self, client):
        """Test error on invalid end date format."""
        response = client.get("/api/export-audit?end_date=31-12-2024")
        
        assert response.status_code == 400
        assert "invalid" in response.json()["detail"].lower()

    def test_export_audit_filename_in_header(self, client):
        """Test that response includes filename in Content-Disposition."""
        response = client.get("/api/export-audit?format=csv")
        
        assert "content-disposition" in response.headers
        assert "attachment" in response.headers["content-disposition"]
        assert "cubo_audit_" in response.headers["content-disposition"]

    def test_export_audit_query_hash_for_privacy(self, client):
        """Test that queries are hashed for GDPR privacy."""
        response = client.get("/api/export-audit?format=json")
        
        data = response.json()
        
        for entry in data["audit_entries"]:
            # Query hash should be present if query-related
            query_hash = entry.get("query_hash", "")
            # If there's a hash, it should be in format [hashed:xxx]
            if query_hash:
                assert "[hashed:" in query_hash or query_hash == ""

    def test_export_audit_includes_trace_id(self, client):
        """Test that audit entries include trace IDs."""
        response = client.get("/api/export-audit?format=json")
        
        data = response.json()
        
        for entry in data["audit_entries"]:
            assert "trace_id" in entry

    def test_export_audit_sorted_by_timestamp(self, client):
        """Test that entries are sorted by timestamp."""
        response = client.get("/api/export-audit?format=json")
        
        data = response.json()
        entries = data["audit_entries"]
        
        if len(entries) > 1:
            timestamps = [e["timestamp"] for e in entries]
            assert timestamps == sorted(timestamps)


class TestGDPRAuditExportEdgeCases:
    """Edge cases for audit export."""

    def test_export_audit_empty_logs(self, tmp_path):
        """Test export with no log files."""
        empty_log_dir = tmp_path / "empty_logs"
        empty_log_dir.mkdir()
        
        with patch("cubo.server.api.cubo_app", MagicMock()):
            with patch("cubo.server.api.config") as mock_config:
                mock_config.get.return_value = str(empty_log_dir)
                
                from cubo.server.api import app
                client = TestClient(app)
                
                response = client.get("/api/export-audit?format=json")
                
                assert response.status_code == 200
                data = response.json()
                assert data["count"] is None
                assert data["audit_entries"] == []

    def test_export_audit_malformed_json_lines(self, tmp_path):
        """Test that malformed JSON lines are skipped."""
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        
        log_file = log_dir / "cubo_log.jsonl"
        with open(log_file, "w") as f:
            f.write('{"asctime": "2024-11-30 10:00:00,000", "levelname": "INFO", "name": "test", "message": "valid", "trace_id": ""}\n')
            f.write('this is not valid json\n')
            f.write('{"asctime": "2024-11-30 11:00:00,000", "levelname": "INFO", "name": "test", "message": "also valid", "trace_id": ""}\n')
        
        with patch("cubo.server.api.cubo_app", MagicMock()):
            with patch("cubo.server.api.config") as mock_config:
                mock_config.get.return_value = str(log_dir)
                
                from cubo.server.api import app
                client = TestClient(app)
                
                response = client.get("/api/export-audit?format=json")
                
                assert response.status_code == 200
                data = response.json()
                # Should have 2 valid entries, malformed line skipped
                assert len(data["audit_entries"]) == 2
