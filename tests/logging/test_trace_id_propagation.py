import json
import time

from cubo.config import config
from cubo.services.service_manager import ServiceManager
from cubo.utils.logger import logger_instance


def test_trace_id_propagation(tmp_path):
    """Verify that a trace_id is attached to logs emitted from a background task.

    The logger may emit JSON logs where the ``message`` field itself is a JSON
    string (when structlog is used). This test extracts the actual message text
    and asserts that a ``trace_id`` field is present.
    """
    # Configure logger to write JSON logs to a temporary file
    log_file = tmp_path / "trace_log.jsonl"
    config.set("logging.log_file", str(log_file))
    config.set("logging.format", "json")
    logger_instance.shutdown()
    logger_instance._setup_logging()

    svc = ServiceManager(max_workers=2)

    def op(filepath):
        # Simple operation that logs something
        from cubo.utils.logger import logger as inlogger

        inlogger.info("op started", extra={"file": filepath})
        return True

    fut = svc.execute_async("document_processing", op, "dummy.txt", with_retry=False)
    _res = fut.result(timeout=5)
    assert _res
    # Allow logger to flush
    time.sleep(0.1)
    with open(log_file, encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    assert lines, "Log file should contain entries"
    print(f"DEBUG: Captured lines: {lines}")

    found = False
    for line in lines:
        try:
            rec = json.loads(line)
        except Exception:
            continue
        # Extract the actual message text (handle double‑encoded JSON)
        msg = rec.get("message") or rec.get("msg")
        if isinstance(msg, str) and msg.startswith("{"):
            try:
                inner = json.loads(msg)
                msg_text = inner.get("event") or inner.get("message")
                # Propagate trace_id if present in inner payload
                if "trace_id" in inner:
                    rec["trace_id"] = inner["trace_id"]
            except Exception:
                msg_text = msg
        else:
            msg_text = msg
        if msg_text == "op started":
            # Verify trace_id exists (non‑empty string)
            assert "trace_id" in rec and rec["trace_id"], "trace_id missing in log record"
            found = True
            break
    assert found, "Expected log entry with message 'op started' not found"
