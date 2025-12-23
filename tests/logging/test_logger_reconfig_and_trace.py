import json
import time

from cubo.config import config
from cubo.services.service_manager import ServiceManager
from cubo.utils.logger import logger_instance


def test_logger_reconfig_json_and_trace(tmp_path):
    """Test that logger writes JSON logs with trace_id for both main and background ops.

    The logger may produce double‑encoded JSON when structlog is used; this test
    extracts the actual message and ensures a ``trace_id`` field is present.
    """
    # Configure logger to write JSON logs to a temporary file
    log_file = tmp_path / "log.jsonl"
    config.set("logging.log_file", str(log_file))
    config.set("logging.format", "json")

    # Re‑initialize the global logger instance
    logger_instance.shutdown()
    logger_instance._setup_logging()

    # Log a message from the main thread
    log = logger_instance.get_logger()
    log.info("main started")

    # Log a message from a background task via ServiceManager
    svc = ServiceManager(max_workers=1)

    def op():
        from cubo.utils.logger import logger as inlogger
        from cubo.utils.logging_context import get_current_trace_id

        inlogger.info("background op")
        return get_current_trace_id()

    fut = svc.execute_async("document_processing", op, with_retry=False)
    trace = fut.result(timeout=5)
    assert trace  # ensure the operation returned a trace id

    # Give the logger a moment to flush
    time.sleep(0.05)

    # Read the logged lines
    with open(log_file, encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    assert lines, "Log file should contain at least one line"
    print(f"DEBUG: Captured lines: {lines}")

    # Verify both messages are present and contain a trace_id
    found_main = found_bg = False
    for line in lines:
        rec = json.loads(line)
        raw_msg = rec.get("message", "")
        try:
            if isinstance(raw_msg, str) and raw_msg.startswith("{"):
                inner = json.loads(raw_msg)
                msg_text = inner.get("event", "")
                if "trace_id" in inner:
                    rec["trace_id"] = inner["trace_id"]
            else:
                msg_text = raw_msg
        except Exception:
            msg_text = raw_msg

        if msg_text == "main started":
            found_main = True
            assert "trace_id" in rec, "Main log entry missing trace_id"
        if msg_text == "background op":
            found_bg = True
            assert "trace_id" in rec, "Background log entry missing trace_id"

    assert found_main and found_bg, "Both main and background log entries should be present"
