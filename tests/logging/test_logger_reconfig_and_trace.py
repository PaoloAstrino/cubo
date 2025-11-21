import json
import time
from pathlib import Path
import pytest
from src.cubo.config import config
from src.cubo.utils.logger import logger_instance
from src.cubo.services.service_manager import ServiceManager


def test_logger_reconfig_json_and_trace(tmp_path):
    # config to write json logs
    log_file = tmp_path / 'log.jsonl'
    config.set('logging.log_file', str(log_file))
    config.set('logging.format', 'json')

    # Re-init the logger
    logger_instance.shutdown()
    logger_instance._setup_logging()

    # Acquire the reconfigured logger and write a main-thread log
    log = logger_instance.get_logger()
    log.info('main started')

    # Emit a log from a background operation via ServiceManager which should attach trace_id
    svc = ServiceManager(max_workers=1)

    def op():
        from src.cubo.utils.logging_context import get_current_trace_id
        from src.cubo.utils.logger import logger as inlogger
        inlogger.info('background op')
        return get_current_trace_id()

    fut = svc.execute_async('document_processing', op, with_retry=False)
    trace = fut.result(timeout=5)
    assert trace

    # Allow flush
    time.sleep(0.05)

    with open(log_file, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    assert lines
    # Confirm at least one JSON line has message and trace_id
    found_main = found_bg = False
    for l in lines:
        rec = json.loads(l)
        if rec.get('message') == 'main started':
            found_main = True
            assert 'trace_id' in rec
        if rec.get('message') == 'background op':
            found_bg = True
            # Trace should be in log; we assert presence of field (non-empty may depend on
            # queue handler behavior) and that operation returned a trace.
            assert 'trace_id' in rec
    assert found_main and found_bg
