import json
import time
from pathlib import Path
import pytest
from src.cubo.services.service_manager import ServiceManager
from src.cubo.config import config
from src.cubo.utils.logger import logger_instance


def test_trace_id_propagation(tmp_path):
    # configure JSON log file
    log_file = tmp_path / 'trace_log.jsonl'
    config.set('logging.log_file', str(log_file))
    config.set('logging.format', 'json')
    logger_instance.shutdown()
    logger_instance._setup_logging()

    svc = ServiceManager(max_workers=2)

    def op(filepath):
        # simple operation logs something
        from src.cubo.utils.logger import logger
        logger.info('op started', extra={'file': filepath})
        return True

    fut = svc.execute_async('document_processing', op, 'dummy.txt', with_retry=False)
    res = fut.result(timeout=5)
    time.sleep(0.1)
    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    assert lines
    found = False
    for l in lines:
        try:
            rec = json.loads(l)
            if rec.get('message') == 'op started' or rec.get('msg') == 'op started':
                # verify trace_id exists
                assert 'trace_id' in rec or 'trace_id' in rec.get('context', {})
                found = True
                break
        except Exception:
            continue
    assert found
