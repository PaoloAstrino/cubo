import json
import tempfile
import os
from pathlib import Path
import pytest
from src.cubo.utils.logger import logger_instance, logger
from src.cubo.config import config


def test_json_log_format(tmp_path):
    # Ensure we write to a test file
    log_file = tmp_path / 'test_log.jsonl'
    config.set('logging.log_file', str(log_file))
    config.set('logging.format', 'json')
    config.set('logging.enable_queue', False)
    # Re-init logger
    logger_instance.shutdown()
    logger_instance._setup_logging()
    log = logger.bind(component='test', trace_id='abc123') if hasattr(logger, 'bind') else logger
    # Write a log
    log.info('Test message', extra={'foo': 'bar'}) if isinstance(log, type(logger)) and not hasattr(log, 'bind') else log.info('Test message')
    # Ensure file exists and is valid JSON
    with open(log_file, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    assert lines and lines[0]
    line = lines[0]
    # If JSON lines produced, parse
    try:
        rec = json.loads(line)
        # Required keys
        assert 'level' in rec or 'levelname' in rec
        assert 'message' in rec or 'msg' in rec
    except Exception:
        # Not JSON — fail
        pytest.fail('Log entry was not valid JSON')
