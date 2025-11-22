from pathlib import Path
from cubo.config import config
from cubo.utils.logger import logger_instance, logger
from cubo.services.service_manager import ServiceManager
import time

log_file = Path(r'C:/Users/paolo/AppData/Local/Temp/pytest-of-paolo/pytest-206/test_logger_reconfig_json_and_0/log.jsonl')
config.set('logging.log_file', str(log_file))
config.set('logging.format', 'json')
logger_instance.shutdown(); logger_instance._setup_logging()

svc = ServiceManager(max_workers=1)

def op():
    from cubo.utils.logging_context import get_current_trace_id
    from cubo.utils.logger import logger as inlogger
    inlogger.info('background op')
    return get_current_trace_id()

fut = svc.execute_async('document_processing', op, with_retry=False)
trace = fut.result(timeout=5)
print('trace returned:', trace)

# Wait for log flush
for _ in range(3):
    time.sleep(0.05)

print('Log exists:', log_file.exists())
if log_file.exists():
    with open(log_file, 'r', encoding='utf-8') as f:
        for l in f:
            print('LINE:', l.strip())
