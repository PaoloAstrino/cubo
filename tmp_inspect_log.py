from pathlib import Path
from src.cubo.config import config
from src.cubo.utils.logger import logger_instance
from src.cubo.services.service_manager import ServiceManager
import time

log_file = Path(r'C:/Users/paolo/AppData/Local/Temp/pytest-of-paolo/pytest-197/test_trace_id_propagation0/trace_log.jsonl')
config.set('logging.log_file', str(log_file))
config.set('logging.format', 'json')
logger_instance.shutdown()
logger_instance._setup_logging()

svc = ServiceManager(max_workers=2)

def op(filepath):
    from src.cubo.utils.logger import logger
    logger.info('op started', extra={'file': filepath})
    return True

fut = svc.execute_async('document_processing', op, 'dummy.txt', with_retry=False)
res = fut.result(timeout=5)

# Wait a bit for listener
import time

# slightly longer wait
for _ in range(3):
    time.sleep(0.1)

print('Log file exists:', log_file.exists())
if log_file.exists():
    with open(log_file, 'r', encoding='utf-8') as f:
        for l in f:
            print('LINE:', l.strip())
