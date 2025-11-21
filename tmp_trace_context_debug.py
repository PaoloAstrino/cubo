from src.cubo.config import config
from src.cubo.utils.logger import logger_instance
from src.cubo.services.service_manager import ServiceManager
from src.cubo.utils.logging_context import get_current_trace_id

config.set('logging.log_file', './tmp_trace_log.jsonl')
config.set('logging.format', 'json')
logger_instance.shutdown(); logger_instance._setup_logging()
svc = ServiceManager(max_workers=2)

def op(filepath):
    from src.cubo.utils.logger import logger
    print('IN OP - current trace id in context:', get_current_trace_id())
    logger.info('op started', extra={'file': filepath})
    return True

fut = svc.execute_async('document_processing', op, 'dummy.txt', with_retry=False)
res = fut.result(timeout=5)
print('Done')
