from pathlib import Path
from cubo.config import config
from cubo.utils.logger import logger_instance
from cubo.main import CUBOApp

log_file = Path('./tmp_debug_log_scrub.jsonl')
config.set('logging.log_file', str(log_file))
config.set('logging.format', 'json')
config.set('logging.scrub_queries', True)

logger_instance.shutdown(); logger_instance._setup_logging()

app = CUBOApp()
app._display_command_line_results('this is my secret query', ['doc1'], 'ok')

print('exists', log_file.exists())
if log_file.exists():
    with open(log_file, 'r', encoding='utf-8') as f:
        for l in f:
            print('LINE:', l.strip())
