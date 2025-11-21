from src.cubo.utils.error_recovery import ErrorRecoveryManager
from src.cubo.utils.logging_context import trace_context, get_current_trace_id

mgr = ErrorRecoveryManager()

def op():
    print('IN OP - Trace id:', get_current_trace_id())
    return True

# call from main thread with trace_context
with trace_context('abc-123'):
    mgr.execute_with_recovery('document_processing', op)

# call from a thread with trace_context
import threading

def thread_func():
    with trace_context('thread-987'):
        mgr.execute_with_recovery('document_processing', op)

th = threading.Thread(target=thread_func)
th.start()
th.join()

print('done')
