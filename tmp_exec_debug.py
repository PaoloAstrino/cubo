from src.cubo.utils.logging_context import trace_context, get_current_trace_id
from src.cubo.workers.thread_manager import ThreadManager
from src.cubo.services.service_manager import ServiceManager
from src.cubo.utils.error_recovery import ErrorRecoveryManager
import threading

print('\n=== Direct ThreadManager test ===')

def tm_op():
    print('tm_op in thread - trace:', get_current_trace_id())

# Submit a task via ThreadManager
from src.cubo.services.service_manager import ServiceManager
svc_tm = ThreadManager(max_workers=2)

def wrapped_tm():
    with trace_context('tm-123'):
        print('wrapped_tm context:', get_current_trace_id())
        # Directly run within same thread
        tm_op()

def wrapped_tm_inner():
    with trace_context('tm-456'):
        print('wrapped_tm_inner context:', get_current_trace_id())
        # Spawn new inner thread with copy_context inside ThreadManager
        import contextvars
        ctx = contextvars.copy_context()
        def inner():
            ctx.run(tm_op)
        t = threading.Thread(target=inner)
        t.start(); t.join()

from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=2) as ex:
    ex.submit(wrapped_tm).result()
    ex.submit(wrapped_tm_inner).result()

print('\n=== ServiceManager test ===')
# ServiceManager version of the problem
svc = ServiceManager(max_workers=2)

def op(filepath):
    print('op start - trace:', get_current_trace_id())
    return True

fut = svc.execute_async('document_processing', op, 'dummy.txt', with_retry=False)
print('fut result', fut.result(timeout=5))

print('done')
