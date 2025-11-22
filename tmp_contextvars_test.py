import contextvars
import threading
from cubo.utils.logging_context import trace_context, get_current_trace_id


def worker():
    print('IN WORKER - trace id:', get_current_trace_id())

# Spawn thread with trace_context in main thread
with trace_context('abc123'):
    print('MAIN thread trace:', get_current_trace_id())
    # Context var not propagated to new thread by default
    t = threading.Thread(target=worker)
    t.start()
    t.join()

# Now propagate context via copy_context
with trace_context('xyz789'):
    ctx = contextvars.copy_context()
    def run_with_ctx():
        ctx.run(worker)
    t2 = threading.Thread(target=run_with_ctx)
    t2.start()
    t2.join()
