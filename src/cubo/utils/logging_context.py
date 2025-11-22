import contextvars
import uuid
from contextlib import contextmanager

TRACE_ID_CTX = contextvars.ContextVar('trace_id', default=None)


def generate_trace_id() -> str:
    return str(uuid.uuid4())


def get_current_trace_id() -> str | None:
    return TRACE_ID_CTX.get()


@contextmanager
def trace_context(trace_id: str | None = None):
    old = TRACE_ID_CTX.set(trace_id or generate_trace_id())
    try:
        yield TRACE_ID_CTX.get()
    finally:
        TRACE_ID_CTX.reset(old)
