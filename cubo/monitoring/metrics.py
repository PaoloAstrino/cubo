from threading import Lock

_metrics = {}
_lock = Lock()


def record(metric: str, value: int = 1) -> None:
    """Increment a counter metric by value."""
    with _lock:
        _metrics[metric] = _metrics.get(metric, 0) + int(value)


def observe(metric: str, value: float) -> None:
    """Record an observed value (adds to a total)."""
    with _lock:
        _metrics[metric] = _metrics.get(metric, 0) + float(value)


def get_metrics() -> dict:
    """Return a snapshot of all metrics."""
    with _lock:
        return dict(_metrics)


def reset_metrics() -> None:
    """Reset all metrics (useful for tests)."""
    with _lock:
        _metrics.clear()
