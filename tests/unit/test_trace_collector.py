import time

from cubo.utils.trace_collector import TraceCollector


def test_trace_collector_basic():
    tc = TraceCollector(max_events_per_trace=5, keep_seconds=60)
    trace_id = "t1"
    for i in range(6):
        tc.record(trace_id, "test", f"event_{i}", {"i": i})
    events = tc.get_trace(trace_id)
    # max_events_per_trace=5 means only last 5 events should be kept
    assert len(events) == 5
    assert events[0]["event"] == "event_1"


def test_trace_collector_ttl():
    tc = TraceCollector(max_events_per_trace=10, keep_seconds=1)
    trace_id = "t2"
    tc.record(trace_id, "test", "one", {})
    time.sleep(2)
    events = tc.get_trace(trace_id)
    assert events == []
