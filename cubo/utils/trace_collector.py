"""In-memory trace collector to aggregate events keyed by trace_id.

This is a low-overhead local-only collector for debugging developer runs.
It keeps a bounded number of events per trace and optionally supports TTL.

The trace event schema is a simple dict:
{ "timestamp": <iso>, "component": str, "event": str, "details": dict }
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional

from cubo.config import config


@dataclass
class TraceEvent:
    timestamp: float
    component: str
    event: str
    details: Dict[str, Any]


class TraceCollector:
    def __init__(self, max_events_per_trace: int = 200, keep_seconds: int = 600):
        self._traces: Dict[str, Deque[TraceEvent]] = {}
        self._locks: Dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()
        self._max_events = max_events_per_trace
        self._keep_seconds = keep_seconds

    def record(
        self, trace_id: str, component: str, event: str, details: Optional[Dict[str, Any]] = None
    ) -> None:
        if not trace_id:
            return
        details = details or {}
        ev = TraceEvent(time.time(), component, event, details)

        with self._global_lock:
            if trace_id not in self._traces:
                self._traces[trace_id] = deque(maxlen=self._max_events)
                self._locks[trace_id] = threading.Lock()
            # append in trace-specific lock
            deque_ref = self._traces[trace_id]

        with self._locks[trace_id]:
            deque_ref.append(ev)

    def get_trace(self, trace_id: str) -> Optional[List[Dict[str, Any]]]:
        with self._global_lock:
            dq = self._traces.get(trace_id)
            if dq is None:
                return None
            # copy to avoid concurrency issues
            with self._locks[trace_id]:
                now = time.time()
                events = [
                    {
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(e.timestamp)),
                        "component": e.component,
                        "event": e.event,
                        "details": e.details,
                    }
                    for e in list(dq)
                    if (now - e.timestamp) <= self._keep_seconds
                ]
                return events

    def clear_trace(self, trace_id: str) -> None:
        with self._global_lock:
            if trace_id in self._traces:
                with self._locks[trace_id]:
                    self._traces[trace_id].clear()
                del self._traces[trace_id]
                del self._locks[trace_id]


# Singleton
_trace_collector_config = config.get("trace_collector", {})
_max_events = int(_trace_collector_config.get("max_events_per_trace", 200))
_keep_seconds = int(_trace_collector_config.get("keep_seconds", 600))
trace_collector = TraceCollector(max_events_per_trace=_max_events, keep_seconds=_keep_seconds)
