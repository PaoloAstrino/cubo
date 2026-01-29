"""Lightweight pub/sub wrapper with Redis fallback.

Provides publish_event(topic, payload) with a best-effort Redis pub/sub connection
and a no-op fallback for local dev and tests.
"""
from typing import Any, Dict
import json
import logging

logger = logging.getLogger(__name__)


def publish_event(topic: str, payload: Dict[str, Any]) -> None:
    """Publish an event to the given topic.

    Attempts to use Redis if available, otherwise logs the event.
    This function is intentionally best-effort and should not raise for failures.
    """
    try:
        import redis

        r = redis.Redis.from_url("redis://localhost:6379/0")
        r.publish(topic, json.dumps(payload))
        return
    except Exception as e:  # pragma: no cover - environment specific
        logger.debug(f"Redis publish failed ({e}), falling back to logger")

    # Fallback: log the event so test harnesses can inspect logs if needed
    logger.info(f"PUBSUB {topic} -> {json.dumps(payload, default=str)}")
