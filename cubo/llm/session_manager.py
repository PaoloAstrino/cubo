"""Session manager for LLM conversation contexts.

Provides in-memory session clearing and publishes invalidation events via pubsub.
Designed to be simple and safe: functions are idempotent and best-effort.
"""

from threading import Lock
from typing import Set
import logging

from cubo.utils.pubsub import publish_event

logger = logging.getLogger(__name__)


class SessionManager:
    def __init__(self):
        self._lock = Lock()
        # Track invalidated sessions for in-process checks
        self._invalidated_sessions: Set[str] = set()
        self._invalidated_collections: Set[str] = set()

    def clear_session(self, conversation_id: str) -> None:
        """Clear in-memory session context for a conversation and publish event."""
        if not conversation_id:
            return
        with self._lock:
            if conversation_id in self._invalidated_sessions:
                # already invalidated
                return
            self._invalidated_sessions.add(conversation_id)
        try:
            publish_event("conversation.invalidated", {"id": conversation_id})
            logger.info(f"Invalidated session {conversation_id}")
        except Exception as e:
            logger.warning(f"Failed to publish invalidation for {conversation_id}: {e}")

    def clear_sessions_by_collection(self, collection_id: str) -> None:
        """Invalidate all sessions that belong to a collection (best-effort).

        Note: Conversations are not currently mapped to collections in DB,
        so this is a best-effort publication that UIs and services can listen to.
        """
        if not collection_id:
            return
        with self._lock:
            if collection_id in self._invalidated_collections:
                return
            self._invalidated_collections.add(collection_id)
        try:
            publish_event("collection.invalidated", {"collection_id": collection_id})
            logger.info(f"Invalidated sessions for collection {collection_id}")
        except Exception as e:
            logger.warning(f"Failed to publish collection invalidation for {collection_id}: {e}")


# Expose a module level session manager
_manager: SessionManager | None = None


def get_session_manager() -> SessionManager:
    global _manager
    if _manager is None:
        _manager = SessionManager()
    return _manager
