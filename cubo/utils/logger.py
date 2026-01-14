import logging
import os
import sys
import time
from logging.handlers import (
    QueueHandler,
    QueueListener,
    RotatingFileHandler,
    TimedRotatingFileHandler,
)
from queue import Queue
from typing import Any, Optional

from cubo.config import config
from cubo.utils.logging_context import get_current_trace_id

try:
    import structlog

    STRUCTLOG_AVAILABLE = True
except Exception:
    STRUCTLOG_AVAILABLE = False

try:
    try:
        from pythonjsonlogger import json as jsonlogger
    except ImportError:
        from pythonjsonlogger import jsonlogger
    JSONLOGGER_AVAILABLE = True
except Exception:
    JSONLOGGER_AVAILABLE = False


class WindowsSafeRotatingFileHandler(RotatingFileHandler):
    """Windows-safe rotating file handler that handles file locking issues."""

    def doRollover(self):
        """Override to handle Windows file locking gracefully."""
        if self.stream:
            self.stream.close()
            self.stream = None

        # Try rotation with retry logic
        import time

        for attempt in range(3):
            try:
                super().doRollover()
                break
            except (OSError, PermissionError):
                if attempt < 2:
                    time.sleep(0.1)
                else:
                    # If rotation fails, just continue writing to current file
                    pass

        # Reopen stream
        if not self.stream:
            self.stream = self._open()


class WindowsSafeTimedRotatingFileHandler(TimedRotatingFileHandler):
    """Windows-safe timed rotating file handler that handles file locking issues."""

    def doRollover(self):
        """Override to handle Windows file locking gracefully."""
        if self.stream:
            self.stream.close()
            self.stream = None

        # Try rotation with retry logic
        import time

        for attempt in range(3):
            try:
                super().doRollover()
                break
            except (OSError, PermissionError):
                if attempt < 2:
                    time.sleep(0.1)
                else:
                    # If rotation fails, just continue writing to current file
                    pass

        # Reopen stream
        if not self.stream:
            self.stream = self._open()


def _ensure_log_dir(path: str):
    log_dir = os.path.dirname(path) or "."
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)


def _cfg_logging_or(key: str, default: Any = None) -> Any:
    """Try reading 'logging.<key>', fallback to common top-level keys like <key> and log_<key>.

    This helps compatibility with both the old and new config formats.
    """
    v = config.get(f"logging.{key}")
    if v is not None:
        return v
    # Common alternate top-level names
    candidates = [key, f"log_{key}", f"{key}_file"]
    for c in candidates:
        v = config.get(c)
        if v is not None:
            return v
    return default


class Logger:
    """Structured logger wrapper that provides backward compatibility and trace-id injection.

    This class configures either structlog (preferred) or stdlib logging (fallback) and ensures
    logs are written in JSON format where possible, with file rotation and optional queueing.
    """

    def __init__(self):
        self.logger = None
        self._queue: Optional[Queue] = None
        self._listener: Optional[QueueListener] = None
        self._setup_logging()

    def _get_formatter(self):
        fmt = _cfg_logging_or("format", "json")
        if fmt == "json" and JSONLOGGER_AVAILABLE:
            # Include a 'trace_id' field if present
            fmt_str = "%(asctime)s %(levelname)s %(name)s %(message)s %(trace_id)s"
            return jsonlogger.JsonFormatter(fmt_str)
        if fmt == "json" and not JSONLOGGER_AVAILABLE:
            # Fallback JSON formatter if python-json-logger is not available. This
            # emits minimal JSON objects with fields used by tests: level, message,
            # name, asctime and trace_id.
            class SimpleJsonFormatter(logging.Formatter):
                def format(self, record):
                    try:
                        trace_id = getattr(record, "trace_id", "")
                    except Exception:
                        trace_id = ""
                    payload = {
                        "asctime": self.formatTime(record, datefmt=None),
                        "level": record.levelname,
                        "name": record.name,
                        "message": record.getMessage(),
                        "trace_id": trace_id or "",
                    }
                    try:
                        import json as _json

                        return _json.dumps(payload, ensure_ascii=False)
                    except Exception:
                        return str(payload)

            return SimpleJsonFormatter()
        # Default human readable format
        return logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")

    def _setup_handlers(self):
        path = str(_cfg_logging_or("log_file", "./logs/cubo_log.jsonl"))
        _ensure_log_dir(path)
        # Ensure the log file exists so tests and external tools can open it immediately
        try:
            with open(path, "a", encoding="utf-8"):
                pass
        except Exception:
            pass

        rotate_method = _cfg_logging_or("rotate_method", "midnight")
        _retention_days = int(_cfg_logging_or("retention_days", 30))
        _compress_rotated = bool(_cfg_logging_or("compress_rotated", True))

        # Use Windows-safe rotation handlers
        if rotate_method == "size":
            size = int(_cfg_logging_or("rotate_size", 100 * 1024 * 1024))
            if sys.platform == "win32":
                # Use Windows-safe rotating handler with delay to avoid file locking
                handler = WindowsSafeRotatingFileHandler(
                    path, maxBytes=size, backupCount=10, encoding="utf-8", delay=True
                )
            else:
                handler = RotatingFileHandler(path, maxBytes=size, backupCount=10, encoding="utf-8")
        elif rotate_method == "time" or rotate_method == "midnight":
            when = _cfg_logging_or("rotate_when", "midnight")
            if sys.platform == "win32":
                # Use Windows-safe timed rotating handler with delay to avoid file locking
                handler = WindowsSafeTimedRotatingFileHandler(
                    path, when=when, backupCount=10, encoding="utf-8", delay=True
                )
            else:
                handler = TimedRotatingFileHandler(
                    path, when=when, backupCount=10, encoding="utf-8"
                )
        else:
            handler = logging.FileHandler(path, encoding="utf-8")

        handler.setFormatter(self._get_formatter())
        return handler

    def _configure_root_logger(self):
        """Configure root logger with level from config."""
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, _cfg_logging_or("log_level", "INFO")))
        return root_logger

    def _create_trace_filter(self):
        """Create TraceIDFilter class for attaching trace_id to log records."""
        class TraceIDFilter(logging.Filter):
            def filter(self, record):
                trace = get_current_trace_id()
                try:
                    record.trace_id = trace or ""
                except Exception:
                    pass
                return True
        return TraceIDFilter()

    def _setup_queue_handler(self, root_logger, handler, trace_filter):
        """Setup queue-based async logging with listener."""
        self._queue = Queue(-1)
        qhandler = QueueHandler(self._queue)
        try:
            qhandler.addFilter(trace_filter)
        except Exception:
            pass
        root_logger.addHandler(qhandler)
        self._listener = QueueListener(self._queue, handler, respect_handler_level=True)
        self._listener.start()
        # Give the listener a short moment to fully start
        try:
            time.sleep(0.01)
        except Exception:
            pass

    def _setup_direct_handler(self, root_logger, handler):
        """Setup direct file handler (non-queued logging)."""
        root_logger.addHandler(handler)

    def _configure_structlog(self, trace_filter):
        """Configure structlog if available with trace_id injection."""
        if not STRUCTLOG_AVAILABLE:
            self.logger = logging.getLogger("cubo")
            return
        
        # Add processor to attach trace_id from ContextVar to every event dict
        def _structlog_add_trace_id(logger, method_name, event_dict):
            try:
                trace = get_current_trace_id()
                if trace:
                    event_dict["trace_id"] = trace
            except Exception:
                pass
            return event_dict

        structlog.configure(
            processors=[
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.add_log_level,
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                _structlog_add_trace_id,
                (
                    structlog.processors.JSONRenderer()
                    if _cfg_logging_or("format", "json") == "json"
                    else structlog.dev.ConsoleRenderer()
                ),
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        self.logger = structlog.get_logger("cubo")
        trace = get_current_trace_id()
        if trace:
            self.logger = self.logger.bind(trace_id=trace)

    def _setup_logging(self):
        # Avoid re-configuring if already set up
        if self.logger and getattr(self.logger, "handlers", None):
            return

        root_logger = self._configure_root_logger()
        queue_enabled = bool(_cfg_logging_or("enable_queue", True))
        handler = self._setup_handlers()
        trace_filter = self._create_trace_filter()

        # Setup handler (queued or direct)
        if queue_enabled:
            self._setup_queue_handler(root_logger, handler, trace_filter)
        else:
            self._setup_direct_handler(root_logger, handler)

        # Add trace_id filter to root logger and handler
        root_logger.addFilter(trace_filter)
        try:
            handler.addFilter(trace_filter)
        except Exception:
            pass
        
        # Attach filter to logger instance if available
        try:
            if hasattr(self.logger, "logger"):
                try:
                    self.logger.logger.addFilter(trace_filter)
                except Exception:
                    pass
            else:
                try:
                    self.logger.addFilter(trace_filter)
                except Exception:
                    pass
        except Exception:
            pass

        # Configure structlog
        self._configure_structlog(trace_filter)
        
        # Export logger as module-level variable
        try:
            import sys
            module = sys.modules[__name__]
            module.logger = self.logger
        except Exception:
            pass

    def get_logger(self):
        return self.logger

    def shutdown(self):
        if self._listener:
            try:
                self._listener.stop()
            except Exception:
                pass


# Global logger instance
logger_instance = Logger()
logger = logger_instance.get_logger()
