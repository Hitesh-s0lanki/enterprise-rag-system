"""Application-owned JSON logging with rotating files and credential redaction."""

import json
import logging
import os
import re
import threading
import traceback
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOCK = threading.RLock()
_CONFIGURATION = None
_LOGGER_NAME = "enterprise_rag"
_DEFAULT_FILE = Path(__file__).resolve().parents[2] / ".local/logs/app.jsonl"
_FIELDS = (
    "run_id", "agent", "stage", "model", "tool", "source_run_id", "sequence", "error_code",
    "duration_ms", "model_calls", "tool_calls", "input_tokens", "output_tokens", "total_tokens",
    "cache_source", "cache_status", "cache_write_status", "pipeline_runs", "equivalence_checks",
    "memory_search_latency_ms", "memory_fetch_latency_ms", "memory_write_latency_ms",
    "equivalence_latency_ms", "pipeline_latency_ms", "total_latency_ms",
)


def redact(text: str) -> str:
    for key, value in sorted(os.environ.items(), key=lambda item: len(item[1]), reverse=True):
        if len(value) >= 6 and any(word in key.upper() for word in ("API_KEY", "TOKEN", "SECRET", "PASSWORD")):
            text = text.replace(value, "[REDACTED]")
    text = re.sub(r"\bsk-[A-Za-z0-9_-]+", "[REDACTED]", text)
    text = re.sub(r"(?i)(bearer\s+)[^\s\"',;]+", r"\1[REDACTED]", text)
    return re.sub(
        r'''(?i)((?:api[_-]?key|access[_-]?token|refresh[_-]?token|password|secret|authorization)["']?\s*[:=]\s*)("[^"]*"|'[^']*'|[^\s,;}]+)''',
        r"\1[REDACTED]", text,
    )


def _exception(exc: BaseException, seen: set[int] | None = None) -> dict:
    seen = set() if seen is None else seen
    seen.add(id(exc))
    result = {"type": type(exc).__name__, "message": redact(str(exc))[:2000],
              "frames": [{"file": frame.filename, "line": frame.lineno, "function": frame.name}
                         for frame in traceback.extract_tb(exc.__traceback__)]}
    cause = exc.__cause__ or (exc.__context__ if not exc.__suppress_context__ else None)
    if cause is not None and id(cause) not in seen and len(seen) < 5:
        result["cause"] = _exception(cause, seen)
    if isinstance(exc, BaseExceptionGroup):
        result["exceptions"] = [_exception(item, set(seen)) for item in exc.exceptions[:5] if id(item) not in seen]
    return result


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {"timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
                   "level": record.levelname, "logger": record.name, "event": redact(record.getMessage())}
        for field in _FIELDS:
            value = getattr(record, field, None)
            if isinstance(value, (str, int, float, bool)):
                payload[field] = redact(value) if isinstance(value, str) else value
        if record.exc_info and record.exc_info[1] is not None:
            payload["exception"] = _exception(record.exc_info[1])
        return json.dumps(payload, ensure_ascii=False, allow_nan=False)


def configure_logging() -> None:
    """Idempotent across Streamlit reruns; do not alter third-party root logging."""
    global _CONFIGURATION
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        level = "INFO"
    filename = os.getenv("LOG_FILE", str(_DEFAULT_FILE)).strip()
    max_bytes = int(os.getenv("LOG_MAX_BYTES", "5242880"))
    backups = int(os.getenv("LOG_BACKUP_COUNT", "3"))
    if max_bytes <= 0 or backups < 0:
        raise ValueError("Log rotation size must be positive and backup count non-negative")
    configuration = (level, filename, max_bytes, backups)
    with _LOCK:
        logger = logging.getLogger(_LOGGER_NAME)
        if configuration == _CONFIGURATION and logger.handlers:
            return
        handlers: list[logging.Handler] = [logging.StreamHandler()]
        file_error = False
        if filename:
            try:
                path = Path(filename).expanduser()
                path.parent.mkdir(parents=True, exist_ok=True)
                handlers.append(RotatingFileHandler(path, maxBytes=max_bytes, backupCount=backups, encoding="utf-8"))
            except OSError:
                file_error = True
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()
        for handler in handlers:
            handler.setFormatter(JsonLogFormatter())
            logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
        _CONFIGURATION = configuration
        if file_error:
            logger.warning("logging.file_unavailable_using_stderr")


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"{_LOGGER_NAME}.{name}")


def current_run_id() -> str | None:
    """LangChain propagates this metadata into synchronous tool worker threads."""
    from langchain_core.runnables.config import ensure_config
    return ensure_config().get("metadata", {}).get("agent_run_id")
