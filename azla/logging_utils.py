import logging
import os
import sys
import json
from typing import Any, Dict, Optional

from .utils import json_dumps, timestamp_ms


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = {
            "ts": timestamp_ms(),
            "level": record.levelname,
            "name": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            base["exc_info"] = self.formatException(record.exc_info)
        if hasattr(record, "extra") and isinstance(record.extra, dict):
            base.update(record.extra)
        reserved = {
            "name",
            "msg",
            "args",
            "levelname",
            "levelno",
            "pathname",
            "filename",
            "module",
            "exc_info",
            "exc_text",
            "stack_info",
            "lineno",
            "funcName",
            "created",
            "msecs",
            "relativeCreated",
            "thread",
            "threadName",
            "processName",
            "process",
            "message",
        }
        for k, v in record.__dict__.items():
            if k in reserved or k in base or k.startswith("_"):
                continue
            try:
                json.dumps(v)
                base[k] = v
            except Exception:
                base[k] = str(v)
        return json_dumps(base)


def configure_logging(level: str = "INFO") -> None:
    lvl = getattr(logging, level.upper(), logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(lvl)
    for h in list(logger.handlers):
        logger.removeHandler(h)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
