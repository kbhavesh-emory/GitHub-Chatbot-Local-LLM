# backend/logging_config.py
# Opinionated logging config with JSON option and per-module levels.

import json
import logging
import os
from typing import Optional

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
JSON_LOGS = os.getenv("JSON_LOGS", "0") == "1"

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = {
            "lvl": record.levelname,
            "ts": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S"),
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            base["exc"] = self.formatException(record.exc_info)
        return json.dumps(base, ensure_ascii=False)

def configure_logging(level: Optional[str] = None, json_logs: Optional[bool] = None) -> None:
    lvl = getattr(logging, (level or LOG_LEVEL).upper(), logging.INFO)
    fmt = JsonFormatter() if (json_logs if json_logs is not None else JSON_LOGS) else logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    h = logging.StreamHandler()
    h.setFormatter(fmt)
    root.addHandler(h)
    root.setLevel(lvl)

    # Per-module knobs
    logging.getLogger("rag-chat").setLevel(lvl)
    logging.getLogger("chat-api").setLevel(lvl)
    logging.getLogger("uvicorn").setLevel(lvl)
    logging.getLogger("uvicorn.error").setLevel(lvl)
    logging.getLogger("uvicorn.access").setLevel(lvl)

# Auto-configure on import (can be overridden by explicit call)
configure_logging()
