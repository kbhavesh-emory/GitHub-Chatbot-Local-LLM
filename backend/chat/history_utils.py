# backend/chat/history_utils.py
# Lightweight, robust conversation history utilities.
# - JSONL storage (append-only) with rotation
# - In-memory ring buffer per chat_id
# - Safe read/write with file locks (fcntl on POSIX; noop on Windows)

from __future__ import annotations
import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Optional, Iterable

HIST_DIR = Path(os.getenv("HISTORY_DIR", "backend/history"))
HIST_DIR.mkdir(parents=True, exist_ok=True)

MAX_TURNS = int(os.getenv("HISTORY_MAX_TURNS", "20"))
ROTATE_BYTES = int(os.getenv("HISTORY_ROTATE_BYTES", "8_000_000"))  # ~8MB
FLUSH_EVERY = int(os.getenv("HISTORY_FLUSH_EVERY", "1"))  # flush frequency (lines)

try:
    import fcntl  # type: ignore
    _HAVE_FCNTL = True
except Exception:
    _HAVE_FCNTL = False

@dataclass
class ChatTurn:
    role: str  # "user" | "assistant" | "system"
    content: str
    ts: float

# ---------------- File lock helpers ----------------
class _Lock:
    def __init__(self, path: Path):
        self.path = path
        self._fh = None

    def __enter__(self):
        self._fh = self.path.open("a+")
        if _HAVE_FCNTL:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX)
        return self._fh

    def __exit__(self, exc_type, exc, tb):
        try:
            if _HAVE_FCNTL and self._fh:
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        finally:
            if self._fh:
                self._fh.close()
            self._fh = None

# ---------------- Public API ----------------
def hist_path(chat_id: str) -> Path:
    safe = "".join(c for c in chat_id if c.isalnum() or c in ("-", "_")).strip() or "default"
    return HIST_DIR / f"{safe}.jsonl"

def append_turn(chat_id: str, role: str, content: str) -> None:
    """Append one turn to JSONL and rotate if large."""
    path = hist_path(chat_id)
    rec = ChatTurn(role=role, content=content, ts=time.time())
    line = json.dumps(asdict(rec), ensure_ascii=False)
    with _Lock(path):
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        if path.stat().st_size >= ROTATE_BYTES:
            rotate(path)

def rotate(path: Path) -> None:
    idx = 1
    while True:
        p = path.with_suffix(f".jsonl.{idx}")
        if not p.exists():
            path.rename(p)
            break
        idx += 1

def read_history(chat_id: str, max_turns: int = MAX_TURNS) -> List[Dict[str, str]]:
    """Return last max_turns as [{'role':..., 'content':...}]"""
    path = hist_path(chat_id)
    if not path.exists():
        return []
    lines = []
    with _Lock(path):
        with path.open("r", encoding="utf-8") as f:
            for ln in f:
                try:
                    obj = json.loads(ln)
                    lines.append({"role": obj.get("role", "user"), "content": obj.get("content", "")})
                except Exception:
                    continue
    return lines[-max_turns:]

def clear_history(chat_id: str) -> int:
    """Delete JSONL; return 1 if removed, 0 if absent."""
    path = hist_path(chat_id)
    if path.exists():
        path.unlink()
        return 1
    return 0
