"""Persistent OCR cache.

OCR is expensive (~15 s per scanned page on CPU) but perfectly
deterministic: the same PDF always produces the same text. So we hash
the file bytes and store the extracted text once.

After a one-time warm-up pass (scripts/warm_ocr_cache.py), scanned
resumes resolve in microseconds instead of seconds — which is what
makes a live demo over a scanned-PDF corpus possible at all.

Storage: a single SQLite file at models/ocr_cache.db
  key   = sha256 of the raw file bytes
  value = extracted text
"""
from __future__ import annotations

import hashlib
import sqlite3
import threading
from pathlib import Path

from ..config import MODELS_DIR

DB_PATH = MODELS_DIR / "ocr_cache.db"

_local = threading.local()


def _conn() -> sqlite3.Connection:
    """One connection per thread — sqlite3 objects aren't thread-safe."""
    c = getattr(_local, "conn", None)
    if c is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        c = sqlite3.connect(str(DB_PATH), timeout=30)
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA synchronous=NORMAL")
        c.execute(
            "CREATE TABLE IF NOT EXISTS ocr ("
            "  key TEXT PRIMARY KEY,"
            "  text TEXT NOT NULL,"
            "  chars INTEGER NOT NULL"
            ")"
        )
        c.commit()
        _local.conn = c
    return c


def key_for(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def get(data: bytes) -> str | None:
    """Return cached OCR text for these bytes, or None."""
    try:
        row = _conn().execute(
            "SELECT text FROM ocr WHERE key = ?", (key_for(data),)
        ).fetchone()
        return row[0] if row else None
    except Exception:
        return None


def put(data: bytes, text: str) -> None:
    try:
        c = _conn()
        c.execute(
            "INSERT OR REPLACE INTO ocr (key, text, chars) VALUES (?, ?, ?)",
            (key_for(data), text, len(text)),
        )
        c.commit()
    except Exception:
        pass


def stats() -> dict:
    try:
        row = _conn().execute(
            "SELECT COUNT(*), COALESCE(SUM(chars), 0) FROM ocr"
        ).fetchone()
        return {"entries": row[0], "total_chars": row[1],
                "db_mb": round(DB_PATH.stat().st_size / 1048576, 1) if DB_PATH.exists() else 0}
    except Exception:
        return {"entries": 0, "total_chars": 0, "db_mb": 0}


def is_warm() -> bool:
    return stats()["entries"] > 0
