"""Session-cookie admin auth.

Deliberately simple for a final-year project: one admin account whose
credentials come from environment variables (or a local .env-style
dev default). The password is stored as a PBKDF2-SHA256 hash, checked
with constant-time compare, and the login endpoint is rate-limited.

Environment (recommended for real use):
    RESUMEAI_ADMIN_USER=admin
    RESUMEAI_ADMIN_PASS=<your password>
    RESUMEAI_SECRET=<32+ random chars>

If not set, defaults are  admin / resume-ai-2026  (change these before
demo day; the login page prints a note when defaults are in use).
"""
from __future__ import annotations

import hashlib
import hmac
import os
import time
from typing import Optional

from fastapi import HTTPException, Request

ADMIN_USER = os.getenv("RESUMEAI_ADMIN_USER", "admin")
ADMIN_PASS = os.getenv("RESUMEAI_ADMIN_PASS", "resume-ai-2026")
IS_DEFAULT_PASS = ADMIN_PASS == "resume-ai-2026"

# 24-hour session
SESSION_TTL_SECONDS = 24 * 3600

_login_attempts: dict[str, list[float]] = {}
_MAX_ATTEMPTS = 6
_WINDOW = 300  # 5 min


def _hash_pw(pw: str) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", pw.encode(), b"resume-ai-salt", 120_000)


_ADMIN_HASH = _hash_pw(ADMIN_PASS)


def check_credentials(user: str, pw: str) -> bool:
    if user != ADMIN_USER:
        return False
    return hmac.compare_digest(_hash_pw(pw), _ADMIN_HASH)


def rate_limit(ip: str) -> None:
    now = time.time()
    hits = [t for t in _login_attempts.get(ip, []) if now - t < _WINDOW]
    if len(hits) >= _MAX_ATTEMPTS:
        raise HTTPException(429, "Too many login attempts. Try again in a few minutes.")
    hits.append(now)
    _login_attempts[ip] = hits


def is_authed(request: Request) -> bool:
    s = request.session
    if not s.get("admin"):
        return False
    ts = s.get("ts", 0)
    if time.time() - ts > SESSION_TTL_SECONDS:
        s.clear()
        return False
    return True


def require_admin(request: Request) -> None:
    if not is_authed(request):
        # Signal 401 so the front-end can redirect to /login
        raise HTTPException(401, "Login required")


def login_session(request: Request, user: str) -> None:
    request.session["admin"] = user
    request.session["ts"] = int(time.time())


def logout_session(request: Request) -> None:
    request.session.clear()
