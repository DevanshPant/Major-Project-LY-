"""FastAPI entry point.

- Preloads ML models on startup (so the first /api/match doesn't hang).
- Session-cookie admin auth.
- Login page + protected /admin route.
"""
from __future__ import annotations

import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from .api.routes import router as api_router
from . import auth

APP_DIR = Path(__file__).resolve().parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up the models so the first request doesn't hang for 60s."""
    print("[startup] preloading MiniLM bi-encoder …")
    try:
        from .core import embedder
        embedder.encode(["warm-up"])  # forces model load
        print("[startup]   bi-encoder ready")
    except Exception as e:
        print(f"[startup]   bi-encoder FAILED: {e}")

    print("[startup] preloading cross-encoder reranker …")
    try:
        from .core import reranker
        reranker.score("warm-up", ["warm-up"])
        print(f"[startup]   reranker ready ({'finetuned' if reranker.is_finetuned() else 'pretrained'})")
    except Exception as e:
        print(f"[startup]   reranker FAILED: {e}")

    print("[startup] models ready — server accepting requests")
    yield
    print("[shutdown] bye")


app = FastAPI(title="ResumeAI — Semantic Resume Screening", lifespan=lifespan)

app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("RESUMEAI_SECRET", secrets.token_hex(32)),
    session_cookie="resumeai_session",
    max_age=auth.SESSION_TTL_SECONDS,
    same_site="lax",
    https_only=False,
)

app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")
templates = Jinja2Templates(directory=APP_DIR / "templates")
app.include_router(api_router)


# ---------- public pages ----------
@app.get("/", response_class=HTMLResponse)
async def landing(request: Request):
    return templates.TemplateResponse(request, "landing.html",
                                      {"is_admin": auth.is_authed(request)})


@app.get("/compare", response_class=HTMLResponse)
async def compare_page(request: Request):
    return templates.TemplateResponse(request, "compare.html",
                                      {"is_admin": auth.is_authed(request)})


# ---------- admin auth ----------
@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, error: str | None = None):
    if auth.is_authed(request):
        return RedirectResponse("/admin", status_code=303)
    return templates.TemplateResponse(request, "login.html", {
        "error": error,
        "default_creds": auth.IS_DEFAULT_PASS,
    })


@app.post("/login")
async def login_submit(request: Request,
                       username: str = Form(...),
                       password: str = Form(...)):
    ip = request.client.host if request.client else "?"
    auth.rate_limit(ip)
    if not auth.check_credentials(username, password):
        return RedirectResponse("/login?error=1", status_code=303)
    auth.login_session(request, username)
    return RedirectResponse("/admin", status_code=303)


@app.post("/logout")
async def logout(request: Request):
    auth.logout_session(request)
    return RedirectResponse("/", status_code=303)


@app.get("/admin", response_class=HTMLResponse)
async def admin(request: Request):
    if not auth.is_authed(request):
        return RedirectResponse("/login", status_code=303)
    return templates.TemplateResponse(request, "admin.html",
                                      {"is_admin": True})
