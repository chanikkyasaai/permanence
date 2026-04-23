"""
PERMANENCE — FastAPI application for OpenEnv deployment.

Exposes the environment as an HTTP server with:
  POST /reset    → start new episode
  POST /step     → execute action
  GET  /state    → get current episode metadata
  GET  /health   → health check (required by HuggingFace)
  GET  /         → info

Deploy locally:
  uvicorn server.app:app --host 0.0.0.0 --port 7860

Deploy via OpenEnv CLI:
  openenv push --repo-id chane35/permanence
"""
from __future__ import annotations

import os
import threading
import time
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from models import (
    PermanenceObservation,
    PermanenceState,
    ResetRequest,
    StepRequest,
)
from server.permanence_server import PermanenceServer


# ─────────────────────────────────────────────────────────────────────────────
# Session management
# Session TTL: 30 minutes of inactivity → auto-evict
# ─────────────────────────────────────────────────────────────────────────────

SESSION_TTL_SECONDS = 1800
SUPPORTS_CONCURRENT_SESSIONS: bool = True

_sessions: Dict[str, Dict[str, Any]] = {}
_sessions_lock = threading.Lock()


def _get_or_create_session(session_id: str) -> PermanenceServer:
    with _sessions_lock:
        now = time.time()

        # Evict expired sessions
        expired = [
            sid for sid, s in _sessions.items()
            if now - s["last_active"] > SESSION_TTL_SECONDS
        ]
        for sid in expired:
            _sessions.pop(sid, None)

        if session_id not in _sessions:
            _sessions[session_id] = {
                "env": PermanenceServer(),
                "last_active": now,
            }
        else:
            _sessions[session_id]["last_active"] = now

        return _sessions[session_id]["env"]


def _get_session(session_id: str) -> PermanenceServer:
    with _sessions_lock:
        if session_id not in _sessions:
            raise HTTPException(
                status_code=404,
                detail=f"Session '{session_id}' not found. Call /reset first.",
            )
        _sessions[session_id]["last_active"] = time.time()
        return _sessions[session_id]["env"]


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="PERMANENCE Environment",
    description=(
        "OpenEnv-compatible environment that trains agents to predict "
        "action reversibility before acting. R-levels computed at runtime "
        "from world state — not static tags."
    ),
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Health (REQUIRED — HuggingFace pings this)
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> Dict[str, Any]:
    """Health check endpoint. Must return 200 for HuggingFace Space validation."""
    return {
        "status": "ok",
        "environment": "permanence",
        "version": "1.1.0",
        "supports_concurrent_sessions": SUPPORTS_CONCURRENT_SESSIONS,
        "active_sessions": len(_sessions),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Root info
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
async def root() -> Dict[str, Any]:
    return {
        "name": "PERMANENCE",
        "description": "RL environment for action reversibility training",
        "tasks": [
            "task_correction",
            "task_conflict",
            "task_launch",
            "task_crisis",
            "task_cascade",
        ],
        "endpoints": {
            "POST /reset": "Start new episode",
            "POST /step": "Execute agent action",
            "GET /state": "Get episode metadata",
            "GET /health": "Health check",
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Core OpenEnv endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/reset")
async def reset(
    request: ResetRequest = ResetRequest(),
    session_id: str = "default",
) -> PermanenceObservation:
    """
    Start a new episode.

    Empty body {} uses default task (task_correction, random seed).
    This is the behavior the OpenEnv validator expects.
    """
    env = _get_or_create_session(session_id)
    try:
        return env.reset(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step")
async def step(
    request: StepRequest,
    session_id: str = "default",
) -> PermanenceObservation:
    """
    Execute one agent action.

    The action.text field contains the agent's full response
    including <thinking>, <action>, and <reversibility> tags.
    """
    env = _get_session(session_id)
    try:
        return env.step(request.action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
async def state(session_id: str = "default") -> PermanenceState:
    """Return current episode metadata without advancing the episode."""
    env = _get_session(session_id)
    try:
        return env.state()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/session")
async def close_session(session_id: str = "default") -> Dict[str, str]:
    """Close and evict a session."""
    with _sessions_lock:
        if session_id in _sessions:
            _sessions[session_id]["env"].close()
            del _sessions[session_id]
    return {"status": "closed", "session_id": session_id}
