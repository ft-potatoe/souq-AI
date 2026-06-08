"""
api/endpoints/health.py
GET /health
"""

from __future__ import annotations

import logging

import httpx
from fastapi import APIRouter

from analytics._loader import load_features
from api._dates import model_versions_snapshot
from api.models import HealthResponse

log = logging.getLogger(__name__)

router = APIRouter()

_OLLAMA_URL = "http://localhost:11434/api/tags"


async def _check_ollama() -> bool:
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(_OLLAMA_URL)
            return r.status_code == 200
    except Exception:
        return False


@router.get("/health", response_model=HealthResponse)
async def get_health() -> HealthResponse:
    features_loaded = False
    features_rows = 0
    try:
        df = load_features()
        features_loaded = True
        features_rows = len(df)
    except Exception as exc:
        log.warning("Health check: features not loadable: %s", exc)

    ollama_ok = await _check_ollama()

    return HealthResponse(
        status="ok" if (features_loaded and ollama_ok) else "degraded",
        features_loaded=features_loaded,
        features_rows=features_rows,
        ollama_reachable=ollama_ok,
        model_versions=model_versions_snapshot(),
    )
