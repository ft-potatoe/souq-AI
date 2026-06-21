"""
api/endpoints/health.py
GET /health
"""

from __future__ import annotations

import logging
import os

import httpx
from fastapi import APIRouter

from analytics._loader import load_features
from api._dates import model_versions_snapshot
from api.models import HealthResponse

log = logging.getLogger(__name__)

router = APIRouter()

_OLLAMA_URL = "http://localhost:11434/api/tags"
_GROQ_MODELS_URL = "https://api.groq.com/openai/v1/models"


async def _check_llm() -> tuple[bool, str]:
    """Check whichever LLM backend is active. Returns (reachable, backend_name)."""
    groq_key = os.environ.get("GROQ_API_KEY")
    if groq_key:
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                r = await client.get(
                    _GROQ_MODELS_URL,
                    headers={"Authorization": f"Bearer {groq_key}"},
                )
                return r.status_code == 200, "groq"
        except Exception:
            return False, "groq"
    else:
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                r = await client.get(_OLLAMA_URL)
                return r.status_code == 200, "ollama"
        except Exception:
            return False, "ollama"


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

    llm_ok, llm_backend = await _check_llm()

    return HealthResponse(
        status="ok" if (features_loaded and llm_ok) else "degraded",
        features_loaded=features_loaded,
        features_rows=features_rows,
        ollama_reachable=llm_ok,
        llm_backend=llm_backend,
        model_versions=model_versions_snapshot(),
    )
