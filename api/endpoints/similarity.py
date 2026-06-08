"""
api/endpoints/similarity.py
GET /similarity/{date}
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from ml import similarity_ranker
from api._dates import resolve_date

log = logging.getLogger(__name__)

router = APIRouter()


@router.get("/similarity/{date}")
async def get_similarity(date: str) -> dict:
    try:
        resolved = resolve_date(date)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    try:
        return similarity_ranker.rank(resolved.strftime("%Y-%m-%d"), None)
    except Exception as exc:
        log.error("similarity/%s failed: %s", date, exc)
        raise HTTPException(status_code=500, detail=str(exc))
