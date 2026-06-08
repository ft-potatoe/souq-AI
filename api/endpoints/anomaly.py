"""
api/endpoints/anomaly.py
GET /anomaly/{date}
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from ml import anomaly_scorer
from api._dates import resolve_date

log = logging.getLogger(__name__)

router = APIRouter()


@router.get("/anomaly/{date}")
async def get_anomaly(date: str) -> dict:
    try:
        resolved = resolve_date(date)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    try:
        return anomaly_scorer.score(resolved.strftime("%Y-%m-%d"), None)
    except Exception as exc:
        log.error("anomaly/%s failed: %s", date, exc)
        raise HTTPException(status_code=500, detail=str(exc))
