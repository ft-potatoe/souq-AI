"""
api/endpoints/features.py
GET /features/today
GET /features/{date}
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from analytics._loader import load_features, row_for_date
from api._dates import resolve_date

log = logging.getLogger(__name__)

router = APIRouter()


def _row_to_dict(row) -> dict:
    result = {}
    for k, v in row.items():
        if hasattr(v, "item"):          # numpy scalar -> Python native
            v = v.item()
        elif hasattr(v, "isoformat"):   # Timestamp / date
            v = v.isoformat()
        result[k] = v
    return result


@router.get("/features/today")
async def get_features_today() -> dict:
    try:
        data_date = resolve_date(None)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    try:
        row = row_for_date(data_date.strftime("%Y-%m-%d"))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        log.error("features/today failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    return _row_to_dict(row)


@router.get("/features/{date}")
async def get_features_by_date(date: str) -> dict:
    try:
        resolved = resolve_date(date)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    try:
        row = row_for_date(resolved.strftime("%Y-%m-%d"))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        log.error("features/%s failed: %s", date, exc)
        raise HTTPException(status_code=500, detail=str(exc))

    return _row_to_dict(row)
