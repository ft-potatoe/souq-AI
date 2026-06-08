"""
api/endpoints/regime.py
GET /regime/current
GET /regime/history
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, HTTPException

from analytics import regime as regime_mod
from analytics._loader import load_features
from api._dates import resolve_date

log = logging.getLogger(__name__)

router = APIRouter()


@router.get("/regime/current")
async def get_regime_current() -> dict:
    try:
        data_date = resolve_date(None)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    try:
        return await asyncio.to_thread(regime_mod.run, data_date.strftime("%Y-%m-%d"), {})
    except Exception as exc:
        log.error("regime/current failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


def _build_regime_history() -> dict:
    """
    Decode the full HMM state sequence in a single pass and return
    per-session regime labels.  Avoids calling regime.run() once per row
    (which would re-run model.predict() for an ever-growing prefix each time).
    """
    df = load_features()
    if df.empty:
        raise ValueError("features_master is empty")

    _FEATURES = regime_mod._FEATURES
    _MIN_SESSIONS = regime_mod._MIN_SESSIONS

    clean = df[_FEATURES + ["date"]].dropna().reset_index(drop=True)
    n = len(clean)

    if n < _MIN_SESSIONS:
        # Return what we have with null labels; mirrors regime.run() behaviour.
        history = [
            {"date": str(clean["date"].iloc[i].date()), "regime": None, "regime_probability": None}
            for i in range(n)
        ]
        return {"history": history, "total_sessions": n,
                "note": f"Regime labels require at least {_MIN_SESSIONS} sessions; only {n} available."}

    state_seq, log_post, _, model = regime_mod._decode(clean)
    label_map = regime_mod._assign_state_labels(model, _FEATURES)

    history = []
    for i in range(n):
        state = int(state_seq[i])
        history.append({
            "date": str(clean["date"].iloc[i].date()),
            "regime": label_map[state],
            "regime_probability": round(float(log_post[i, state]), 4),
        })

    return {"history": history, "total_sessions": n}


@router.get("/regime/history")
async def get_regime_history() -> dict:
    try:
        return await asyncio.to_thread(_build_regime_history)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        log.error("regime/history failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
