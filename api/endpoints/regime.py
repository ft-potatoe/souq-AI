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
from analytics import volatility_regime as vol_regime_mod
from analytics._loader import load_features
from api._dates import resolve_date

log = logging.getLogger(__name__)

router = APIRouter()


def _merge_vol(trend_result: dict, date_str: str) -> dict:
    """Fetch vol regime and merge its fields into *trend_result* in-place."""
    try:
        vol = vol_regime_mod.run(date_str, {})
        trend_result["vol_regime"] = vol.get("vol_regime")
        trend_result["vol_regime_probability"] = vol.get("vol_regime_probability")
        trend_result["vol_regime_sessions"] = vol.get("vol_regime_sessions")
        trend_result["vol_regime_start_date"] = vol.get("vol_regime_start_date")
        trend_result["prior_vol_regime"] = vol.get("prior_vol_regime")
        trend_result["prior_vol_regime_duration_sessions"] = vol.get("prior_vol_regime_duration_sessions")
        trend_result["volatility_20d_current"] = vol.get("volatility_20d_current")
        trend_result["volatility_20d_percentile"] = vol.get("volatility_20d_percentile")
        trend_result["volatility_60d_current"] = vol.get("volatility_60d_current")
        trend_result["vol_regime_distribution"] = vol.get("vol_regime_distribution")
        trend_result["vol_model_version"] = vol.get("model_version")
    except Exception as exc:
        log.warning("vol regime unavailable for %s: %s", date_str, exc)
        trend_result["vol_regime"] = None
    return trend_result


def _current_combined(date_str: str) -> dict:
    result = regime_mod.run(date_str, {})
    return _merge_vol(result, date_str)


@router.get("/regime/current")
async def get_regime_current() -> dict:
    try:
        data_date = resolve_date(None)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    try:
        return await asyncio.to_thread(_current_combined, data_date.strftime("%Y-%m-%d"))
    except Exception as exc:
        log.error("regime/current failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


def _build_regime_history() -> dict:
    """
    Decode both HMM state sequences in single passes and return per-session
    trend + vol regime labels.
    """
    df = load_features()
    if df.empty:
        raise ValueError("features_master is empty")

    _FEATURES = regime_mod._FEATURES
    _VOL_FEATURES = vol_regime_mod._FEATURES
    _MIN_SESSIONS = regime_mod._MIN_SESSIONS

    # Trend HMM: use rows clean on trend features
    trend_clean = df[_FEATURES + ["date"]].dropna().reset_index(drop=True)
    n = len(trend_clean)

    if n < _MIN_SESSIONS:
        history = [
            {
                "date": str(trend_clean["date"].iloc[i].date()),
                "regime": None,
                "regime_probability": None,
                "vol_regime": None,
                "vol_regime_probability": None,
            }
            for i in range(n)
        ]
        return {
            "history": history,
            "total_sessions": n,
            "note": f"Regime labels require at least {_MIN_SESSIONS} sessions; only {n} available.",
        }

    trend_seq, trend_post, _, trend_model = regime_mod._decode(trend_clean)
    trend_label_map = regime_mod._assign_state_labels(trend_model, _FEATURES)

    # Vol HMM: decode separately using its own clean set, then align by date
    vol_by_date: dict[str, tuple[str, float]] = {}
    try:
        vol_all_features = _VOL_FEATURES + ["date"]
        vol_clean = df[vol_all_features].dropna().reset_index(drop=True)
        if len(vol_clean) >= _MIN_SESSIONS:
            vol_seq, vol_post, _, vol_model = vol_regime_mod._decode(vol_clean)
            vol_label_map = vol_regime_mod._assign_state_labels(vol_model)
            for i in range(len(vol_seq)):
                d = str(vol_clean["date"].iloc[i].date())
                state = int(vol_seq[i])
                vol_by_date[d] = (vol_label_map[state], round(float(vol_post[i, state]), 4))
    except Exception as exc:
        log.warning("vol HMM history decode failed (non-fatal): %s", exc)

    history = []
    for i in range(n):
        date_str = str(trend_clean["date"].iloc[i].date())
        t_state = int(trend_seq[i])
        vol_entry = vol_by_date.get(date_str)
        history.append({
            "date": date_str,
            "regime": trend_label_map[t_state],
            "regime_probability": round(float(trend_post[i, t_state]), 4),
            "vol_regime": vol_entry[0] if vol_entry else None,
            "vol_regime_probability": vol_entry[1] if vol_entry else None,
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
