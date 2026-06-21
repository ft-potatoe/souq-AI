"""
analytics/summary.py
Deterministic historical summary statistics.

Computes all-time, 52-week, and YTD high/low records for price, volume,
value_traded, and return metrics, plus longest streaks and biggest
single-day moves.

run(date, params) -> dict
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from analytics._loader import history_up_to, row_for_date

# Metrics covered by this module
PRICE_COLS    = ["close", "open", "high", "low"]
VOLUME_COLS   = ["volume", "value_traded", "total_trades"]
RETURN_COLS   = ["return_1d", "return_5d", "return_20d"]
ALL_COLS      = PRICE_COLS + VOLUME_COLS + RETURN_COLS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe(val: Any) -> Any:
    """Convert NaN/inf to None for clean JSON output."""
    if val is None:
        return None
    try:
        if math.isnan(val) or math.isinf(val):
            return None
    except TypeError:
        pass
    return val


def _record(series: pd.Series, dates: pd.Series, fn: str) -> dict:
    """Return {value, date} for the max or min of series."""
    clean = series.dropna()
    if clean.empty:
        return {"value": None, "date": None}
    idx = clean.idxmax() if fn == "max" else clean.idxmin()
    return {
        "value": _safe(round(float(clean[idx]), 6)),
        "date":  str(dates[idx].date()),
    }


def _window_hist(hist: pd.DataFrame, days: int | None, ytd: bool = False) -> pd.DataFrame:
    """Slice history to a rolling window or YTD."""
    if ytd:
        year = hist["date"].iloc[-1].year
        return hist[hist["date"].dt.year == year].copy()
    if days is not None:
        return hist.tail(days).copy()
    return hist.copy()


def _streak(series: pd.Series, positive: bool) -> int:
    """Consecutive trailing sessions above (positive=True) or below zero."""
    vals = series.dropna().values[::-1]
    count = 0
    for v in vals:
        if (positive and v > 0) or (not positive and v < 0):
            count += 1
        else:
            break
    return count


def _records_for_window(hist: pd.DataFrame, col: str) -> dict:
    """High and low record for a single column over a history slice."""
    return {
        "high": _record(hist[col], hist["date"], "max"),
        "low":  _record(hist[col], hist["date"], "min"),
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(date: str, params: dict[str, Any]) -> dict:
    """
    params: (all optional)
        metrics (list[str]): columns to include; default = all
    """
    requested: list[str] = params.get("metrics", ALL_COLS)
    # Filter to known columns only
    requested = [c for c in requested if c in ALL_COLS]
    if not requested:
        requested = ALL_COLS

    hist = history_up_to(date)
    if hist.empty:
        raise ValueError(f"No history available up to {date}")

    # Drop columns not present in features (safety)
    requested = [c for c in requested if c in hist.columns]

    # Three windows
    hist_alltime = _window_hist(hist, days=None)
    hist_52w     = _window_hist(hist, days=252)
    hist_ytd     = _window_hist(hist, days=None, ytd=True)

    # ── Per-metric records ────────────────────────────────────────────────
    records: dict[str, dict] = {}
    for col in requested:
        records[col] = {
            "all_time": _records_for_window(hist_alltime, col),
            "52_week":  _records_for_window(hist_52w,     col),
            "ytd":      _records_for_window(hist_ytd,     col),
        }

    # ── Return-based streaks & extremes (return_1d) ───────────────────────
    r1d = hist_alltime["return_1d"].dropna() if "return_1d" in hist_alltime.columns else pd.Series(dtype=float)

    winning_streak  = _streak(r1d, positive=True)
    losing_streak   = _streak(r1d, positive=False)

    best_day  = _record(r1d, hist_alltime["date"], "max") if not r1d.empty else {"value": None, "date": None}
    worst_day = _record(r1d, hist_alltime["date"], "min") if not r1d.empty else {"value": None, "date": None}

    # YTD versions
    r1d_ytd = hist_ytd["return_1d"].dropna() if "return_1d" in hist_ytd.columns else pd.Series(dtype=float)
    best_day_ytd  = _record(r1d_ytd, hist_ytd["date"], "max") if not r1d_ytd.empty else {"value": None, "date": None}
    worst_day_ytd = _record(r1d_ytd, hist_ytd["date"], "min") if not r1d_ytd.empty else {"value": None, "date": None}

    # ── Average daily return ──────────────────────────────────────────────
    avg_return_alltime = _safe(round(float(r1d.mean()), 6))        if not r1d.empty      else None
    avg_return_ytd     = _safe(round(float(r1d_ytd.mean()), 6))    if not r1d_ytd.empty  else None

    r1d_52w = hist_52w["return_1d"].dropna() if "return_1d" in hist_52w.columns else pd.Series(dtype=float)
    avg_return_52w = _safe(round(float(r1d_52w.mean()), 6)) if not r1d_52w.empty else None

    # ── Session counts ────────────────────────────────────────────────────
    total_sessions = len(hist_alltime)
    ytd_sessions   = len(hist_ytd)
    sessions_52w   = len(hist_52w)

    up_days_alltime   = int((r1d > 0).sum())
    down_days_alltime = int((r1d < 0).sum())
    up_days_ytd       = int((r1d_ytd > 0).sum())
    down_days_ytd     = int((r1d_ytd < 0).sum())

    return {
        "as_of_date":   date,
        "total_sessions": total_sessions,
        "ytd_sessions":   ytd_sessions,
        "sessions_52w":   sessions_52w,

        # High/low records per metric across 3 windows
        "records": records,

        # Single-day return extremes
        "best_day":       best_day,
        "worst_day":      worst_day,
        "best_day_ytd":   best_day_ytd,
        "worst_day_ytd":  worst_day_ytd,

        # Streaks (trailing from as_of_date)
        "current_winning_streak": winning_streak,
        "current_losing_streak":  losing_streak,

        # Average daily returns
        "avg_daily_return_alltime": avg_return_alltime,
        "avg_daily_return_52w":     avg_return_52w,
        "avg_daily_return_ytd":     avg_return_ytd,

        # Up/down day counts
        "up_days_alltime":   up_days_alltime,
        "down_days_alltime": down_days_alltime,
        "up_days_ytd":       up_days_ytd,
        "down_days_ytd":     down_days_ytd,
    }
