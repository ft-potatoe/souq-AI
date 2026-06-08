"""
analytics/seasonality.py
Deterministic seasonality analysis.

run(date, params) -> dict

Also exposes map_to_hijri() as a stub for Ramadan detection — the hard-coded
RAMADAN_RANGES from build_features.py are the authoritative source; this module
reads is_ramadan directly from features_master.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from analytics._loader import history_up_to, row_for_date

# QSE day-of-week encoding: 0=Sunday … 4=Thursday
_QSE_DOW_NAMES = {0: "Sunday", 1: "Monday", 2: "Tuesday", 3: "Wednesday", 4: "Thursday"}
_MONTH_NAMES = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}

SUPPORTED_METRICS = {
    "volume", "value_traded", "total_trades", "return_1d",
    "foreign_net", "domestic_net", "breadth_ratio", "foreign_participation",
    "volatility_20d",
}


# ---------------------------------------------------------------------------
# Stub kept for public API compatibility with §6.4
# ---------------------------------------------------------------------------

def map_to_hijri(gregorian_date: pd.Timestamp) -> dict[str, int]:
    """
    Stub: returns an approximate Hijri year/month derived from the known
    Ramadan boundary lookup in features_master (is_ramadan flag).
    Full Hijri conversion is out of scope for this phase.
    """
    return {"note": "Ramadan detection uses hard-coded Gregorian ranges in build_features.py"}


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def day_of_week_profile(metric: str, hist: pd.DataFrame) -> dict[str, Any]:
    """
    Mean of *metric* grouped by QSE day_of_week (0=Sun … 4=Thu).
    Returns the group means and the ordinal rank of today's day.
    """
    grp = hist.groupby("day_of_week")[metric].mean()
    means: dict[str, float] = {}
    for dow, name in _QSE_DOW_NAMES.items():
        v = grp.get(dow)
        means[name] = round(float(v), 6) if v is not None and not math.isnan(v) else None

    return means


def monthly_profile(metric: str, hist: pd.DataFrame) -> dict[str, Any]:
    """Mean of *metric* grouped by calendar month."""
    grp = hist.groupby("month")[metric].mean()
    means: dict[str, float | None] = {}
    for m, name in _MONTH_NAMES.items():
        v = grp.get(m)
        means[name] = round(float(v), 6) if v is not None and not math.isnan(v) else None
    return means


def ramadan_effect(metric: str, hist: pd.DataFrame) -> dict[str, Any]:
    """Compare *metric* mean during Ramadan vs non-Ramadan sessions."""
    ram = hist[hist["is_ramadan"] == 1][metric].dropna()
    non_ram = hist[hist["is_ramadan"] == 0][metric].dropna()

    ram_mean = float(ram.mean()) if not ram.empty else None
    non_ram_mean = float(non_ram.mean()) if not non_ram.empty else None

    pct_diff: float | None = None
    if ram_mean is not None and non_ram_mean is not None and non_ram_mean != 0:
        pct_diff = round((ram_mean - non_ram_mean) / abs(non_ram_mean) * 100, 2)

    return {
        "ramadan_mean": round(ram_mean, 6) if ram_mean is not None else None,
        "non_ramadan_mean": round(non_ram_mean, 6) if non_ram_mean is not None else None,
        "pct_difference": pct_diff,
        "ramadan_sessions": int(len(ram)),
        "non_ramadan_sessions": int(len(non_ram)),
    }


def _ordinal(n: int) -> str:
    if 11 <= (n % 100) <= 13:
        return f"{n}th"
    return f"{n}{['th', 'st', 'nd', 'rd', 'th'][min(n % 10, 4)]}"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(date: str, params: dict[str, Any]) -> dict:
    """
    params:
        metric (str): one of SUPPORTED_METRICS (default "volume")
    """
    metric: str = params.get("metric", "volume")
    if metric not in SUPPORTED_METRICS:
        raise ValueError(
            f"Unsupported metric '{metric}'. Choose from: {sorted(SUPPORTED_METRICS)}"
        )

    hist = history_up_to(date)
    if hist.empty:
        raise ValueError(f"No history available up to {date}")

    row = row_for_date(date)

    today_dow: int = int(row["day_of_week"])
    today_dow_name: str = _QSE_DOW_NAMES.get(today_dow, str(today_dow))
    today_month: int = int(row["month"])
    today_month_name: str = _MONTH_NAMES.get(today_month, str(today_month))
    is_ramadan_today: bool = bool(int(row["is_ramadan"]) == 1)

    dow_means = day_of_week_profile(metric, hist)
    month_means = monthly_profile(metric, hist)
    ram_effect = ramadan_effect(metric, hist)

    # Today's value
    today_val = row[metric]
    today_val_safe = (
        None if isinstance(today_val, float) and math.isnan(today_val)
        else float(today_val)
    )

    # Day-of-week rank (1 = highest)
    today_dow_mean = dow_means.get(today_dow_name)
    dow_rank: str | None = None
    if today_dow_mean is not None:
        valid_dow = {k: v for k, v in dow_means.items() if v is not None}
        sorted_dow = sorted(valid_dow.items(), key=lambda x: x[1], reverse=True)
        rank_num = next(
            (i + 1 for i, (k, _) in enumerate(sorted_dow) if k == today_dow_name), None
        )
        total_dow = len(sorted_dow)
        dow_rank = f"{_ordinal(rank_num)} of {total_dow}" if rank_num else None

    # Monthly rank (1 = highest)
    today_month_mean = month_means.get(today_month_name)
    month_rank: str | None = None
    if today_month_mean is not None:
        valid_month = {k: v for k, v in month_means.items() if v is not None}
        sorted_month = sorted(valid_month.items(), key=lambda x: x[1], reverse=True)
        rank_num_m = next(
            (i + 1 for i, (k, _) in enumerate(sorted_month) if k == today_month_name), None
        )
        total_months = len(sorted_month)
        month_rank = (
            f"{_ordinal(rank_num_m)} of {total_months}" if rank_num_m else None
        )

    return {
        "metric": metric,
        "date": str(pd.Timestamp(date).date()),
        "today_day_of_week": today_dow_name,
        "day_of_week_mean": round(today_dow_mean, 6) if today_dow_mean is not None else None,
        "day_of_week_rank": dow_rank,
        "day_of_week_profile": dow_means,
        "monthly_mean_this_month": (
            round(today_month_mean, 6) if today_month_mean is not None else None
        ),
        "monthly_rank": month_rank,
        "monthly_profile": month_means,
        "ramadan_effect": {**ram_effect, "is_ramadan_today": is_ramadan_today},
    }
