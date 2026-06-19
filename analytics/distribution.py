"""
analytics/distribution.py
Deterministic distribution analysis.

run(date, params) -> dict
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from analytics._loader import history_up_to, row_for_date

SUPPORTED_METRICS = {
    "volume",
    "value_traded",
    "total_trades",
    "breadth_ratio",
    "foreign_net",
    "domestic_net",
    "return_1d",
    "volatility_20d",
    "foreign_participation",
}

_ROLLING_WINDOWS = [20, 60, 252]


# ---------------------------------------------------------------------------
# Core helpers (also importable by other modules)
# ---------------------------------------------------------------------------

def percentile_rank(series: pd.Series, value: float) -> float:
    """Fraction of *series* values strictly below *value*, expressed 0-100."""
    valid = series.dropna()
    if valid.empty:
        return float("nan")
    return float((valid < value).sum() / len(valid) * 100)


def historical_frequency(
    series: pd.Series, value: float, direction: str = "above"
) -> float:
    """Fraction of sessions where *series* is above/below *value*."""
    valid = series.dropna()
    if valid.empty:
        return float("nan")
    if direction == "above":
        count = (valid >= value).sum()
    else:
        count = (valid <= value).sum()
    return float(count / len(valid))


def rolling_stats(series: pd.Series, windows: list[int] = _ROLLING_WINDOWS) -> dict:
    """
    For each window, compute the rolling mean and std *ending at the last
    available observation* in *series*.
    """
    result: dict[str, dict] = {}
    for w in windows:
        tail = series.dropna().tail(w)
        if tail.empty:
            result[f"{w}d"] = {"mean": None, "std": None}
        else:
            result[f"{w}d"] = {
                "mean": round(float(tail.mean()), 6),
                "std": round(float(tail.std(ddof=1)) if len(tail) > 1 else 0.0, 6),
            }
    return result


def _last_comparable_date(
    hist: pd.DataFrame, metric: str, today_value: float
) -> str | None:
    """Most recent date (before today) where metric was within 1 % of today_value."""
    if len(hist) < 2:
        return None
    past = hist.iloc[:-1].reset_index(drop=True)  # exclude today, reset so .loc is safe
    col = past[metric].dropna()
    if col.empty or today_value == 0:
        return None
    rel = (col - today_value).abs() / abs(today_value)
    close_mask = rel < 0.01
    if not close_mask.any():
        # fall back to nearest single value — use col's own index
        best_pos = int(rel.values.argmin())
        idx = col.index[best_pos]
    else:
        # idxmax on col's sub-index; that index maps directly into past
        idx = col[close_mask].index[-1]  # last (most recent) close match
    dt = past.at[idx, "date"]
    return str(dt.date())


def _extremes(hist: pd.DataFrame, metric: str) -> dict:
    """
    Dataset-wide min and max of *metric* over the supplied history, each paired
    with the date it occurred on. Answers "biggest drop / highest volume / worst
    day" style questions without the LLM ever computing the extremum itself.
    """
    valid = hist[[metric, "date"]].dropna(subset=[metric])
    if valid.empty:
        return {
            "min_value": None, "min_date": None,
            "max_value": None, "max_date": None,
        }
    min_row = valid.loc[valid[metric].idxmin()]
    max_row = valid.loc[valid[metric].idxmax()]
    return {
        "min_value": round(float(min_row[metric]), 6),
        "min_date": str(min_row["date"].date()),
        "max_value": round(float(max_row[metric]), 6),
        "max_date": str(max_row["date"].date()),
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(date: str, params: dict[str, Any]) -> dict:
    """
    params:
        metric (str): one of SUPPORTED_METRICS
        direction (str, optional): "above" | "below" for historical_frequency
    """
    metric: str = params.get("metric", "volume")
    if metric not in SUPPORTED_METRICS:
        raise ValueError(
            f"Unsupported metric '{metric}'. Choose from: {sorted(SUPPORTED_METRICS)}"
        )
    direction: str = params.get("direction", "above")

    hist = history_up_to(date)
    if hist.empty:
        raise ValueError(f"No history available up to {date}")

    row = row_for_date(date)
    today_value = row[metric]

    series = hist[metric]

    pct_rank = percentile_rank(series, today_value)
    freq = historical_frequency(series, today_value, direction)
    rstats = rolling_stats(series)
    last_comparable = _last_comparable_date(hist, metric, today_value)

    valid = series.dropna()
    if direction == "above":
        sessions_extreme = int((valid >= today_value).sum())
    else:
        sessions_extreme = int((valid <= today_value).sum())

    skewness = round(float(valid.skew()), 4) if len(valid) >= 3 else None
    kurtosis = round(float(valid.kurt()), 4) if len(valid) >= 4 else None
    pct25 = round(float(valid.quantile(0.25)), 6) if not valid.empty else None
    pct50 = round(float(valid.quantile(0.50)), 6) if not valid.empty else None
    pct75 = round(float(valid.quantile(0.75)), 6) if not valid.empty else None

    extremes = _extremes(hist, metric)

    return {
        "metric": metric,
        "today_value": (
            None if (isinstance(today_value, float) and math.isnan(today_value))
            else today_value
        ),
        "percentile_rank": round(pct_rank, 2) if not math.isnan(pct_rank) else None,
        f"historical_frequency_{direction}": round(freq, 4) if not math.isnan(freq) else None,
        "rolling_stats": rstats,
        "last_comparable_date": last_comparable,
        "sessions_above_today" if direction == "above" else "sessions_below_today": sessions_extreme,
        "total_sessions": len(valid),
        "skewness": skewness,
        "kurtosis": kurtosis,
        "percentiles": {"p25": pct25, "p50": pct50, "p75": pct75},
        "extremes": extremes,
    }
