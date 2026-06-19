"""
analytics/flows.py
Deterministic flow analysis.

run(date, params) -> dict
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from analytics._loader import history_up_to, row_for_date
from analytics.trend import linear_slope

_CUMULATIVE_WINDOWS = [5, 10, 20]


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def cumulative_pressure(net_series: pd.Series, windows: list[int]) -> dict[str, float | None]:
    """Rolling sum of *net_series* over each window, using the last available obs."""
    result: dict[str, float | None] = {}
    vals = net_series.dropna()
    for w in windows:
        tail = vals.tail(w)
        if tail.empty:
            result[f"{w}d"] = None
        else:
            result[f"{w}d"] = round(float(tail.sum()), 2)
    return result


def participation_ratio(
    foreign_buy: float,
    foreign_sell: float,
    value_traded: float,
) -> float | None:
    """(foreign_buy + foreign_sell) / value_traded as a percentage."""
    total_flow = foreign_buy + foreign_sell
    if value_traded == 0 or math.isnan(value_traded):
        return None
    ratio = total_flow / value_traded * 100
    return round(ratio, 2)


def flow_dominance(foreign_net: float, domestic_net: float) -> str:
    """
    Classify who is the dominant actor today.
    Returns one of: "foreign_buying", "foreign_selling",
    "domestic_buying", "domestic_selling", "balanced".
    """
    threshold = 0.0
    if abs(foreign_net) <= threshold and abs(domestic_net) <= threshold:
        return "balanced"
    # The larger absolute net determines dominance
    if abs(foreign_net) >= abs(domestic_net):
        return "foreign_buying" if foreign_net > 0 else "foreign_selling"
    return "domestic_buying" if domestic_net > 0 else "domestic_selling"


def pressure_trend(net_series: pd.Series, window: int = 10) -> str:
    """
    OLS slope of *net_series* over the last *window* sessions.
    Returns a human-readable direction string.
    """
    slope = linear_slope(net_series, window)
    if math.isnan(slope):
        return "insufficient_data"
    if slope > 0:
        # Are the recent values themselves positive or negative?
        recent_mean = float(net_series.dropna().tail(window).mean())
        if recent_mean >= 0:
            return "increasing_buying"
        return "decreasing_selling"
    else:
        recent_mean = float(net_series.dropna().tail(window).mean())
        if recent_mean >= 0:
            return "decreasing_buying"
        return "increasing_selling"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(date: str, params: dict[str, Any]) -> dict:
    """
    params:
        cumulative_windows (list[int], optional): default [5, 10, 20]
        pressure_window (int, optional): window for pressure_trend (default 10)
        date_from (str, optional): ISO date — if provided, aggregate flows from
            date_from to date (inclusive) and return range_aggregates instead of
            rolling windows
    """
    cumulative_windows: list[int] = params.get("cumulative_windows", _CUMULATIVE_WINDOWS)
    pressure_window: int = int(params.get("pressure_window", 10))
    date_from: str | None = params.get("date_from")

    hist = history_up_to(date)
    if hist.empty:
        raise ValueError(f"No history available up to {date}")

    row = row_for_date(date)

    foreign_net_today: float = float(row["foreign_net"])
    domestic_net_today: float = float(row["domestic_net"])
    foreign_buy_today: float = float(row["foreign_buy"])
    foreign_sell_today: float = float(row["foreign_sell"])
    value_traded_today: float = float(row["value_traded"])
    foreign_flow_zscore: float = float(row["foreign_flow_zscore"])
    domestic_flow_zscore: float = float(row["domestic_flow_zscore"])

    dominant = flow_dominance(foreign_net_today, domestic_net_today)

    cum_foreign = cumulative_pressure(hist["foreign_net"], cumulative_windows)
    cum_domestic = cumulative_pressure(hist["domestic_net"], cumulative_windows)

    fp = participation_ratio(foreign_buy_today, foreign_sell_today, value_traded_today)

    trend_str = pressure_trend(hist["foreign_net"], pressure_window)

    def _safe(v: float) -> float | None:
        return None if math.isnan(v) else round(v, 4)

    out: dict[str, Any] = {
        "date": str(pd.Timestamp(date).date()),
        "data_through": str(pd.Timestamp(date).date()),
        "foreign_net_today": round(foreign_net_today, 2),
        "domestic_net_today": round(domestic_net_today, 2),
        "dominant_flow": dominant,
        "cumulative_foreign_net": cum_foreign,
        "cumulative_domestic_net": cum_domestic,
        "foreign_participation_pct": fp,
        f"flow_pressure_trend_{pressure_window}d": trend_str,
        "foreign_flow_zscore": _safe(foreign_flow_zscore),
        "domestic_flow_zscore": _safe(domestic_flow_zscore),
    }

    if date_from:
        ts_from = pd.Timestamp(date_from)
        ts_to = pd.Timestamp(date)
        rng = hist[(hist["date"] >= ts_from) & (hist["date"] <= ts_to)]
        if not rng.empty:
            out["range_aggregates"] = {
                "date_from": str(ts_from.date()),
                "date_to": str(ts_to.date()),
                "trading_sessions": len(rng),
                "total_foreign_buy": round(float(rng["foreign_buy"].sum()), 2),
                "total_foreign_sell": round(float(rng["foreign_sell"].sum()), 2),
                "total_foreign_net": round(float(rng["foreign_net"].sum()), 2),
                "total_domestic_buy": round(float(rng["domestic_buy"].sum()), 2),
                "total_domestic_sell": round(float(rng["domestic_sell"].sum()), 2),
                "total_domestic_net": round(float(rng["domestic_net"].sum()), 2),
            }

    return out
