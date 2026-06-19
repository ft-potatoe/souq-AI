"""
analytics/trend.py
Deterministic trend analysis.

run(date, params) -> dict
"""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from analytics._loader import history_up_to, row_for_date

SUPPORTED_METRICS = {
    "foreign_net",
    "domestic_net",
    "volume",
    "value_traded",
    "total_trades",
    "breadth_ratio",
    "return_1d",
    "foreign_flow_zscore",
    "foreign_participation",
    "close",
}

_MOMENTUM_WINDOWS = [5, 10, 20]


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def linear_slope(series: pd.Series, window: int) -> float:
    """OLS slope over the last *window* observations (same units as series/period)."""
    tail = series.dropna().tail(window)
    if len(tail) < 2:
        return float("nan")
    x = np.arange(len(tail), dtype=float)
    result = sp_stats.linregress(x, tail.values)
    return float(result.slope)


def streak_count(series: pd.Series, condition_fn: Callable[[float], bool]) -> int:
    """
    Count of consecutive trailing sessions (newest first) satisfying *condition_fn*.
    Returns 0 if the most recent value does not satisfy the condition.
    """
    vals = series.dropna().values[::-1]  # newest first
    count = 0
    for v in vals:
        if condition_fn(v):
            count += 1
        else:
            break
    return count


def momentum_direction(
    series: pd.Series, windows: list[int] = _MOMENTUM_WINDOWS
) -> dict[str, str]:
    """
    For each window: compare last value to the value *window* periods ago.
    Returns "up", "down", or "flat" (within 0.5% relative change).
    """
    result: dict[str, str] = {}
    vals = series.dropna()
    last = vals.iloc[-1] if not vals.empty else float("nan")
    for w in windows:
        if len(vals) < w + 1:
            result[f"{w}d"] = "insufficient_data"
            continue
        prior = vals.iloc[-(w + 1)]
        if prior == 0:
            result[f"{w}d"] = "flat"
            continue
        change = (last - prior) / abs(prior)
        if change > 0.005:
            result[f"{w}d"] = "up"
        elif change < -0.005:
            result[f"{w}d"] = "down"
        else:
            result[f"{w}d"] = "flat"
    return result


def sma_crossover(
    fast_series: pd.Series, slow_series: pd.Series
) -> tuple[str | None, str | None]:
    """
    Detect the most recent golden/death cross between fast and slow SMA.
    Returns (event_type, date_str) or (None, None) if no cross in available history.
    event_type: "golden_cross" | "death_cross"
    """
    df = pd.DataFrame({"fast": fast_series.values, "slow": slow_series.values})
    df = df.dropna()
    if len(df) < 2:
        return None, None

    # A cross occurs when sign of (fast - slow) changes between consecutive rows
    diff = df["fast"] - df["slow"]
    sign = np.sign(diff)
    cross = sign.diff().fillna(0) != 0

    # Find last crossing
    cross_indices = cross[cross].index.tolist()
    if not cross_indices:
        return None, None

    last_idx = cross_indices[-1]
    event = "golden_cross" if sign.iloc[last_idx] > 0 else "death_cross"
    return event, None  # date not accessible here; caller supplies if needed


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(date: str, params: dict[str, Any]) -> dict:
    """
    params:
        metric (str): column to analyse
        streak_condition (str, optional): "positive" | "negative" | "above_mean"
        slope_window (int, optional): window for linear_slope (default 10)
        momentum_windows (list[int], optional)
        fast_sma_col (str, optional): features column for fast SMA
        slow_sma_col (str, optional): features column for slow SMA
        date_from (str, optional): ISO date — restrict history to on/after this date
    """
    metric: str = params.get("metric", "foreign_net")
    slope_window: int = int(params.get("slope_window", 10))
    streak_cond: str = params.get("streak_condition", "negative")
    momentum_windows: list[int] = params.get("momentum_windows", _MOMENTUM_WINDOWS)
    fast_col: str | None = params.get("fast_sma_col")
    slow_col: str | None = params.get("slow_sma_col")
    date_from: str | None = params.get("date_from")

    hist = history_up_to(date)
    if hist.empty:
        raise ValueError(f"No history available up to {date}")

    if date_from:
        hist = hist[hist["date"] >= pd.Timestamp(date_from)].reset_index(drop=True)
        if hist.empty:
            raise ValueError(f"No history in range {date_from} to {date}")

    series = hist[metric].reset_index(drop=True)

    # Slope
    slope_val = linear_slope(series, slope_window)
    if math.isnan(slope_val):
        slope_direction = None
    elif slope_val > 0:
        slope_direction = "increasing"
    elif slope_val < 0:
        slope_direction = "decreasing"
    else:
        slope_direction = "flat"

    # Streak
    if streak_cond == "positive":
        cond_fn: Callable[[float], bool] = lambda v: v > 0
        cond_label = "positive"
    elif streak_cond == "negative":
        cond_fn = lambda v: v < 0
        cond_label = "negative"
    else:
        mean_val = float(series.dropna().mean())
        cond_fn = lambda v: v > mean_val
        cond_label = "above_mean"

    streak = streak_count(series, cond_fn)

    # Momentum
    mom = momentum_direction(series, momentum_windows)

    # SMA crossover
    crossover_event: str | None = None
    crossover_date: str | None = None
    if fast_col and slow_col and fast_col in hist.columns and slow_col in hist.columns:
        event, _ = sma_crossover(hist[fast_col], hist[slow_col])
        if event:
            crossover_event = event
            # Find the actual date of the last cross
            fast_s = hist[fast_col].reset_index(drop=True)
            slow_s = hist[slow_col].reset_index(drop=True)
            diff = (fast_s - slow_s).dropna()
            sign = np.sign(diff)
            cross_mask = sign.diff().fillna(0) != 0
            cross_positions = cross_mask[cross_mask].index.tolist()
            if cross_positions:
                last_pos = cross_positions[-1]
                crossover_date = str(hist["date"].iloc[last_pos].date())

    out: dict[str, Any] = {
        "metric": metric,
        f"slope_{slope_window}d": round(slope_val, 4) if not math.isnan(slope_val) else None,
        "slope_direction": slope_direction,
        "streak": {"condition": cond_label, "count": streak},
        "momentum": {k.replace("d", ""): v for k, v in mom.items()},
        "sma_crossover": crossover_event,
        "crossover_date": crossover_date,
    }
    if date_from:
        out["period_date_from"] = date_from
        out["period_sessions"] = len(hist)
    return out
