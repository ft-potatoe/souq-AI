"""
analytics/correlation.py
Deterministic correlation analysis.

run(date, params) -> dict
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from analytics._loader import history_up_to, load_gcc_raw
from analytics.distribution import percentile_rank

_ROLLING_WINDOWS = [20, 60]

# Map raw market_name values to display names used in the output
_MARKET_DISPLAY = {
    "TASI": "Tadawul",
    "ADX": "ADX",
    "DFM": "DFM",
    "KSE": "KSE",
    "MSM": "MSM",
    "BSE": "BSE",
}


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def rolling_correlation(
    series_a: pd.Series,
    series_b: pd.Series,
    windows: list[int] = _ROLLING_WINDOWS,
) -> dict[str, float | None]:
    """Pearson correlation over the last *window* valid pairs for each window."""
    result: dict[str, float | None] = {}
    for w in windows:
        paired = pd.concat(
            [series_a.rename("a"), series_b.rename("b")], axis=1
        ).dropna()
        tail = paired.tail(w)
        if len(tail) < 4:
            result[f"{w}d"] = None
            continue
        r = float(tail["a"].corr(tail["b"]))
        result[f"{w}d"] = round(r, 4) if not math.isnan(r) else None
    return result


def current_vs_historical_corr(
    series_a: pd.Series,
    series_b: pd.Series,
    window: int = 20,
) -> dict[str, Any]:
    """
    Rolling window correlation over the full history, then describe where
    the most-recent window sits in that distribution.
    """
    paired = pd.concat(
        [series_a.rename("a"), series_b.rename("b")], axis=1
    ).dropna()

    if len(paired) < window + 1:
        return {
            "current": None,
            "historical_mean": None,
            "historical_std": None,
            "percentile_of_current": None,
        }

    roll_corr = (
        paired["a"].rolling(window, min_periods=window // 2).corr(paired["b"])
    ).dropna()

    current_corr = float(roll_corr.iloc[-1]) if not roll_corr.empty else None
    hist_mean = float(roll_corr.mean()) if not roll_corr.empty else None
    hist_std = float(roll_corr.std(ddof=1)) if len(roll_corr) > 1 else None

    pct = None
    if current_corr is not None and not math.isnan(current_corr):
        pct = round(percentile_rank(roll_corr, current_corr), 2)

    return {
        "current": round(current_corr, 4) if current_corr is not None else None,
        "historical_mean": round(hist_mean, 4) if hist_mean is not None else None,
        "historical_std": round(hist_std, 4) if hist_std is not None else None,
        "percentile_of_current": pct,
    }


def gcc_peer_correlations(
    qse_returns: pd.Series,
    gcc_df: pd.DataFrame,
    window: int = 60,
) -> dict[str, float | None]:
    """
    60-day rolling Pearson corr between QSE return_1d and each peer's daily return.
    gcc_df must have columns: date, market_name, daily_change_pct
    """
    result: dict[str, float | None] = {}
    peers = gcc_df[gcc_df["market_name"].str.upper() != "QSE"].copy()
    peers = peers.copy()
    peers["ret"] = peers["daily_change_pct"] / 100.0

    for mkt, grp in peers.groupby("market_name"):
        display = _MARKET_DISPLAY.get(mkt.upper(), mkt)
        peer_ret = grp.set_index("date")["ret"].sort_index()
        # Align on QSE dates
        aligned = pd.concat(
            [qse_returns.rename("qse"), peer_ret.rename("peer")], axis=1
        ).dropna()
        tail = aligned.tail(window)
        if len(tail) < 4:
            result[display] = None
            continue
        r = float(tail["qse"].corr(tail["peer"]))
        result[display] = round(r, 4) if not math.isnan(r) else None

    return result


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(date: str, params: dict[str, Any]) -> dict:
    """
    params:
        metric_a (str): first column (default "foreign_net")
        metric_b (str): second column (default "return_1d")
        rolling_windows (list[int], optional)
        current_window (int, optional): window for current_vs_historical (default 20)
        include_gcc (bool, optional): include gcc_peer_correlations (default True)
        gcc_window (int, optional): window for GCC peer correlations (default 60)
    """
    metric_a: str = params.get("metric_a", "foreign_net")
    metric_b: str = params.get("metric_b", "return_1d")
    rolling_windows: list[int] = params.get("rolling_windows", _ROLLING_WINDOWS)
    current_window: int = int(params.get("current_window", 20))
    include_gcc: bool = bool(params.get("include_gcc", True))
    gcc_window: int = int(params.get("gcc_window", 60))

    hist = history_up_to(date)
    if hist.empty:
        raise ValueError(f"No history available up to {date}")

    series_a = hist[metric_a].reset_index(drop=True)
    series_b = hist[metric_b].reset_index(drop=True)

    roll_corrs = rolling_correlation(series_a, series_b, rolling_windows)
    cv_hist = current_vs_historical_corr(series_a, series_b, current_window)

    gcc_corrs: dict[str, float | None] = {}
    if include_gcc:
        gcc_raw = load_gcc_raw()
        gcc_hist = gcc_raw[gcc_raw["date"] <= pd.Timestamp(date)].copy()
        # Index QSE return by date for alignment
        qse_ret = hist.set_index("date")["return_1d"]
        gcc_corrs = gcc_peer_correlations(qse_ret, gcc_hist, gcc_window)

    pair_label = f"{metric_a} vs {metric_b}"

    out: dict[str, Any] = {
        "pair": pair_label,
    }
    for w in rolling_windows:
        out[f"rolling_corr_{w}d"] = roll_corrs.get(f"{w}d")

    out["historical_mean_corr"] = cv_hist["historical_mean"]
    out["historical_std_corr"] = cv_hist["historical_std"]
    out["percentile_of_current_corr"] = cv_hist["percentile_of_current"]

    if include_gcc:
        out[f"gcc_correlations_{gcc_window}d"] = gcc_corrs

    return out
