"""
analytics/gcc.py
Deterministic GCC benchmarking analysis.

run(date, params) -> dict
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from analytics._loader import history_up_to, load_gcc_raw, row_for_date

_HORIZONS = [1, 5, 20]

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

def peer_relative_performance(
    hist: pd.DataFrame,
    gcc_hist: pd.DataFrame,
    horizons: list[int] = _HORIZONS,
) -> dict[str, float | None]:
    """
    QSE cumulative return vs GCC peer average over each horizon (trading days).
    hist: features up to date (sorted ascending, includes return_1d and qse_vs_gcc_spread)
    Returns {f"qse_vs_gcc_spread_{h}d": value, ...}
    """
    result: dict[str, float | None] = {}
    for h in horizons:
        if h == 1:
            # Use pre-computed spread for today
            if hist.empty:
                result["qse_vs_gcc_spread_1d"] = None
                continue
            val = float(hist["qse_vs_gcc_spread"].iloc[-1])
            result["qse_vs_gcc_spread_1d"] = round(val, 6) if not math.isnan(val) else None
        else:
            # Sum of spread over last h sessions
            spread_series = hist["qse_vs_gcc_spread"].dropna()
            if len(spread_series) < h:
                result[f"qse_vs_gcc_spread_{h}d"] = None
                continue
            cumulative = float(spread_series.tail(h).sum())
            result[f"qse_vs_gcc_spread_{h}d"] = round(cumulative, 6)
    return result


def peer_rank(qse_return: float, gcc_returns_dict: dict[str, float]) -> int | None:
    """
    Rank of QSE among all markets (including QSE itself) by return, 1 = best.
    """
    if math.isnan(qse_return):
        return None
    all_returns = list(gcc_returns_dict.values()) + [qse_return]
    valid = [v for v in all_returns if not math.isnan(v)]
    if not valid:
        return None
    sorted_desc = sorted(valid, reverse=True)
    try:
        return sorted_desc.index(qse_return) + 1
    except ValueError:
        return None


def rolling_outperformance_rate(
    qse_returns: pd.Series,
    gcc_avg: pd.Series,
    window: int = 60,
) -> float | None:
    """
    Fraction of sessions in the last *window* where QSE return_1d > gcc_avg_return_1d.
    """
    paired = pd.concat(
        [qse_returns.rename("qse"), gcc_avg.rename("gcc")], axis=1
    ).dropna()
    tail = paired.tail(window)
    if tail.empty:
        return None
    rate = float((tail["qse"] > tail["gcc"]).sum() / len(tail))
    return round(rate, 4)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(date: str, params: dict[str, Any]) -> dict:
    """
    params:
        horizons (list[int], optional): spread horizons in trading days (default [1,5,20])
        outperformance_window (int, optional): window for rolling_outperformance_rate (default 60)
    """
    horizons: list[int] = params.get("horizons", _HORIZONS)
    op_window: int = int(params.get("outperformance_window", 60))

    hist = history_up_to(date)
    if hist.empty:
        raise ValueError(f"No history available up to {date}")

    row = row_for_date(date)
    ts = pd.Timestamp(date)

    # Per-peer returns for today from gcc_daily raw
    gcc_raw = load_gcc_raw()
    gcc_today = gcc_raw[gcc_raw["date"] == ts].copy()
    peer_returns: dict[str, float | None] = {}
    qse_gcc_return_1d: float | None = None

    for _, prow in gcc_today.iterrows():
        mkt = str(prow["market_name"]).upper()
        ret = float(prow["daily_change_pct"]) / 100.0
        display = _MARKET_DISPLAY.get(mkt, mkt)
        if mkt == "QSE":
            qse_gcc_return_1d = round(ret * 100, 4)
        else:
            peer_returns[display] = round(ret * 100, 4)

    # Fall back to features if gcc_daily doesn't have today
    qse_return_1d_feat = float(row["return_1d"])
    if qse_gcc_return_1d is None:
        qse_gcc_return_1d = (
            round(qse_return_1d_feat, 6) if not math.isnan(qse_return_1d_feat) else None
        )

    gcc_avg_today = float(row["gcc_avg_return_1d"])
    qse_vs_gcc_today = float(row["qse_vs_gcc_spread"])

    # Rank QSE
    rank = peer_rank(
        qse_return_1d_feat,
        {k: v for k, v in peer_returns.items() if v is not None},
    )
    total_peers = len(peer_returns) + 1  # +1 for QSE itself

    # Relative performance spreads
    rel_perf = peer_relative_performance(hist, gcc_raw, horizons)

    # Rolling outperformance rate
    op_rate = rolling_outperformance_rate(
        hist.set_index("date")["return_1d"],
        hist.set_index("date")["gcc_avg_return_1d"],
        op_window,
    )

    def _safe(v: float) -> float | None:
        return None if (v is None or math.isnan(v)) else v

    op_interpretation = (
        "underperforming" if (op_rate is not None and op_rate < 0.50) else
        "outperforming" if (op_rate is not None and op_rate >= 0.50) else None
    )

    def _pct(v: float) -> float | None:
        return None if (v is None or math.isnan(v)) else round(v * 100, 4)

    out: dict[str, Any] = {
        "date": str(ts.date()),
        "units": "all return and spread values are in percent (%)",
        "qse_return_1d_pct": _pct(qse_return_1d_feat),
        "gcc_avg_return_1d_pct": _pct(gcc_avg_today),
        "qse_vs_gcc_spread_1d_pct": _pct(qse_vs_gcc_today),
        "qse_rank_among_all_markets_including_qse": rank,
        "total_markets_including_qse": total_peers,
        "peer_returns_pct": peer_returns,
        f"rolling_outperformance_rate_{op_window}d": op_rate,
        f"rolling_outperformance_interpretation_{op_window}d": op_interpretation,
    }

    # Add multi-horizon spreads in pct
    for h in horizons:
        if h != 1:
            raw = rel_perf.get(f"qse_vs_gcc_spread_{h}d")
            out[f"qse_vs_gcc_spread_{h}d_pct"] = round(raw * 100, 4) if raw is not None else None

    return out
