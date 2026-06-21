"""
analytics/relationships.py
Deterministic relationship / association discovery.

Unlike analytics/correlation.py (which compares two USER-NAMED features), this module
brute-force scans every numeric feature pair for monotonic co-movement (Spearman),
ranks the strongest, and produces human-readable conditional-decile statements
("when foreign_net is in its bottom 10%, breadth_ratio is above its median on 73%
of those days vs 50% baseline").

Strictly descriptive: associations are co-movement, NOT causation, and never a forecast.

run(date, params) -> dict
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from analytics._loader import history_up_to

# ---------------------------------------------------------------------------
# Guardrails / configuration
# ---------------------------------------------------------------------------

_MIN_OVERLAP = 60        # minimum non-NaN paired observations for a pair to count
_MIN_STRENGTH = 0.4      # keep only |spearman| >= this
_TOP_PRIMARY = 5         # primary_relationships size
_TOP_STRONGEST = 15      # fuller ranked list size
_DECILE = 0.10           # bottom-decile threshold for conditional analysis

# Columns excluded from the scan. The goal is to surface NON-OBVIOUS relationships,
# so we drop columns that are trivially or definitionally tied to another column --
# otherwise the ranking fills up with identities (~1.0) that teach nothing.
#  - forward returns: look-ahead / leakage, never an "observed" relationship
#  - date: not numeric-meaningful here
#  - calendar / identifier columns: spurious ordinal "relationships"
#  - raw OHLC + SMA: trivially ~1.0 with close and with each other
#  - derived breadth/flow duplicates: each is an algebraic transform of a kept column
#    (advance_decline & breadth_net <- gainers/losers; *_zscore <- *_net;
#     domestic_net ~ -foreign_net since daily flows net to ~0; cumulative <- net)
_EXCLUDE_COLS = {
    "forward_return_5d",
    "forward_return_10d",
    "forward_return_20d",
    "date",
    "day_of_week",
    "month",
    "quarter",
    "is_ramadan",
    "trading_day_of_month",
    "above_sma_20",
    "above_sma_200",
    "open",
    "high",
    "low",
    "close",
    "sma_20",
    "sma_50",
    "sma_200",
    # derived / definitional duplicates
    "advance_decline",
    "breadth_net",
    "gainers",
    "losers",
    "unchanged",
    "total_listed",
    "total_traded",
    # domestic_flow_zscore is a z-score transform of domestic_net — trivial pair
    # domestic_net, domestic_buy, domestic_sell retained: they are the key columns
    # for counterparty analysis (does domestic absorb foreign selling?).
    # domestic_net ~ -foreign_net is a meaningful *discovery*, not a definitional identity.
    "domestic_flow_zscore",
    "foreign_net_cumulative_5d",
    "foreign_net_cumulative_20d",
    "value_zscore",
    "trades_zscore",
    "value_traded",
    "total_trades",
    # foreign_buy and foreign_sell are NOT excluded — they carry asymmetric
    # gross-flow information that foreign_net alone cannot represent
    # (heavy sell + low buy differs from balanced two-way flow with same net).
}

_NOTE = "Associations only -- co-movement, not causation."


# ---------------------------------------------------------------------------
# Phrasing helpers
# ---------------------------------------------------------------------------

def _humanise(col: str) -> str:
    """Turn a feature column name into readable words."""
    return col.replace("_", " ")


def _pair_plain(a: str, b: str, rho: float) -> str:
    """Plain-language description of a pairwise relationship."""
    ha, hb = _humanise(a), _humanise(b)
    if rho < 0:
        return f"When {ha} falls, {hb} tends to rise"
    return f"When {ha} rises, {hb} also tends to rise"


# ---------------------------------------------------------------------------
# Core scan
# ---------------------------------------------------------------------------

def _candidate_columns(df: pd.DataFrame) -> list[str]:
    """Numeric columns eligible for the scan, in stable order."""
    numeric = df.select_dtypes("number").columns
    return [c for c in numeric if c not in _EXCLUDE_COLS]


def _is_trivial_pair(a: str, b: str) -> bool:
    """
    True when two columns are obviously the same quantity under a transform
    (e.g. 'volume' vs 'volume_zscore', 'volatility_20d' vs 'volatility_60d',
    'foreign_net' vs 'foreign_flow_zscore'). Such pairs are definitional, not
    discoveries, so they are dropped from the scan.
    """
    # Strip common transform / window suffixes to compare base quantities.
    suffixes = ("_zscore", "_20d", "_60d", "_5d", "_10d", "_1d", "_slope_10d",
                "_flow_zscore", "_participation", "_pct", "_cumulative_5d",
                "_cumulative_20d")

    def base(name: str) -> str:
        stem = name
        for s in suffixes:
            if stem.endswith(s):
                stem = stem[: -len(s)]
        return stem.rstrip("_")

    ba, bb = base(a), base(b)
    if not ba or not bb:
        return False
    # Same base quantity, or one base is a prefix of the other (e.g. "foreign"
    # vs "foreign_flow"): treat as the same underlying series family.
    return ba == bb or ba.startswith(bb) or bb.startswith(ba)


def scan_relationships(
    df: pd.DataFrame,
    columns: list[str],
    min_overlap: int = _MIN_OVERLAP,
    min_strength: float = _MIN_STRENGTH,
) -> list[dict[str, Any]]:
    """
    Spearman scan over all column pairs. Returns a list of relationship dicts
    sorted by |spearman| descending, filtered by overlap and strength.
    """
    results: list[dict[str, Any]] = []
    for a, b in combinations(columns, 2):
        if _is_trivial_pair(a, b):
            continue
        paired = df[[a, b]].dropna()
        n = len(paired)
        if n < min_overlap:
            continue
        # Spearman = Pearson on ranks; guard against zero-variance columns.
        ra = paired[a].rank()
        rb = paired[b].rank()
        if ra.nunique() < 2 or rb.nunique() < 2:
            continue
        rho = float(ra.corr(rb))
        if math.isnan(rho) or abs(rho) < min_strength:
            continue
        results.append(
            {
                "feature_a": a,
                "feature_b": b,
                "spearman": round(rho, 4),
                "direction": "inverse" if rho < 0 else "direct",
                "sample_size": int(n),
                "plain": _pair_plain(a, b, rho),
            }
        )
    results.sort(key=lambda r: abs(r["spearman"]), reverse=True)
    return results


# ---------------------------------------------------------------------------
# Conditional decile analysis
# ---------------------------------------------------------------------------

def conditional_decile(
    df: pd.DataFrame,
    given: str,
    observed: str,
    decile: float = _DECILE,
    direction: str = "bottom",
) -> dict[str, Any] | None:
    """
    On days where *given* is in its bottom (or top) decile, how often is
    *observed* above its all-history median? Compared against the 50% baseline.

    direction: "bottom" (default) | "top"
    """
    paired = df[[given, observed]].dropna()
    if len(paired) < _MIN_OVERLAP:
        return None

    # If the observed column is essentially constant, the comparison is meaningless.
    if paired[observed].nunique() < 2:
        return None

    n_extreme = max(1, int(round(len(paired) * decile)))
    if direction == "top":
        extreme_days = paired.nlargest(n_extreme, given)
        condition_label = f"top {int(decile * 100)}%"
        direction_word = "highest"
    else:
        extreme_days = paired.nsmallest(n_extreme, given)
        condition_label = f"bottom {int(decile * 100)}%"
        direction_word = "lowest"

    obs_median = paired[observed].median()
    n = len(extreme_days)
    above = int((extreme_days[observed] > obs_median).sum())
    result_pct = round(100.0 * above / n, 1)

    h_given, h_obs = _humanise(given), _humanise(observed)
    plain = (
        f"On the {n} days when {h_given} was in its {direction_word} "
        f"{int(decile * 100)}%, {h_obs} was above its median "
        f"{result_pct}% of the time (vs 50% baseline)."
    )
    return {
        "given": given,
        "condition": condition_label,
        "observed": observed,
        "result_pct": result_pct,
        "baseline_pct": 50.0,
        "sample_size": n,
        "plain": plain,
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(date: str, params: dict[str, Any]) -> dict:
    """
    params (all optional):
        min_overlap (int):   minimum paired observations per pair      [60]
        min_strength (float):minimum |spearman| to keep                [0.4]
        given (str):         focus feature for conditional analysis
                             (defaults to feature_a of strongest pair)
        observed (str):      observed feature for conditional analysis
                             (defaults to feature_b of strongest pair)

    Returns a JSON-serialisable dict. Relationships are descriptive co-movement
    only -- never causation, never a forecast.
    """
    min_overlap = int(params.get("min_overlap", _MIN_OVERLAP))
    min_strength = float(params.get("min_strength", _MIN_STRENGTH))
    date_from: str | None = params.get("date_from")

    hist = history_up_to(date)
    if hist.empty:
        raise ValueError(f"No history available up to {date}")

    if date_from:
        hist = hist[hist["date"] >= pd.Timestamp(date_from)].reset_index(drop=True)
        if hist.empty:
            raise ValueError(f"No history in range {date_from} to {date}")

    data_through = str(hist["date"].iloc[-1].date())
    columns = _candidate_columns(hist)

    ranked = scan_relationships(hist, columns, min_overlap, min_strength)

    primary = ranked[:_TOP_PRIMARY]
    strongest = ranked[:_TOP_STRONGEST]

    # Conditional analysis: default to the strongest pair's features.
    given = params.get("given")
    observed = params.get("observed")
    if (given is None or observed is None) and ranked:
        given = given or ranked[0]["feature_a"]
        observed = observed or ranked[0]["feature_b"]

    conditional = None
    conditional_high = None
    if given and observed and given in hist.columns and observed in hist.columns:
        conditional = conditional_decile(hist, given, observed, direction="bottom")
        conditional_high = conditional_decile(hist, given, observed, direction="top")

    out = {
        "date": str(pd.Timestamp(date).date()),
        "data_through": data_through,
        "primary_relationships": primary,
        "strongest_relationships": strongest,
        "conditional": conditional,
        "conditional_high": conditional_high,
        "note": _NOTE,
    }
    if date_from:
        out["period_date_from"] = date_from
        out["period_sessions"] = len(hist)
    return out
