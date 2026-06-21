"""
Tests for analytics/relationships.py — automated relationship discovery.

Most tests drive the pure helpers (scan_relationships, conditional_decile,
_is_trivial_pair) with synthetic data so the assertions are deterministic and
independent of the live features_master parquet.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analytics.relationships import (
    scan_relationships,
    conditional_decile,
    _is_trivial_pair,
    _candidate_columns,
    run,
    _MIN_OVERLAP,
)


# ---------------------------------------------------------------------------
# scan_relationships
# ---------------------------------------------------------------------------

def test_inverse_relationship_has_negative_spearman():
    n = 200
    rng = np.random.default_rng(0)
    a = pd.Series(np.linspace(0, 1, n))
    b = pd.Series(-np.linspace(0, 1, n) + rng.normal(0, 0.01, n))  # strong inverse
    df = pd.DataFrame({"alpha": a, "beta": b})
    res = scan_relationships(df, ["alpha", "beta"], min_overlap=_MIN_OVERLAP, min_strength=0.4)
    assert len(res) == 1
    rel = res[0]
    assert rel["spearman"] < 0
    assert rel["direction"] == "inverse"
    assert "falls" in rel["plain"] and "rise" in rel["plain"]


def test_direct_relationship_has_positive_spearman():
    n = 150
    a = pd.Series(np.arange(n, dtype=float))
    df = pd.DataFrame({"alpha": a, "beta": a * 2.0})  # perfectly direct
    res = scan_relationships(df, ["alpha", "beta"], min_overlap=_MIN_OVERLAP, min_strength=0.4)
    assert res[0]["direction"] == "direct"
    assert res[0]["spearman"] > 0


def test_min_strength_filters_weak_pairs():
    n = 300
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "alpha": rng.normal(0, 1, n),
        "beta": rng.normal(0, 1, n),  # independent -> near-zero correlation
    })
    res = scan_relationships(df, ["alpha", "beta"], min_overlap=_MIN_OVERLAP, min_strength=0.4)
    assert res == []


def test_min_overlap_filters_short_pairs():
    n = 30  # below default _MIN_OVERLAP=60
    a = pd.Series(np.arange(n, dtype=float))
    df = pd.DataFrame({"alpha": a, "beta": a})
    res = scan_relationships(df, ["alpha", "beta"], min_overlap=_MIN_OVERLAP, min_strength=0.4)
    assert res == []


def test_results_sorted_by_absolute_strength():
    n = 200
    base = pd.Series(np.linspace(0, 1, n))
    rng = np.random.default_rng(2)
    df = pd.DataFrame({
        "x": base,
        "strong": -base + rng.normal(0, 0.005, n),   # ~ -1
        "weaker": base * 0.5 + rng.normal(0, 0.25, n),  # weaker positive
    })
    res = scan_relationships(df, ["x", "strong", "weaker"], min_overlap=_MIN_OVERLAP, min_strength=0.4)
    strengths = [abs(r["spearman"]) for r in res]
    assert strengths == sorted(strengths, reverse=True)


# ---------------------------------------------------------------------------
# _is_trivial_pair
# ---------------------------------------------------------------------------

def test_trivial_pair_detects_zscore_transform():
    assert _is_trivial_pair("volume", "volume_zscore") is True


def test_trivial_pair_detects_window_variants():
    assert _is_trivial_pair("volatility_20d", "volatility_60d") is True


def test_trivial_pair_allows_genuine_pair():
    assert _is_trivial_pair("return_1d", "breadth_ratio") is False


# ---------------------------------------------------------------------------
# conditional_decile
# ---------------------------------------------------------------------------

def test_conditional_decile_detects_inverse_pattern():
    # Construct: when 'driver' is low, 'observed' is high (perfect inverse).
    n = 200
    driver = pd.Series(np.linspace(0, 1, n))
    observed = pd.Series(-np.linspace(0, 1, n))
    df = pd.DataFrame({"driver": driver, "observed": observed})
    cond = conditional_decile(df, "driver", "observed", decile=0.10)
    assert cond is not None
    # Bottom decile of driver => observed is at its highest => above median ~100%.
    assert cond["result_pct"] >= 90.0
    assert cond["baseline_pct"] == 50.0
    assert cond["sample_size"] == int(round(n * 0.10))


def test_conditional_decile_none_when_observed_constant():
    n = 100
    df = pd.DataFrame({"driver": np.arange(n, dtype=float), "observed": np.ones(n)})
    assert conditional_decile(df, "driver", "observed") is None


def test_conditional_decile_none_when_too_few_rows():
    df = pd.DataFrame({"driver": [1.0, 2.0, 3.0], "observed": [3.0, 2.0, 1.0]})
    assert conditional_decile(df, "driver", "observed") is None


# ---------------------------------------------------------------------------
# run() — integration against the live features_master
# ---------------------------------------------------------------------------

def test_run_excludes_forward_returns():
    # Forward-return columns must never appear in any discovered relationship.
    res = run("2026-06-19", {})
    for rel in res["strongest_relationships"]:
        assert "forward_return" not in rel["feature_a"]
        assert "forward_return" not in rel["feature_b"]


def test_run_output_is_json_serialisable_with_note():
    import json
    res = run("2026-06-19", {})
    assert res["note"]  # non-empty associations-not-causation note
    assert "primary_relationships" in res
    json.dumps(res)  # must not raise


def test_candidate_columns_drops_forward_returns_and_calendar():
    from analytics._loader import load_features
    cols = _candidate_columns(load_features())
    assert "forward_return_5d" not in cols
    assert "day_of_week" not in cols
    assert "month" not in cols
