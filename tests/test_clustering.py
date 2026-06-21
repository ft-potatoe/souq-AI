"""
Tests for ml/clustering.py — HDBSCAN market-state discovery.

Training/inference tests run against the live features_master parquet (the model
auto-trains on first call, mirroring the other ML modules' test style). Pure-logic
tests (label assignment, min-session guard) use synthetic frames.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.clustering import (
    train_and_save,
    cluster,
    _load_artifact,
    _label_for_profile,
    _FEATURES,
    _MIN_SESSIONS,
)
from analytics._loader import load_features


# Silence the sklearn HDBSCAN `copy` FutureWarning during the test session.
warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# Training + gate
# ---------------------------------------------------------------------------

def test_train_and_save_produces_at_least_two_clusters():
    train_and_save()  # gate raises if it fails
    scaler, model, meta = _load_artifact()
    assert meta["n_clusters"] >= 2
    assert meta["silhouette"] >= 0.20
    assert meta["noise_fraction"] <= 0.40


def test_artifact_round_trips():
    train_and_save()
    loaded = _load_artifact()
    assert loaded is not None
    scaler, model, meta = loaded
    assert set(meta["feature_names"]) == set(_FEATURES)
    assert "cluster_profiles" in meta


def test_train_gate_raises_on_degenerate_single_cluster():
    # A frame with effectively one tight blob -> HDBSCAN cannot form >=2 clusters.
    n = 400
    rng = np.random.default_rng(0)
    data = {f: rng.normal(0, 1e-6, n) for f in _FEATURES}
    df = pd.DataFrame(data)
    df["date"] = pd.date_range("2020-01-01", periods=n, freq="D")
    with pytest.raises(ValueError):
        train_and_save(df)


def test_train_raises_below_min_sessions():
    n = _MIN_SESSIONS - 10
    rng = np.random.default_rng(1)
    data = {f: rng.normal(0, 1, n) for f in _FEATURES}
    df = pd.DataFrame(data)
    df["date"] = pd.date_range("2020-01-01", periods=n, freq="D")
    with pytest.raises(ValueError):
        train_and_save(df)


# ---------------------------------------------------------------------------
# cluster() inference
# ---------------------------------------------------------------------------

def test_cluster_assigns_valid_id_and_label():
    train_and_save()
    res = cluster("2026-06-19", {})
    assert res["current_cluster_id"] is not None
    assert isinstance(res["current_cluster_label"], str)
    assert res["all_clusters"]  # non-empty
    assert res["note"]


def test_cluster_output_is_json_serialisable():
    import json
    train_and_save()
    json.dumps(cluster("2026-06-19", {}))


def test_cluster_below_min_sessions_returns_none_with_note():
    # Use a very early date so history_up_to has < 250 rows.
    df = load_features()
    early_date = str(df["date"].iloc[50].date())
    res = cluster(early_date, {})
    assert res["current_cluster_id"] is None
    assert res["note"] and "least" in res["note"]


# ---------------------------------------------------------------------------
# Label assignment determinism
# ---------------------------------------------------------------------------

def test_label_for_profile_is_deterministic_and_descriptive():
    z = {"volatility_20d": 1.0, "return_1d": -0.8, "breadth_ratio": -0.9}
    label = _label_for_profile(z)
    assert label == _label_for_profile(z)  # deterministic
    assert "High-vol" in label
    assert "selloff" in label


def test_label_for_profile_handles_flat_profile():
    z = {f: 0.0 for f in _FEATURES}
    assert _label_for_profile(z)  # non-empty fallback label
