"""
tests/test_anomaly_scorer.py

Tests for ml/anomaly_scorer.py covering:
  - build_anomaly_labels  (bootstrap logic + feedback overrides)
  - train_and_save        (gates, degradation check, artifact on disk)
  - score                 (output schema, value types, invariants)

Task-specific assertions (per build spec):
  - Bootstrap labels produce at least some anomalies when known sessions
    have >= 2 z-scores above 2.0
  - Feedback labels correctly override bootstrap labels
  - Output schema has all required keys
  - Anomaly score is between 0 and 1
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.anomaly_scorer import (
    FEATURES,
    _ZSCORE_COLS,
    _ZSCORE_MIN_COUNT,
    _ZSCORE_THRESHOLD,
    _cross_val_metrics,
    _passes_degradation_check,
    _passes_gates,
    build_anomaly_labels,
    score,
    train_and_save,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _qse_days(start: str, n: int) -> pd.DatetimeIndex:
    days, d = [], pd.Timestamp(start)
    while len(days) < n:
        if d.dayofweek not in (4, 5):
            days.append(d)
        d += pd.Timedelta(days=1)
    return pd.DatetimeIndex(days)


def _make_features(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """
    Build a feature DataFrame with all FEATURES columns present.
    ~20 % of rows are forced anomalous (>= 2 z-scores above |2.0|) so
    the classifier has a learnable signal and the validation gates pass.
    """
    rng = np.random.default_rng(seed)
    dates = _qse_days("2023-01-01", n)
    df = pd.DataFrame({"date": pd.DatetimeIndex(dates)})

    # Normal z-score features: N(0,1)
    for col in _ZSCORE_COLS:
        df[col] = rng.normal(0, 1, n)

    # Force ~20 % of rows anomalous: set 3 z-scores to ±3
    anomaly_rows = rng.choice(n, size=n // 5, replace=False)
    for row in anomaly_rows:
        for col in _ZSCORE_COLS[:3]:
            df.loc[row, col] = float(rng.choice([-3.5, 3.5]))

    # Non-z-score features
    close = 10_000.0
    prices = []
    for _ in range(n):
        close *= 1 + rng.normal(0, 0.008)
        prices.append(close)

    ret = pd.Series(prices).pct_change(1)
    df["return_1d"] = ret.values
    df["volatility_20d"] = ret.rolling(20, min_periods=10).std().values * np.sqrt(252)

    delta = ret.fillna(0)
    gain = delta.clip(lower=0).rolling(14, min_periods=14).mean()
    loss = (-delta).clip(lower=0).rolling(14, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi_14"] = (100 - (100 / (1 + rs))).values

    df["qse_vs_gcc_spread"] = rng.normal(0, 0.005, n)
    df["foreign_participation"] = rng.uniform(0.2, 0.8, n)
    df["above_sma_200"] = rng.integers(0, 2, n).astype(float)
    df["price_vs_sma20_pct"] = rng.normal(0, 0.02, n)

    return df.reset_index(drop=True)


@pytest.fixture(scope="module")
def feat() -> pd.DataFrame:
    return _make_features(300)


@pytest.fixture(scope="module")
def last_date(feat) -> str:
    return str(feat["date"].iloc[-1].date())


# ---------------------------------------------------------------------------
# Helpers to redirect model I/O to a temp dir
# ---------------------------------------------------------------------------

def _tmp_model_patches(tmp: str):
    """Patch all module-level path constants to point at *tmp*."""
    p = Path(tmp)
    return patch.multiple(
        "ml.anomaly_scorer",
        _MODEL_DIR=p,
        _MODEL_PATH=p / "rf_anomaly_v1.pkl",
        _SYMLINK=p / "rf_anomaly_current",
    )


def _patch_loader(feat: pd.DataFrame, date: str):
    ts = pd.Timestamp(date)

    def _history_up_to(d):
        return feat[feat["date"] <= pd.Timestamp(d)].copy()

    return patch.multiple(
        "ml.anomaly_scorer",
        history_up_to=_history_up_to,
    )


# ===========================================================================
# 1. build_anomaly_labels — bootstrap logic
# ===========================================================================

class TestBuildAnomalyLabels_Bootstrap:

    def test_all_normal_returns_zero(self):
        df = pd.DataFrame({c: [0.5, -0.5, 1.0] for c in _ZSCORE_COLS})
        df["date"] = pd.date_range("2024-01-01", periods=3)
        labels = build_anomaly_labels(df)
        assert (labels == 0).all()

    def test_two_z_scores_above_threshold_returns_one(self):
        """Exactly _ZSCORE_MIN_COUNT columns above threshold -> anomalous."""
        row = {c: 0.0 for c in _ZSCORE_COLS}
        row[_ZSCORE_COLS[0]] = _ZSCORE_THRESHOLD + 0.1
        row[_ZSCORE_COLS[1]] = -(_ZSCORE_THRESHOLD + 0.1)
        df = pd.DataFrame([row])
        df["date"] = pd.Timestamp("2024-01-01")
        assert build_anomaly_labels(df).iloc[0] == 1

    def test_one_z_score_above_threshold_returns_zero(self):
        """Only _ZSCORE_MIN_COUNT - 1 columns -> normal."""
        row = {c: 0.0 for c in _ZSCORE_COLS}
        row[_ZSCORE_COLS[0]] = _ZSCORE_THRESHOLD + 0.5
        df = pd.DataFrame([row])
        df["date"] = pd.Timestamp("2024-01-01")
        assert build_anomaly_labels(df).iloc[0] == 0

    def test_output_is_int_series(self, feat):
        labels = build_anomaly_labels(feat)
        assert labels.dtype in (np.int32, np.int64, int)
        assert set(labels.unique()).issubset({0, 1})

    def test_length_matches_input(self, feat):
        labels = build_anomaly_labels(feat)
        assert len(labels) == len(feat)

    def test_roughly_twenty_percent_anomalous(self, feat):
        """Fixture injects ~20 % anomalies; rate should be between 10 % and 35 %."""
        labels = build_anomaly_labels(feat)
        rate = labels.mean()
        assert 0.10 <= rate <= 0.35, f"Unexpected anomaly rate: {rate:.2%}"


# ===========================================================================
# 2. build_anomaly_labels — feedback overrides
# ===========================================================================

class TestBuildAnomalyLabels_Feedback:

    def _base_df(self):
        row = {c: 0.0 for c in _ZSCORE_COLS}
        df = pd.DataFrame([row, row.copy()])
        df["date"] = [pd.Timestamp("2024-01-07"), pd.Timestamp("2024-01-08")]
        return df

    def test_confirm_overrides_normal_to_one(self):
        df = self._base_df()
        fb = pd.DataFrame({
            "date": [pd.Timestamp("2024-01-07")],
            "label_type": ["anomaly_confirm"],
        })
        labels = build_anomaly_labels(df, fb)
        assert labels.iloc[0] == 1
        assert labels.iloc[1] == 0

    def test_reject_overrides_anomalous_to_zero(self):
        row = {c: _ZSCORE_THRESHOLD + 1 for c in _ZSCORE_COLS}
        df = pd.DataFrame([row])
        df["date"] = pd.Timestamp("2024-01-07")
        fb = pd.DataFrame({
            "date": [pd.Timestamp("2024-01-07")],
            "label_type": ["anomaly_reject"],
        })
        labels = build_anomaly_labels(df, fb)
        assert labels.iloc[0] == 0

    def test_unknown_date_in_feedback_is_ignored(self):
        df = self._base_df()
        fb = pd.DataFrame({
            "date": [pd.Timestamp("2099-01-01")],
            "label_type": ["anomaly_confirm"],
        })
        labels_with = build_anomaly_labels(df, fb)
        labels_without = build_anomaly_labels(df, None)
        pd.testing.assert_series_equal(labels_with, labels_without, check_names=False)

    def test_empty_feedback_same_as_none(self, feat):
        fb_empty = pd.DataFrame(columns=["date", "label_type"])
        labels_none = build_anomaly_labels(feat, None)
        labels_empty = build_anomaly_labels(feat, fb_empty)
        pd.testing.assert_series_equal(labels_none, labels_empty, check_names=False)

    def test_feedback_does_not_mutate_input(self, feat):
        feat_copy = feat.copy()
        fb = pd.DataFrame({"date": [feat["date"].iloc[0]], "label_type": ["anomaly_confirm"]})
        build_anomaly_labels(feat, fb)
        pd.testing.assert_frame_equal(feat, feat_copy)


# ===========================================================================
# 3. Validation helpers
# ===========================================================================

class TestValidationHelpers:

    def test_passes_gates_all_good(self):
        ok, _ = _passes_gates({"precision": 0.70, "recall": 0.65, "auc_roc": 0.75})
        assert ok

    def test_passes_gates_low_precision(self):
        ok, failures = _passes_gates({"precision": 0.60, "recall": 0.65, "auc_roc": 0.75})
        assert not ok
        assert any("precision" in f for f in failures)

    def test_passes_gates_low_recall(self):
        ok, failures = _passes_gates({"precision": 0.70, "recall": 0.55, "auc_roc": 0.75})
        assert not ok
        assert any("recall" in f for f in failures)

    def test_passes_gates_low_auc(self):
        ok, failures = _passes_gates({"precision": 0.70, "recall": 0.65, "auc_roc": 0.68})
        assert not ok
        assert any("auc_roc" in f for f in failures)

    def test_passes_gates_multiple_failures(self):
        ok, failures = _passes_gates({"precision": 0.50, "recall": 0.50, "auc_roc": 0.60})
        assert not ok
        assert len(failures) == 3

    def test_degradation_check_passes_when_no_prior(self):
        ok, _ = _passes_degradation_check(
            {"precision": 0.70, "recall": 0.65, "auc_roc": 0.75}, {}
        )
        assert ok

    def test_degradation_check_fails_at_16_percent(self):
        prior = {"precision": 0.80, "recall": 0.75, "auc_roc": 0.82}
        # 16 % degradation on precision
        new = {"precision": 0.672, "recall": 0.75, "auc_roc": 0.82}
        ok, failures = _passes_degradation_check(new, prior)
        assert not ok
        assert any("precision" in f for f in failures)

    def test_degradation_check_passes_at_14_percent(self):
        prior = {"precision": 0.80, "recall": 0.75, "auc_roc": 0.82}
        new = {"precision": 0.688, "recall": 0.75, "auc_roc": 0.82}
        ok, _ = _passes_degradation_check(new, prior)
        assert ok

    def test_degradation_check_zero_prior_skips(self):
        """Zero prior metric should not cause division by zero."""
        ok, _ = _passes_degradation_check(
            {"precision": 0.0, "recall": 0.0, "auc_roc": 0.0},
            {"precision": 0.0, "recall": 0.0, "auc_roc": 0.0},
        )
        assert ok


# ===========================================================================
# 4. train_and_save
# ===========================================================================

class TestTrainAndSave:

    def test_saves_artifact_to_disk(self, feat):
        with tempfile.TemporaryDirectory() as tmp:
            with _tmp_model_patches(tmp):
                path = train_and_save(feat)
                assert Path(path).exists()

    def test_raises_on_single_class(self):
        """If all labels are 0, training must fail with a clear error."""
        df = _make_features(50, seed=7)
        # Force all z-scores below threshold so bootstrap produces only zeros
        for col in _ZSCORE_COLS:
            df[col] = 0.5
        with tempfile.TemporaryDirectory() as tmp:
            with _tmp_model_patches(tmp):
                with pytest.raises(ValueError, match="one class"):
                    train_and_save(df)

    def test_artifact_contains_expected_keys(self, feat):
        import joblib
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            with _tmp_model_patches(tmp):
                train_and_save(feat)
            artifact = joblib.load(p / "rf_anomaly_v1.pkl")
        assert "model" in artifact
        assert "meta" in artifact
        meta = artifact["meta"]
        for key in ("cv_metrics", "feature_names", "n_train", "n_anomalous", "version"):
            assert key in meta, f"Missing meta key: {key}"

    def test_meta_feature_names_match_module(self, feat):
        import joblib
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            with _tmp_model_patches(tmp):
                train_and_save(feat)
            artifact = joblib.load(p / "rf_anomaly_v1.pkl")
        assert artifact["meta"]["feature_names"] == FEATURES

    def test_degradation_check_blocks_worse_model(self, feat):
        """train_and_save must raise when _passes_degradation_check fails,
        and must call it with (new_metrics, prior_cv_metrics)."""
        import joblib
        from unittest.mock import MagicMock
        prior_cv = {"precision": 0.80, "recall": 0.75, "auc_roc": 0.82}
        fake_prior_meta = {
            "cv_metrics": prior_cv,
            "feature_names": FEATURES,
            "n_train": 300,
            "n_anomalous": 60,
            "version": "rf_anomaly_v1",
        }
        from sklearn.ensemble import RandomForestClassifier as RFC
        dummy_model = RFC(n_estimators=1, random_state=0)
        labels = build_anomaly_labels(feat).values
        X = feat[FEATURES].fillna(0).values
        dummy_model.fit(X, labels)

        mock_check = MagicMock(
            return_value=(False, ["precision degraded by 20.0% (prior=0.800, new=0.640)"])
        )

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            p.mkdir(exist_ok=True)
            joblib.dump({"model": dummy_model, "meta": fake_prior_meta}, p / "rf_anomaly_v1.pkl")
            with _tmp_model_patches(tmp), \
                 patch("ml.anomaly_scorer._passes_degradation_check", mock_check):
                with pytest.raises(ValueError, match="degraded"):
                    train_and_save(feat)

        # Verify it was called once with the prior cv_metrics as the second arg
        mock_check.assert_called_once()
        _, call_prior = mock_check.call_args.args
        assert call_prior == prior_cv

    def test_feedback_overrides_used_in_training(self, feat):
        """Providing feedback should not crash training."""
        fb = pd.DataFrame({
            "date": feat["date"].iloc[:5].values,
            "label_type": ["anomaly_confirm"] * 5,
        })
        with tempfile.TemporaryDirectory() as tmp:
            with _tmp_model_patches(tmp):
                path = train_and_save(feat, fb)
                # Check existence inside the context manager while tmp dir is live
                assert Path(path).exists()


# ===========================================================================
# 5. score — output schema and invariants
# ===========================================================================

class TestScore:

    def _run(self, feat, last_date):
        with tempfile.TemporaryDirectory() as tmp:
            with _tmp_model_patches(tmp), _patch_loader(feat, last_date):
                return score(last_date)

    def test_required_keys_present(self, feat, last_date):
        r = self._run(feat, last_date)
        required = {
            "date", "anomaly_score", "anomaly_label", "confidence",
            "top_contributing_features", "model_version", "bootstrap_label_used",
        }
        missing = required - r.keys()
        assert not missing, f"Missing keys: {missing}"

    def test_date_matches_input(self, feat, last_date):
        r = self._run(feat, last_date)
        assert r["date"] == last_date

    def test_anomaly_score_in_unit_interval(self, feat, last_date):
        r = self._run(feat, last_date)
        assert isinstance(r["anomaly_score"], float)
        assert 0.0 <= r["anomaly_score"] <= 1.0

    def test_anomaly_label_is_binary(self, feat, last_date):
        r = self._run(feat, last_date)
        assert r["anomaly_label"] in (0, 1)

    def test_label_consistent_with_score(self, feat, last_date):
        r = self._run(feat, last_date)
        if r["anomaly_score"] >= 0.5:
            assert r["anomaly_label"] == 1
        else:
            assert r["anomaly_label"] == 0

    def test_confidence_in_unit_interval(self, feat, last_date):
        r = self._run(feat, last_date)
        assert isinstance(r["confidence"], float)
        assert 0.0 <= r["confidence"] <= 1.0

    def test_top_contributing_features_is_list(self, feat, last_date):
        r = self._run(feat, last_date)
        assert isinstance(r["top_contributing_features"], list)
        assert len(r["top_contributing_features"]) > 0

    def test_top_feature_entry_schema(self, feat, last_date):
        r = self._run(feat, last_date)
        for entry in r["top_contributing_features"]:
            assert "feature" in entry
            assert "importance" in entry
            assert "session_value" in entry
            assert entry["feature"] in FEATURES

    def test_top_feature_importances_sum_leq_one(self, feat, last_date):
        r = self._run(feat, last_date)
        total = sum(e["importance"] for e in r["top_contributing_features"])
        # Top-5 importances out of 13 features; total must be <= 1.0
        assert total <= 1.0 + 1e-9

    def test_model_version_is_string(self, feat, last_date):
        r = self._run(feat, last_date)
        assert isinstance(r["model_version"], str)

    def test_bootstrap_label_is_binary(self, feat, last_date):
        r = self._run(feat, last_date)
        assert r["bootstrap_label_used"] in (0, 1)

    def test_missing_date_raises(self, feat):
        with tempfile.TemporaryDirectory() as tmp:
            with _tmp_model_patches(tmp), _patch_loader(feat, "2099-01-01"):
                with pytest.raises(KeyError):
                    score("2099-01-01")

    def test_nan_features_produce_warning(self, feat, last_date):
        """A row where all features are NaN should still return a result with a warning."""
        nan_feat = feat.copy()
        last_ts = pd.Timestamp(last_date)
        mask = nan_feat["date"] == last_ts
        for col in FEATURES:
            if col in nan_feat.columns:
                nan_feat.loc[mask, col] = np.nan

        with tempfile.TemporaryDirectory() as tmp:
            with _tmp_model_patches(tmp), patch("ml.anomaly_scorer.history_up_to",
                    side_effect=lambda d: nan_feat[nan_feat["date"] <= pd.Timestamp(d)].copy()):
                r = score(last_date)
        assert "warning" in r


# ===========================================================================
# 6. Cross-val metrics helper
# ===========================================================================

class TestCrossValMetrics:

    def test_returns_three_metric_keys(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(100, 5))
        y = (X[:, 0] > 0.5).astype(int)
        metrics = _cross_val_metrics(X, y)
        assert set(metrics.keys()) == {"precision", "recall", "auc_roc"}

    def test_metrics_in_unit_interval(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(80, 4))
        y = (X[:, 0] > 0).astype(int)
        metrics = _cross_val_metrics(X, y)
        for key, val in metrics.items():
            assert 0.0 <= val <= 1.0, f"{key}={val} out of [0,1]"

    def test_sparse_positives_does_not_crash(self):
        """Only 1 positive should not crash (falls back to train-on-all path)."""
        X = np.zeros((20, 3))
        y = np.zeros(20, dtype=int)
        y[0] = 1
        metrics = _cross_val_metrics(X, y)
        assert isinstance(metrics["precision"], float)


# ===========================================================================
# 7. score — auto-train-on-first-call path
# ===========================================================================

class TestScore_AutoTrain:
    """Verify score() trains a model from scratch when no artifact exists."""

    def test_score_trains_on_first_call_and_returns_valid_result(self, feat, last_date):
        """Fresh temp dir (no pkl) — score() must auto-train and still return a dict."""
        with tempfile.TemporaryDirectory() as tmp:
            with _tmp_model_patches(tmp), _patch_loader(feat, last_date):
                # Confirm no artifact exists before the call
                assert not (Path(tmp) / "rf_anomaly_v1.pkl").exists()
                r = score(last_date)
                # Artifact must now exist
                assert (Path(tmp) / "rf_anomaly_v1.pkl").exists()

        required = {
            "date", "anomaly_score", "anomaly_label", "confidence",
            "top_contributing_features", "model_version", "bootstrap_label_used",
        }
        assert required <= r.keys()
        assert 0.0 <= r["anomaly_score"] <= 1.0

    def test_score_raises_runtime_error_when_training_fails(self, feat, last_date):
        """When auto-training raises ValueError, score() must re-raise as RuntimeError."""
        with tempfile.TemporaryDirectory() as tmp:
            with _tmp_model_patches(tmp), _patch_loader(feat, last_date), \
                 patch("ml.anomaly_scorer.train_and_save",
                       side_effect=ValueError("Label vector has only one class")):
                with pytest.raises(RuntimeError, match="auto-training failed"):
                    score(last_date)
