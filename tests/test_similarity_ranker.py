"""
tests/test_similarity_ranker.py

Tests for ml/similarity_ranker.py covering:

  safe_forward_returns:
    - Returns (None, None) when candidate is within 10 QSE trading days of today
    - Returns real values when candidate is 15+ trading days before today
    - Returns (None, None) when candidate is on today or in the future
    - Boundary: candidate at exactly day 10 is still masked

  rank() pipeline:
    - Top-10 results are a subset of the original top-40 k-NN candidates
    - No future candidate (date > query date) appears in results
    - Output schema has all required keys with correct types
    - forward_return_5d/10d in matches are None when candidate is recent
    - knn_candidates_retrieved <= _KNN_CANDIDATES

  build_similarity_labels:
    - Cold-start produces all 0.5 when no feedback
    - Feedback ratings are normalised correctly (1->0.0, 3->0.5, 5->1.0)
    - Non-similarity feedback rows are ignored

  Validation helpers:
    - _passes_gates fails below 0.70
    - _passes_degradation fails at >10% relative drop

  train_and_save:
    - Artifact saved to disk with expected keys
    - Raises when fewer sessions than minimum
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

from ml.similarity_ranker import (
    KNN_FEATURES,
    PAIRWISE_FEATURES,
    _KNN_CANDIDATES,
    _LEAKAGE_WINDOW_DAYS,
    _TOP_K,
    _passes_degradation,
    _passes_gates,
    build_similarity_labels,
    rank,
    safe_forward_returns,
    train_and_save,
)


# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------

def _qse_days(start: str, n: int) -> pd.DatetimeIndex:
    days, d = [], pd.Timestamp(start)
    while len(days) < n:
        if d.dayofweek not in (4, 5):   # skip Fri=4, Sat=5
            days.append(d)
        d += pd.Timedelta(days=1)
    return pd.DatetimeIndex(days)


def _make_features(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """
    Synthetic feature DataFrame with all KNN_FEATURES columns and the
    forward-return / regime columns the ranker expects.
    n must be large enough to exceed _KNN_CANDIDATES + a holdout margin (>=100).
    """
    rng = np.random.default_rng(seed)
    dates = _qse_days("2022-01-02", n)
    df = pd.DataFrame({"date": pd.DatetimeIndex(dates)})

    # k-NN embedding features — uncorrelated noise is fine for unit tests
    df["return_1d"] = rng.normal(0, 0.01, n)
    df["volume_zscore"] = rng.normal(0, 1, n)
    df["foreign_flow_zscore"] = rng.normal(0, 1, n)
    df["breadth_ratio"] = rng.uniform(0.3, 0.7, n)
    df["volatility_20d"] = rng.uniform(0.005, 0.025, n)
    df["rsi_14"] = rng.uniform(30, 70, n)

    # Forward returns — deterministic so leakage guard tests are predictable
    df["forward_return_5d"] = rng.normal(0, 0.01, n)
    df["forward_return_10d"] = rng.normal(0, 0.015, n)
    df["forward_return_20d"] = rng.normal(0, 0.02, n)

    # Regime labels (bear/sideways/bull)
    regimes = ["bear", "sideways", "bull"]
    df["regime"] = [regimes[i % 3] for i in range(n)]

    return df.reset_index(drop=True)


@pytest.fixture(scope="module")
def feat() -> pd.DataFrame:
    return _make_features(300)


@pytest.fixture(scope="module")
def query_date(feat) -> str:
    # Use the last date so all earlier sessions are candidates
    return str(feat["date"].iloc[-1].date())


# ---------------------------------------------------------------------------
# Path-patching helpers (mirror anomaly_scorer test pattern)
# ---------------------------------------------------------------------------

def _tmp_model_patches(tmp: str):
    p = Path(tmp)
    return patch.multiple(
        "ml.similarity_ranker",
        _MODEL_DIR=p,
        _MODEL_PATH=p / "xgb_ranker_v1.pkl",
        _SYMLINK=p / "xgb_ranker_current",
    )


def _patch_loader(feat: pd.DataFrame):
    def _history_up_to(d):
        return feat[feat["date"] <= pd.Timestamp(d)].copy()

    return patch("ml.similarity_ranker.history_up_to", side_effect=_history_up_to)


# Module-scoped trained artifact so TestRank tests load rather than retrain.
# Training is expensive (XGBRanker x 300 sessions x many query loops), so we
# pay the cost once per module and share the temp dir for the lifetime of the
# test session.

@pytest.fixture(scope="module", autouse=False)
def trained_artifact(feat) -> str:
    """
    Train once, yield the temp dir path.  TestRank._run() patches model paths
    to this directory so every rank() call loads the existing artifact.
    """
    with tempfile.TemporaryDirectory() as tmp:
        with _tmp_model_patches(tmp):
            train_and_save(feat)
        yield tmp


# ===========================================================================
# 1. safe_forward_returns — leakage guard
# ===========================================================================

class TestSafeForwardReturns:

    # Reference anchor: a fixed "today" well inside a trading week
    TODAY = pd.Timestamp("2024-11-14")   # Thursday

    # -- within the leakage window ------------------------------------------

    def test_same_day_returns_none_none(self):
        r = safe_forward_returns(self.TODAY, self.TODAY, 0.05, 0.08)
        assert r == (None, None)

    def test_one_day_before_returns_none_none(self):
        r = safe_forward_returns(self.TODAY - pd.Timedelta(days=1), self.TODAY, 0.01, 0.02)
        assert r == (None, None)

    def test_five_trading_days_before_returns_none_none(self):
        # 7 calendar days back lands on a Thu, which is within the 10-day window
        candidate = pd.Timestamp("2024-11-07")   # Thu, 5 QSE trading days before
        r = safe_forward_returns(candidate, self.TODAY, 0.05, 0.08)
        assert r == (None, None)

    def test_exactly_ten_trading_days_is_still_masked(self):
        # Thu 2024-10-31 is exactly 10 QSE trading days before Thu 2024-11-14
        # (Sun-Thu calendar: Oct31,Nov3,4,5,6,7,10,11,12,13,14 = 10 gaps)
        candidate = pd.Timestamp("2024-10-31")
        r = safe_forward_returns(candidate, self.TODAY, 0.03, 0.06)
        assert r == (None, None), (
            "Candidate exactly at the 10-trading-day boundary must still be masked"
        )

    def test_future_candidate_returns_none_none(self):
        future = self.TODAY + pd.Timedelta(days=5)
        r = safe_forward_returns(future, self.TODAY, 0.01, 0.02)
        assert r == (None, None)

    # -- outside the leakage window -----------------------------------------

    def test_fifteen_trading_days_before_returns_values(self):
        # 21 calendar days back: 2024-10-24 is Thu — count from 2024-11-14:
        # Nov 13(W), 12(T), 11(M), 10(Su), 7(Th), 6(W), 5(T), 4(M), 3(Su),
        # Oct 31(Th), 30(W), 29(T), 28(M), 27(Su), 24(Th) = 15 trading days
        candidate = pd.Timestamp("2024-10-24")
        r = safe_forward_returns(candidate, self.TODAY, 0.05, 0.08)
        assert r == (0.05, 0.08), f"Expected real values, got {r}"

    def test_thirty_trading_days_before_returns_values(self):
        candidate = pd.Timestamp("2024-10-03")   # well outside any window
        r = safe_forward_returns(candidate, self.TODAY, -0.02, 0.03)
        assert r == (-0.02, 0.03)

    def test_none_inputs_pass_through_when_outside_window(self):
        candidate = pd.Timestamp("2024-10-03")
        r = safe_forward_returns(candidate, self.TODAY, None, None)
        assert r == (None, None)

    def test_zero_fwd_returns_are_not_confused_with_none(self):
        candidate = pd.Timestamp("2024-10-03")
        r = safe_forward_returns(candidate, self.TODAY, 0.0, 0.0)
        assert r == (0.0, 0.0)

    def test_return_types_are_float_or_none(self):
        candidate = pd.Timestamp("2024-10-03")
        a, b = safe_forward_returns(candidate, self.TODAY, 0.05, 0.08)
        assert isinstance(a, float) and isinstance(b, float)

    def test_accepts_string_dates(self):
        r = safe_forward_returns("2024-10-03", "2024-11-14", 0.01, 0.02)
        assert r == (0.01, 0.02)


# ===========================================================================
# 2. build_similarity_labels
# ===========================================================================

class TestBuildSimilarityLabels:

    def test_cold_start_all_half(self, feat):
        labels = build_similarity_labels(feat)
        assert (labels == 0.5).all()

    def test_length_matches_input(self, feat):
        labels = build_similarity_labels(feat)
        assert len(labels) == len(feat)

    def test_feedback_rating_1_maps_to_zero(self, feat):
        fb = pd.DataFrame({
            "date": [feat["date"].iloc[10]],
            "feedback_type": ["similarity_rating"],
            "rating": [1],
        })
        labels = build_similarity_labels(feat, fb)
        assert labels.iloc[10] == pytest.approx(0.0)

    def test_feedback_rating_3_maps_to_half(self, feat):
        fb = pd.DataFrame({
            "date": [feat["date"].iloc[20]],
            "feedback_type": ["similarity_rating"],
            "rating": [3],
        })
        labels = build_similarity_labels(feat, fb)
        assert labels.iloc[20] == pytest.approx(0.5)

    def test_feedback_rating_5_maps_to_one(self, feat):
        fb = pd.DataFrame({
            "date": [feat["date"].iloc[30]],
            "feedback_type": ["similarity_rating"],
            "rating": [5],
        })
        labels = build_similarity_labels(feat, fb)
        assert labels.iloc[30] == pytest.approx(1.0)

    def test_non_similarity_feedback_ignored(self, feat):
        fb = pd.DataFrame({
            "date": [feat["date"].iloc[5]],
            "feedback_type": ["anomaly_confirm"],
            "rating": [5],
        })
        labels_with = build_similarity_labels(feat, fb)
        labels_without = build_similarity_labels(feat, None)
        pd.testing.assert_series_equal(labels_with, labels_without, check_names=False)

    def test_unknown_date_in_feedback_ignored(self, feat):
        fb = pd.DataFrame({
            "date": [pd.Timestamp("2099-01-01")],
            "feedback_type": ["similarity_rating"],
            "rating": [5],
        })
        labels = build_similarity_labels(feat, fb)
        assert (labels == 0.5).all()

    def test_empty_feedback_same_as_none(self, feat):
        fb_empty = pd.DataFrame(columns=["date", "feedback_type", "rating"])
        l1 = build_similarity_labels(feat, None)
        l2 = build_similarity_labels(feat, fb_empty)
        pd.testing.assert_series_equal(l1, l2, check_names=False)

    def test_two_feedback_rows_applied_independently(self, feat):
        """Two feedback entries on different dates must not corrupt each other,
        and all other rows must remain at the cold-start value of 0.5."""
        fb = pd.DataFrame({
            "date": [feat["date"].iloc[10], feat["date"].iloc[50]],
            "feedback_type": ["similarity_rating", "similarity_rating"],
            "rating": [1, 5],
        })
        labels = build_similarity_labels(feat, fb)
        assert labels.iloc[10] == pytest.approx(0.0), "Rating 1 should map to 0.0"
        assert labels.iloc[50] == pytest.approx(1.0), "Rating 5 should map to 1.0"
        # Every other row must be untouched
        mask = pd.Series(True, index=feat.index)
        mask.iloc[10] = False
        mask.iloc[50] = False
        assert (labels[mask] == 0.5).all(), "Unrated rows must remain 0.5"


# ===========================================================================
# 3. Validation helpers
# ===========================================================================

class TestValidationHelpers:

    def test_passes_gates_at_threshold(self):
        ok, _ = _passes_gates(0.70)
        assert ok

    def test_passes_gates_above_threshold(self):
        ok, _ = _passes_gates(0.85)
        assert ok

    def test_fails_gates_below_threshold(self):
        ok, failures = _passes_gates(0.65)
        assert not ok
        assert any("NDCG" in f for f in failures)

    def test_passes_degradation_within_limit(self):
        ok, _ = _passes_degradation(0.72, 0.75)  # 4 % drop, under 10 %
        assert ok

    def test_fails_degradation_above_limit(self):
        ok, failures = _passes_degradation(0.63, 0.75)  # 16 % drop
        assert not ok
        assert any("NDCG" in f for f in failures)

    def test_degradation_zero_prior_skips(self):
        ok, _ = _passes_degradation(0.0, 0.0)
        assert ok


# ===========================================================================
# 4. train_and_save
# ===========================================================================

@pytest.fixture(scope="module")
def _trained_artifact_no_feedback(feat):
    """One training run shared by the artifact-inspection tests."""
    import joblib
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp)
        with _tmp_model_patches(tmp):
            train_and_save(feat, None)
        yield joblib.load(p / "xgb_ranker_v1.pkl")


class TestTrainAndSave:

    def test_saves_artifact_to_disk(self, feat):
        with tempfile.TemporaryDirectory() as tmp:
            with _tmp_model_patches(tmp):
                path = train_and_save(feat)
            assert Path(path).exists()

    def test_artifact_has_expected_keys(self, _trained_artifact_no_feedback):
        artifact = _trained_artifact_no_feedback
        assert {"scaler", "model", "meta"} <= artifact.keys()
        for key in ("ndcg_at_10", "knn_features", "pairwise_features",
                    "n_train_pairs", "n_query_sessions", "version"):
            assert key in artifact["meta"], f"Missing meta key: {key}"

    def test_meta_feature_lists_match_module_constants(self, _trained_artifact_no_feedback):
        meta = _trained_artifact_no_feedback["meta"]
        assert meta["knn_features"] == KNN_FEATURES
        assert meta["pairwise_features"] == PAIRWISE_FEATURES

    def test_raises_when_too_few_sessions(self):
        tiny = _make_features(n=20)
        with tempfile.TemporaryDirectory() as tmp:
            with _tmp_model_patches(tmp):
                with pytest.raises(ValueError, match="at least"):
                    train_and_save(tiny)

    def test_feedback_informed_false_without_feedback(self, _trained_artifact_no_feedback):
        assert _trained_artifact_no_feedback["meta"]["feedback_informed"] is False

    def test_feedback_informed_flag_set_when_feedback_provided(self, feat):
        import joblib
        fb = pd.DataFrame({
            "date": feat["date"].iloc[:3].values,
            "feedback_type": ["similarity_rating"] * 3,
            "rating": [4, 2, 5],
        })
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            with _tmp_model_patches(tmp):
                train_and_save(feat, fb)
            artifact = joblib.load(p / "xgb_ranker_v1.pkl")
        assert artifact["meta"]["feedback_informed"] is True


# ===========================================================================
# 5. rank() — output schema and pipeline invariants
# ===========================================================================

class TestRank:
    """
    All tests share one trained artifact via `trained_artifact` so training
    runs once per module (~10 s) instead of once per test (~10 min total).
    Tests that need a genuinely empty dir (auto-train, error paths) bring
    their own tempfile.TemporaryDirectory.
    """

    _QBDAY = pd.offsets.CustomBusinessDay(weekmask="Sun Mon Tue Wed Thu")

    def _run(self, feat: pd.DataFrame, date: str, trained_artifact: str) -> dict:
        with _tmp_model_patches(trained_artifact), _patch_loader(feat):
            return rank(date)

    # -- output schema -------------------------------------------------------

    def test_required_top_level_keys(self, feat, query_date, trained_artifact):
        r = self._run(feat, query_date, trained_artifact)
        required = {
            "date", "query_regime", "top_matches",
            "forward_return_stats", "model_version",
            "feedback_informed", "knn_candidates_retrieved",
        }
        assert required <= r.keys(), f"Missing: {required - r.keys()}"

    def test_date_matches_input(self, feat, query_date, trained_artifact):
        r = self._run(feat, query_date, trained_artifact)
        assert r["date"] == query_date

    def test_top_matches_is_list(self, feat, query_date, trained_artifact):
        r = self._run(feat, query_date, trained_artifact)
        assert isinstance(r["top_matches"], list)

    def test_top_matches_count_at_most_top_k(self, feat, query_date, trained_artifact):
        r = self._run(feat, query_date, trained_artifact)
        assert len(r["top_matches"]) <= _TOP_K

    def test_match_entry_schema(self, feat, query_date, trained_artifact):
        r = self._run(feat, query_date, trained_artifact)
        required_match_keys = {
            "rank", "date", "ranker_score", "knn_cosine_score",
            "regime", "regime_match", "forward_return_5d", "forward_return_10d",
        }
        for m in r["top_matches"]:
            missing = required_match_keys - m.keys()
            assert not missing, f"Match entry missing keys: {missing}"

    def test_ranks_are_sequential_from_one(self, feat, query_date, trained_artifact):
        r = self._run(feat, query_date, trained_artifact)
        ranks = [m["rank"] for m in r["top_matches"]]
        assert ranks == list(range(1, len(ranks) + 1))

    def test_ranker_scores_are_floats(self, feat, query_date, trained_artifact):
        r = self._run(feat, query_date, trained_artifact)
        for m in r["top_matches"]:
            assert isinstance(m["ranker_score"], float)

    def test_knn_cosine_scores_in_unit_interval(self, feat, query_date, trained_artifact):
        r = self._run(feat, query_date, trained_artifact)
        for m in r["top_matches"]:
            assert 0.0 <= m["knn_cosine_score"] <= 1.0, (
                f"cosine score {m['knn_cosine_score']} outside [0,1]"
            )

    def test_regime_match_is_binary(self, feat, query_date, trained_artifact):
        r = self._run(feat, query_date, trained_artifact)
        for m in r["top_matches"]:
            assert m["regime_match"] in (0, 1)

    def test_model_version_is_string(self, feat, query_date, trained_artifact):
        r = self._run(feat, query_date, trained_artifact)
        assert isinstance(r["model_version"], str)

    def test_feedback_informed_is_bool(self, feat, query_date, trained_artifact):
        r = self._run(feat, query_date, trained_artifact)
        assert isinstance(r["feedback_informed"], bool)

    def test_forward_return_stats_keys(self, feat, query_date, trained_artifact):
        r = self._run(feat, query_date, trained_artifact)
        fwd = r["forward_return_stats"]
        assert {"forward_return_5d", "forward_return_10d", "forward_return_20d"} <= fwd.keys()

    def test_forward_return_stats_structure_when_present(self, feat, query_date, trained_artifact):
        r = self._run(feat, query_date, trained_artifact)
        for window_key in ("forward_return_5d", "forward_return_10d", "forward_return_20d"):
            stats = r["forward_return_stats"][window_key]
            if stats is not None:
                assert {"mean", "median", "n"} <= stats.keys()
                assert isinstance(stats["n"], int) and stats["n"] > 0

    def test_forward_return_stats_non_none_for_old_dataset(self, feat, query_date, trained_artifact):
        """With a dataset anchored in 2022, all candidates are well outside the
        10-trading-day window, so at least one window must return non-None stats."""
        r = self._run(feat, query_date, trained_artifact)
        fwd = r["forward_return_stats"]
        non_none = [k for k, v in fwd.items() if v is not None]
        assert len(non_none) > 0, (
            "Expected at least one forward-return window to have stats for "
            f"a 2022-anchored dataset, but all were None: {fwd}"
        )
        for k in non_none:
            s = fwd[k]
            assert isinstance(s["mean"], float)
            assert isinstance(s["median"], float)
            assert isinstance(s["n"], int) and s["n"] > 0

    # -- pipeline invariants -------------------------------------------------

    def test_top10_are_subset_of_knn40_candidates(self, feat, query_date, trained_artifact):
        """Every match date must have appeared in the k-NN retrieval pool."""
        from ml.similarity_ranker import _knn_candidates as _orig_knn

        captured: list[pd.DataFrame] = []

        def _capturing(knn, scaler, features_df, query_vec):
            result = _orig_knn(knn, scaler, features_df, query_vec)
            captured.append(result)
            return result

        with _tmp_model_patches(trained_artifact), _patch_loader(feat), \
             patch("ml.similarity_ranker._knn_candidates", side_effect=_capturing):
            r = rank(query_date)

        assert captured, "k-NN candidates were never collected"
        knn_dates = {str(pd.Timestamp(d).date()) for d in captured[0]["date"]}
        match_dates = {m["date"] for m in r["top_matches"]}
        assert match_dates <= knn_dates, (
            f"Match dates not a subset of k-NN candidates: {match_dates - knn_dates}"
        )

    def test_knn_candidates_retrieved_leq_forty(self, feat, query_date, trained_artifact):
        r = self._run(feat, query_date, trained_artifact)
        assert r["knn_candidates_retrieved"] <= _KNN_CANDIDATES

    def test_no_future_candidate_in_results(self, feat, query_date, trained_artifact):
        r = self._run(feat, query_date, trained_artifact)
        query_ts = pd.Timestamp(query_date)
        for m in r["top_matches"]:
            assert pd.Timestamp(m["date"]) < query_ts, (
                f"Match {m['date']} is >= query date {query_date}"
            )

    def test_no_same_day_candidate_in_results(self, feat, query_date, trained_artifact):
        r = self._run(feat, query_date, trained_artifact)
        for m in r["top_matches"]:
            assert m["date"] != query_date, "Query session must not appear as its own match"

    def test_forward_returns_none_for_recent_candidates(self):
        """
        Any match within 10 QSE trading days of the query must have
        forward_return_5d and forward_return_10d set to None.

        Uses its own artifact trained on the same tiny dataset so the scaler
        distribution matches and the kNN pool is well-defined.  The dataset is
        sized so that roughly half the kNN pool (40 candidates) falls within the
        10-trading-day window, guaranteeing the guard is exercised in practice.
        We assert the guard fired on at least one match rather than relying on
        an if-branch that could silently pass with zero assertions.
        """
        # 65 sessions: last 40 are candidates, last ~13 of those within window
        tiny = _make_features(n=65, seed=7)
        tiny["date"] = pd.DatetimeIndex(_qse_days("2024-09-15", 65))
        tiny_last = str(tiny["date"].iloc[-1].date())
        query_ts = pd.Timestamp(tiny_last)

        with tempfile.TemporaryDirectory() as tmp:
            with _tmp_model_patches(tmp), _patch_loader(tiny):
                train_and_save(tiny)
                r = rank(tiny_last)

        # Partition matches by whether they fall inside the leakage window
        inside = [
            m for m in r["top_matches"]
            if (len(pd.bdate_range(
                start=pd.Timestamp(m["date"]), end=query_ts, freq=self._QBDAY
            )) - 1) <= 10
        ]
        outside = [
            m for m in r["top_matches"]
            if (len(pd.bdate_range(
                start=pd.Timestamp(m["date"]), end=query_ts, freq=self._QBDAY
            )) - 1) > 10
        ]

        # The guard must have fired on at least one match — test is not vacuous
        assert len(inside) > 0, (
            "No match fell within the 10-trading-day leakage window; "
            "the guard was never exercised. Adjust the dataset size."
        )

        for m in inside:
            gap = len(pd.bdate_range(
                start=pd.Timestamp(m["date"]), end=query_ts, freq=self._QBDAY
            )) - 1
            assert m["forward_return_5d"] is None, (
                f"Candidate {m['date']} is {gap} trading days before query "
                f"but forward_return_5d={m['forward_return_5d']} is not None"
            )
            assert m["forward_return_10d"] is None, (
                f"Candidate {m['date']} is {gap} trading days before query "
                f"but forward_return_10d={m['forward_return_10d']} is not None"
            )

        # Outside-window matches must have real (non-None) forward returns
        for m in outside:
            assert m["forward_return_5d"] is not None, (
                f"Candidate {m['date']} is outside the leakage window "
                f"but forward_return_5d is unexpectedly None"
            )

    # -- error paths ---------------------------------------------------------

    def test_missing_date_raises_key_error(self, feat, trained_artifact):
        with _tmp_model_patches(trained_artifact), _patch_loader(feat):
            with pytest.raises(KeyError):
                rank("2099-01-01")

    def test_too_few_candidates_raises_value_error(self, trained_artifact):
        # n=1: the single row becomes the query date, leaving valid_pool empty,
        # which triggers the "Not enough historical sessions" ValueError.
        one_row = _make_features(n=1)
        with _tmp_model_patches(trained_artifact), _patch_loader(one_row):
            with pytest.raises(ValueError, match="enough historical sessions"):
                rank(str(one_row["date"].iloc[0].date()))

    def test_auto_train_on_first_call(self, feat, query_date):
        """rank() must auto-train when no artifact exists (isolated empty dir)."""
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            with _tmp_model_patches(tmp), _patch_loader(feat):
                assert not (p / "xgb_ranker_v1.pkl").exists()
                r = rank(query_date)
                assert (p / "xgb_ranker_v1.pkl").exists()

        assert isinstance(r["top_matches"], list)
        assert len(r["top_matches"]) > 0
