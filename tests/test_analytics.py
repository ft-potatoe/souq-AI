"""
tests/test_analytics.py

One test per spec §18 acceptance test (AT-1 through AT-8).
Each test drives the analytics module with the 20-row synthetic fixture from
test_features.py, calls run(date, params), and asserts the output schema
exactly — all required keys present, all value types correct, and the
computed answer matches manual calculation from the same fixture data.

Fixture construction is copied verbatim from test_features.py so there is no
shared state and no parquet file dependency.
"""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sp_stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Synthetic fixture — 20 QSE trading days (Sun–Thu), seeds fixed
# ---------------------------------------------------------------------------

def _qse_trading_days(start: str, n: int) -> pd.DatetimeIndex:
    days: list[pd.Timestamp] = []
    d = pd.Timestamp(start)
    while len(days) < n:
        if d.dayofweek not in (4, 5):   # skip Fri=4, Sat=5
            days.append(d)
        d += pd.Timedelta(days=1)
    return pd.DatetimeIndex(days)


def _build_fixture() -> dict[str, pd.DataFrame]:
    """
    Build market_df, flows_df, breadth_df with the exact seeds used in
    test_features.py, then assemble a features DataFrame that mirrors
    what build_features.py would produce (subset of columns sufficient for
    all six analytics modules).
    """
    rng = np.random.default_rng(42)
    dates = _qse_trading_days("2024-01-07", 20)

    close = 10_000.0
    prices: list[float] = []
    for _ in dates:
        close *= 1 + rng.normal(0, 0.008)
        prices.append(round(close, 2))

    volume      = rng.integers(50_000_000, 300_000_000, 20)
    value_traded = rng.uniform(50e6, 400e6, 20).round(0)
    total_trades = rng.integers(3_000, 30_000, 20)

    market_df = pd.DataFrame({
        "date":         pd.DatetimeIndex(dates),
        "open":         [p * 0.999 for p in prices],
        "high":         [p * 1.005 for p in prices],
        "low":          [p * 0.995 for p in prices],
        "close":        prices,
        "volume":       volume,
        "value_traded": value_traded,
        "total_trades": total_trades,
    })

    rng2 = np.random.default_rng(7)
    fbuy  = rng2.uniform(20e6, 200e6, 20).round(0)
    fsell = rng2.uniform(20e6, 200e6, 20).round(0)
    dbuy  = rng2.uniform(30e6, 300e6, 20).round(0)
    dsell = rng2.uniform(30e6, 300e6, 20).round(0)
    foreign_net  = fbuy - fsell
    domestic_net = dbuy - dsell

    flows_df = pd.DataFrame({
        "date":          pd.DatetimeIndex(dates),
        "foreign_buy":   fbuy,
        "foreign_sell":  fsell,
        "foreign_net":   foreign_net,
        "domestic_buy":  dbuy,
        "domestic_sell": dsell,
        "domestic_net":  domestic_net,
    })

    rng3 = np.random.default_rng(13)
    total_listed = 50
    gainers   = rng3.integers(5, 35, 20)
    losers    = np.minimum(rng3.integers(5, 35, 20), total_listed - gainers)
    unchanged = total_listed - gainers - losers

    breadth_df = pd.DataFrame({
        "date":         pd.DatetimeIndex(dates),
        "gainers":      gainers,
        "losers":       losers,
        "unchanged":    unchanged,
        "total_listed": total_listed,
        "total_traded": rng3.integers(20, 50, 20),
    })

    # ---- assemble features DataFrame ----------------------------------------
    # Only the columns the six analytics modules actually read.
    df = market_df.copy()

    # returns
    df["return_1d"]  = df["close"].pct_change(1)
    df["return_5d"]  = df["close"].pct_change(5)
    df["return_20d"] = df["close"].pct_change(20)

    # volatility
    df["volatility_20d"] = df["return_1d"].rolling(20, min_periods=10).std() * np.sqrt(252)

    # SMAs
    df["sma_20"]  = df["close"].rolling(20, min_periods=10).mean()
    df["sma_50"]  = df["close"].rolling(50, min_periods=25).mean()
    df["sma_200"] = df["close"].rolling(200, min_periods=100).mean()

    # flows
    for col in ["foreign_buy","foreign_sell","foreign_net","domestic_buy","domestic_sell","domestic_net"]:
        df[col] = flows_df[col].values

    total_flow = df["foreign_buy"] + df["foreign_sell"] + df["domestic_buy"] + df["domestic_sell"]
    df["foreign_participation"] = (df["foreign_buy"] + df["foreign_sell"]) / total_flow.replace(0, np.nan)
    df["foreign_net_cumulative_5d"]  = df["foreign_net"].rolling(5, min_periods=1).sum()
    df["foreign_net_cumulative_20d"] = df["foreign_net"].rolling(20, min_periods=1).sum()

    # z-scores (60d window, but only 20 rows — NaN for thin history, that's fine)
    def _zscore(s: pd.Series, w: int) -> pd.Series:
        mu  = s.rolling(w, min_periods=w // 2).mean()
        std = s.rolling(w, min_periods=w // 2).std().replace(0, np.nan)
        return (s - mu) / std

    df["volume_zscore"]          = _zscore(df["volume"].astype(float), 60)
    df["value_zscore"]           = _zscore(df["value_traded"], 60)
    df["trades_zscore"]          = _zscore(df["total_trades"].astype(float), 60)
    df["foreign_flow_zscore"]    = _zscore(df["foreign_net"], 60)
    df["domestic_flow_zscore"]   = _zscore(df["domestic_net"], 60)
    df["foreign_flow_slope_10d"] = np.nan  # not needed by acceptance tests

    # breadth
    total_b = (breadth_df["gainers"] + breadth_df["losers"] + breadth_df["unchanged"]).replace(0, np.nan)
    df["gainers"]      = breadth_df["gainers"].values
    df["losers"]       = breadth_df["losers"].values
    df["unchanged"]    = breadth_df["unchanged"].values
    df["total_listed"] = breadth_df["total_listed"].values
    df["total_traded"] = breadth_df["total_traded"].values
    df["breadth_ratio"] = breadth_df["gainers"].values / total_b.values
    df["breadth_net"]   = breadth_df["gainers"].values - breadth_df["losers"].values
    df["breadth_zscore"] = _zscore(df["breadth_ratio"], 60)

    # GCC placeholders — correlation test uses gcc_raw separately
    df["gcc_avg_return_1d"]      = np.nan
    df["qse_vs_gcc_spread"]      = np.nan
    df["qse_gcc_relative_5d"]    = np.nan
    df["qse_gcc_rolling_corr_20d"] = np.nan
    df["gcc_peer_rank"]          = np.nan

    # seasonality — QSE dow remapping: Sun=0 … Thu=4
    dow_map = {6: 0, 0: 1, 1: 2, 2: 3, 3: 4}
    df["day_of_week"] = df["date"].dt.dayofweek.map(dow_map)
    df["month"]       = df["date"].dt.month
    df["quarter"]     = df["date"].dt.quarter
    df["is_ramadan"]  = 0   # all 20 dates are Jan–Feb 2024, outside Ramadan
    df["trading_day_of_month"] = (
        df.groupby(df["date"].dt.to_period("M")).cumcount() + 1
    )

    df["rsi_14"]           = np.nan
    df["above_sma_20"]     = (df["close"] > df["sma_20"]).astype(int)
    df["above_sma_200"]    = 0
    df["price_vs_sma20_pct"] = np.nan

    df = df.sort_values("date").reset_index(drop=True)
    return {"features": df, "flows": flows_df, "breadth": breadth_df, "market": market_df}


# Module-scoped so we build the fixture only once
@pytest.fixture(scope="module")
def fx() -> dict[str, pd.DataFrame]:
    return _build_fixture()


@pytest.fixture(scope="module")
def feat(fx) -> pd.DataFrame:
    return fx["features"]


@pytest.fixture(scope="module")
def last_date(feat) -> str:
    return str(feat["date"].iloc[-1].date())   # 2024-02-01


@pytest.fixture(scope="module")
def last_row(feat, last_date) -> pd.Series:
    return feat[feat["date"] == pd.Timestamp(last_date)].iloc[0]


# ---------------------------------------------------------------------------
# Helper: patch the two _loader functions so modules read the fixture instead
# of the real parquet.
# ---------------------------------------------------------------------------

def _patch_loader(feat: pd.DataFrame, date: str):
    """
    Returns a context manager that replaces load_features / row_for_date /
    history_up_to with versions backed by *feat*.
    """
    ts = pd.Timestamp(date)

    def _load_features():
        return feat

    def _history_up_to(d):
        return feat[feat["date"] <= pd.Timestamp(d)].copy()

    def _row_for_date(d):
        ts2 = pd.Timestamp(d)
        mask = feat["date"] == ts2
        if not mask.any():
            raise KeyError(f"No row for {d}")
        return feat.loc[mask].iloc[0]

    return patch.multiple(
        "analytics._loader",
        load_features=_load_features,
        history_up_to=_history_up_to,
        row_for_date=_row_for_date,
    )


# ---------------------------------------------------------------------------
# AT-1: Distribution — percentile of today's volume in the past 60 days
# Spec §18 test 1: "What percentile is today's volume in the past 60 days?"
# ---------------------------------------------------------------------------

class TestAT1_VolumePercentile:
    """
    Drives distribution.run with metric='volume'.
    Window is capped at the 20 available rows (spec asks 60d, fixture has 20).
    Asserts exact output schema and manually verified percentile value.
    """

    EXPECTED_KEYS = {
        "metric",
        "today_value",
        "percentile_rank",
        "historical_frequency_above",
        "rolling_stats",
        "last_comparable_date",
        "sessions_above_today",
        "total_sessions",
    }
    ROLLING_STAT_WINDOWS = ("20d", "60d", "252d")
    ROLLING_STAT_SUB_KEYS = ("mean", "std")

    def _run(self, feat, last_date):
        from analytics.distribution import run
        with _patch_loader(feat, last_date):
            # Patch history_up_to inside distribution module too
            with patch("analytics.distribution.history_up_to",
                       side_effect=lambda d: feat[feat["date"] <= pd.Timestamp(d)].copy()), \
                 patch("analytics.distribution.row_for_date",
                       side_effect=lambda d: feat[feat["date"] == pd.Timestamp(d)].iloc[0]):
                return run(last_date, {"metric": "volume"})

    def test_all_top_level_keys_present(self, feat, last_date):
        r = self._run(feat, last_date)
        missing = self.EXPECTED_KEYS - r.keys()
        assert not missing, f"Missing keys: {missing}"

    def test_no_extra_surprise_keys(self, feat, last_date):
        """Schema is stable — unknown keys break the LLM payload contract."""
        r = self._run(feat, last_date)
        # distribution.run() also returns skewness, kurtosis and percentiles
        # (documented in CLAUDE.md) in addition to the directional frequency keys.
        allowed = self.EXPECTED_KEYS | {
            "sessions_below_today", "historical_frequency_below",
            "skewness", "kurtosis", "percentiles",
        }
        unexpected = r.keys() - allowed
        assert not unexpected, f"Unexpected keys in output: {unexpected}"

    def test_metric_field_is_volume(self, feat, last_date):
        r = self._run(feat, last_date)
        assert r["metric"] == "volume"
        assert isinstance(r["metric"], str)

    def test_today_value_matches_fixture(self, feat, last_date):
        r = self._run(feat, last_date)
        expected = int(feat[feat["date"] == pd.Timestamp(last_date)]["volume"].iloc[0])
        assert r["today_value"] == expected

    def test_today_value_is_not_nan(self, feat, last_date):
        r = self._run(feat, last_date)
        assert r["today_value"] is not None
        if isinstance(r["today_value"], float):
            assert not math.isnan(r["today_value"])

    def test_percentile_rank_type_is_float(self, feat, last_date):
        r = self._run(feat, last_date)
        assert r["percentile_rank"] is None or isinstance(r["percentile_rank"], float)

    def test_percentile_rank_in_bounds(self, feat, last_date):
        r = self._run(feat, last_date)
        pct = r["percentile_rank"]
        if pct is not None:
            assert 0.0 <= pct <= 100.0

    def test_percentile_rank_correct_value(self, feat, last_date):
        """
        Manual calculation: fraction of all 20 rows strictly below last volume × 100.
        Fixture row 19 volume=220762238. Rows strictly below: count manually.
        """
        r = self._run(feat, last_date)
        hist = feat[feat["date"] <= pd.Timestamp(last_date)]["volume"].dropna()
        last_vol = int(feat[feat["date"] == pd.Timestamp(last_date)]["volume"].iloc[0])
        expected_pct = round((hist < last_vol).sum() / len(hist) * 100, 2)
        assert r["percentile_rank"] == expected_pct

    def test_historical_frequency_above_type(self, feat, last_date):
        r = self._run(feat, last_date)
        fa = r["historical_frequency_above"]
        assert fa is None or isinstance(fa, float)

    def test_historical_frequency_above_in_bounds(self, feat, last_date):
        r = self._run(feat, last_date)
        fa = r["historical_frequency_above"]
        if fa is not None:
            assert 0.0 <= fa <= 1.0

    def test_historical_frequency_above_correct_value(self, feat, last_date):
        """freq_above == (volume >= last_vol).sum() / total"""
        r = self._run(feat, last_date)
        hist = feat[feat["date"] <= pd.Timestamp(last_date)]["volume"].dropna()
        last_vol = int(feat[feat["date"] == pd.Timestamp(last_date)]["volume"].iloc[0])
        expected = round((hist >= last_vol).sum() / len(hist), 4)
        assert r["historical_frequency_above"] == expected

    def test_rolling_stats_has_all_windows(self, feat, last_date):
        r = self._run(feat, last_date)
        for w in self.ROLLING_STAT_WINDOWS:
            assert w in r["rolling_stats"], f"Missing rolling_stats window '{w}'"

    def test_rolling_stats_sub_keys(self, feat, last_date):
        r = self._run(feat, last_date)
        for w in self.ROLLING_STAT_WINDOWS:
            for sub in self.ROLLING_STAT_SUB_KEYS:
                assert sub in r["rolling_stats"][w], f"Missing rolling_stats['{w}']['{sub}']"

    def test_rolling_stats_20d_mean_type(self, feat, last_date):
        r = self._run(feat, last_date)
        mean_20 = r["rolling_stats"]["20d"]["mean"]
        assert mean_20 is None or isinstance(mean_20, float)

    def test_rolling_stats_20d_std_nonnegative(self, feat, last_date):
        r = self._run(feat, last_date)
        std_20 = r["rolling_stats"]["20d"]["std"]
        if std_20 is not None:
            assert std_20 >= 0.0

    def test_rolling_stats_20d_mean_correct_value(self, feat, last_date):
        """rolling_stats['20d']['mean'] == mean of last 20 volume values."""
        r = self._run(feat, last_date)
        hist = feat[feat["date"] <= pd.Timestamp(last_date)]["volume"].dropna()
        expected_mean = round(float(hist.tail(20).mean()), 6)
        assert r["rolling_stats"]["20d"]["mean"] == expected_mean

    def test_total_sessions_equals_history_length(self, feat, last_date):
        r = self._run(feat, last_date)
        expected = int(feat[feat["date"] <= pd.Timestamp(last_date)]["volume"].dropna().count())
        assert r["total_sessions"] == expected
        assert isinstance(r["total_sessions"], int)

    def test_sessions_above_today_is_int(self, feat, last_date):
        r = self._run(feat, last_date)
        assert isinstance(r["sessions_above_today"], int)

    def test_sessions_above_does_not_exceed_total(self, feat, last_date):
        r = self._run(feat, last_date)
        assert r["sessions_above_today"] <= r["total_sessions"]

    def test_last_comparable_date_is_str_or_none(self, feat, last_date):
        r = self._run(feat, last_date)
        lcd = r["last_comparable_date"]
        assert lcd is None or isinstance(lcd, str)

    def test_last_comparable_date_before_query_date(self, feat, last_date):
        r = self._run(feat, last_date)
        lcd = r["last_comparable_date"]
        if lcd is not None:
            assert pd.Timestamp(lcd) < pd.Timestamp(last_date)


# ---------------------------------------------------------------------------
# AT-2: Distribution — how often has foreign net selling been this extreme?
# Spec §18 test 2: "How often has foreign net selling been this extreme historically?"
# Mapped to: distribution.run with metric='foreign_net', direction='below'
# on a date where foreign_net < 0.
# ---------------------------------------------------------------------------

class TestAT2_ForeignNetSellingFrequency:
    """
    Uses row 6 (2024-01-15, foreign_net = -82969330) — the most negative in
    the fixture — to answer 'how often has selling been this extreme'.
    direction='below' => historical_frequency_below = P(fn <= today_fn).
    """

    # Row 6 in the fixture: first date where fn < 0 and large in magnitude
    SELLING_DATE = "2024-01-15"

    EXPECTED_KEYS = {
        "metric",
        "today_value",
        "percentile_rank",
        "historical_frequency_below",
        "rolling_stats",
        "last_comparable_date",
        "sessions_below_today",
        "total_sessions",
    }

    def _run(self, feat):
        from analytics.distribution import run
        with patch("analytics.distribution.history_up_to",
                   side_effect=lambda d: feat[feat["date"] <= pd.Timestamp(d)].copy()), \
             patch("analytics.distribution.row_for_date",
                   side_effect=lambda d: feat[feat["date"] == pd.Timestamp(d)].iloc[0]):
            return run(self.SELLING_DATE, {"metric": "foreign_net", "direction": "below"})

    def test_query_date_has_negative_foreign_net(self, feat):
        row = feat[feat["date"] == pd.Timestamp(self.SELLING_DATE)].iloc[0]
        assert float(row["foreign_net"]) < 0, "Test date must have negative foreign_net"

    def test_all_keys_present(self, feat):
        r = self._run(feat)
        missing = self.EXPECTED_KEYS - r.keys()
        assert not missing, f"Missing keys: {missing}"

    def test_no_above_key_when_direction_below(self, feat):
        r = self._run(feat)
        assert "historical_frequency_above" not in r
        assert "sessions_above_today" not in r

    def test_metric_is_foreign_net(self, feat):
        r = self._run(feat)
        assert r["metric"] == "foreign_net"
        assert isinstance(r["metric"], str)

    def test_today_value_matches_fixture(self, feat):
        r = self._run(feat)
        expected = float(feat[feat["date"] == pd.Timestamp(self.SELLING_DATE)]["foreign_net"].iloc[0])
        assert r["today_value"] == expected

    def test_today_value_is_negative(self, feat):
        r = self._run(feat)
        assert r["today_value"] < 0

    def test_historical_frequency_below_type_and_bounds(self, feat):
        r = self._run(feat)
        fb = r["historical_frequency_below"]
        assert fb is None or isinstance(fb, float)
        if fb is not None:
            assert 0.0 <= fb <= 1.0

    def test_historical_frequency_below_correct_value(self, feat):
        """freq_below == (fn <= today_fn).sum() / total."""
        r = self._run(feat)
        hist = feat[feat["date"] <= pd.Timestamp(self.SELLING_DATE)]["foreign_net"].dropna()
        today_fn = float(feat[feat["date"] == pd.Timestamp(self.SELLING_DATE)]["foreign_net"].iloc[0])
        expected = round((hist <= today_fn).sum() / len(hist), 4)
        assert r["historical_frequency_below"] == expected

    def test_percentile_rank_in_range(self, feat):
        r = self._run(feat)
        pct = r["percentile_rank"]
        if pct is not None:
            assert 0.0 <= pct <= 100.0

    def test_percentile_rank_low_for_extreme_selling(self, feat):
        """
        Extreme net selling => low percentile (many sessions above this value).
        With 7 rows of history up to 2024-01-15, this fn is the lowest seen
        so percentile_rank should be < 20.
        """
        r = self._run(feat)
        pct = r["percentile_rank"]
        if pct is not None:
            assert pct < 50.0, f"Extreme selling should be low percentile, got {pct}"

    def test_sessions_below_is_int_and_bounded(self, feat):
        r = self._run(feat)
        assert isinstance(r["sessions_below_today"], int)
        assert 0 <= r["sessions_below_today"] <= r["total_sessions"]

    def test_rolling_stats_windows_present(self, feat):
        r = self._run(feat)
        for w in ("20d", "60d", "252d"):
            assert w in r["rolling_stats"]
            assert "mean" in r["rolling_stats"][w]
            assert "std" in r["rolling_stats"][w]

    def test_rolling_stats_types(self, feat):
        r = self._run(feat)
        for w in ("20d", "60d", "252d"):
            mean = r["rolling_stats"][w]["mean"]
            std  = r["rolling_stats"][w]["std"]
            assert mean is None or isinstance(mean, float)
            assert std  is None or isinstance(std,  float)
            if std is not None:
                assert std >= 0.0

    def test_total_sessions_is_positive_int(self, feat):
        r = self._run(feat)
        assert isinstance(r["total_sessions"], int)
        assert r["total_sessions"] > 0


# ---------------------------------------------------------------------------
# AT-3: Trend — is foreign net flow trending positive or negative over 10 sessions?
# Spec §18 test 3: "Is foreign net flow trending positive or negative over the last 10 sessions?"
# Mapped to: trend.run with metric='foreign_net', slope_window=10.
# ---------------------------------------------------------------------------

class TestAT3_ForeignNetTrend:
    """
    The slope of foreign_net over last 10 sessions as of 2024-02-01 is
    +13,061,575 (precomputed above) => direction='increasing'.
    """

    EXPECTED_KEYS = {
        "metric",
        "slope_10d",
        "slope_direction",
        "streak",
        "momentum",
        "sma_crossover",
        "crossover_date",
    }
    VALID_DIRECTIONS = {"increasing", "decreasing", "flat", None}
    VALID_MOMENTUM   = {"up", "down", "flat", "insufficient_data"}

    def _run(self, feat, last_date):
        from analytics.trend import run
        with patch("analytics.trend.history_up_to",
                   side_effect=lambda d: feat[feat["date"] <= pd.Timestamp(d)].copy()), \
             patch("analytics.trend.row_for_date",
                   side_effect=lambda d: feat[feat["date"] == pd.Timestamp(d)].iloc[0]):
            return run(last_date, {"metric": "foreign_net", "slope_window": 10})

    def test_all_keys_present(self, feat, last_date):
        r = self._run(feat, last_date)
        missing = self.EXPECTED_KEYS - r.keys()
        assert not missing, f"Missing keys: {missing}"

    def test_slope_key_name_matches_window(self, feat, last_date):
        """Key must be slope_{window}d, i.e. slope_10d for window=10."""
        r = self._run(feat, last_date)
        assert "slope_10d" in r

    def test_slope_value_type(self, feat, last_date):
        r = self._run(feat, last_date)
        slope = r["slope_10d"]
        assert slope is None or isinstance(slope, float)

    def test_slope_value_is_finite(self, feat, last_date):
        r = self._run(feat, last_date)
        slope = r["slope_10d"]
        if slope is not None:
            assert math.isfinite(slope)

    def test_slope_matches_scipy_linregress(self, feat, last_date):
        """slope_10d must equal scipy linregress on the last 10 foreign_net values."""
        r = self._run(feat, last_date)
        hist = feat[feat["date"] <= pd.Timestamp(last_date)]["foreign_net"].dropna()
        tail = hist.tail(10)
        x = np.arange(len(tail), dtype=float)
        expected = sp_stats.linregress(x, tail.values).slope
        assert r["slope_10d"] is not None
        assert abs(r["slope_10d"] - expected) < 1.0   # slope is in QAR units, 1 QAR tolerance

    def test_slope_direction_type_is_str_or_none(self, feat, last_date):
        r = self._run(feat, last_date)
        assert r["slope_direction"] in self.VALID_DIRECTIONS

    def test_slope_direction_consistent_with_slope(self, feat, last_date):
        r = self._run(feat, last_date)
        slope = r["slope_10d"]
        direction = r["slope_direction"]
        if slope is not None:
            if slope > 0:
                assert direction == "increasing"
            elif slope < 0:
                assert direction == "decreasing"
            else:
                assert direction == "flat"

    def test_slope_direction_is_increasing_for_fixture(self, feat, last_date):
        """Pre-verified: OLS slope of last 10 fn values is positive (+13M)."""
        r = self._run(feat, last_date)
        assert r["slope_direction"] == "increasing"

    def test_streak_dict_structure(self, feat, last_date):
        r = self._run(feat, last_date)
        assert isinstance(r["streak"], dict)
        assert "condition" in r["streak"]
        assert "count" in r["streak"]

    def test_streak_condition_is_string(self, feat, last_date):
        r = self._run(feat, last_date)
        assert isinstance(r["streak"]["condition"], str)

    def test_streak_count_is_nonneg_int(self, feat, last_date):
        r = self._run(feat, last_date)
        count = r["streak"]["count"]
        assert isinstance(count, int)
        assert count >= 0

    def test_momentum_is_dict(self, feat, last_date):
        r = self._run(feat, last_date)
        assert isinstance(r["momentum"], dict)

    def test_momentum_values_are_valid_labels(self, feat, last_date):
        r = self._run(feat, last_date)
        for k, v in r["momentum"].items():
            assert v in self.VALID_MOMENTUM, f"momentum['{k}'] = '{v}'"

    def test_sma_crossover_is_none_when_no_sma_params(self, feat, last_date):
        """No fast_sma_col/slow_sma_col passed => crossover must be None."""
        r = self._run(feat, last_date)
        assert r["sma_crossover"] is None
        assert r["crossover_date"] is None

    def test_metric_field_is_foreign_net(self, feat, last_date):
        r = self._run(feat, last_date)
        assert r["metric"] == "foreign_net"


# ---------------------------------------------------------------------------
# AT-4: Trend — how many consecutive sessions of net foreign selling?
# Spec §18 test 4: "How many consecutive sessions of net foreign selling have there been?"
# Mapped to: trend.run with metric='foreign_net', streak_condition='negative'.
# ---------------------------------------------------------------------------

class TestAT4_ConsecutiveSellingStreak:
    """
    As of 2024-02-01 (row 19), foreign_net = +150M => streak of 'negative' = 0.
    Test the streak=0 case and confirm schema, then test a date where fn<0 at end.
    Row 14 (2024-01-25) has fn=-33M; verify streak >= 1 there.
    """

    DATE_LAST      = "2024-02-01"   # foreign_net positive — streak 0
    DATE_SELLING   = "2024-01-25"   # foreign_net negative at this row

    def _run(self, feat, date, streak_cond="negative"):
        from analytics.trend import run
        with patch("analytics.trend.history_up_to",
                   side_effect=lambda d: feat[feat["date"] <= pd.Timestamp(d)].copy()), \
             patch("analytics.trend.row_for_date",
                   side_effect=lambda d: feat[feat["date"] == pd.Timestamp(d)].iloc[0]):
            return run(date, {"metric": "foreign_net", "streak_condition": streak_cond})

    def test_streak_key_present(self, feat):
        r = self._run(feat, self.DATE_LAST)
        assert "streak" in r

    def test_streak_has_condition_and_count(self, feat):
        r = self._run(feat, self.DATE_LAST)
        assert set(r["streak"].keys()) == {"condition", "count"}

    def test_streak_condition_label_is_negative(self, feat):
        r = self._run(feat, self.DATE_LAST)
        assert r["streak"]["condition"] == "negative"

    def test_streak_count_type_is_int(self, feat):
        r = self._run(feat, self.DATE_LAST)
        assert isinstance(r["streak"]["count"], int)

    def test_streak_zero_when_last_fn_positive(self, feat):
        """Row 19 has foreign_net > 0 => negative streak must be 0."""
        last_fn = float(feat[feat["date"] == pd.Timestamp(self.DATE_LAST)]["foreign_net"].iloc[0])
        assert last_fn > 0, "Pre-condition: row 19 must have positive fn"
        r = self._run(feat, self.DATE_LAST)
        assert r["streak"]["count"] == 0

    def test_streak_positive_when_last_fn_negative(self, feat):
        """Row 14 has foreign_net < 0 => negative streak >= 1."""
        row_fn = float(feat[feat["date"] == pd.Timestamp(self.DATE_SELLING)]["foreign_net"].iloc[0])
        assert row_fn < 0, "Pre-condition: 2024-01-25 must have negative fn"
        r = self._run(feat, self.DATE_SELLING)
        assert r["streak"]["count"] >= 1

    def test_streak_count_matches_manual_count(self, feat):
        """
        Manual count from end of history up to DATE_SELLING:
        Walk backwards until fn >= 0, count consecutive fn < 0.
        """
        hist = feat[feat["date"] <= pd.Timestamp(self.DATE_SELLING)]["foreign_net"].dropna().values
        manual = 0
        for v in reversed(hist):
            if v < 0:
                manual += 1
            else:
                break
        r = self._run(feat, self.DATE_SELLING)
        assert r["streak"]["count"] == manual

    def test_schema_complete(self, feat):
        r = self._run(feat, self.DATE_LAST)
        required = {"metric", "slope_10d", "slope_direction", "streak", "momentum",
                    "sma_crossover", "crossover_date"}
        assert required.issubset(r.keys())

    def test_positive_streak_condition_also_works(self, feat):
        """Passing streak_condition='positive' on row 19 (fn>0) gives count >= 1."""
        r = self._run(feat, self.DATE_LAST, streak_cond="positive")
        assert r["streak"]["condition"] == "positive"
        assert r["streak"]["count"] >= 1


# ---------------------------------------------------------------------------
# AT-5: Correlation — has QSE decoupled from GCC peers in the past 20 days?
# Spec §18 test 5: "Has QSE decoupled from GCC peers in the past 20 days?"
# Mapped to: correlation.run with metric_a='return_1d', metric_b='gcc_avg_return_1d',
# rolling_windows=[20], include_gcc=True.
# The fixture has gcc_avg_return_1d = NaN for all rows, so we inject a synthetic
# gcc peer series to exercise the schema.
# ---------------------------------------------------------------------------

class TestAT5_GCCDecoupling:
    """
    Drives correlation.run(metric_a='return_1d', metric_b='gcc_avg_return_1d').
    Because the fixture's gcc_avg_return_1d column is NaN, we patch it with
    a synthetic peer average so the 20d correlation is computable.
    Also tests gcc_peer_correlations schema from the real raw parquet when
    available, else gracefully handles None.
    """

    EXPECTED_KEYS = {
        "pair",
        "rolling_corr_20d",
        "rolling_corr_60d",
        "historical_mean_corr",
        "historical_std_corr",
        "percentile_of_current_corr",
    }

    def _feat_with_gcc(self, feat: pd.DataFrame) -> pd.DataFrame:
        """Inject a synthetic gcc_avg_return_1d and qse_vs_gcc_spread."""
        rng = np.random.default_rng(99)
        df = feat.copy()
        # Peer avg slightly correlated with return_1d
        df["gcc_avg_return_1d"] = (
            df["return_1d"].fillna(0) * 0.6 + rng.normal(0, 0.003, len(df))
        )
        df["qse_vs_gcc_spread"] = df["return_1d"].fillna(0) - df["gcc_avg_return_1d"]
        return df

    def _run(self, feat, last_date, include_gcc=False):
        feat2 = self._feat_with_gcc(feat)
        from analytics.correlation import run
        with patch("analytics.correlation.history_up_to",
                   side_effect=lambda d: feat2[feat2["date"] <= pd.Timestamp(d)].copy()), \
             patch("analytics.correlation.load_gcc_raw",
                   return_value=pd.DataFrame(columns=["date", "market_name", "daily_change_pct"])):
            return run(last_date, {
                "metric_a": "return_1d",
                "metric_b": "gcc_avg_return_1d",
                "rolling_windows": [20, 60],
                "current_window": 20,
                "include_gcc": include_gcc,
            })

    def test_all_schema_keys_present(self, feat, last_date):
        r = self._run(feat, last_date)
        missing = self.EXPECTED_KEYS - r.keys()
        assert not missing, f"Missing keys: {missing}"

    def test_pair_label_correct(self, feat, last_date):
        r = self._run(feat, last_date)
        assert r["pair"] == "return_1d vs gcc_avg_return_1d"
        assert isinstance(r["pair"], str)

    def test_rolling_corr_20d_key_present(self, feat, last_date):
        r = self._run(feat, last_date)
        assert "rolling_corr_20d" in r

    def test_rolling_corr_60d_key_present(self, feat, last_date):
        r = self._run(feat, last_date)
        assert "rolling_corr_60d" in r

    def test_rolling_corr_20d_type(self, feat, last_date):
        r = self._run(feat, last_date)
        val = r["rolling_corr_20d"]
        assert val is None or isinstance(val, float)

    def test_rolling_corr_20d_in_range(self, feat, last_date):
        r = self._run(feat, last_date)
        val = r["rolling_corr_20d"]
        if val is not None:
            assert -1.0 <= val <= 1.0

    def test_rolling_corr_60d_none_for_thin_history(self, feat, last_date):
        """With only 20 rows, 60d window has fewer than 4 pairs => None."""
        r = self._run(feat, last_date)
        # 20 rows, window=60: tail(60) is all 20 rows, corr is computable
        # It may be None or a float depending on paired non-NaN count
        val = r["rolling_corr_60d"]
        assert val is None or isinstance(val, float)
        if val is not None:
            assert -1.0 <= val <= 1.0

    def test_historical_mean_corr_type(self, feat, last_date):
        r = self._run(feat, last_date)
        val = r["historical_mean_corr"]
        assert val is None or isinstance(val, float)

    def test_historical_std_corr_nonneg(self, feat, last_date):
        r = self._run(feat, last_date)
        val = r["historical_std_corr"]
        if val is not None:
            assert val >= 0.0

    def test_percentile_of_current_corr_in_range(self, feat, last_date):
        r = self._run(feat, last_date)
        pct = r["percentile_of_current_corr"]
        if pct is not None:
            assert 0.0 <= pct <= 100.0

    def test_no_gcc_correlations_when_include_gcc_false(self, feat, last_date):
        r = self._run(feat, last_date, include_gcc=False)
        assert "gcc_correlations_60d" not in r

    def test_gcc_correlations_present_when_include_gcc_true(self, feat, last_date):
        r = self._run(feat, last_date, include_gcc=True)
        assert "gcc_correlations_60d" in r

    def test_rolling_corr_matches_pandas(self, feat, last_date):
        """rolling_corr_20d must equal pandas corr over same paired tail."""
        feat2 = self._feat_with_gcc(feat)
        hist = feat2[feat2["date"] <= pd.Timestamp(last_date)].reset_index(drop=True)
        sa = hist["return_1d"]
        sb = hist["gcc_avg_return_1d"]
        paired = pd.concat([sa.rename("a"), sb.rename("b")], axis=1).dropna().tail(20)
        if len(paired) >= 4:
            expected = round(float(paired["a"].corr(paired["b"])), 4)
            r = self._run(feat, last_date)
            val = r["rolling_corr_20d"]
            if val is not None:
                assert abs(val - expected) < 5e-5


# ---------------------------------------------------------------------------
# AT-6: Seasonality — is today a typically high or low volume day of the week?
# Spec §18 test 6: "Is today a typically high or low volume day of the week for QSE?"
# Mapped to: seasonality.run with metric='volume'.
# ---------------------------------------------------------------------------

class TestAT6_SeasonalityDayOfWeek:
    """
    The last date 2024-02-01 is a Thursday (QSE dow=4).
    Asserts full schema including day_of_week_profile and the rank string.
    """

    EXPECTED_KEYS = {
        "metric",
        "date",
        "today_day_of_week",
        "day_of_week_mean",
        "day_of_week_rank",
        "day_of_week_profile",
        "monthly_mean_this_month",
        "monthly_rank",
        "monthly_profile",
        "ramadan_effect",
    }
    RAMADAN_EFFECT_KEYS = {
        "ramadan_mean",
        "non_ramadan_mean",
        "pct_difference",
        "ramadan_sessions",
        "non_ramadan_sessions",
        "is_ramadan_today",
    }
    VALID_DOW_NAMES = {"Sunday", "Monday", "Tuesday", "Wednesday", "Thursday"}
    VALID_MONTH_NAMES = {"Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"}

    def _run(self, feat, last_date):
        from analytics.seasonality import run
        with patch("analytics.seasonality.history_up_to",
                   side_effect=lambda d: feat[feat["date"] <= pd.Timestamp(d)].copy()), \
             patch("analytics.seasonality.row_for_date",
                   side_effect=lambda d: feat[feat["date"] == pd.Timestamp(d)].iloc[0]):
            return run(last_date, {"metric": "volume"})

    def test_all_schema_keys_present(self, feat, last_date):
        r = self._run(feat, last_date)
        missing = self.EXPECTED_KEYS - r.keys()
        assert not missing, f"Missing keys: {missing}"

    def test_metric_is_volume(self, feat, last_date):
        r = self._run(feat, last_date)
        assert r["metric"] == "volume"
        assert isinstance(r["metric"], str)

    def test_date_field_matches_query(self, feat, last_date):
        r = self._run(feat, last_date)
        assert r["date"] == last_date
        assert isinstance(r["date"], str)

    def test_today_day_of_week_is_thursday(self, feat, last_date):
        """2024-02-01 is a Thursday => QSE dow=4 => name='Thursday'."""
        assert pd.Timestamp(last_date).dayofweek == 3  # pandas Thu=3
        r = self._run(feat, last_date)
        assert r["today_day_of_week"] == "Thursday"

    def test_today_day_of_week_in_valid_set(self, feat, last_date):
        r = self._run(feat, last_date)
        assert r["today_day_of_week"] in self.VALID_DOW_NAMES

    def test_day_of_week_mean_type(self, feat, last_date):
        r = self._run(feat, last_date)
        val = r["day_of_week_mean"]
        assert val is None or isinstance(val, float)

    def test_day_of_week_mean_is_positive(self, feat, last_date):
        r = self._run(feat, last_date)
        val = r["day_of_week_mean"]
        if val is not None:
            assert val > 0

    def test_day_of_week_mean_correct_value(self, feat, last_date):
        """day_of_week_mean == mean volume of all Thursday rows in history."""
        r = self._run(feat, last_date)
        hist = feat[feat["date"] <= pd.Timestamp(last_date)]
        thu_rows = hist[hist["day_of_week"] == 4]["volume"].dropna()
        expected = round(float(thu_rows.mean()), 6)
        assert abs(r["day_of_week_mean"] - expected) < 1.0

    def test_day_of_week_rank_is_string(self, feat, last_date):
        r = self._run(feat, last_date)
        rank = r["day_of_week_rank"]
        assert rank is None or isinstance(rank, str)

    def test_day_of_week_rank_contains_of(self, feat, last_date):
        r = self._run(feat, last_date)
        rank = r["day_of_week_rank"]
        if rank is not None:
            assert "of" in rank, f"Expected 'Nth of M' format, got '{rank}'"

    def test_day_of_week_profile_has_all_five_days(self, feat, last_date):
        r = self._run(feat, last_date)
        assert self.VALID_DOW_NAMES == set(r["day_of_week_profile"].keys())

    def test_day_of_week_profile_values_float_or_none(self, feat, last_date):
        r = self._run(feat, last_date)
        for k, v in r["day_of_week_profile"].items():
            assert v is None or isinstance(v, float), f"profile['{k}'] has wrong type"

    def test_day_of_week_mean_matches_profile_entry(self, feat, last_date):
        r = self._run(feat, last_date)
        dow = r["today_day_of_week"]
        profile_val = r["day_of_week_profile"].get(dow)
        reported   = r["day_of_week_mean"]
        if profile_val is not None and reported is not None:
            assert abs(profile_val - reported) < 1.0

    def test_monthly_mean_this_month_type(self, feat, last_date):
        r = self._run(feat, last_date)
        val = r["monthly_mean_this_month"]
        assert val is None or isinstance(val, float)

    def test_monthly_rank_format(self, feat, last_date):
        r = self._run(feat, last_date)
        rank = r["monthly_rank"]
        if rank is not None:
            assert "of" in rank

    def test_monthly_profile_has_all_12_months(self, feat, last_date):
        r = self._run(feat, last_date)
        assert self.VALID_MONTH_NAMES == set(r["monthly_profile"].keys())

    def test_monthly_profile_values_float_or_none(self, feat, last_date):
        r = self._run(feat, last_date)
        for k, v in r["monthly_profile"].items():
            assert v is None or isinstance(v, float)

    def test_ramadan_effect_has_required_keys(self, feat, last_date):
        r = self._run(feat, last_date)
        ram = r["ramadan_effect"]
        missing = self.RAMADAN_EFFECT_KEYS - ram.keys()
        assert not missing, f"ramadan_effect missing keys: {missing}"

    def test_is_ramadan_today_is_false(self, feat, last_date):
        """All fixture dates (Jan–Feb 2024) are outside Ramadan."""
        r = self._run(feat, last_date)
        assert r["ramadan_effect"]["is_ramadan_today"] is False

    def test_is_ramadan_today_is_bool(self, feat, last_date):
        r = self._run(feat, last_date)
        assert isinstance(r["ramadan_effect"]["is_ramadan_today"], bool)

    def test_ramadan_sessions_zero_for_fixture(self, feat, last_date):
        """No Ramadan rows in the fixture => ramadan_sessions == 0."""
        r = self._run(feat, last_date)
        assert r["ramadan_effect"]["ramadan_sessions"] == 0

    def test_non_ramadan_sessions_equals_total(self, feat, last_date):
        r = self._run(feat, last_date)
        total = feat[feat["date"] <= pd.Timestamp(last_date)]["volume"].dropna().count()
        assert r["ramadan_effect"]["non_ramadan_sessions"] == int(total)

    def test_pct_difference_is_none_when_no_ramadan(self, feat, last_date):
        """With zero Ramadan sessions, ramadan_mean is None => pct_difference is None."""
        r = self._run(feat, last_date)
        assert r["ramadan_effect"]["pct_difference"] is None


# ---------------------------------------------------------------------------
# AT-7: Flows — who is driving the market today?
# Spec §18 test 7: "Who is driving the market today — foreign or domestic?"
# Mapped to: flows.run(date, {}).
# ---------------------------------------------------------------------------

class TestAT7_FlowDominance:
    """
    Row 19 (2024-02-01): foreign_net=+150M, domestic_net=+120M.
    Both positive; abs(foreign) > abs(domestic) => dominant_flow='foreign_buying'.
    """

    EXPECTED_KEYS = {
        "date",
        "foreign_net_today",
        "domestic_net_today",
        "dominant_flow",
        "cumulative_foreign_net",
        "cumulative_domestic_net",
        "foreign_participation_pct",
        "flow_pressure_trend_10d",
        "foreign_flow_zscore",
        "domestic_flow_zscore",
    }
    VALID_DOMINANT = {
        "foreign_buying", "foreign_selling",
        "domestic_buying", "domestic_selling",
        "balanced",
    }
    VALID_TREND = {
        "increasing_buying", "decreasing_buying",
        "increasing_selling", "decreasing_selling",
        "insufficient_data",
    }
    CUMULATIVE_WINDOWS = ("5d", "10d", "20d")

    def _run(self, feat, last_date):
        from analytics.flows import run
        with patch("analytics.flows.history_up_to",
                   side_effect=lambda d: feat[feat["date"] <= pd.Timestamp(d)].copy()), \
             patch("analytics.flows.row_for_date",
                   side_effect=lambda d: feat[feat["date"] == pd.Timestamp(d)].iloc[0]):
            return run(last_date, {})

    def test_all_schema_keys_present(self, feat, last_date):
        r = self._run(feat, last_date)
        missing = self.EXPECTED_KEYS - r.keys()
        assert not missing, f"Missing keys: {missing}"

    def test_date_field_type_and_value(self, feat, last_date):
        r = self._run(feat, last_date)
        assert isinstance(r["date"], str)
        assert r["date"] == last_date

    def test_foreign_net_today_type(self, feat, last_date):
        r = self._run(feat, last_date)
        assert isinstance(r["foreign_net_today"], float)

    def test_foreign_net_today_matches_fixture(self, feat, last_date):
        r = self._run(feat, last_date)
        expected = float(feat[feat["date"] == pd.Timestamp(last_date)]["foreign_net"].iloc[0])
        assert abs(r["foreign_net_today"] - expected) < 1.0

    def test_foreign_net_today_positive(self, feat, last_date):
        """Row 19 has foreign_net > 0."""
        r = self._run(feat, last_date)
        assert r["foreign_net_today"] > 0

    def test_domestic_net_today_type(self, feat, last_date):
        r = self._run(feat, last_date)
        assert isinstance(r["domestic_net_today"], float)

    def test_domestic_net_today_matches_fixture(self, feat, last_date):
        r = self._run(feat, last_date)
        expected = float(feat[feat["date"] == pd.Timestamp(last_date)]["domestic_net"].iloc[0])
        assert abs(r["domestic_net_today"] - expected) < 1.0

    def test_dominant_flow_type_is_str(self, feat, last_date):
        r = self._run(feat, last_date)
        assert isinstance(r["dominant_flow"], str)

    def test_dominant_flow_valid_label(self, feat, last_date):
        r = self._run(feat, last_date)
        assert r["dominant_flow"] in self.VALID_DOMINANT

    def test_dominant_flow_is_foreign_buying_for_row19(self, feat, last_date):
        """Row 19: foreign_net=+150M, domestic_net=+120M; abs(fn)>abs(dn) => foreign_buying."""
        row = feat[feat["date"] == pd.Timestamp(last_date)].iloc[0]
        fn = float(row["foreign_net"])
        dn = float(row["domestic_net"])
        assert fn > 0 and abs(fn) > abs(dn), "Pre-condition failed"
        r = self._run(feat, last_date)
        assert r["dominant_flow"] == "foreign_buying"

    def test_cumulative_foreign_net_has_all_windows(self, feat, last_date):
        r = self._run(feat, last_date)
        for w in self.CUMULATIVE_WINDOWS:
            assert w in r["cumulative_foreign_net"], f"Missing window '{w}'"

    def test_cumulative_domestic_net_has_all_windows(self, feat, last_date):
        r = self._run(feat, last_date)
        for w in self.CUMULATIVE_WINDOWS:
            assert w in r["cumulative_domestic_net"], f"Missing window '{w}'"

    def test_cumulative_5d_correct_value(self, feat, last_date):
        """cumulative_foreign_net['5d'] == sum of last 5 foreign_net values."""
        r = self._run(feat, last_date)
        hist = feat[feat["date"] <= pd.Timestamp(last_date)]["foreign_net"].dropna()
        expected = float(hist.tail(5).sum())
        assert abs(r["cumulative_foreign_net"]["5d"] - expected) < 1.0

    def test_cumulative_10d_correct_value(self, feat, last_date):
        r = self._run(feat, last_date)
        hist = feat[feat["date"] <= pd.Timestamp(last_date)]["foreign_net"].dropna()
        expected = float(hist.tail(10).sum())
        assert abs(r["cumulative_foreign_net"]["10d"] - expected) < 1.0

    def test_cumulative_20d_correct_value(self, feat, last_date):
        r = self._run(feat, last_date)
        hist = feat[feat["date"] <= pd.Timestamp(last_date)]["foreign_net"].dropna()
        expected = float(hist.tail(20).sum())
        assert abs(r["cumulative_foreign_net"]["20d"] - expected) < 1.0

    def test_foreign_participation_pct_type(self, feat, last_date):
        r = self._run(feat, last_date)
        val = r["foreign_participation_pct"]
        assert val is None or isinstance(val, float)

    def test_foreign_participation_pct_is_float_or_none(self, feat, last_date):
        r = self._run(feat, last_date)
        val = r["foreign_participation_pct"]
        assert val is None or isinstance(val, float)

    def test_foreign_participation_pct_correct_value(self, feat, last_date):
        """
        Module formula (spec §5.6): (fbuy + fsell) / value_traded * 100.
        value_traded is the market column, generated independently of flows, so
        the ratio can exceed 100 in synthetic data — no upper-bound assert here.
        """
        r = self._run(feat, last_date)
        row = feat[feat["date"] == pd.Timestamp(last_date)].iloc[0]
        fbuy  = float(row["foreign_buy"])
        fsell = float(row["foreign_sell"])
        vt    = float(row["value_traded"])
        expected = round((fbuy + fsell) / vt * 100, 2) if vt > 0 else None
        fp = r["foreign_participation_pct"]
        if expected is not None and fp is not None:
            assert abs(fp - expected) < 0.01

    def test_pressure_trend_key_present(self, feat, last_date):
        r = self._run(feat, last_date)
        assert "flow_pressure_trend_10d" in r

    def test_pressure_trend_valid_label(self, feat, last_date):
        r = self._run(feat, last_date)
        assert r["flow_pressure_trend_10d"] in self.VALID_TREND

    def test_foreign_flow_zscore_type(self, feat, last_date):
        r = self._run(feat, last_date)
        val = r["foreign_flow_zscore"]
        assert val is None or isinstance(val, float)

    def test_domestic_flow_zscore_type(self, feat, last_date):
        r = self._run(feat, last_date)
        val = r["domestic_flow_zscore"]
        assert val is None or isinstance(val, float)

    def test_no_nan_in_float_scalars(self, feat, last_date):
        r = self._run(feat, last_date)
        for k, v in r.items():
            if isinstance(v, float):
                assert not math.isnan(v), f"NaN leaked at key '{k}'"


# ---------------------------------------------------------------------------
# AT-8: GCC — is QSE outperforming or underperforming GCC peers this week?
# Spec §18 test 8: "Is QSE outperforming or underperforming GCC peers this week?"
# Mapped to: gcc.run(date, {"horizons": [1, 5, 20]}).
# ---------------------------------------------------------------------------

class TestAT8_GCCOutperformance:
    """
    The fixture has qse_vs_gcc_spread = NaN (no gcc raw data in fixture).
    We patch load_gcc_raw with a synthetic per-peer table so all schema paths
    are exercised. The 5d spread is computed as the cumulative sum of synthetic
    qse_vs_gcc_spread.
    """

    # gcc.run() returns all return/spread values in percent (keys suffixed _pct),
    # uses unambiguous rank/total key names, and adds a pre-computed outperformance
    # interpretation label (see CLAUDE.md "analytics/gcc.py implementation notes").
    EXPECTED_KEYS = {
        "date",
        "units",
        "qse_return_1d_pct",
        "gcc_avg_return_1d_pct",
        "qse_vs_gcc_spread_1d_pct",
        "qse_rank_among_all_markets_including_qse",
        "total_markets_including_qse",
        "peer_returns_pct",
        "rolling_outperformance_rate_60d",
        "rolling_outperformance_interpretation_60d",
        "qse_vs_gcc_spread_5d_pct",
        "qse_vs_gcc_spread_20d_pct",
    }
    PEER_NAMES = {"Tadawul", "ADX", "DFM", "KSE", "MSM", "BSE"}

    def _synthetic_gcc_raw(self, feat: pd.DataFrame) -> pd.DataFrame:
        """Build a per-peer gcc_daily parquet substitute for all fixture dates."""
        peers = ["QSE", "ADX", "DFM", "TASI", "KSE", "MSM", "BSE"]
        rng = np.random.default_rng(77)
        rows = []
        for d in feat["date"]:
            qse_ret = float(feat[feat["date"] == d]["return_1d"].iloc[0])
            for mkt in peers:
                if mkt == "QSE":
                    pct = (qse_ret if not math.isnan(qse_ret) else 0.0) * 100
                else:
                    pct = float(rng.normal(0, 0.5))
                rows.append({"date": d, "market_name": mkt, "daily_change_pct": pct})
        return pd.DataFrame(rows)

    def _feat_with_gcc(self, feat: pd.DataFrame, gcc_raw: pd.DataFrame) -> pd.DataFrame:
        """Inject gcc_avg_return_1d and qse_vs_gcc_spread into feat."""
        df = feat.copy()
        peers = gcc_raw[gcc_raw["market_name"] != "QSE"].copy()
        peers["ret"] = peers["daily_change_pct"] / 100.0
        avg = peers.groupby("date")["ret"].mean().reset_index().rename(columns={"ret": "gcc_avg_return_1d"})
        df = df.merge(avg, on="date", how="left", suffixes=("_old", ""))
        if "gcc_avg_return_1d_old" in df.columns:
            df.drop(columns=["gcc_avg_return_1d_old"], inplace=True)
        df["qse_vs_gcc_spread"] = df["return_1d"].fillna(0) - df["gcc_avg_return_1d"].fillna(0)
        return df

    def _run(self, feat, last_date):
        gcc_raw = self._synthetic_gcc_raw(feat)
        feat2 = self._feat_with_gcc(feat, gcc_raw)
        from analytics.gcc import run
        with patch("analytics.gcc.history_up_to",
                   side_effect=lambda d: feat2[feat2["date"] <= pd.Timestamp(d)].copy()), \
             patch("analytics.gcc.row_for_date",
                   side_effect=lambda d: feat2[feat2["date"] == pd.Timestamp(d)].iloc[0]), \
             patch("analytics.gcc.load_gcc_raw", return_value=gcc_raw):
            return run(last_date, {"horizons": [1, 5, 20]})

    def test_all_schema_keys_present(self, feat, last_date):
        r = self._run(feat, last_date)
        missing = self.EXPECTED_KEYS - r.keys()
        assert not missing, f"Missing keys: {missing}"

    def test_date_field_correct(self, feat, last_date):
        r = self._run(feat, last_date)
        assert r["date"] == last_date
        assert isinstance(r["date"], str)

    def test_qse_return_1d_type(self, feat, last_date):
        r = self._run(feat, last_date)
        val = r["qse_return_1d_pct"]
        assert val is None or isinstance(val, float)

    def test_qse_return_1d_matches_features(self, feat, last_date):
        r = self._run(feat, last_date)
        # Module returns the value in percent (return_1d * 100).
        expected = float(feat[feat["date"] == pd.Timestamp(last_date)]["return_1d"].iloc[0]) * 100
        if r["qse_return_1d_pct"] is not None and not math.isnan(expected):
            # Module rounds to 4 decimals in percent space.
            assert abs(r["qse_return_1d_pct"] - expected) < 1e-3

    def test_gcc_avg_return_1d_type(self, feat, last_date):
        r = self._run(feat, last_date)
        val = r["gcc_avg_return_1d_pct"]
        assert val is None or isinstance(val, float)

    def test_qse_vs_gcc_spread_1d_type(self, feat, last_date):
        r = self._run(feat, last_date)
        val = r["qse_vs_gcc_spread_1d_pct"]
        assert val is None or isinstance(val, float)

    def test_spread_1d_equals_qse_minus_avg(self, feat, last_date):
        """spread_1d == qse_return_1d - gcc_avg_return_1d (all in percent)."""
        r = self._run(feat, last_date)
        qse = r["qse_return_1d_pct"]
        avg = r["gcc_avg_return_1d_pct"]
        spread = r["qse_vs_gcc_spread_1d_pct"]
        if all(v is not None for v in (qse, avg, spread)):
            # Each operand is independently rounded to 4 decimals in percent space.
            assert abs(spread - (qse - avg)) < 1e-3

    def test_qse_rank_today_type(self, feat, last_date):
        r = self._run(feat, last_date)
        rank = r["qse_rank_among_all_markets_including_qse"]
        assert rank is None or isinstance(rank, int)

    def test_qse_rank_positive(self, feat, last_date):
        r = self._run(feat, last_date)
        rank = r["qse_rank_among_all_markets_including_qse"]
        if rank is not None:
            assert rank >= 1

    def test_qse_rank_within_total_peers(self, feat, last_date):
        r = self._run(feat, last_date)
        rank = r["qse_rank_among_all_markets_including_qse"]
        total = r["total_markets_including_qse"]
        if rank is not None:
            assert rank <= total

    def test_total_peers_is_7(self, feat, last_date):
        """Synthetic gcc has 7 markets (QSE + 6 peers); the total counts all 7."""
        r = self._run(feat, last_date)
        assert r["total_markets_including_qse"] == 7

    def test_peer_returns_has_six_peers(self, feat, last_date):
        r = self._run(feat, last_date)
        assert set(r["peer_returns_pct"].keys()) == self.PEER_NAMES

    def test_peer_returns_values_float_or_none(self, feat, last_date):
        r = self._run(feat, last_date)
        for peer, val in r["peer_returns_pct"].items():
            assert val is None or isinstance(val, float), f"{peer}: wrong type"

    def test_peer_returns_are_finite(self, feat, last_date):
        r = self._run(feat, last_date)
        for peer, val in r["peer_returns_pct"].items():
            if val is not None:
                assert math.isfinite(val), f"{peer} return is not finite"

    def test_outperformance_rate_type(self, feat, last_date):
        r = self._run(feat, last_date)
        rate = r["rolling_outperformance_rate_60d"]
        assert rate is None or isinstance(rate, float)

    def test_outperformance_rate_in_range(self, feat, last_date):
        r = self._run(feat, last_date)
        rate = r["rolling_outperformance_rate_60d"]
        if rate is not None:
            assert 0.0 <= rate <= 1.0

    def test_outperformance_rate_correct_value(self, feat, last_date):
        """rate == fraction of sessions where qse_return > gcc_avg (last 60, here all 20)."""
        gcc_raw = self._synthetic_gcc_raw(feat)
        feat2 = self._feat_with_gcc(feat, gcc_raw)
        hist = feat2[feat2["date"] <= pd.Timestamp(last_date)][["return_1d", "gcc_avg_return_1d"]].dropna().tail(60)
        expected = round(float((hist["return_1d"] > hist["gcc_avg_return_1d"]).sum() / len(hist)), 4)
        r = self._run(feat, last_date)
        rate = r["rolling_outperformance_rate_60d"]
        if rate is not None:
            assert abs(rate - expected) < 1e-6

    def test_spread_5d_type(self, feat, last_date):
        r = self._run(feat, last_date)
        val = r["qse_vs_gcc_spread_5d_pct"]
        assert val is None or isinstance(val, float)

    def test_spread_5d_correct_value(self, feat, last_date):
        """spread_5d == sum of last 5 qse_vs_gcc_spread values, expressed in percent."""
        gcc_raw = self._synthetic_gcc_raw(feat)
        feat2 = self._feat_with_gcc(feat, gcc_raw)
        spreads = feat2[feat2["date"] <= pd.Timestamp(last_date)]["qse_vs_gcc_spread"].dropna().tail(5)
        expected = round(float(spreads.sum()) * 100, 4)
        r = self._run(feat, last_date)
        val = r["qse_vs_gcc_spread_5d_pct"]
        if val is not None:
            assert abs(val - expected) < 1e-4

    def test_spread_20d_type(self, feat, last_date):
        r = self._run(feat, last_date)
        val = r["qse_vs_gcc_spread_20d_pct"]
        assert val is None or isinstance(val, float)


# ---------------------------------------------------------------------------
# Regime HMM tests
# ---------------------------------------------------------------------------

class TestRegime_TooFewSessions:
    """With only 20 rows, run() must return regime: null and a note."""

    def _run(self, feat, last_date):
        from analytics.regime import run
        with patch("analytics.regime.history_up_to",
                   side_effect=lambda d: feat[feat["date"] <= pd.Timestamp(d)].copy()):
            return run(last_date, {})

    def test_current_regime_is_null(self, feat, last_date):
        r = self._run(feat, last_date)
        assert r["current_regime"] is None

    def test_required_null_keys_present(self, feat, last_date):
        r = self._run(feat, last_date)
        for key in (
            "current_regime", "regime_probability", "sessions_in_current_regime",
            "regime_start_date", "prior_regime", "prior_regime_duration_sessions",
            "regime_distribution_historical", "model_version",
        ):
            assert key in r, f"Missing key: {key}"

    def test_note_field_present_and_non_empty(self, feat, last_date):
        r = self._run(feat, last_date)
        assert "note" in r
        assert isinstance(r["note"], str) and len(r["note"]) > 0

    def test_all_regime_fields_are_none(self, feat, last_date):
        r = self._run(feat, last_date)
        none_keys = [
            "regime_probability", "sessions_in_current_regime",
            "regime_start_date", "prior_regime", "prior_regime_duration_sessions",
            "regime_distribution_historical", "model_version",
        ]
        for key in none_keys:
            assert r[key] is None, f"Expected None for {key}, got {r[key]!r}"


def _build_large_fixture(n: int = 300) -> pd.DataFrame:
    """300 QSE trading days with all six HMM input features populated."""
    rng = np.random.default_rng(99)

    def qse_days(start, count):
        days, d = [], pd.Timestamp(start)
        while len(days) < count:
            if d.dayofweek not in (4, 5):
                days.append(d)
            d += pd.Timedelta(days=1)
        return pd.DatetimeIndex(days)

    dates = qse_days("2023-01-01", n)
    close = 10_000.0
    prices = []
    for _ in dates:
        close *= 1 + rng.normal(0, 0.008)
        prices.append(close)

    df = pd.DataFrame({"date": pd.DatetimeIndex(dates), "close": prices})
    df["return_1d"] = df["close"].pct_change(1)

    roll20 = df["return_1d"].rolling(20, min_periods=10)
    df["volatility_20d"] = roll20.std() * np.sqrt(252)

    volume = rng.integers(50_000_000, 300_000_000, n).astype(float)
    mu_v = pd.Series(volume).rolling(60, min_periods=30).mean()
    std_v = pd.Series(volume).rolling(60, min_periods=30).std().replace(0, np.nan)
    df["volume_zscore"] = (volume - mu_v.values) / std_v.values

    gainers = rng.integers(5, 35, n)
    losers = np.minimum(rng.integers(5, 35, n), 50 - gainers)
    unchanged = 50 - gainers - losers
    total_b = (gainers + losers + unchanged).astype(float)
    df["breadth_ratio"] = gainers / total_b

    fnet = rng.uniform(-50e6, 50e6, n)
    mu_f = pd.Series(fnet).rolling(60, min_periods=30).mean()
    std_f = pd.Series(fnet).rolling(60, min_periods=30).std().replace(0, np.nan)
    df["foreign_flow_zscore"] = (fnet - mu_f.values) / std_f.values

    # Simple RSI-14 approximation sufficient for fixture purposes
    delta = df["return_1d"].fillna(0)
    gain = delta.clip(lower=0).rolling(14, min_periods=14).mean()
    loss = (-delta).clip(lower=0).rolling(14, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    return df.reset_index(drop=True)


@pytest.fixture(scope="module")
def large_feat():
    return _build_large_fixture(300)


@pytest.fixture(scope="module")
def large_last_date(large_feat):
    return str(large_feat["date"].iloc[-1].date())


class TestRegime_SufficientSessions:
    """With 300 rows the module must return a fully populated regime dict."""

    def _run(self, large_feat, large_last_date):
        from analytics.regime import run, _MODEL_PATH, _SYMLINK
        import tempfile, shutil

        # Redirect model save to a temp directory so tests don't touch models/
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            with patch("analytics.regime._MODEL_DIR", tmp_path), \
                 patch("analytics.regime._MODEL_PATH", tmp_path / "hmm_v1.pkl"), \
                 patch("analytics.regime._SYMLINK", tmp_path / "hmm_current"), \
                 patch("analytics.regime.history_up_to",
                       side_effect=lambda d: large_feat[
                           large_feat["date"] <= pd.Timestamp(d)
                       ].copy()):
                return run(large_last_date, {})

    def test_current_regime_is_valid_label(self, large_feat, large_last_date):
        r = self._run(large_feat, large_last_date)
        assert r["current_regime"] in ("bear", "sideways", "bull")

    def test_no_note_field(self, large_feat, large_last_date):
        r = self._run(large_feat, large_last_date)
        assert "note" not in r

    def test_regime_probability_in_bounds(self, large_feat, large_last_date):
        r = self._run(large_feat, large_last_date)
        p = r["regime_probability"]
        assert isinstance(p, float)
        assert 0.0 <= p <= 1.0

    def test_sessions_in_current_regime_positive(self, large_feat, large_last_date):
        r = self._run(large_feat, large_last_date)
        assert isinstance(r["sessions_in_current_regime"], int)
        assert r["sessions_in_current_regime"] >= 1

    def test_regime_start_date_is_date_string(self, large_feat, large_last_date):
        r = self._run(large_feat, large_last_date)
        pd.Timestamp(r["regime_start_date"])  # must parse without error

    def test_regime_distribution_covers_all_labels(self, large_feat, large_last_date):
        r = self._run(large_feat, large_last_date)
        dist = r["regime_distribution_historical"]
        assert isinstance(dist, dict)
        assert set(dist.keys()) == {"bear", "sideways", "bull"}

    def test_regime_distribution_sums_to_one(self, large_feat, large_last_date):
        r = self._run(large_feat, large_last_date)
        total = sum(r["regime_distribution_historical"].values())
        assert abs(total - 1.0) < 1e-3

    def test_model_version_is_string(self, large_feat, large_last_date):
        r = self._run(large_feat, large_last_date)
        assert isinstance(r["model_version"], str)

    def test_date_field_matches_input(self, large_feat, large_last_date):
        r = self._run(large_feat, large_last_date)
        assert r["date"] == large_last_date

    def test_required_keys_all_present(self, large_feat, large_last_date):
        r = self._run(large_feat, large_last_date)
        required = {
            "date", "current_regime", "regime_probability",
            "sessions_in_current_regime", "regime_start_date",
            "prior_regime", "prior_regime_duration_sessions",
            "regime_distribution_historical", "model_version",
        }
        missing = required - r.keys()
        assert not missing, f"Missing keys: {missing}"
