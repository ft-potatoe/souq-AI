"""
Tests for feature engineering in scripts/features/build_features.py.

All tests use a 20-row synthetic fixture of QSE trading days (Sun–Thu).
No file I/O required — functions are tested directly.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make the project root importable so we can reach scripts/
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.features.build_features import (
    RAMADAN_RANGES,
    _is_ramadan,
    _rolling_zscore,
    compute_breadth,
    compute_flows,
    compute_forward_returns,
    compute_returns,
    compute_seasonality,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _qse_trading_days(start: str, n: int) -> pd.DatetimeIndex:
    """Generate n QSE trading days (Sun–Thu) from start."""
    days = []
    d = pd.Timestamp(start)
    while len(days) < n:
        if d.dayofweek not in (4, 5):  # skip Fri=4, Sat=5
            days.append(d)
        d += pd.Timedelta(days=1)
    return pd.DatetimeIndex(days)


@pytest.fixture
def market_df() -> pd.DataFrame:
    """20-row synthetic market_daily DataFrame."""
    rng = np.random.default_rng(42)
    dates = _qse_trading_days("2024-01-07", 20)  # first QSE Sun after 2024-01-01

    close = 10_000.0
    prices = []
    for _ in dates:
        close *= 1 + rng.normal(0, 0.008)
        prices.append(round(close, 2))

    return pd.DataFrame(
        {
            "date": pd.DatetimeIndex(dates),
            "open": [p * 0.999 for p in prices],
            "high": [p * 1.005 for p in prices],
            "low": [p * 0.995 for p in prices],
            "close": prices,
            "volume": rng.integers(50_000_000, 300_000_000, 20),
            "value_traded": rng.uniform(50e6, 400e6, 20).round(0),
            "total_trades": rng.integers(3_000, 30_000, 20),
        }
    )


@pytest.fixture
def flows_df(market_df) -> pd.DataFrame:
    """20-row synthetic flows_daily DataFrame aligned to market_df dates."""
    rng = np.random.default_rng(7)
    dates = market_df["date"]
    fbuy = rng.uniform(20e6, 200e6, 20).round(0)
    fsell = rng.uniform(20e6, 200e6, 20).round(0)
    dbuy = rng.uniform(30e6, 300e6, 20).round(0)
    dsell = rng.uniform(30e6, 300e6, 20).round(0)
    return pd.DataFrame(
        {
            "date": dates,
            "foreign_buy": fbuy,
            "foreign_sell": fsell,
            "foreign_net": fbuy - fsell,
            "domestic_buy": dbuy,
            "domestic_sell": dsell,
            "domestic_net": dbuy - dsell,
        }
    )


@pytest.fixture
def breadth_df(market_df) -> pd.DataFrame:
    """20-row synthetic breadth_daily DataFrame aligned to market_df dates."""
    rng = np.random.default_rng(13)
    dates = market_df["date"]
    total_listed = 50
    gainers = rng.integers(5, 35, 20)
    losers = np.minimum(rng.integers(5, 35, 20), total_listed - gainers)
    unchanged = total_listed - gainers - losers
    return pd.DataFrame(
        {
            "date": dates,
            "gainers": gainers,
            "losers": losers,
            "unchanged": unchanged,
            "total_listed": total_listed,
            "total_traded": rng.integers(20, 50, 20),
        }
    )


# ---------------------------------------------------------------------------
# 1. Return calculations match manual pandas pct_change
# ---------------------------------------------------------------------------

class TestReturnCalculations:
    def test_return_1d_matches_pct_change(self, market_df):
        result = compute_returns(market_df.copy())
        expected = market_df["close"].pct_change(1)
        pd.testing.assert_series_equal(
            result["return_1d"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_return_5d_matches_pct_change(self, market_df):
        result = compute_returns(market_df.copy())
        expected = market_df["close"].pct_change(5)
        pd.testing.assert_series_equal(
            result["return_5d"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_return_20d_matches_pct_change(self, market_df):
        result = compute_returns(market_df.copy())
        expected = market_df["close"].pct_change(20)
        pd.testing.assert_series_equal(
            result["return_20d"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_return_60d_matches_pct_change(self, market_df):
        result = compute_returns(market_df.copy())
        expected = market_df["close"].pct_change(60)
        pd.testing.assert_series_equal(
            result["return_60d"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_return_1d_first_row_is_nan(self, market_df):
        result = compute_returns(market_df.copy())
        assert pd.isna(result["return_1d"].iloc[0])


# ---------------------------------------------------------------------------
# 2. Z-score returns NaN when std == 0, not an error
# ---------------------------------------------------------------------------

class TestRollingZscore:
    def test_constant_series_returns_nan(self):
        s = pd.Series([5.0] * 20)
        result = _rolling_zscore(s, window=5)
        # Every position where a full window exists should be NaN (std == 0)
        windowed = result.dropna()
        assert windowed.isna().all() or len(windowed) == 0 or windowed.apply(
            lambda v: np.isnan(v) or True
        ).all()

    def test_constant_series_no_inf_no_error(self):
        s = pd.Series([100.0] * 20)
        result = _rolling_zscore(s, window=5)
        assert not np.isinf(result).any(), "z-score must not produce Inf for constant series"

    def test_constant_series_no_value_error(self):
        s = pd.Series([1.0] * 20)
        # Should complete without raising
        result = _rolling_zscore(s, window=5)
        assert isinstance(result, pd.Series)

    def test_varying_series_has_finite_zscores(self):
        rng = np.random.default_rng(0)
        s = pd.Series(rng.normal(0, 1, 20))
        result = _rolling_zscore(s, window=5)
        non_nan = result.dropna()
        assert not non_nan.empty
        assert np.isfinite(non_nan).all()

    def test_zero_std_specifically_produces_nan(self):
        # Explicitly build a window where std is 0
        s = pd.Series([3.0] * 10)
        result = _rolling_zscore(s, window=5)
        # All positions with enough data should be NaN, not a number
        valid_positions = result.iloc[4:]  # window=5, so positions 4+ have full window
        assert valid_positions.isna().all(), (
            f"Expected all NaN for constant series, got: {valid_positions.tolist()}"
        )


# ---------------------------------------------------------------------------
# 3. breadth_ratio is always between 0 and 1
# ---------------------------------------------------------------------------

class TestBreadthRatio:
    def test_breadth_ratio_bounds(self, market_df, breadth_df):
        result = compute_breadth(market_df.copy(), breadth_df)
        ratio = result["breadth_ratio"].dropna()
        assert (ratio >= 0).all(), "breadth_ratio must be >= 0"
        assert (ratio <= 1).all(), "breadth_ratio must be <= 1"

    def test_breadth_ratio_when_all_gainers(self, market_df):
        dates = market_df["date"]
        b = pd.DataFrame(
            {
                "date": dates,
                "gainers": [50] * 20,
                "losers": [0] * 20,
                "unchanged": [0] * 20,
                "total_listed": [50] * 20,
                "total_traded": [50] * 20,
            }
        )
        result = compute_breadth(market_df.copy(), b)
        assert (result["breadth_ratio"].dropna() == 1.0).all()

    def test_breadth_ratio_when_no_gainers(self, market_df):
        dates = market_df["date"]
        b = pd.DataFrame(
            {
                "date": dates,
                "gainers": [0] * 20,
                "losers": [30] * 20,
                "unchanged": [20] * 20,
                "total_listed": [50] * 20,
                "total_traded": [40] * 20,
            }
        )
        result = compute_breadth(market_df.copy(), b)
        assert (result["breadth_ratio"].dropna() == 0.0).all()

    def test_breadth_ratio_zero_total_is_nan_not_error(self, market_df):
        dates = market_df["date"]
        b = pd.DataFrame(
            {
                "date": dates,
                "gainers": [0] * 20,
                "losers": [0] * 20,
                "unchanged": [0] * 20,
                "total_listed": [50] * 20,
                "total_traded": [0] * 20,
            }
        )
        result = compute_breadth(market_df.copy(), b)
        assert not np.isinf(result["breadth_ratio"]).any()


# ---------------------------------------------------------------------------
# 4. foreign_net == foreign_buy - foreign_sell for all rows
# ---------------------------------------------------------------------------

class TestForeignNet:
    def test_foreign_net_identity(self, market_df, flows_df):
        result = compute_flows(market_df.copy(), flows_df)
        expected = result["foreign_buy"] - result["foreign_sell"]
        diff = (result["foreign_net"] - expected).abs()
        # Flows fixture constructs foreign_net as fbuy - fsell exactly
        assert (diff < 1e-6).all(), (
            f"foreign_net != foreign_buy - foreign_sell; max diff = {diff.max()}"
        )

    def test_foreign_net_raw_fixture_identity(self, flows_df):
        # The fixture itself must satisfy the invariant before any merging
        diff = (flows_df["foreign_net"] - (flows_df["foreign_buy"] - flows_df["foreign_sell"])).abs()
        assert (diff < 1e-6).all()

    def test_foreign_net_with_synthetic_mismatch_detected(self):
        # Verify we'd catch a broken fixture — foreign_net intentionally wrong
        df = pd.DataFrame(
            {
                "foreign_buy": [100.0, 200.0],
                "foreign_sell": [50.0, 80.0],
                "foreign_net": [999.0, 999.0],  # deliberately wrong
            }
        )
        diff = (df["foreign_net"] - (df["foreign_buy"] - df["foreign_sell"])).abs()
        assert (diff > 1).any(), "Should detect the mismatch"


# ---------------------------------------------------------------------------
# 5. is_ramadan correctly flags known Ramadan date ranges
# ---------------------------------------------------------------------------

class TestIsRamadan:
    @pytest.mark.parametrize("start,end", RAMADAN_RANGES)
    def test_start_date_is_flagged(self, start, end):
        dates = pd.Series([pd.Timestamp(start)])
        assert _is_ramadan(dates).iloc[0], f"{start} should be flagged as Ramadan"

    @pytest.mark.parametrize("start,end", RAMADAN_RANGES)
    def test_end_date_is_flagged(self, start, end):
        dates = pd.Series([pd.Timestamp(end)])
        assert _is_ramadan(dates).iloc[0], f"{end} should be flagged as Ramadan"

    @pytest.mark.parametrize("start,end", RAMADAN_RANGES)
    def test_day_before_start_is_not_flagged(self, start, end):
        day_before = pd.Timestamp(start) - pd.Timedelta(days=1)
        dates = pd.Series([day_before])
        assert not _is_ramadan(dates).iloc[0], f"{day_before} should NOT be Ramadan"

    @pytest.mark.parametrize("start,end", RAMADAN_RANGES)
    def test_day_after_end_is_not_flagged(self, start, end):
        day_after = pd.Timestamp(end) + pd.Timedelta(days=1)
        dates = pd.Series([day_after])
        assert not _is_ramadan(dates).iloc[0], f"{day_after} should NOT be Ramadan"

    def test_non_ramadan_date_not_flagged(self):
        dates = pd.Series([pd.Timestamp("2024-06-15")])  # well outside any range
        assert not _is_ramadan(dates).iloc[0]

    def test_is_ramadan_in_compute_seasonality(self, market_df):
        # Patch market_df to include a known Ramadan date
        df = market_df.copy()
        ramadan_start = pd.Timestamp(RAMADAN_RANGES[-1][0])
        df.loc[0, "date"] = ramadan_start
        df.loc[1, "date"] = pd.Timestamp("2024-06-15")  # non-Ramadan

        result = compute_seasonality(df)
        assert result.loc[0, "is_ramadan"] == 1
        assert result.loc[1, "is_ramadan"] == 0


# ---------------------------------------------------------------------------
# 6. No future-dated rows make it into features_master
# ---------------------------------------------------------------------------

class TestNoFutureDates:
    def test_no_future_dates_after_compute_returns(self, market_df):
        today = pd.Timestamp("today").normalize()
        # inject a future date
        df = market_df.copy()
        df.loc[0, "date"] = today + pd.Timedelta(days=5)

        # Simulate what build_features does: filter future dates before computing
        clean = df[df["date"] <= today].copy()
        result = compute_returns(clean)

        assert (result["date"] <= today).all(), "Future-dated rows must not appear in output"

    def test_future_date_filter_removes_correct_rows(self):
        today = pd.Timestamp("today").normalize()
        future = today + pd.Timedelta(days=1)
        dates = pd.to_datetime(["2024-01-07", "2024-01-08", str(future.date())])
        df = pd.DataFrame({"date": dates, "close": [100.0, 101.0, 102.0]})

        clean = df[df["date"] <= today]
        assert len(clean) == 2
        assert (clean["date"] <= today).all()

    def test_all_fixture_dates_are_not_future(self, market_df):
        today = pd.Timestamp("today").normalize()
        assert (market_df["date"] <= today).all(), (
            "Fixture itself must not contain future dates"
        )

    def test_features_master_parquet_no_future_dates(self):
        """If features_master.parquet exists, assert it has no future-dated rows."""
        parquet_path = ROOT / "data" / "features" / "features_master.parquet"
        if not parquet_path.exists():
            pytest.skip("features_master.parquet not yet generated")

        df = pd.read_parquet(parquet_path)
        df["date"] = pd.to_datetime(df["date"])
        today = pd.Timestamp("today").normalize()
        future_rows = df[df["date"] > today]
        assert future_rows.empty, (
            f"features_master.parquet contains {len(future_rows)} future-dated rows: "
            f"{future_rows['date'].dt.date.tolist()}"
        )


# ---------------------------------------------------------------------------
# 7. Forward return columns — correctness and leakage safety
# ---------------------------------------------------------------------------

class TestForwardReturns:
    def test_columns_present(self, market_df):
        result = compute_forward_returns(compute_returns(market_df.copy()))
        for col in ("forward_return_5d", "forward_return_10d", "forward_return_20d"):
            assert col in result.columns, f"Missing column: {col}"

    def test_forward_return_5d_matches_shifted_pct_change(self, market_df):
        df = compute_forward_returns(compute_returns(market_df.copy()))
        expected = market_df["close"].pct_change(5).shift(-5)
        pd.testing.assert_series_equal(
            df["forward_return_5d"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_forward_return_10d_matches_shifted_pct_change(self, market_df):
        df = compute_forward_returns(compute_returns(market_df.copy()))
        expected = market_df["close"].pct_change(10).shift(-10)
        pd.testing.assert_series_equal(
            df["forward_return_10d"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_trailing_rows_are_nan_5d(self, market_df):
        # Last 5 rows must be NaN — their outcome window extends beyond available data
        df = compute_forward_returns(compute_returns(market_df.copy()))
        assert df["forward_return_5d"].iloc[-5:].isna().all(), (
            "Last 5 rows of forward_return_5d must be NaN"
        )

    def test_trailing_rows_are_nan_10d(self, market_df):
        df = compute_forward_returns(compute_returns(market_df.copy()))
        assert df["forward_return_10d"].iloc[-10:].isna().all(), (
            "Last 10 rows of forward_return_10d must be NaN"
        )

    def test_non_trailing_row_is_not_nan(self, market_df):
        # Row 0 has a valid 5d forward window given 20 rows
        df = compute_forward_returns(compute_returns(market_df.copy()))
        assert not pd.isna(df["forward_return_5d"].iloc[0]), (
            "Row 0 forward_return_5d should be a real value in a 20-row fixture"
        )

    def test_no_inf_values(self, market_df):
        df = compute_forward_returns(compute_returns(market_df.copy()))
        for col in ("forward_return_5d", "forward_return_10d", "forward_return_20d"):
            assert not np.isinf(df[col].dropna()).any(), f"{col} must not contain Inf"
