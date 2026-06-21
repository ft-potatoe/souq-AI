"""
Tests for scripts/ingest/load_raw.py — focused on the merge-on-write behaviour
that preserves history when incremental/overlapping slices are ingested.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.ingest.load_raw as lr


def _trading(start: str, end: str) -> pd.DatetimeIndex:
    d = pd.date_range(start, end, freq="D")
    return d[~d.dayofweek.isin([4, 5])]  # exclude Fri/Sat (QSE week is Sun-Thu)


@pytest.fixture
def raw_dir(tmp_path, monkeypatch):
    """Redirect the module's RAW_DIR to a temp dir for isolation."""
    monkeypatch.setattr(lr, "RAW_DIR", tmp_path)
    return tmp_path


def test_merge_preserves_history_and_extends(raw_dir):
    """Overlapping slice keeps old history, updates overlap, extends the range."""
    existing = pd.DataFrame({"date": _trading("2024-01-01", "2026-06-30")})
    existing["close"] = 10.0
    existing.to_parquet(raw_dir / "market_daily.parquet", index=False)

    new = pd.DataFrame({"date": _trading("2026-03-01", "2026-09-30")})
    new["close"] = 99.0

    merged = lr._merge_with_existing(new, "market_daily")

    assert not merged["date"].duplicated().any()
    # History start preserved (first trading day of the existing range).
    assert merged["date"].min() == existing["date"].min()
    # Range extended into Sep 2026 by the new slice.
    assert merged["date"].max() >= pd.Timestamp("2026-09-01")
    # Pre-overlap history retained (old value)
    assert merged.loc[merged["date"] == "2024-05-15", "close"].iloc[0] == 10.0
    # Overlap takes the NEW value
    assert merged.loc[merged["date"] == "2026-04-15", "close"].iloc[0] == 99.0


def test_merge_is_idempotent(raw_dir):
    """Re-merging identical data must not duplicate or drop rows."""
    df = pd.DataFrame({"date": _trading("2024-01-01", "2024-03-31")})
    df["close"] = 5.0
    df.to_parquet(raw_dir / "market_daily.parquet", index=False)

    merged = lr._merge_with_existing(df.copy(), "market_daily")
    assert len(merged) == len(df)
    assert not merged["date"].duplicated().any()


def test_merge_with_no_existing_file_returns_input(raw_dir):
    """First-ever ingest (no existing parquet) returns the new data unchanged."""
    new = pd.DataFrame({"date": _trading("2024-01-01", "2024-02-01")})
    new["close"] = 1.0
    merged = lr._merge_with_existing(new, "market_daily")
    assert len(merged) == len(new)


def test_gcc_merge_keys_on_date_and_market(raw_dir):
    """gcc_daily dedups on (date, market_name), so peers on the same day coexist."""
    dates = _trading("2024-01-01", "2024-01-31")
    rows = []
    for d in dates:
        for mkt in ["QSE", "ADX", "DFM"]:
            rows.append({"date": d, "market_name": mkt, "daily_change_pct": 1.0})
    existing = pd.DataFrame(rows)
    existing.to_parquet(raw_dir / "gcc_daily.parquet", index=False)

    # New slice updates only QSE for the same dates.
    new_rows = [{"date": d, "market_name": "QSE", "daily_change_pct": 9.0} for d in dates]
    new = pd.DataFrame(new_rows)

    merged = lr._merge_with_existing(new, "gcc_daily")

    # All three markets still present per day (no collapse to one key).
    assert set(merged["market_name"].unique()) == {"QSE", "ADX", "DFM"}
    # QSE updated to new value; ADX untouched.
    qse = merged[(merged["market_name"] == "QSE") & (merged["date"] == dates[0])]
    adx = merged[(merged["market_name"] == "ADX") & (merged["date"] == dates[0])]
    assert qse["daily_change_pct"].iloc[0] == 9.0
    assert adx["daily_change_pct"].iloc[0] == 1.0


def test_dedup_key_selects_market_for_gcc():
    assert lr._dedup_key("gcc_daily") == ["date", "market_name"]
    assert lr._dedup_key("market_daily") == ["date"]
