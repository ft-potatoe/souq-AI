"""
Shared parquet loader used by all analytics modules.
Returns a cached DataFrame for the lifetime of a process.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
_FEATURES_PATH = _ROOT / "data" / "features" / "features_master.parquet"
_GCC_RAW_PATH = _ROOT / "data" / "raw" / "gcc_daily.parquet"


@lru_cache(maxsize=1)
def load_features() -> pd.DataFrame:
    df = pd.read_parquet(_FEATURES_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


@lru_cache(maxsize=1)
def load_gcc_raw() -> pd.DataFrame:
    df = pd.read_parquet(_GCC_RAW_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["date", "market_name"]).reset_index(drop=True)


def row_for_date(date: str | pd.Timestamp) -> pd.Series:
    """Return the single features row for *date*; raise KeyError if missing."""
    df = load_features()
    ts = pd.Timestamp(date)
    mask = df["date"] == ts
    if not mask.any():
        raise KeyError(f"No features row for date {ts.date()}")
    return df.loc[mask].iloc[0]


def history_up_to(date: str | pd.Timestamp) -> pd.DataFrame:
    """All feature rows with date <= *date*, sorted ascending."""
    df = load_features()
    ts = pd.Timestamp(date)
    return df[df["date"] <= ts].copy()
