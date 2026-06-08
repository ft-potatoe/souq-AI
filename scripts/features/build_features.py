"""
Build feature master from data/raw/ parquets and write to
data/features/features_master.parquet.

Run:
  python scripts/features/build_features.py
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
FEAT_DIR = ROOT / "data" / "features"
LOG_DIR = ROOT / "logs"
FEAT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("features")
logger.setLevel(logging.DEBUG)
_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
_fh = logging.FileHandler(LOG_DIR / "feature_build.log", encoding="utf-8")
_fh.setLevel(logging.WARNING)
_fh.setFormatter(_fmt)
_sh = logging.StreamHandler(sys.stdout)
_sh.setLevel(logging.INFO)
_sh.setFormatter(_fmt)
logger.addHandler(_fh)
logger.addHandler(_sh)

# ---------------------------------------------------------------------------
# Ramadan date ranges (extend as needed)
# ---------------------------------------------------------------------------
RAMADAN_RANGES = [
    ("2018-05-16", "2018-06-14"),
    ("2019-05-05", "2019-06-03"),
    ("2020-04-23", "2020-05-23"),
    ("2021-04-12", "2021-05-12"),
    ("2022-04-02", "2022-05-01"),
    ("2023-03-22", "2023-04-20"),
    ("2024-03-10", "2024-04-08"),
    ("2025-02-28", "2025-03-29"),
    ("2026-02-17", "2026-03-18"),
]


def _is_ramadan(dates: pd.Series) -> pd.Series:
    result = pd.Series(False, index=dates.index)
    for start, end in RAMADAN_RANGES:
        result |= dates.between(pd.Timestamp(start), pd.Timestamp(end))
    return result


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _rolling_zscore(s: pd.Series, window: int) -> pd.Series:
    """Rolling z-score; returns NaN when std == 0 instead of dividing by zero."""
    mean = s.rolling(window, min_periods=window // 2).mean()
    std = s.rolling(window, min_periods=window // 2).std()
    std = std.replace(0, np.nan)
    return (s - mean) / std


def _ols_slope(s: pd.Series, window: int) -> pd.Series:
    """Rolling OLS slope over `window` periods."""
    x = np.arange(window, dtype=float)

    def _slope(y):
        if np.isnan(y).any():
            return np.nan
        return sp_stats.linregress(x, y).slope

    return s.rolling(window, min_periods=window).apply(_slope, raw=True)


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI without external TA library dependency."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ---------------------------------------------------------------------------
# Feature blocks
# ---------------------------------------------------------------------------

def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    """return_1d, return_5d, return_20d, return_60d."""
    px = df["close"]
    df["return_1d"] = px.pct_change(1)
    df["return_5d"] = px.pct_change(5)
    df["return_20d"] = px.pct_change(20)
    df["return_60d"] = px.pct_change(60)
    return df


def compute_forward_returns(df: pd.DataFrame) -> pd.DataFrame:
    """forward_return_5d, forward_return_10d, forward_return_20d.

    Trailing rows within the lookahead window are NaN — the similarity
    ranker's safe_forward_returns() guard masks them again at inference
    time, but NaN here is the ground truth: those outcomes are unknown.
    """
    px = df["close"]
    df["forward_return_5d"] = px.pct_change(5).shift(-5)
    df["forward_return_10d"] = px.pct_change(10).shift(-10)
    df["forward_return_20d"] = px.pct_change(20).shift(-20)
    return df


def compute_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """Annualised rolling std × √252."""
    r = df["return_1d"]
    df["volatility_20d"] = r.rolling(20, min_periods=10).std() * np.sqrt(252)
    df["volatility_60d"] = r.rolling(60, min_periods=30).std() * np.sqrt(252)
    return df


def compute_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """RSI-14, SMAs, above_sma_20, above_sma_200, price_vs_sma20_pct."""
    px = df["close"]
    df["rsi_14"] = _rsi(px, 14)
    df["sma_20"] = px.rolling(20, min_periods=10).mean()
    df["sma_50"] = px.rolling(50, min_periods=25).mean()
    df["sma_200"] = px.rolling(200, min_periods=100).mean()
    df["above_sma_20"] = (px > df["sma_20"]).astype(int)
    df["above_sma_200"] = (px > df["sma_200"]).astype(int)
    df["price_vs_sma20_pct"] = (px - df["sma_20"]) / df["sma_20"].replace(0, np.nan)
    return df


def compute_volume_zscores(df: pd.DataFrame) -> pd.DataFrame:
    """Z-scores for volume, value_traded, total_trades (60d window)."""
    df["volume_zscore"] = _rolling_zscore(df["volume"], 60)
    df["value_zscore"] = _rolling_zscore(df["value_traded"], 60)
    df["trades_zscore"] = _rolling_zscore(df["total_trades"], 60)
    return df


def compute_breadth(df_market: pd.DataFrame, df_breadth: pd.DataFrame) -> pd.DataFrame:
    """
    Merge breadth into market df, then compute breadth features.
    Returns enriched market df.
    """
    b = df_breadth[["date", "gainers", "losers", "unchanged", "total_listed", "total_traded"]].copy()
    df = df_market.merge(b, on="date", how="left")

    total = df["gainers"] + df["losers"] + df["unchanged"]
    total = total.replace(0, np.nan)

    df["breadth_ratio"] = df["gainers"] / total
    df["breadth_net"] = df["gainers"] - df["losers"]
    df["advance_decline"] = df["gainers"] / df["losers"].replace(0, np.nan)
    df["breadth_zscore"] = _rolling_zscore(df["breadth_ratio"], 60)
    return df


def compute_flows(df_market: pd.DataFrame, df_flows: pd.DataFrame) -> pd.DataFrame:
    """
    Merge flows into market df and compute flow features.
    Returns enriched market df.
    """
    f = df_flows[["date", "foreign_buy", "foreign_sell", "foreign_net",
                  "domestic_buy", "domestic_sell", "domestic_net"]].copy()
    df = df_market.merge(f, on="date", how="left")

    df["foreign_net_cumulative_5d"] = df["foreign_net"].rolling(5, min_periods=1).sum()
    df["foreign_net_cumulative_20d"] = df["foreign_net"].rolling(20, min_periods=1).sum()

    total_flow = (df["foreign_buy"] + df["foreign_sell"] +
                  df["domestic_buy"] + df["domestic_sell"])
    df["foreign_participation"] = (
        (df["foreign_buy"] + df["foreign_sell"]) / total_flow.replace(0, np.nan)
    )

    df["foreign_flow_slope_10d"] = _ols_slope(df["foreign_net"], 10)
    df["foreign_flow_zscore"] = _rolling_zscore(df["foreign_net"], 60)
    df["domestic_flow_zscore"] = _rolling_zscore(df["domestic_net"], 60)
    return df


def compute_gcc_features(df_market: pd.DataFrame, df_gcc: pd.DataFrame) -> pd.DataFrame:
    """
    Compute GCC-relative features and merge into market df.
    """
    gcc = df_gcc.copy()
    gcc["date"] = pd.to_datetime(gcc["date"])
    gcc = gcc.sort_values(["market_name", "date"])

    # daily_change_pct is already the 1d return per spec §16
    gcc["gcc_ret_1d"] = gcc["daily_change_pct"] / 100.0  # normalise pct to decimal

    # Exclude QSE from peer average (QSE is in market_daily)
    peers = gcc[gcc["market_name"].str.upper() != "QSE"].copy()

    peer_avg = (
        peers.groupby("date")["gcc_ret_1d"]
        .mean()
        .reset_index()
        .rename(columns={"gcc_ret_1d": "gcc_avg_return_1d"})
    )

    df = df_market.merge(peer_avg, on="date", how="left")
    df["qse_vs_gcc_spread"] = df["return_1d"] - df["gcc_avg_return_1d"]

    # 5d cumulative spread
    df["qse_gcc_relative_5d"] = df["qse_vs_gcc_spread"].rolling(5, min_periods=3).sum()

    # 20d rolling correlation between QSE return_1d and gcc_avg_return_1d
    df["qse_gcc_rolling_corr_20d"] = (
        df["return_1d"]
        .rolling(20, min_periods=10)
        .corr(df["gcc_avg_return_1d"])
    )

    # Rank of QSE among GCC peers on each date (1 = best)
    qse_ret = df[["date", "return_1d"]].rename(columns={"return_1d": "qse_ret"})
    all_rets = peers[["date", "gcc_ret_1d"]].copy()

    def _rank_qse(date_val, qse_r):
        if pd.isna(qse_r):
            return np.nan
        peers_day = all_rets.loc[all_rets["date"] == date_val, "gcc_ret_1d"].dropna()
        if peers_day.empty:
            return np.nan
        all_combined = list(peers_day) + [qse_r]
        all_combined_sorted = sorted(all_combined, reverse=True)
        return all_combined_sorted.index(qse_r) + 1

    df["gcc_peer_rank"] = [
        _rank_qse(row["date"], row["return_1d"])
        for _, row in df[["date", "return_1d"]].iterrows()
    ]

    return df


def compute_seasonality(df: pd.DataFrame) -> pd.DataFrame:
    """day_of_week, month, quarter, is_ramadan, trading_day_of_month."""
    dates = df["date"]
    # QSE: Sunday=0, Monday=1, ..., Thursday=4  (dayofweek: Mon=0..Sun=6 → remap)
    dow_pandas = dates.dt.dayofweek  # Mon=0..Sun=6
    qse_dow_map = {6: 0, 0: 1, 1: 2, 2: 3, 3: 4}  # Sun→0, Mon→1 ... Thu→4
    df["day_of_week"] = dow_pandas.map(qse_dow_map)

    df["month"] = dates.dt.month
    df["quarter"] = dates.dt.quarter
    df["is_ramadan"] = _is_ramadan(dates).astype(int)

    # Trading day of month: rank within calendar month (Sun–Thu only)
    df["_ym"] = dates.dt.to_period("M")
    df["trading_day_of_month"] = (
        df.groupby("_ym").cumcount() + 1
    )
    df.drop(columns=["_ym"], inplace=True)
    return df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def load_parquet(name: str) -> pd.DataFrame:
    path = RAW_DIR / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Raw parquet not found: {path}")
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def build_features() -> pd.DataFrame:
    logger.info("Loading raw parquets...")
    df_market = load_parquet("market_daily")
    df_flows = load_parquet("flows_daily")
    df_gcc = load_parquet("gcc_daily")
    df_breadth = load_parquet("breadth_daily")

    logger.info("Computing returns & volatility...")
    df = compute_returns(df_market.copy())
    df = compute_forward_returns(df)
    df = compute_volatility(df)

    logger.info("Computing momentum indicators...")
    df = compute_momentum(df)

    logger.info("Computing volume z-scores...")
    df = compute_volume_zscores(df)

    logger.info("Computing breadth features...")
    df = compute_breadth(df, df_breadth)

    logger.info("Computing flow features...")
    df = compute_flows(df, df_flows)

    logger.info("Computing GCC relative features...")
    df = compute_gcc_features(df, df_gcc)

    logger.info("Computing seasonality features...")
    df = compute_seasonality(df)

    # Final sort and reset
    df = df.sort_values("date").reset_index(drop=True)

    out_path = FEAT_DIR / "features_master.parquet"
    df.to_parquet(out_path, index=False)
    logger.info("Wrote %d rows x %d cols -> %s", len(df), len(df.columns), out_path)

    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        df = build_features()
        print(f"\nFeature matrix shape: {df.shape}")
        print(f"Columns:\n{list(df.columns)}")
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        sys.exit(1)
