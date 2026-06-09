"""
Ingest CSV/Excel source files, validate, and write parquet to data/raw/.

Datasets handled:
  market_daily   – daily index/market summary
  flows_daily    – foreign & domestic money flows
  gcc_daily      – GCC peer market prices
  breadth_daily  – advance/decline breadth data

Run:
  python scripts/ingest/load_raw.py --src <source_dir>
"""

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
LOG_DIR = ROOT / "logs"
RAW_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("ingest")
logger.setLevel(logging.DEBUG)

_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

_fh = logging.FileHandler(LOG_DIR / "ingestion_errors.log", encoding="utf-8")
_fh.setLevel(logging.WARNING)
_fh.setFormatter(_fmt)

_sh = logging.StreamHandler(sys.stdout)
_sh.setLevel(logging.INFO)
_sh.setFormatter(_fmt)

logger.addHandler(_fh)
logger.addHandler(_sh)

# ---------------------------------------------------------------------------
# Schemas  (required columns per dataset)
# ---------------------------------------------------------------------------
SCHEMAS = {
    "market_daily": [
        "date", "open", "high", "low", "close",
        "volume", "value_traded", "total_trades",
    ],
    "flows_daily": [
        "date", "foreign_buy", "foreign_sell", "foreign_net",
        "domestic_buy", "domestic_sell", "domestic_net",
    ],
    "gcc_daily": [
        "date", "market_name", "daily_change_pct",
    ],
    "breadth_daily": [
        "date", "gainers", "losers", "unchanged",
        "total_listed", "total_traded",
    ],
}

KNOWN_DATASETS = set(SCHEMAS.keys())

# GCC markets that must appear on every trading day
GCC_ACTIVE_MARKETS = {"QSE", "ADX", "DFM", "TASI", "KSE"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_file(path: Path) -> pd.DataFrame:
    """Read CSV or Excel into a DataFrame."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, parse_dates=["date"])
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path, parse_dates=["date"])
    raise ValueError(f"Unsupported file type: {suffix}")


def _check_required_columns(df: pd.DataFrame, dataset: str, src: Path) -> bool:
    required = set(SCHEMAS[dataset])
    missing = required - set(df.columns)
    if missing:
        logger.error("[%s] %s — missing columns: %s", dataset, src.name, missing)
        return False
    return True


def _trading_days(df: pd.DataFrame) -> pd.Series:
    """Return boolean mask for rows that are weekday (Sun–Thu for QSE)."""
    dow = df["date"].dt.dayofweek  # Mon=0 … Sun=6
    # QSE trades Sun(6)–Thu(3); exclude Fri(4) and Sat(5)
    return ~dow.isin([4, 5])


# ---------------------------------------------------------------------------
# Validation functions  — each returns (clean_df, quarantine_df)
# ---------------------------------------------------------------------------

def _validate_no_future_dates(df: pd.DataFrame, dataset: str):
    today = pd.Timestamp(date.today())
    mask = df["date"] > today
    if mask.any():
        logger.warning(
            "[%s] %d future-dated rows quarantined: %s",
            dataset, mask.sum(),
            df.loc[mask, "date"].dt.date.tolist(),
        )
    return df[~mask].copy(), df[mask].copy()


def _validate_no_duplicate_dates(df: pd.DataFrame, dataset: str):
    # gcc_daily is keyed by (date, market_name)
    key = ["date", "market_name"] if dataset == "gcc_daily" else ["date"]
    dupes = df.duplicated(subset=key, keep=False)
    if dupes.any():
        logger.warning(
            "[%s] %d duplicate-key rows quarantined.", dataset, dupes.sum()
        )
    return df[~dupes].copy(), df[dupes].copy()


def _validate_positive_trading_cols(df: pd.DataFrame, dataset: str):
    """volume, value_traded, total_trades must be > 0 on trading days."""
    if dataset not in ("market_daily", "gcc_daily"):
        return df, pd.DataFrame()

    trading = _trading_days(df)
    cols = [c for c in ["volume", "value_traded", "total_trades"] if c in df.columns]
    bad = trading & (df[cols] <= 0).any(axis=1)
    if bad.any():
        logger.warning(
            "[%s] %d rows with non-positive trading columns quarantined.",
            dataset, bad.sum(),
        )
    return df[~bad].copy(), df[bad].copy()


def _validate_breadth_sum(df: pd.DataFrame, dataset: str):
    if dataset != "breadth_daily":
        return df, pd.DataFrame()

    computed = df["gainers"] + df["losers"] + df["unchanged"]
    diff = (computed - df["total_listed"]).abs()
    bad = diff > 2
    if bad.any():
        logger.warning(
            "[%s] %d rows where gainers+losers+unchanged != total_listed (±2) quarantined.",
            dataset, bad.sum(),
        )
    return df[~bad].copy(), df[bad].copy()


def _validate_foreign_net(df: pd.DataFrame, dataset: str):
    if dataset != "flows_daily":
        return df, pd.DataFrame()

    computed_foreign = df["foreign_buy"] - df["foreign_sell"]
    diff_f = (computed_foreign - df["foreign_net"]).abs()

    computed_domestic = df["domestic_buy"] - df["domestic_sell"]
    diff_d = (computed_domestic - df["domestic_net"]).abs()

    bad = (diff_f > 1000) | (diff_d > 1000)
    if bad.any():
        logger.warning(
            "[%s] %d rows with net flow mismatch (>1000 QAR tolerance) quarantined.",
            dataset, bad.sum(),
        )
    return df[~bad].copy(), df[bad].copy()


def _validate_gcc_coverage(df: pd.DataFrame, dataset: str):
    """Every trading day must have records for all active GCC markets."""
    if dataset != "gcc_daily":
        return df, pd.DataFrame()

    trading_dates = df.loc[_trading_days(df), "date"].dt.normalize().unique()
    bad_rows = pd.Index([])

    for d in trading_dates:
        day_df = df[df["date"].dt.normalize() == d]
        present = set(day_df["market_name"].str.upper().unique())
        missing_markets = GCC_ACTIVE_MARKETS - present
        if missing_markets:
            logger.warning(
                "[gcc_daily] %s — missing GCC markets: %s", d.date(), missing_markets
            )
            bad_rows = bad_rows.union(day_df.index)

    quarantine = df.loc[df.index.isin(bad_rows)].copy()
    clean = df.loc[~df.index.isin(bad_rows)].copy()
    return clean, quarantine


# ---------------------------------------------------------------------------
# Consecutive missing days check
# ---------------------------------------------------------------------------

def _check_consecutive_missing(df: pd.DataFrame, dataset: str, threshold: int = 7):
    """Halt if more than `threshold` consecutive trading days are absent."""
    if df.empty:
        return

    key_col = "date"
    dates = pd.to_datetime(df[key_col]).dt.normalize().unique()
    dates = sorted(dates)

    if not dates:
        return

    # Build expected trading day range (Sun–Thu)
    start, end = dates[0], dates[-1]
    all_days = pd.date_range(start, end, freq="D")
    expected_trading = [d for d in all_days if d.dayofweek not in (4, 5)]  # excl Fri/Sat

    present_set = set(pd.Timestamp(d) for d in dates)

    max_gap = 0
    current_gap = 0
    gap_start = None

    for d in expected_trading:
        if d not in present_set:
            if current_gap == 0:
                gap_start = d
            current_gap += 1
            max_gap = max(max_gap, current_gap)
        else:
            current_gap = 0

    if max_gap > threshold:
        msg = (
            f"[{dataset}] ALERT: {max_gap} consecutive trading days missing "
            f"(threshold={threshold}). Pipeline halted."
        )
        logger.error(msg)
        raise RuntimeError(msg)


# ---------------------------------------------------------------------------
# Per-dataset pipeline
# ---------------------------------------------------------------------------

VALIDATORS = [
    _validate_no_future_dates,
    _validate_no_duplicate_dates,
    _validate_positive_trading_cols,
    _validate_breadth_sum,
    _validate_foreign_net,
    _validate_gcc_coverage,
]


def process_dataset(src_path: Path, dataset: str) -> bool:
    logger.info("Processing [%s] from %s", dataset, src_path.name)

    try:
        df = _load_file(src_path)
    except Exception as exc:
        logger.error("[%s] Failed to load %s: %s", dataset, src_path.name, exc)
        return False

    if not _check_required_columns(df, dataset, src_path):
        return False

    # Normalise date column
    df["date"] = pd.to_datetime(df["date"])

    all_quarantine = []
    clean = df.copy()

    for validator in VALIDATORS:
        clean, q = validator(clean, dataset)
        if not q.empty:
            all_quarantine.append(q)

    # Consecutive missing days check — halts pipeline on violation
    _check_consecutive_missing(clean, dataset)

    # Write clean parquet
    out_path = RAW_DIR / f"{dataset}.parquet"
    clean.to_parquet(out_path, index=False)
    logger.info("[%s] Wrote %d clean rows -> %s", dataset, len(clean), out_path)

    # Write quarantine parquet if any bad rows
    if all_quarantine:
        quarantine_df = pd.concat(all_quarantine).drop_duplicates()
        q_path = RAW_DIR / f"{dataset}_quarantine.parquet"
        quarantine_df.to_parquet(q_path, index=False)
        logger.warning(
            "[%s] Quarantined %d rows -> %s", dataset, len(quarantine_df), q_path
        )

    return True


# ---------------------------------------------------------------------------
# Discovery: map files in source dir to known dataset names
# ---------------------------------------------------------------------------

def _discover_files(src_dir: Path) -> dict[str, Path]:
    """
    Match files whose stem contains a known dataset name.
    e.g. market_daily_2024.csv -> market_daily
    """
    mapping: dict[str, Path] = {}
    for f in src_dir.iterdir():
        if f.suffix.lower() not in (".csv", ".xlsx", ".xls"):
            continue
        for ds in KNOWN_DATASETS:
            if ds in f.stem.lower():
                if ds in mapping:
                    logger.warning(
                        "Multiple files match dataset '%s'; using first found (%s).",
                        ds, mapping[ds].name,
                    )
                else:
                    mapping[ds] = f
    return mapping


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Ingest raw market data files.")
    parser.add_argument(
        "--src", required=True, type=Path,
        help="Directory containing source CSV/Excel files.",
    )
    parser.add_argument(
        "--dataset", default=None,
        choices=list(KNOWN_DATASETS),
        help="Process a single dataset (default: all discovered).",
    )
    args = parser.parse_args()

    src_dir: Path = args.src
    if not src_dir.is_dir():
        logger.error("Source directory does not exist: %s", src_dir)
        sys.exit(1)

    file_map = _discover_files(src_dir)

    if args.dataset:
        if args.dataset not in file_map:
            logger.error("Dataset '%s' not found in %s", args.dataset, src_dir)
            sys.exit(1)
        file_map = {args.dataset: file_map[args.dataset]}

    if not file_map:
        logger.error("No recognisable dataset files found in %s", src_dir)
        sys.exit(1)

    success = True
    for dataset, path in file_map.items():
        try:
            ok = process_dataset(path, dataset)
            if not ok:
                success = False
        except RuntimeError as exc:
            # consecutive missing days — already logged
            logger.error("Pipeline halted: %s", exc)
            sys.exit(2)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
