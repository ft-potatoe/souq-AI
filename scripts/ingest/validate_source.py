"""
Pre-ingestion validator for raw Bloomberg Excel source files.

Checks all four datasets for structural, logical, and data-quality issues
BEFORE the prep/reshape step so problems can be fixed at the source.

Usage:
    python scripts/ingest/validate_source.py --src <folder_with_excel_files>

    # validate a single dataset
    python scripts/ingest/validate_source.py --src <folder> --dataset market_daily

    # validate already-reshaped files (gcc_daily already in long format)
    python scripts/ingest/validate_source.py --src <folder> --gcc-long

Options:
    --src           Directory containing Excel/CSV files
    --dataset       Validate one dataset only (default: all)
    --gcc-long      gcc_daily file is already in long format (date, market_name, daily_change_pct)
                    Default: wide format (date, QSE, ADX, DFM, TASI, KSE, ...)
    --date-format   Expected date format string e.g. "%Y-%m-%d" (default: auto-detect)

Exit codes:
    0  all datasets pass
    1  one or more datasets have issues
"""

import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

# Required columns after renaming (pre-reshape)
REQUIRED_COLS = {
    "market_daily": ["date", "open", "high", "low", "close",
                     "volume", "value_traded", "total_trades"],
    "flows_daily":  ["date", "foreign_buy", "foreign_sell", "foreign_net",
                     "domestic_buy", "domestic_sell", "domestic_net"],
    "gcc_daily":    ["date", "market_name", "daily_change_pct"],   # long format
    "breadth_daily":["date", "gainers", "losers", "unchanged",
                     "total_listed", "total_traded"],
}

GCC_REQUIRED_MARKETS = {"QSE", "ADX", "DFM", "TASI", "KSE"}
GCC_WIDE_RENAME = {
    "TADAWUL": "TASI",
    "KBS":     "KSE",
    "KWSEPM":  "KSE",
    "DFMGI":   "DFM",
    "ADSMI":   "ADX",
    "QE":      "QSE",
}

# Net flow tolerance (QAR)
FLOW_TOLERANCE = 1000.0

# Breadth sum tolerance
BREADTH_TOLERANCE = 2

# Consecutive missing trading days threshold
GAP_THRESHOLD = 7

# daily_change_pct sanity range — if ALL values are < this, likely still in decimal
PCT_DECIMAL_THRESHOLD = 0.10


# ---------------------------------------------------------------------------
# Colour helpers (ANSI — skipped on Windows if not supported)
# ---------------------------------------------------------------------------

def _green(s: str) -> str:
    return f"\033[92m{s}\033[0m"

def _red(s: str) -> str:
    return f"\033[91m{s}\033[0m"

def _yellow(s: str) -> str:
    return f"\033[93m{s}\033[0m"

def _bold(s: str) -> str:
    return f"\033[1m{s}\033[0m"


# ---------------------------------------------------------------------------
# Issue collector
# ---------------------------------------------------------------------------

class IssueList:
    def __init__(self):
        self.errors   = []   # blocking — will cause ingestion failure
        self.warnings = []   # non-blocking — worth investigating

    def error(self, msg: str):
        self.errors.append(msg)

    def warn(self, msg: str):
        self.warnings.append(msg)

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0


# ---------------------------------------------------------------------------
# File discovery (mirrors load_raw.py logic)
# ---------------------------------------------------------------------------

def _discover_files(src_dir: Path) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for f in sorted(src_dir.iterdir()):
        if f.suffix.lower() not in (".csv", ".xlsx", ".xls"):
            continue
        for ds in REQUIRED_COLS:
            if ds in f.stem.lower():
                if ds not in mapping:
                    mapping[ds] = f
    return mapping


def _load_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type: {suffix}")


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

def _parse_dates(series: pd.Series, issues: IssueList) -> pd.Series:
    """Try to parse dates; warn on ambiguous formats."""
    raw = series.astype(str).str.strip()

    # Detect ambiguous MM/DD/YY or DD/MM/YY patterns
    slash_pattern = raw.str.match(r"^\d{1,2}/\d{1,2}/\d{2,4}$")
    if slash_pattern.any():
        issues.warn(
            f"Date format looks like MM/DD/YYYY or DD/MM/YYYY "
            f"(e.g. '{raw[slash_pattern].iloc[0]}'). "
            f"This is ambiguous — convert to YYYY-MM-DD to be safe."
        )

    parsed = pd.to_datetime(series, errors="coerce")
    n_failed = parsed.isna().sum()
    if n_failed > 0:
        issues.error(f"{n_failed} date values could not be parsed.")
    return parsed


def _qse_trading_days(start: pd.Timestamp, end: pd.Timestamp) -> list:
    """All expected QSE trading days (Sun-Thu) between start and end inclusive."""
    all_days = pd.date_range(start, end, freq="D")
    return [d for d in all_days if d.dayofweek not in (4, 5)]  # excl Fri=4, Sat=5


def _check_gaps(dates: pd.Series, dataset: str, issues: IssueList):
    """Check for consecutive missing QSE trading days."""
    clean = pd.to_datetime(dates).dt.normalize().dropna().unique()
    if len(clean) < 2:
        return
    clean = sorted(clean)
    expected = _qse_trading_days(clean[0], clean[-1])
    present = set(pd.Timestamp(d) for d in clean)

    max_gap = 0
    current_gap = 0
    gap_examples = []

    for d in expected:
        if d not in present:
            current_gap += 1
            if current_gap == GAP_THRESHOLD + 1:
                gap_examples.append(str(d.date()))
            max_gap = max(max_gap, current_gap)
        else:
            current_gap = 0

    if max_gap > GAP_THRESHOLD:
        issues.error(
            f"Max consecutive missing trading days = {max_gap} "
            f"(threshold={GAP_THRESHOLD}). "
            f"First violation around: {gap_examples[0] if gap_examples else 'unknown'}. "
            f"Pipeline will halt at ingestion."
        )
    elif max_gap > 0:
        issues.warn(f"Max consecutive missing trading days = {max_gap} (within threshold).")


# ---------------------------------------------------------------------------
# Dataset-specific validators
# ---------------------------------------------------------------------------

def _validate_market_daily(df: pd.DataFrame, issues: IssueList):
    # Required columns
    missing = [c for c in REQUIRED_COLS["market_daily"] if c not in df.columns]
    if missing:
        issues.error(f"Missing columns: {missing}")
        return  # can't continue without columns

    dates = _parse_dates(df["date"], issues)

    # Future dates
    today = pd.Timestamp(date.today())
    future = (dates > today).sum()
    if future:
        issues.error(f"{future} rows have future dates.")

    # Duplicate dates
    dupes = df.duplicated(subset=["date"], keep=False).sum()
    if dupes:
        issues.error(f"{dupes} duplicate date rows found.")

    # NaN in critical columns
    for col in ["close", "volume", "value_traded", "total_trades"]:
        n = df[col].isna().sum()
        if n:
            issues.error(f"'{col}' has {n} NaN values.")

    # Non-positive on trading columns
    trading_mask = pd.to_datetime(df["date"], errors="coerce").dt.dayofweek.isin([6, 0, 1, 2, 3])
    for col in ["volume", "value_traded", "total_trades"]:
        if col not in df.columns:
            continue
        bad = (trading_mask & (pd.to_numeric(df[col], errors="coerce") <= 0)).sum()
        if bad:
            issues.error(f"'{col}' has {bad} rows with non-positive values on trading days.")

    # OHLC sanity
    if all(c in df.columns for c in ["open", "high", "low", "close"]):
        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        bad_high = (df["high"] < df["close"]).sum()
        bad_low  = (df["low"]  > df["close"]).sum()
        if bad_high:
            issues.warn(f"{bad_high} rows where high < close.")
        if bad_low:
            issues.warn(f"{bad_low} rows where low > close.")

    _check_gaps(dates, "market_daily", issues)

    # Summary stats
    issues.warn(f"INFO: close range = {df['close'].min():.2f} - {df['close'].max():.2f}")
    issues.warn(f"INFO: {len(df)} rows | {dates.min().date()} to {dates.max().date()}")


def _validate_flows_daily(df: pd.DataFrame, issues: IssueList):
    missing = [c for c in REQUIRED_COLS["flows_daily"] if c not in df.columns]
    if missing:
        issues.error(f"Missing columns: {missing}")
        return

    dates = _parse_dates(df["date"], issues)

    future = (pd.to_datetime(df["date"], errors="coerce") > pd.Timestamp(date.today())).sum()
    if future:
        issues.error(f"{future} rows have future dates.")

    dupes = df.duplicated(subset=["date"], keep=False).sum()
    if dupes:
        issues.error(f"{dupes} duplicate date rows found.")

    # NaN check
    for col in REQUIRED_COLS["flows_daily"]:
        if col == "date":
            continue
        n = pd.to_numeric(df[col], errors="coerce").isna().sum()
        if n:
            issues.error(f"'{col}' has {n} NaN / non-numeric values.")

    # Convert to numeric for math checks
    for col in REQUIRED_COLS["flows_daily"]:
        if col != "date":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # foreign_net = foreign_buy - foreign_sell
    computed_f = df["foreign_buy"] - df["foreign_sell"]
    diff_f = (computed_f - df["foreign_net"]).abs()
    bad_f = (diff_f > FLOW_TOLERANCE).sum()
    if bad_f:
        issues.error(
            f"{bad_f} rows where |foreign_buy - foreign_sell - foreign_net| > {FLOW_TOLERANCE}. "
            f"Max deviation: {diff_f.max():.2f}"
        )

    # domestic_net = domestic_buy - domestic_sell
    computed_d = df["domestic_buy"] - df["domestic_sell"]
    diff_d = (computed_d - df["domestic_net"]).abs()
    bad_d = (diff_d > FLOW_TOLERANCE).sum()
    if bad_d:
        issues.error(
            f"{bad_d} rows where |domestic_buy - domestic_sell - domestic_net| > {FLOW_TOLERANCE}. "
            f"Max deviation: {diff_d.max():.2f}"
        )

    # foreign_net + domestic_net ~ 0 (informational)
    total_net = (df["foreign_net"] + df["domestic_net"]).abs()
    max_total = total_net.max()
    if max_total > FLOW_TOLERANCE:
        issues.warn(
            f"foreign_net + domestic_net max deviation from 0 = {max_total:.2f}. "
            f"Expected ~0 (closed market)."
        )

    _check_gaps(dates, "flows_daily", issues)
    issues.warn(f"INFO: {len(df)} rows | {dates.min().date()} to {dates.max().date()}")


def _validate_gcc_daily_long(df: pd.DataFrame, issues: IssueList):
    """Validate gcc_daily already in long format (date, market_name, daily_change_pct)."""
    missing = [c for c in REQUIRED_COLS["gcc_daily"] if c not in df.columns]
    if missing:
        issues.error(f"Missing columns: {missing}")
        return

    dates = _parse_dates(df["date"], issues)

    future = (pd.to_datetime(df["date"], errors="coerce") > pd.Timestamp(date.today())).sum()
    if future:
        issues.error(f"{future} rows have future dates.")

    dupes = df.duplicated(subset=["date", "market_name"], keep=False).sum()
    if dupes:
        issues.error(f"{dupes} duplicate (date, market_name) rows found.")

    # Normalise market names
    df["market_name"] = df["market_name"].str.strip().str.upper()
    df["market_name"] = df["market_name"].replace(GCC_WIDE_RENAME)

    # Unknown markets
    known = GCC_REQUIRED_MARKETS | set(GCC_WIDE_RENAME.values())
    unknown = set(df["market_name"].unique()) - known
    if unknown:
        issues.warn(f"Unrecognised market_name values (will be ignored): {unknown}")

    # Coverage per trading day
    df["_date_norm"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    trading_dates = df.loc[
        df["_date_norm"].dt.dayofweek.isin([6, 0, 1, 2, 3]), "_date_norm"
    ].unique()

    missing_coverage = []
    for d in sorted(trading_dates):
        day_markets = set(
            df.loc[df["_date_norm"] == d, "market_name"].str.upper().unique()
        )
        missing_markets = GCC_REQUIRED_MARKETS - day_markets
        if missing_markets:
            missing_coverage.append((str(d.date()), missing_markets))

    if missing_coverage:
        n = len(missing_coverage)
        sample = missing_coverage[:3]
        issues.error(
            f"{n} trading days missing one or more required GCC markets. "
            f"First 3: {[(d, list(m)) for d, m in sample]}"
        )

    # daily_change_pct sanity
    pct = pd.to_numeric(df["daily_change_pct"], errors="coerce")
    n_nan = pct.isna().sum()
    if n_nan:
        issues.error(f"daily_change_pct has {n_nan} NaN / non-numeric values.")

    abs_max = pct.abs().max()
    abs_median = pct.abs().median()
    if abs_median < PCT_DECIMAL_THRESHOLD:
        issues.error(
            f"daily_change_pct looks like it is still in decimal form "
            f"(median abs value = {abs_median:.5f}). Multiply by 100 before ingestion."
        )
    if abs_max > 20:
        issues.warn(
            f"daily_change_pct max absolute value = {abs_max:.2f}. "
            f"Values > 20% are unusual — verify these are not index levels."
        )

    _check_gaps(
        df.loc[df["market_name"] == "QSE", "date"] if "QSE" in df["market_name"].values else dates,
        "gcc_daily",
        issues,
    )
    issues.warn(f"INFO: {len(df)} rows | {len(trading_dates)} trading days | "
                f"{dates.min().date()} to {dates.max().date()}")


def _validate_gcc_daily_wide(df: pd.DataFrame, issues: IssueList):
    """Validate gcc_daily in wide format (date per row, one column per market)."""
    if "date" not in df.columns:
        issues.error("Missing 'date' column.")
        return

    dates = _parse_dates(df["date"], issues)

    # Rename known aliases
    df = df.rename(columns=GCC_WIDE_RENAME)

    # Check required markets exist as columns
    present_markets = set(df.columns) & GCC_REQUIRED_MARKETS
    missing_markets = GCC_REQUIRED_MARKETS - present_markets
    if missing_markets:
        issues.error(f"Missing market columns: {missing_markets}")

    for mkt in present_markets:
        col = pd.to_numeric(df[mkt], errors="coerce")
        n_nan = col.isna().sum()
        n_total = len(col)
        pct_missing = n_nan / n_total * 100
        if pct_missing > 30:
            issues.warn(f"'{mkt}' has {n_nan}/{n_total} ({pct_missing:.1f}%) blank values.")

        abs_median = col.abs().median()
        if pd.notna(abs_median) and abs_median < PCT_DECIMAL_THRESHOLD:
            issues.error(
                f"'{mkt}' daily_change_pct looks like decimal form "
                f"(median abs = {abs_median:.5f}). Multiply by 100."
            )
        abs_max = col.abs().max()
        if pd.notna(abs_max) and abs_max > 20:
            issues.warn(
                f"'{mkt}' max absolute value = {abs_max:.2f}. "
                f"Values > 20% are unusual — verify these are not index levels."
            )

    dupes = df.duplicated(subset=["date"], keep=False).sum()
    if dupes:
        issues.error(f"{dupes} duplicate date rows found.")

    future = (pd.to_datetime(df["date"], errors="coerce") > pd.Timestamp(date.today())).sum()
    if future:
        issues.error(f"{future} rows have future dates.")

    _check_gaps(dates, "gcc_daily", issues)
    issues.warn(f"INFO: {len(df)} rows | {dates.min().date()} to {dates.max().date()}")


def _validate_breadth_daily(df: pd.DataFrame, issues: IssueList):
    missing = [c for c in REQUIRED_COLS["breadth_daily"] if c not in df.columns]
    if missing:
        issues.error(f"Missing columns: {missing}")
        return

    dates = _parse_dates(df["date"], issues)

    future = (pd.to_datetime(df["date"], errors="coerce") > pd.Timestamp(date.today())).sum()
    if future:
        issues.error(f"{future} rows have future dates.")

    dupes = df.duplicated(subset=["date"], keep=False).sum()
    if dupes:
        issues.error(f"{dupes} duplicate date rows found.")

    for col in ["gainers", "losers", "unchanged", "total_listed", "total_traded"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        n = df[col].isna().sum()
        if n:
            issues.error(f"'{col}' has {n} NaN / non-numeric values.")

    # gainers + losers + unchanged = total_listed within ±2
    computed = df["gainers"] + df["losers"] + df["unchanged"]
    diff = (computed - df["total_listed"]).abs()
    bad = (diff > BREADTH_TOLERANCE).sum()
    if bad:
        worst = diff.max()
        issues.error(
            f"{bad} rows where gainers+losers+unchanged != total_listed (tolerance ±{BREADTH_TOLERANCE}). "
            f"Max deviation: {worst:.0f}. "
            f"Sample dates: {df.loc[diff > BREADTH_TOLERANCE, 'date'].head(3).tolist()}"
        )

    # Negative values
    for col in ["gainers", "losers", "unchanged", "total_listed", "total_traded"]:
        neg = (df[col] < 0).sum()
        if neg:
            issues.error(f"'{col}' has {neg} negative values.")

    # total_traded <= total_listed
    bad_traded = (df["total_traded"] > df["total_listed"]).sum()
    if bad_traded:
        issues.error(f"{bad_traded} rows where total_traded > total_listed.")

    # gainers + losers should not exceed total_listed
    bad_sum = ((df["gainers"] + df["losers"]) > df["total_listed"]).sum()
    if bad_sum:
        issues.error(f"{bad_sum} rows where gainers + losers > total_listed.")

    _check_gaps(dates, "breadth_daily", issues)

    total_listed_range = f"{df['total_listed'].min():.0f} - {df['total_listed'].max():.0f}"
    issues.warn(f"INFO: {len(df)} rows | {dates.min().date()} to {dates.max().date()} | "
                f"total_listed range: {total_listed_range}")


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

VALIDATORS = {
    "market_daily":  _validate_market_daily,
    "flows_daily":   _validate_flows_daily,
    "breadth_daily": _validate_breadth_daily,
}


def validate_dataset(
    dataset: str,
    path: Path,
    gcc_long: bool = False,
) -> IssueList:
    issues = IssueList()

    try:
        df = _load_file(path)
    except Exception as exc:
        issues.error(f"Failed to load file: {exc}")
        return issues

    # Normalise column names: strip whitespace, lowercase — except gcc_daily wide
    # which needs market columns preserved as uppercase for matching
    if dataset == "gcc_daily" and not gcc_long:
        df.columns = [
            "date" if str(c).strip().lower() == "date"
            else str(c).strip().upper().replace(" ", "_")
            for c in df.columns
        ]
    else:
        df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    # Remove entirely blank rows
    df = df.dropna(how="all").reset_index(drop=True)

    if dataset == "gcc_daily":
        if gcc_long:
            _validate_gcc_daily_long(df, issues)
        else:
            _validate_gcc_daily_wide(df, issues)
    else:
        VALIDATORS[dataset](df, issues)

    return issues


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

def _print_report(dataset: str, path: Path, issues: IssueList):
    print()
    status = _green("PASS") if issues.ok else _red("FAIL")
    print(_bold(f"[{dataset}]") + f"  {status}  ({path.name})")

    # Filter out INFO lines for clean display
    info_lines   = [w for w in issues.warnings if w.startswith("INFO:")]
    warn_lines   = [w for w in issues.warnings if not w.startswith("INFO:")]

    for info in info_lines:
        print(f"  {_bold(info[5:].strip())}")   # strip "INFO:" prefix

    for err in issues.errors:
        print(f"  {_red('ERROR')}   {err}")

    for warn in warn_lines:
        print(f"  {_yellow('WARN')}    {warn}")

    if issues.ok and not warn_lines:
        print(f"  {_green('No issues found.')}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Validate Bloomberg source files before ingestion."
    )
    parser.add_argument("--src", required=True, type=Path,
                        help="Directory containing source Excel/CSV files.")
    parser.add_argument("--dataset", default=None,
                        choices=list(REQUIRED_COLS.keys()),
                        help="Validate a single dataset only.")
    parser.add_argument("--gcc-long", action="store_true",
                        help="gcc_daily is already in long format.")
    args = parser.parse_args()

    src_dir: Path = args.src
    if not src_dir.is_dir():
        print(_red(f"ERROR: Source directory does not exist: {src_dir}"))
        sys.exit(1)

    file_map = _discover_files(src_dir)

    if args.dataset:
        if args.dataset not in file_map:
            print(_red(f"ERROR: No file found for dataset '{args.dataset}' in {src_dir}"))
            sys.exit(1)
        file_map = {args.dataset: file_map[args.dataset]}

    if not file_map:
        print(_red(f"ERROR: No recognisable dataset files found in {src_dir}"))
        sys.exit(1)

    print(_bold("\n=== Pre-ingestion Validation Report ==="))
    print(f"Source: {src_dir.resolve()}")

    all_pass = True
    for dataset, path in sorted(file_map.items()):
        issues = validate_dataset(dataset, path, gcc_long=args.gcc_long)
        _print_report(dataset, path, issues)
        if not issues.ok:
            all_pass = False

    print()
    if all_pass:
        print(_green(_bold("All datasets passed. Safe to proceed with ingestion.")))
    else:
        print(_red(_bold("One or more datasets have errors. Fix before ingesting.")))
    print()

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
