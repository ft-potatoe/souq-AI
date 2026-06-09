"""
Prepare raw Bloomberg Excel files for ingestion into the pipeline.

What this script does:
  market_daily  — validates columns, writes as-is to CSV
  flows_daily   — validates columns, writes as-is to CSV
  breadth_daily — validates columns, writes as-is to CSV
  gcc_daily     — reshapes wide -> long, renames market columns, writes to CSV

Output CSVs are written to <out_dir> (default: data/ready/) and are safe
to pass directly to load_raw.py.

Usage:
    python scripts/ingest/prepare_source.py --src <folder_with_excel_files>
    python scripts/ingest/prepare_source.py --src <folder> --out <output_folder>

Exit codes:
    0  all datasets prepared successfully
    1  one or more datasets failed
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "data" / "ready"

# ---------------------------------------------------------------------------
# GCC market column rename map (source name -> required pipeline name)
# ---------------------------------------------------------------------------
GCC_MARKET_RENAME = {
    "QSE":     "QSE",
    "ADX":     "ADX",
    "DFM":     "DFM",
    "KSE":     "KSE",
    "TASI":    "TASI",
    "TADAWUL": "TASI",
    "KWSEPM":  "KSE",
    "KBS":     "KSE",
    "DFMGI":   "DFM",
    "ADSMI":   "ADX",
    "QE":      "QSE",
}

GCC_REQUIRED_MARKETS = {"QSE", "ADX", "DFM", "KSE", "TASI"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type: {suffix}")


def _normalise_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Parse date column and format as YYYY-MM-DD string."""
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    bad = df["date"].isna().sum()
    if bad:
        print(f"  WARNING: {bad} rows with unparseable dates will be dropped.")
        df = df.dropna(subset=["date"])
    return df


def _discover_files(src_dir: Path) -> dict[str, Path]:
    datasets = ["market_daily", "flows_daily", "gcc_daily", "breadth_daily"]
    mapping: dict[str, Path] = {}
    for f in sorted(src_dir.iterdir()):
        if f.suffix.lower() not in (".csv", ".xlsx", ".xls"):
            continue
        for ds in datasets:
            if ds in f.stem.lower() and ds not in mapping:
                mapping[ds] = f
    return mapping


# ---------------------------------------------------------------------------
# Per-dataset prep functions
# ---------------------------------------------------------------------------

def prep_market_daily(path: Path, out_dir: Path) -> bool:
    print(f"\n[market_daily] Reading {path.name}...")
    df = _load(path)

    # Normalise column names
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    required = ["date", "open", "high", "low", "close", "volume", "value_traded", "total_trades"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"  ERROR: Missing columns: {missing}")
        return False

    # Drop fully blank rows
    df = df.dropna(how="all").reset_index(drop=True)

    # Normalise dates
    df = _normalise_dates(df)

    # Keep only required columns in order
    df = df[required]

    out_path = out_dir / "market_daily.csv"
    df.to_csv(out_path, index=False)
    print(f"  OK: {len(df)} rows -> {out_path}")
    return True


def prep_flows_daily(path: Path, out_dir: Path) -> bool:
    print(f"\n[flows_daily] Reading {path.name}...")
    df = _load(path)

    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    required = ["date", "foreign_buy", "foreign_sell", "foreign_net",
                "domestic_buy", "domestic_sell", "domestic_net"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"  ERROR: Missing columns: {missing}")
        return False

    df = df.dropna(how="all").reset_index(drop=True)
    df = _normalise_dates(df)
    df = df[required]

    out_path = out_dir / "flows_daily.csv"
    df.to_csv(out_path, index=False)
    print(f"  OK: {len(df)} rows -> {out_path}")
    return True


def prep_breadth_daily(path: Path, out_dir: Path) -> bool:
    print(f"\n[breadth_daily] Reading {path.name}...")
    df = _load(path)

    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    required = ["date", "gainers", "losers", "unchanged", "total_listed", "total_traded"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"  ERROR: Missing columns: {missing}")
        return False

    df = df.dropna(how="all").reset_index(drop=True)
    df = _normalise_dates(df)
    df = df[required]

    out_path = out_dir / "breadth_daily.csv"
    df.to_csv(out_path, index=False)
    print(f"  OK: {len(df)} rows -> {out_path}")
    return True


def prep_gcc_daily(path: Path, out_dir: Path) -> bool:
    print(f"\n[gcc_daily] Reading {path.name}...")
    df = _load(path)

    # Normalise date column name only; keep market columns as-is for now
    df.columns = [
        "date" if str(c).strip().lower() == "date" else str(c).strip().upper()
        for c in df.columns
    ]

    if "date" not in df.columns:
        print("  ERROR: No 'date' column found.")
        return False

    # Drop fully blank rows
    df = df.dropna(how="all").reset_index(drop=True)

    # Rename market columns using alias map
    df = df.rename(columns=GCC_MARKET_RENAME)

    # Keep only date + required market columns
    market_cols = [c for c in df.columns if c in GCC_REQUIRED_MARKETS]
    missing_markets = GCC_REQUIRED_MARKETS - set(market_cols)
    if missing_markets:
        print(f"  ERROR: Missing market columns: {missing_markets}")
        return False

    df = df[["date"] + sorted(market_cols)]

    # Reshape wide -> long
    df = df.melt(id_vars="date", var_name="market_name", value_name="daily_change_pct")

    # Drop rows where market was closed (blank cell = NaN)
    n_before = len(df)
    df = df.dropna(subset=["daily_change_pct"]).reset_index(drop=True)
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"  INFO: Dropped {n_dropped} blank market-day rows (market closed those days).")

    # Normalise dates
    df = _normalise_dates(df)

    # Sort by date then market for clean output
    df = df.sort_values(["date", "market_name"]).reset_index(drop=True)

    # Sanity check: values should be in percent form, not decimal
    abs_median = df["daily_change_pct"].abs().median()
    if abs_median < 0.10:
        print(f"  WARNING: daily_change_pct median abs value = {abs_median:.5f}.")
        print(f"           Values look like decimals — multiplying by 100.")
        df["daily_change_pct"] = df["daily_change_pct"] * 100

    out_path = out_dir / "gcc_daily.csv"
    df.to_csv(out_path, index=False)

    # Summary
    n_days = df["date"].nunique()
    markets_found = sorted(df["market_name"].unique())
    print(f"  OK: {len(df)} rows | {n_days} trading days | markets: {markets_found}")
    print(f"  -> {out_path}")
    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Prepare Bloomberg Excel files for pipeline ingestion."
    )
    parser.add_argument("--src", required=True, type=Path,
                        help="Directory containing source Excel/CSV files.")
    parser.add_argument("--out", default=None, type=Path,
                        help="Output directory for prepared CSVs (default: data/ready/).")
    parser.add_argument("--dataset", default=None,
                        choices=["market_daily", "flows_daily", "gcc_daily", "breadth_daily"],
                        help="Prepare a single dataset only.")
    args = parser.parse_args()

    src_dir: Path = args.src
    out_dir: Path = args.out if args.out else DEFAULT_OUT

    if not src_dir.is_dir():
        print(f"ERROR: Source directory does not exist: {src_dir}")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    file_map = _discover_files(src_dir)

    if args.dataset:
        if args.dataset not in file_map:
            print(f"ERROR: No file found for dataset '{args.dataset}' in {src_dir}")
            sys.exit(1)
        file_map = {args.dataset: file_map[args.dataset]}

    if not file_map:
        print(f"ERROR: No recognisable dataset files found in {src_dir}")
        sys.exit(1)

    preps = {
        "market_daily":  prep_market_daily,
        "flows_daily":   prep_flows_daily,
        "gcc_daily":     prep_gcc_daily,
        "breadth_daily": prep_breadth_daily,
    }

    print(f"\n=== Prepare Source Files ===")
    print(f"Source : {src_dir.resolve()}")
    print(f"Output : {out_dir.resolve()}")

    all_ok = True
    for dataset, path in sorted(file_map.items()):
        ok = preps[dataset](path, out_dir)
        if not ok:
            all_ok = False

    print()
    if all_ok:
        print("All datasets prepared successfully.")
        print()
        print("Next step — run ingestion:")
        print(f"  python scripts/ingest/load_raw.py --src \"{out_dir.resolve()}\"")
    else:
        print("One or more datasets failed. Fix the errors above and re-run.")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
