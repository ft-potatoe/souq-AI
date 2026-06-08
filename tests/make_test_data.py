"""
Generate synthetic CSV test data matching spec §16 schemas.
Writes to a temp source dir: data/test_src/
"""
import random
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "test_src"
OUT.mkdir(parents=True, exist_ok=True)

random.seed(42)
np.random.seed(42)

# Build ~300 QSE trading days (Sun–Thu) starting 2023-01-01
def qse_trading_days(start="2023-01-01", n=300):
    days = []
    d = pd.Timestamp(start)
    while len(days) < n:
        if d.dayofweek not in (4, 5):   # skip Fri=4, Sat=5
            days.append(d)
        d += pd.Timedelta(days=1)
    return pd.DatetimeIndex(days)

dates = qse_trading_days(n=300)

# ── market_daily ──────────────────────────────────────────────────────────────
close = 10000.0
prices = []
for _ in dates:
    close *= (1 + np.random.normal(0, 0.008))
    prices.append(round(close, 2))

market = pd.DataFrame({
    "date":         dates,
    "open":         [p * np.random.uniform(0.998, 1.002) for p in prices],
    "high":         [p * np.random.uniform(1.001, 1.010) for p in prices],
    "low":          [p * np.random.uniform(0.990, 0.999) for p in prices],
    "close":        prices,
    "volume":       np.random.randint(50_000_000, 300_000_000, len(dates)),
    "value_traded": np.random.uniform(50e6, 400e6, len(dates)).round(0),
    "total_trades": np.random.randint(3000, 30000, len(dates)),
})
market.to_csv(OUT / "market_daily.csv", index=False)
print(f"market_daily: {len(market)} rows")

# ── flows_daily ───────────────────────────────────────────────────────────────
fbuy  = np.random.uniform(20e6, 200e6, len(dates))
fsell = np.random.uniform(20e6, 200e6, len(dates))
dbuy  = np.random.uniform(30e6, 300e6, len(dates))
dsell = np.random.uniform(30e6, 300e6, len(dates))

flows = pd.DataFrame({
    "date":          dates,
    "foreign_buy":   fbuy.round(0),
    "foreign_sell":  fsell.round(0),
    "foreign_net":   (fbuy - fsell).round(0),
    "domestic_buy":  dbuy.round(0),
    "domestic_sell": dsell.round(0),
    "domestic_net":  (dbuy - dsell).round(0),
})
flows.to_csv(OUT / "flows_daily.csv", index=False)
print(f"flows_daily:  {len(flows)} rows")

# ── gcc_daily ─────────────────────────────────────────────────────────────────
gcc_markets = ["QSE", "ADX", "DFM", "TASI", "KSE", "MSM", "BSE"]
gcc_rows = []
for d in dates:
    for mkt in gcc_markets:
        gcc_rows.append({
            "date":             d,
            "market_name":      mkt,
            "daily_change_pct": round(np.random.normal(0, 0.8), 4),
            "pe_ratio":         round(np.random.uniform(10, 25), 2),
            "dividend_yield":   round(np.random.uniform(1, 5), 2),
        })
gcc = pd.DataFrame(gcc_rows)
gcc.to_csv(OUT / "gcc_daily.csv", index=False)
print(f"gcc_daily:    {len(gcc)} rows ({len(dates)} dates × {len(gcc_markets)} markets)")

# ── breadth_daily ─────────────────────────────────────────────────────────────
total_listed = 50
gainers  = np.random.randint(5, 35, len(dates))
losers   = np.random.randint(5, 35, len(dates))
# ensure gainers+losers <= total_listed, unchanged fills the rest
losers   = np.minimum(losers, total_listed - gainers)
unchanged = total_listed - gainers - losers

breadth = pd.DataFrame({
    "date":          dates,
    "gainers":       gainers,
    "losers":        losers,
    "unchanged":     unchanged,
    "total_listed":  total_listed,
    "total_traded":  np.random.randint(20, 50, len(dates)),
})
breadth.to_csv(OUT / "breadth_daily.csv", index=False)
print(f"breadth_daily:{len(breadth)} rows")
print(f"\nTest CSVs written to {OUT}")
