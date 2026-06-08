# scripts/features

## build_features.py

Reads clean parquets from `data/raw/`, computes the full feature matrix, and writes
`data/features/features_master.parquet`.

### Usage

```bash
# Activate venv first
venv\Scripts\activate

python scripts/features/build_features.py
```

Requires the following raw parquets to exist (produced by `scripts/ingest/load_raw.py`):
- `data/raw/market_daily.parquet`
- `data/raw/flows_daily.parquet`
- `data/raw/gcc_daily.parquet`
- `data/raw/breadth_daily.parquet`

### Feature catalogue

| Group | Features |
|---|---|
| **Returns** | `return_1d`, `return_5d`, `return_20d`, `return_60d` |
| **Volatility** | `volatility_20d`, `volatility_60d` (annualised, rolling std × √252) |
| **Momentum** | `rsi_14`, `sma_20`, `sma_50`, `sma_200`, `above_sma_20`, `above_sma_200`, `price_vs_sma20_pct` |
| **Volume z-scores** | `volume_zscore`, `value_zscore`, `trades_zscore` (60d rolling) |
| **Breadth** | `breadth_ratio`, `breadth_net`, `advance_decline`, `breadth_zscore` |
| **Flows** | `foreign_net`, `domestic_net`, `foreign_net_cumulative_5d`, `foreign_net_cumulative_20d`, `foreign_participation`, `foreign_flow_slope_10d`, `foreign_flow_zscore`, `domestic_flow_zscore` |
| **GCC Relative** | `gcc_avg_return_1d`, `qse_vs_gcc_spread`, `qse_gcc_relative_5d`, `qse_gcc_rolling_corr_20d`, `gcc_peer_rank` |
| **Seasonality** | `day_of_week` (0=Sunday), `month`, `quarter`, `is_ramadan`, `trading_day_of_month` |

### Implementation notes

- **Price column**: uses `close` from `market_daily` (spec §16: `open, high, low, close`).
- **RSI-14** is computed via Wilder's EMA method (no external TA library dependency).
- **Z-scores** use a 60d rolling window; std == 0 returns `NaN` instead of dividing by zero.
- **breadth_zscore** is computed over `breadth_ratio` (0–1 range), not `breadth_net`.
- **OLS slope** (`foreign_flow_slope_10d`) uses `scipy.stats.linregress` over 10 sessions.
- **QSE calendar**: Sunday=0 through Thursday=4. Friday/Saturday are non-trading days and are excluded from `trading_day_of_month` ranking.
- **Ramadan flag**: Hard-coded date ranges 2018–2026; extend `RAMADAN_RANGES` for future years.
- **GCC data**: uses `daily_change_pct` field directly from `gcc_daily` (divided by 100 to decimal); market column is `market_name`.
- **GCC peer rank**: QSE ranked among all active GCC markets by 1-day return (1 = best performer).

### Output

`data/features/features_master.parquet` — one row per QSE trading day, all features joined.

Logs written to `logs/feature_build.log`.
