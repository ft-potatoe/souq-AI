# scripts/ingest

## load_raw.py

Reads CSV or Excel source files, validates them, and writes clean Parquet to `data/raw/`.

### Usage

```bash
# Activate venv first
source venv/bin/activate   # Windows: venv\Scripts\activate

python scripts/ingest/load_raw.py --src <source_directory>

# Single dataset only
python scripts/ingest/load_raw.py --src <source_directory> --dataset market_daily
```

### File discovery

Files are matched to datasets by their stem name. A file containing `market_daily` anywhere in its filename maps to the `market_daily` dataset. Supported formats: `.csv`, `.xlsx`, `.xls`.

### Datasets & required columns

| Dataset | Required Columns |
|---|---|
| `market_daily` | date, open, high, low, close, volume, value_traded, total_trades |
| `flows_daily` | date, foreign_buy, foreign_sell, foreign_net, domestic_buy, domestic_sell, domestic_net |
| `gcc_daily` | date, market_name, daily_change_pct |
| `breadth_daily` | date, gainers, losers, unchanged, total_listed, total_traded |

### Validation rules

| Rule | Scope | Action on failure |
|---|---|---|
| No future-dated rows | All | Quarantine |
| No duplicate dates (or date+market for gcc_daily) | All | Quarantine |
| volume, value_traded, total_trades > 0 on trading days | market_daily, gcc_daily | Quarantine |
| gainers + losers + unchanged == total_listed (±2) | breadth_daily | Quarantine |
| foreign_net == foreign_buy − foreign_sell (±1000 QAR) | flows_daily | Quarantine |
| domestic_net == domestic_buy − domestic_sell (±1000 QAR) | flows_daily | Quarantine |
| All GCC_ACTIVE_MARKETS present on every trading day | gcc_daily | Quarantine entire day |
| ≤ 3 consecutive missing trading days | All | **Halt pipeline** |

> QSE trading week: Sunday–Thursday. Friday and Saturday are excluded from gap checks.

### Outputs

- `data/raw/<dataset>.parquet` — clean rows
- `data/raw/<dataset>_quarantine.parquet` — rows that failed validation (if any)
- `logs/ingestion_errors.log` — all warnings and errors
