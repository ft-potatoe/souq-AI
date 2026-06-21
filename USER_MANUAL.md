# QSE Market Copilot — User Manual

A practical, operator-focused guide: what this system is, what *you* need to do,
how to add data and retrain models, and the things you must not get wrong.

---

## 1. What this system is (in one minute)

It is a **conversational analytics copilot** for the Qatar Stock Exchange. Analysts ask
plain-language questions about **historical** market activity; the system answers from
**pre-computed analytics and trained ML models**, then a local LLM turns those numbers
into prose.

The fixed pipeline:

```
Raw CSVs -> Ingest -> Features -> Analytics + ML Models -> Structured JSON -> LLM -> Answer
```

**The golden rule of the product:** the LLM never invents numbers and the system never
predicts the future. It describes and discovers what *has* happened — it does not forecast.
Everything below preserves that rule. Do not try to make it forecast.

**What it can do:**
- Answer questions about returns, volume, flows, breadth, GCC peers, regimes, etc.
- Find recurring **day-type clusters** ("today resembles a recurring high-vol state").
- Discover **relationships** ("usually when foreign selling rises, breadth tends to fall").
- Flag **anomalies** and surface **similar historical sessions**.

---

## 2. One-time setup

You need **Python 3.12**, **Node.js** (for the UI), and **Ollama** running locally.

```bash
# 1. Python dependencies
pip install -r requirements.txt

# 2. Ollama + the model (the app uses qwen3:8b)
#    Install Ollama from https://ollama.com, then:
ollama pull qwen3:8b
#    Ollama must be running and reachable at http://localhost:11434

# 3. UI dependencies
cd ui
npm install
cd ..
```

> The model is **qwen3:8b**, set in `llm/interface.py`. If you change the model there,
> pull that model in Ollama too. Larger models give nicer prose but are slower.

---

## 3. The daily / regular workflow (what YOU do)

There are exactly four things you ever do as the operator:

1. **Add new data** (CSV files) — Section 4.
2. **Run ingestion** — Section 5.
3. **Rebuild features** — Section 6.
4. **Retrain models** (weekly, or after big data additions) — Section 7.

Then **start the app** (Section 8) and ask questions. That's the whole job.

A typical "I have new market data" cycle — **one command does all of it**:

```bash
python scripts/refresh.py --src path/to/new_csvs
# then restart the API so it loads the new features:
uvicorn api.main:app --reload --port 8000
```

`refresh.py` runs ingest -> build features -> retrain in order and stops if a step
fails (so you never build features on half-ingested data). Useful flags:
`--skip-retrain` (ingest + features only), `--skip-ingest` (rebuild + retrain only),
`--replace` (overwrite instead of merge — see §4).

Or run the steps individually if you prefer:

```bash
python scripts/ingest/load_raw.py --src path/to/new_csvs   # 1+2 (merges by default)
python scripts/features/build_features.py                  # 3
python scripts/retrain/weekly_retrain.py                   # 4
```

---

## 4. How to add data (the part to get right)

### 4.1 Where it goes
Put your source files in **any folder**, then point the ingester at it with `--src`.
The ingester writes clean Parquet into `data/raw/`. You do **not** edit `data/raw/`
by hand.

### 4.1a Incremental data is safe — ingestion MERGES (important)
By default, ingestion **merges new rows into the existing history** rather than
overwriting it. New rows **win on overlapping dates** (keyed on `date`, or
`date + market_name` for `gcc_daily`). This means:

- You can add an **incremental slice** (e.g. you already loaded Jan 2024–Jun 2026,
  now you add Mar–Sep 2026). Result = the **union**, Jan 2024–Sep 2026, with the
  overlapping Mar–Jun 2026 taking the newly-ingested values. **History is preserved.**
- You do **not** have to re-feed the entire dataset every time (though doing so is
  also safe — re-ingesting identical data is idempotent).

Use `--replace` **only** when you deliberately want to wipe and start fresh:
```bash
python scripts/ingest/load_raw.py --src folder --replace   # overwrite, no merge
```

> Models are separate: adding data never erases trained models. They only change when
> you retrain, and retraining refits from scratch on the full current data (see §7).

### 4.2 File naming
Ingestion matches a file to a dataset if the **dataset name appears in the filename**
(case-insensitive). So these all work:

| File you provide | Recognised as |
|---|---|
| `market_daily_2026.csv` | `market_daily` |
| `qse_flows_daily.csv`   | `flows_daily` |
| `gcc_daily.xlsx`        | `gcc_daily` |
| `breadth_daily_jun.csv` | `breadth_daily` |

CSV and Excel are both accepted.

### 4.3 Required columns (exact names) — the four datasets

These column names are mandatory. Extra columns are ignored; missing ones quarantine the row.

**market_daily**
```
date, open, high, low, close, volume, value_traded, total_trades
```

**flows_daily**
```
date, foreign_buy, foreign_sell, foreign_net, domestic_buy, domestic_sell, domestic_net
```

**gcc_daily**  (one row per peer market per day; long format, not wide)
```
date, market_name, daily_change_pct
```
- `market_name` uses codes: `QSE, TASI, ADX, DFM, KSE, MSM, BSE`
  (`TASI` is displayed as "Tadawul").
- `daily_change_pct` is a **percent** (e.g. `1.25` means +1.25%), not a decimal.

**breadth_daily**
```
date, gainers, losers, unchanged, total_listed, total_traded
```

### 4.4 Date and calendar rules — important
- `date` must be ISO `YYYY-MM-DD`.
- The QSE trading week is **Sunday–Thursday**. Friday/Saturday are non-trading and are
  excluded from gap checks. Do **not** add Fri/Sat rows.
- Data should be roughly continuous. If **more than 3 consecutive trading days** are
  missing, ingestion **halts with exit code 2** (it refuses to silently bridge a gap).
  Fix the source data and re-run.

### 4.5 What happens to bad rows
Rows that fail validation are **not dropped silently** — they are written to
`data/raw/<dataset>_quarantine.parquet` and logged to `logs/ingestion_errors.log`.
After ingesting, **check that log** and the quarantine file if counts look off.

---

## 5. Running ingestion

```bash
# Ingest every recognised file in a folder:
python scripts/ingest/load_raw.py --src data/incoming

# Ingest just one dataset:
python scripts/ingest/load_raw.py --src data/incoming --dataset market_daily
```

**Exit codes you should know:**
- `0` = success
- `1` = some rows quarantined / a dataset failed (check the log)
- `2` = **pipeline halted** — a >3-day trading gap was detected. Nothing downstream
  should run until you resolve this.

Output lands in `data/raw/*.parquet`. Logs: `logs/ingestion_errors.log`.

---

## 6. Rebuilding features

After ingesting, regenerate the feature table the whole system reads from:

```bash
python scripts/features/build_features.py
```

This reads `data/raw/*.parquet` and writes **`data/features/features_master.parquet`**
(58 columns, one row per QSE trading day: returns, RSI, SMAs, volatility, breadth,
flows, GCC-relative, seasonality, plus forward-return columns used internally).

You should see a final line like `Wrote 599 rows x 58 cols`. This step is deterministic
— same input, same output. It never trains anything.

> Note: the system caches the feature file in memory per process. After rebuilding,
> **restart the API** (Section 8) so it picks up the new data.

---

## 7. Training / retraining the models

There are five trainable models. You normally retrain them all at once with the weekly
script; standalone scripts exist for targeted retrains.

### 7.1 The models

| Model | What it does | Retrains when |
|---|---|---|
| Anomaly scorer (RandomForest) | flags unusual days | >= 10 new feedback items |
| Similarity ranker (XGBRanker) | finds analogous past days | >= 20 new feedback items |
| Trend regime HMM | bull / sideways / bear states | every run |
| Volatility regime HMM | low-vol / high-vol states | every run |
| **Clustering (HDBSCAN)** | recurring day-types | every run |

### 7.2 The one command you usually run

```bash
python scripts/retrain/weekly_retrain.py
```

This runs the full pipeline, applies **validation gates**, and only deploys a model if it
passes. It writes a record to **`logs/retrain_log.jsonl`**. Recommended schedule:
**every Sunday 02:00**, or manually after a large data addition.

### 7.3 Standalone retrainers (optional)

```bash
python scripts/retrain/train_anomaly.py     # add --force to bypass the feedback threshold
python scripts/retrain/train_ranker.py      # add --force to bypass the feedback threshold
python scripts/retrain/train_hmm.py         # trend regime
python scripts/retrain/train_vol_hmm.py     # volatility regime
python scripts/retrain/train_clustering.py  # day-type clustering
```

### 7.4 Validation gates — why a model may "fail"
Models must clear quality bars or they are **rejected and the previous version is kept**
(automatic rollback). This is by design — a failing gate protects you from a bad model.

- Anomaly: precision >= 0.65, recall >= 0.60, AUC >= 0.72
- Similarity ranker: NDCG@10 >= 0.70
- Regime / Vol HMM: label-flip rate vs. prior <= 10%
- Clustering: silhouette >= 0.20, noise fraction <= 0.40, at least 2 clusters

### 7.5 How to read the retrain log
Open the last line of `logs/retrain_log.jsonl`. Key fields:
- `status`: `"success"` only if every **core** model deployed.
- `models`: per-model `deployed | skipped | failed`.
- `errors`: the reason for any failure.

**Important nuance:** `status: "failed"` does **not** necessarily mean something is broken.
Only three **core** models drive overall status: anomaly scorer, similarity ranker, trend
regime. The **volatility HMM and clustering are additive/non-blocking** — they can fail
their gate (and keep their prior model) while `status` is still driven by the core three.
So a "failed" overall status caused only by a regime flip-rate gate is a normal,
self-protecting outcome, not an outage.

### 7.6 Minimum data
The HMMs and clustering need **at least 250 trading sessions** before they produce labels.
Below that they return "not enough data" gracefully — that is expected on a fresh dataset,
not an error.

---

## 8. Starting and using the app

```bash
# Terminal 1 — backend API
uvicorn api.main:app --reload --port 8000

# Terminal 2 — frontend UI
cd ui
npm run dev          # opens the dev server (default http://localhost:5173)
```

Then open the UI in a browser and ask questions in plain English. Examples:

- "What was the biggest down day this year?"
- "How does today's volume compare to history?"
- "What kind of day is today compared to history?"  → **clustering**
- "Usually when foreign selling rises, what happens to breadth?"  → **relationships**
- "Has a day like this happened before?"  → **similarity**
- "Is today an anomaly?"  → **anomaly**

**Check system health** any time: `GET http://localhost:8000/health`
(reports whether features are loaded and Ollama is reachable). Model versions:
`GET http://localhost:8000/models/status`.

---

## 9. The feedback loop (optional but valuable)

Thumbs up/down and similarity star-ratings from the UI are stored in
`data/feedback/feedback.db`. They are what unlocks anomaly/ranker retraining (the
>=10 / >=20 thresholds). The more you rate, the better those two models get on the next
weekly retrain. You don't manage this DB by hand.

---

## 10. Things you MUST be aware of (read this)

1. **Ollama must be running** before you start the API, or `/query` returns a 503. The
   analytics still compute; only the prose generation needs the LLM.
2. **The model is qwen3:8b.** If you swap it in `llm/interface.py`, pull that model first.
3. **Rebuild features after every data change, and restart the API** so it reloads them.
4. **A >3-day trading-day gap halts ingestion (exit 2).** The gap is checked against the
   **merged** timeline, so a normal incremental slice won't false-trip it. If it does
   halt, there's a genuine hole in the combined history — resolve the source data; do not
   work around it by inserting fake rows.
5. **Quarantined rows are silent unless you look.** After ingestion, check
   `logs/ingestion_errors.log` and `data/raw/*_quarantine.parquet`.
6. **gcc_daily is long format and in percent.** One row per market per day;
   `daily_change_pct = 1.25` means +1.25%. Getting this wrong corrupts all GCC analytics.
7. **No Friday/Saturday rows.** The calendar is Sun–Thu.
8. **A "failed" weekly retrain is often benign** — see §7.5. Read the `errors` field before
   worrying; a vol-HMM or clustering gate failure does not take the system down.
9. **This is not a forecasting tool, by design.** If someone asks it to predict prices, the
   correct behaviour is that it won't. That is the product working as intended, not a bug.
10. **Discovered relationships are associations, not causation.** The UI and the LLM both
    label them that way on purpose. Don't represent them as cause-and-effect.

---

## 11. Quick troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `/query` returns 503 | Ollama down / model not pulled | Start Ollama; `ollama pull qwen3:8b` |
| Answers cite stale data | API running on old features | Rebuild features, restart `uvicorn` |
| Ingestion exits with code 2 | >3-day trading gap | Fill the gap in source data, re-run |
| Rows missing after ingest | Failed validation -> quarantine | Check `logs/ingestion_errors.log` + `*_quarantine.parquet` |
| Regime / cluster labels are null | < 250 sessions | Add more history; this is expected early on |
| Weekly retrain `status: failed` | A core model missed its gate | Read `errors` in `logs/retrain_log.jsonl`; prior model auto-kept |
| GCC numbers look 100x off | `daily_change_pct` given as decimal | Provide it as percent (1.25, not 0.0125) |

---

## 12. File / directory cheat-sheet

| Path | What it is |
|---|---|
| `data/raw/*.parquet` | Clean ingested data (don't hand-edit) |
| `data/raw/*_quarantine.parquet` | Rows that failed validation |
| `data/features/features_master.parquet` | The 58-feature table everything reads |
| `data/feedback/feedback.db` | User feedback store |
| `models/<name>/<name>_current(.ptr)` | Pointer to the live model for each model |
| `logs/ingestion_errors.log` | Ingestion warnings/errors |
| `logs/feature_build.log` | Feature build warnings/errors |
| `logs/retrain_log.jsonl` | One JSON line per retrain run |
