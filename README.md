# QSE Market Exchange Copilot

**Version:** 2.1
**Scope:** Phase 1 (all features) + Phase 2 Regime Detection + ML Training Layer
**Prepared for:** Qatar Stock Exchange — Market Operations
**LLM Runtime:** Ollama (local machine), Qwen3 14B
**Language:** English
**Last updated:** June 2026
**Changelog:** v2.1 — Fixed data leakage constraint in similarity ranker (§7.2); added token budget and bucket priority to query router (§12.4)

---

## Table of Contents

1. [Objective](#1-objective)
2. [Core Principles](#2-core-principles)
3. [Architecture Overview](#3-architecture-overview)
4. [Data Foundation](#4-data-foundation)
5. [Feature Engineering Layer](#5-feature-engineering-layer)
6. [Deterministic Analytics Layer](#6-deterministic-analytics-layer)
7. [Trained ML Layer](#7-trained-ml-layer)
8. [Regime Detection Layer — HMM](#8-regime-detection-layer--hmm)
9. [Feedback Store](#9-feedback-store)
10. [Weekly Retraining Pipeline](#10-weekly-retraining-pipeline)
11. [Local LLM Interface](#11-local-llm-interface)
12. [API Layer](#12-api-layer)
13. [Chat UI](#13-chat-ui)
14. [Technology Stack](#14-technology-stack)
15. [Project Structure](#15-project-structure)
16. [Data Schema](#16-data-schema)
17. [Feature Definitions Reference](#17-feature-definitions-reference)
18. [Acceptance Tests](#18-acceptance-tests)
19. [Out of Scope](#19-out-of-scope)
20. [Phase 3 Roadmap Reference](#20-phase-3-roadmap-reference)
21. [Quick Start](#21-quick-start)
22. [Build Status](#22-build-status)

---

## 1. Objective

Build a **Market Exchange Copilot** for QSE that answers natural language questions about historical market activity using data-backed, analytics-grounded responses — and that **improves with use** through a weekly ML retraining cycle driven by user feedback.

The system learns in two specific ways:

1. **Anomaly scoring** improves as the trained model sees more QSE data and absorbs analyst feedback about which anomalies were genuinely significant.
2. **Similarity ranking** improves as users indicate which historical sessions they found truly analogous — teaching the system what "structurally similar" means for QSE specifically, beyond raw geometric distance.

This is not a forecasting system. It does not predict future prices or returns.

**Primary audiences:** Market Operations team (daily), Senior leadership / C-suite (demos), individual analyst use.

---

## 2. Core Principles

### Principle 1: Analytics First, LLM Last

```
Raw Data -> Features -> Analytics + ML Models -> Structured JSON -> LLM -> Answer
```

The LLM receives only structured results. It never sees raw data, never performs calculations, and never invents statistics. Every number in the LLM's response must trace back to a computed result.

### Principle 2: Deterministic + Trained — Two Separate Concerns

| Layer | Type | Examples | Retrains? |
|---|---|---|---|
| Feature engineering | Deterministic | RSI, z-scores, slopes | Never |
| Descriptive analytics | Deterministic | Percentile rank, streak count, correlation | Never |
| Anomaly scoring | Trained (RF) | Is today genuinely anomalous for QSE? | Weekly |
| Similarity ranking | Trained (XGB) | Which sessions truly match? | Weekly |
| Regime detection | Trained (HMM) | Bull / bear / sideways | Weekly |

Deterministic layers never change. Trained layers improve weekly.

### Principle 3: Feedback is Training Signal

Every user interaction — thumbs up, thumbs down, a correction, a similarity rating — is stored and used in the next weekly retraining batch. The system gets better the more it is used.

### Principle 4: Model Safety and Rollback

No retrained model is deployed without passing a validation gate. Every deployment is versioned. Rollback to the prior version is a single command.

### Principle 5: Local Deployment, No External APIs

All data, models, and LLM inference run on-premises. No cloud LLM calls, no data leaves the machine.

---

## 3. Architecture Overview

```
+----------------------------------------------------------+
|                   Raw Market Data                        |
|        (5-9 years: OHLCV, flows, breadth, GCC)          |
+---------------------+------------------------------------+
                      | nightly ingestion + validation
+---------------------v------------------------------------+
|              Feature Engineering Layer                   |
|         Deterministic - pandas / scipy / ta              |
|   Returns · volatility · RSI · z-scores · ratios · GCC  |
+------+---------------+---------------+------------------+
       |               |               |
+------v------+ +------v------+ +------v------------------+
|Deterministic| | Trained     | | Trained                 |
|  Analytics  | | Anomaly     | | Similarity              |
|             | | Scorer      | | Ranker                  |
|Distributions| |             | |                         |
|Trends       | |Random       | |LambdaRank /             |
|Correlation  | |Forest       | |XGBoost                  |
|Seasonality  | |             | |                         |
|Flows        | |Learns QSE   | |Learns true match        |
|GCC bench    | |patterns     | |quality from feedback    |
+------+------+ +------+------+ +------+------------------+
       |               |               |
+------v---------------v---------------v------------------+
|                 HMM Regime Layer                         |
|            Bull / Bear / Sideways                        |
+---------------------+------------------------------------+
                      |
+---------------------v------------------------------------+
|              Structured JSON Results                     |
+---------------------+------------------------------------+
                      |
+---------------------v------------------------------------+
|           Local LLM - Qwen3 14B via Ollama               |
|                Explanation only                          |
+---------------------+------------------------------------+
                      |
+---------------------v------------------------------------+
|                    Chat UI                               |
|         FastAPI backend + React frontend                 |
+---------------------+------------------------------------+
                      | user feedback
+---------------------v------------------------------------+
|               Feedback Store (SQLite)                    |
|    thumbs · corrections · similarity ratings            |
+---------------------+------------------------------------+
                      | every Sunday 02:00
+---------------------v------------------------------------+
|            Weekly Retraining Pipeline                    |
|   Anomaly RF · Similarity Ranker · HMM · Validation     |
+----------------------------------------------------------+
```

---

## 4. Data Foundation

### 4.1 Confirmed Available Data

| Dataset | Fields | History |
|---|---|---|
| Daily index | open, high, low, close | 5-9 years |
| Market activity | volume, value_traded, total_trades | 5-9 years |
| Breadth | gainers, losers, unchanged | 5-9 years |
| Foreign flows | buy, sell, net | 5-9 years |
| Domestic flows | buy, sell, net | 5-9 years |
| GCC peers | daily_change_pct, pe_ratio, div_yield | 5-9 years |
| Individual stocks | OHLCV per security | 5-9 years |

Individual stock data is stored and kept clean but is **not used in this phase**. It is available for Phase 3 stock-level analytics.

### 4.2 Data Ingestion

- **Script:** `scripts/ingest/load_raw.py`
- **Schedule:** Nightly at 19:00 (after QSE market close)
- **Storage:** Parquet files under `data/raw/`
- **Format:** CSV or Excel exports from QSE internal systems

### 4.3 Data Quality Rules

Every ingestion run enforces:

```python
RULES = [
    "No future-dated rows",
    "No duplicate dates per dataset",
    "Volume > 0 on trading days",
    "Value_traded > 0 on trading days",
    "Total_trades > 0 on trading days",
    "Gainers + losers + unchanged == total_listed (tolerance +/-2)",
    "Foreign_net == foreign_buy - foreign_sell (tolerance +/-1000 QAR)",
    "Domestic_net == domestic_buy - domestic_sell (tolerance +/-1000 QAR)",
    "GCC peer records present for all active markets on trading days",
]
```

Violations are logged to `logs/ingestion_errors.log`. Rows that fail critical checks are quarantined to `data/raw/<dataset>_quarantine.parquet`, not silently dropped. The pipeline halts and alerts if more than 3 consecutive trading days are missing.

---

## 5. Feature Engineering Layer

All features are pure functions of historical data. This layer never trains and never changes.
Output stored as `data/features/features_master.parquet`, refreshed nightly after ingestion.
**Current output: 55 columns × N trading days.**

### 5.1 Return Features

```python
return_1d   = close.pct_change(1)
return_5d   = close.pct_change(5)
return_20d  = close.pct_change(20)
return_60d  = close.pct_change(60)
```

### 5.2 Volatility Features

```python
volatility_20d  = return_1d.rolling(20).std() * np.sqrt(252)
volatility_60d  = return_1d.rolling(60).std() * np.sqrt(252)
```

### 5.3 Momentum and Technical Features

```python
rsi_14             = wilder_rsi(close, period=14)   # built-in, no pandas-ta
sma_20             = close.rolling(20).mean()
sma_50             = close.rolling(50).mean()
sma_200            = close.rolling(200).mean()
above_sma_20       = (close > sma_20).astype(int)
above_sma_200      = (close > sma_200).astype(int)
price_vs_sma20_pct = (close - sma_20) / sma_20
```

### 5.4 Z-Score Features — Rolling 60-Day Window

```python
def rolling_zscore(series, window=60):
    mu    = series.rolling(window).mean()
    sigma = series.rolling(window).std().replace(0, np.nan)  # no divide-by-zero
    return (series - mu) / sigma

volume_zscore   = rolling_zscore(volume)
value_zscore    = rolling_zscore(value_traded)
trades_zscore   = rolling_zscore(total_trades)
```

### 5.5 Breadth Features

```python
breadth_ratio   = gainers / (gainers + losers + unchanged)
breadth_net     = gainers - losers
advance_decline = gainers / np.maximum(losers, 1)
breadth_zscore  = rolling_zscore(breadth_ratio)   # over ratio, not net
```

### 5.6 Flow Features

```python
foreign_net                 = foreign_buy - foreign_sell
domestic_net                = domestic_buy - domestic_sell
foreign_flow_zscore         = rolling_zscore(foreign_net)
domestic_flow_zscore        = rolling_zscore(domestic_net)
foreign_net_cumulative_5d   = foreign_net.rolling(5).sum()
foreign_net_cumulative_20d  = foreign_net.rolling(20).sum()
foreign_participation       = (foreign_buy + foreign_sell) / value_traded
foreign_flow_slope_10d      = rolling_ols_slope(foreign_net, 10)
```

### 5.7 GCC Relative Features

```python
# gcc_daily.daily_change_pct / 100 gives decimal return directly
gcc_avg_return_1d        = mean(peer daily_change_pct/100 for each non-QSE market)
qse_vs_gcc_spread        = return_1d - gcc_avg_return_1d
qse_gcc_relative_5d      = qse_vs_gcc_spread.rolling(5).sum()
qse_gcc_rolling_corr_20d = rolling_corr(return_1d, gcc_avg_return_1d, 20)
gcc_peer_rank            = rank of QSE among all GCC markets by return_1d (1=best)
```

### 5.8 Seasonality Encodings

```python
day_of_week          = remap(date.dayofweek)   # 0=Sunday for QSE calendar
month                = date.month
quarter              = date.quarter
is_ramadan           = check_ramadan_ranges(date)   # hard-coded 2018-2026
trading_day_of_month = intra_month_trading_day_rank(date)
```

---

## 6. Deterministic Analytics Layer

These modules produce factual, mathematical answers. They never train and never change. Each exposes `run(date, params) -> dict`.

### 6.1 Distribution Analysis — `analytics/distribution.py`

**Questions answered:** How unusual is today's [metric]? What percentile rank? How often has this level been reached?

```python
def percentile_rank(series, value) -> float
def historical_frequency(series, value, direction="above") -> float
def rolling_stats(series, windows=[20, 60, 252]) -> dict
```

**Output schema:**
```json
{
  "metric": "volume",
  "today_value": 142500000,
  "percentile_rank": 87.3,
  "historical_frequency_above": 0.127,
  "rolling_stats": {
    "20d": {"mean": 98000000, "std": 22000000},
    "60d": {"mean": 102000000, "std": 25000000},
    "252d": {"mean": 105000000, "std": 28000000}
  },
  "last_comparable_date": "2022-11-14",
  "sessions_above_today": 47,
  "total_sessions": 1240
}
```

**Supported metrics:** `volume`, `value_traded`, `total_trades`, `breadth_ratio`, `foreign_net`, `domestic_net`, `return_1d`, `volatility_20d`, `foreign_participation`

### 6.2 Trend Analysis — `analytics/trend.py`

**Questions answered:** Is foreign flow trending positive or negative? How many consecutive sessions of net selling? Has volume been increasing or decreasing?

```python
def linear_slope(series, window) -> float
def streak_count(series, condition_fn) -> int
def momentum_direction(series, windows=[5, 10, 20]) -> dict
def sma_crossover(fast_sma, slow_sma) -> str | None
```

**Output schema:**
```json
{
  "metric": "foreign_net",
  "slope_10d": -2340000,
  "slope_direction": "decreasing",
  "streak": {"condition": "negative", "count": 7},
  "momentum": {"5d": "down", "10d": "down", "20d": "flat"},
  "sma_crossover": "death_cross",
  "crossover_date": "2024-03-04"
}
```

### 6.3 Correlation Analysis — `analytics/correlation.py`

**Questions answered:** Is foreign flow correlated with index returns? Has QSE decoupled from GCC peers?

```python
def rolling_correlation(series_a, series_b, windows=[20, 60]) -> dict
def current_vs_historical_corr(series_a, series_b, window=20) -> dict
def gcc_peer_correlations(qse_returns, gcc_dict, window=60) -> dict
```

**Output schema:**
```json
{
  "pair": "foreign_net vs return_1d",
  "rolling_corr_20d": 0.43,
  "rolling_corr_60d": 0.31,
  "historical_mean_corr_60d": 0.38,
  "percentile_of_current_corr": 67.2,
  "gcc_correlations_60d": {"Tadawul": 0.61, "ADX": 0.54, "DFM": 0.49}
}
```

### 6.4 Seasonality Patterns — `analytics/seasonality.py`

**Questions answered:** Is today a typically high or low volume day? How does this month historically perform? Does Ramadan affect trading?

```python
def day_of_week_profile(metric, series) -> dict
def monthly_profile(metric, series) -> dict
def ramadan_effect(metric, series, ramadan_dates) -> dict
```

**Output schema:**
```json
{
  "metric": "volume",
  "today_day_of_week": "Wednesday",
  "day_of_week_mean": 118000000,
  "day_of_week_rank": "2nd highest",
  "monthly_mean_this_month": 105000000,
  "monthly_rank": "4th of 12",
  "ramadan_effect": {
    "ramadan_mean": 82000000,
    "non_ramadan_mean": 112000000,
    "pct_difference": -26.8,
    "is_ramadan_today": false
  }
}
```

### 6.5 Flow Analysis — `analytics/flows.py`

**Questions answered:** Who is driving the market — foreign or domestic? Has net foreign buying been building?

```python
def cumulative_pressure(net_series, windows=[5, 10, 20]) -> dict
def participation_ratio(buy, sell, value) -> float
def flow_dominance(foreign_net, domestic_net) -> str
def pressure_trend(net_series, window=10) -> dict
```

**Output schema:**
```json
{
  "date": "2024-06-05",
  "foreign_net_today": -45000000,
  "dominant_flow": "foreign_selling",
  "cumulative_foreign_net": {"5d": -180000000, "10d": -320000000, "20d": -85000000},
  "foreign_participation_pct": 34.2,
  "flow_pressure_trend_10d": "increasing_selling",
  "foreign_flow_zscore": -2.3,
  "domestic_flow_zscore": 1.1
}
```

### 6.6 GCC Benchmarking — `analytics/gcc.py`

**Questions answered:** Is QSE outperforming or underperforming GCC peers? How has the spread changed this week?

```python
def peer_relative_performance(qse_returns, gcc_returns, horizons=[1, 5, 20]) -> dict
def peer_rank(qse_return, gcc_returns_dict) -> int
def rolling_outperformance_rate(qse_returns, gcc_avg, window=60) -> float
```

**Output schema:**
```json
{
  "date": "2024-06-05",
  "qse_return_1d": 0.42,
  "gcc_avg_return_1d": -0.18,
  "qse_vs_gcc_spread_1d": 0.60,
  "qse_rank_today": 1,
  "total_peers": 6,
  "peer_returns": {"Tadawul": -0.31, "ADX": -0.12, "DFM": 0.08},
  "qse_vs_gcc_spread_5d": 1.2,
  "rolling_outperformance_rate_60d": 0.52
}
```

---

## 7. Trained ML Layer

### 7.1 Anomaly Scorer — `ml/anomaly_scorer.py`

**Model:** `RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=5, class_weight='balanced')`

**Input features:**
```python
ANOMALY_FEATURES = [
    'volume_zscore', 'value_zscore', 'trades_zscore',
    'breadth_zscore', 'foreign_flow_zscore', 'domestic_flow_zscore',
    'return_1d', 'volatility_20d', 'rsi_14',
    'qse_vs_gcc_spread', 'foreign_participation',
    'above_sma_200', 'price_vs_sma20_pct'
]
```

**Bootstrap labels:** Any session with >= 2 z-scores exceeding |2.0| is labelled anomalous=1. User feedback overrides bootstrap labels.

**Validation gate:** Precision >= 0.65, Recall >= 0.60, AUC-ROC >= 0.72, no more than 15% degradation vs prior version.

**Output:**
```json
{
  "anomaly_score": 0.78,
  "anomaly_label": "anomalous",
  "confidence": "high",
  "top_contributing_features": [
    {"feature": "foreign_flow_zscore", "importance": 0.34},
    {"feature": "volume_zscore", "importance": 0.28}
  ],
  "model_version": "rf_anomaly_v12",
  "bootstrap_label_used": false
}
```

### 7.2 Similarity Ranker — `ml/similarity_ranker.py`

**Pipeline:** k-NN top-40 candidates -> XGBRanker re-scores -> top-10 returned.

**Model:** `XGBRanker(objective='rank:pairwise', learning_rate=0.05, n_estimators=200, max_depth=5)`

**Input features (pairwise deltas):**
```python
RANKER_FEATURES = [
    'delta_return_1d', 'delta_volume_zscore', 'delta_foreign_flow_zscore',
    'delta_breadth_ratio', 'delta_volatility_20d', 'delta_rsi_14',
    'same_regime', 'regime_transition_match', 'days_since_candidate',
    'forward_return_5d_candidate', 'forward_return_10d_candidate'
]
```

> **Data leakage guard — never bypass:**
> `forward_return_5d_candidate` and `forward_return_10d_candidate` must be set to
> `None` for any candidate session within 10 trading days of today.
> ```python
> def safe_forward_returns(candidate_date, today, fwd_5d, fwd_10d):
>     trading_days_since = count_trading_days(candidate_date, today)
>     return (
>         fwd_5d  if trading_days_since > 5  else None,
>         fwd_10d if trading_days_since > 10 else None,
>     )
> ```
> Call this for every candidate when building pairwise features — both training and inference.

**Validation gate:** NDCG@10 >= 0.70, no more than 10% degradation vs prior version.

### 7.3 Label Construction

| Model | Primary label | Cold-start fallback |
|---|---|---|
| Anomaly scorer | User feedback (confirm/reject) | Bootstrap: >=2 z-scores above \|2.0\| |
| Similarity ranker | User ratings (1-5 stars) | k-NN cosine similarity score |
| HMM regime | Unsupervised — no labels needed | — |

### 7.4 Model Versioning and Rollback

```
models/
  anomaly_scorer/     rf_anomaly_v1.pkl ... rf_anomaly_current (symlink)
  similarity_ranker/  ranker_v1.pkl     ... ranker_current
  regime_hmm/         hmm_v1.pkl        ... hmm_current
```

```bash
python scripts/models/rollback.py --model anomaly_scorer --version v11
```

---

## 8. Regime Detection Layer — HMM

**Module:** `analytics/regime.py`

```python
from hmmlearn.hmm import GaussianHMM
model = GaussianHMM(n_components=3, covariance_type="full", n_iter=200, random_state=42)

HMM_FEATURES = ['return_1d', 'volatility_20d', 'volume_zscore',
                 'breadth_ratio', 'foreign_flow_zscore', 'rsi_14']
```

All features standardised with `StandardScaler` before fitting. States mapped to labels by mean `return_1d`: lowest = bear, middle = sideways, highest = bull.

**Minimum history:** 250 sessions before surfacing regime labels.

**Output:**
```json
{
  "current_regime": "bear",
  "regime_probability": 0.81,
  "sessions_in_current_regime": 14,
  "regime_start_date": "2024-05-16",
  "prior_regime": "sideways",
  "regime_distribution_historical": {"bull": 0.38, "bear": 0.29, "sideways": 0.33},
  "model_version": "hmm_v6"
}
```

---

## 9. Feedback Store

**Module:** `feedback/store.py` — SQLite at `data/feedback/feedback.db`

```sql
CREATE TABLE feedback (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       DATETIME NOT NULL,
    user_id         TEXT,
    query_date      DATE,
    question        TEXT,
    feedback_type   TEXT NOT NULL,
    -- 'thumbs_up' | 'thumbs_down' | 'anomaly_confirm' | 'anomaly_reject'
    -- | 'similarity_rating' | 'correction'
    target_date     DATE,
    rating          INTEGER,   -- 1-5 for similarity_rating
    correction_text TEXT,
    model_versions  TEXT       -- JSON
);
```

Three UI feedback mechanisms: thumbs up/down on every response, anomaly confirm/reject when anomaly scorer fires, similarity star rating (1-5) on each matched session card.

---

## 10. Weekly Retraining Pipeline

**Script:** `scripts/retrain/weekly_retrain.py` — runs every Sunday at 02:00

**Steps:**
1. Load `features_master.parquet` + `feedback.db`
2. Build training labels (feedback overrides bootstraps)
3. Train anomaly scorer -> validate -> deploy if passes, else keep prior
4. Build ranking pairs from feedback
5. Train similarity ranker -> validate NDCG@10 -> deploy if passes
6. Refit HMM on full history -> validate state consistency
7. Reload API workers with new model symlinks
8. Log summary to `logs/retrain_log.jsonl`

**Minimum feedback thresholds:**

| Model | Minimum new feedback to retrain |
|---|---|
| Anomaly scorer | 10 new anomaly_confirm / anomaly_reject |
| Similarity ranker | 20 new similarity_rating items |
| HMM | Always retrains |

---

## 11. Local LLM Interface

**Module:** `llm/interface.py`

```bash
ollama pull qwen3:14b
ollama serve
```

**System prompt rules (strict):**
1. Never perform calculations — all numbers from analytics JSON only
2. Never invent statistics, dates, or events not in the JSON
3. Never forecast or predict future prices or returns
4. If JSON lacks information, say so explicitly rather than speculating
5. Always cite specific values from the JSON
6. Keep responses to 3-5 paragraphs unless detail is requested
7. Acknowledge top contributing anomaly features by name
8. Mention regime match status when discussing similar sessions
9. Temperature: **0.1** (minimises hallucination)

**Ollama call:**
```python
httpx.post("http://localhost:11434/api/generate", json={
    "model": "qwen3:14b",
    "prompt": prompt,
    "system": system,
    "stream": False,
    "options": {"temperature": 0.1, "top_p": 0.9, "num_predict": 1024}
}, timeout=120.0)
```

---

## 12. API Layer

**Framework:** FastAPI + Uvicorn — `api/main.py`

### Endpoints

```
POST   /query                Natural language question -> full response
GET    /regime/current       Current regime label and probabilities
GET    /regime/history       Full regime label history
GET    /features/today       All computed features for today
GET    /features/{date}      Features for a specific date
GET    /similarity/{date}    Top-N ranked similar sessions
GET    /anomaly/{date}       Anomaly score and contributing features
POST   /feedback             Submit user feedback
GET    /models/status        Current model versions and last retrain timestamp
GET    /health               Service health check
```

### Query Router — `llm/router.py`

```python
BUCKET_KEYWORDS = {
    "distribution": ["unusual", "percentile", "rank", "how high", "how low", "extreme", "rare"],
    "trend":        ["trend", "increasing", "decreasing", "consecutive", "streak", "direction"],
    "anomaly":      ["anomaly", "anomalous", "flag", "signal", "alert", "simultaneous", "today"],
    "similarity":   ["similar", "before", "analogy", "like this", "happened", "analog", "historical"],
    "correlation":  ["correlated", "relationship", "decoupled", "moving with", "together"],
    "seasonality":  ["day of week", "monthly", "ramadan", "seasonal", "typical", "usually"],
    "flows":        ["flow", "foreign", "domestic", "buying", "selling", "pressure", "participation"],
    "gcc":          ["gcc", "peer", "tadawul", "adx", "dfm", "outperform", "underperform", "region"],
    "regime":       ["regime", "bull", "bear", "sideways", "state", "environment", "phase"]
}

BUCKET_PRIORITY = [
    "anomaly",       # 1 - always include if fired
    "regime",        # 2 - always include; essential context
    "flows",         # 3 - core daily operational concern
    "distribution",  # 4 - answers "how unusual" questions
    "trend",         # 5 - directional context
    "similarity",    # 6 - historical analogy; rich but large
    "gcc",           # 7 - regional benchmarking
    "correlation",   # 8 - relationship analysis
    "seasonality",   # 9 - background context; lowest urgency
]

PAYLOAD_TOKEN_BUDGET = 3500   # leaves headroom for system prompt + response
```

Lower-priority buckets are compressed then dropped when the budget is exceeded. Compression strategies: similarity trims to 3 matches, gcc drops per-peer breakdown, seasonality keeps today's day rank only.

---

## 13. Chat UI

**Framework:** React 18 + Vite — `ui/src/`

**Core components:**
- `ChatWindow.jsx` — streaming token-by-token response display
- `RegimeBadge.jsx` — permanently visible header badge (Bull=green, Bear=red, Sideways=amber)
- `AnomalyFeedback.jsx` — confirm/reject widget shown when `anomaly_score > 0.65`
- `SimilarityCard.jsx` — matched session card with 1-5 star rating
- `AnalyticsPanel.jsx` — collapsible raw JSON view for analyst transparency
- `ModelStatus.jsx` — model versions, feedback counts, last retrain timestamp

**Regime badge format:**
```
● BEAR REGIME  81% confidence  |  Active 14 sessions since 16 May 2024  |  v7
```

**Model learning indicator (footer):**
```
Anomaly model: v13 - trained on 287 feedback samples
Similarity model: v9 - trained on 512 rating pairs
Last retrained: Sunday 9 Jun 2024 at 02:14
```

---

## 14. Technology Stack

| Component | Library / Tool | Installed Version |
|---|---|---|
| Language | Python 3.12 | 3.12.1 |
| Data processing | pandas 2.x | 2.2.3 |
| Technical features | ta (replaces pandas-ta) | 0.11.0 |
| Statistics | scipy 1.x | 1.17.1 |
| k-NN candidate pool | scikit-learn 1.x | 1.9.0 |
| Anomaly scorer | scikit-learn RandomForestClassifier | 1.9.0 |
| Similarity ranker | xgboost XGBRanker | 3.2.0 |
| Regime detection | hmmlearn 0.3.x | 0.3.3 |
| Data storage | pyarrow / parquet | 24.0.0 |
| Feedback storage | SQLite (sqlite3) | built-in |
| Model storage | joblib | 1.5.3 |
| API | FastAPI + uvicorn | 0.136.3 / 0.49.0 |
| LLM runtime | Ollama | 0.30.6 |
| LLM model | Qwen3 14B | qwen3:14b (9.3 GB) |
| HTTP client | httpx | 0.28.1 |
| Frontend | React 18 + Vite 5 | Node v20.13.1 |
| Scheduler | APScheduler | 3.11.2 |

> **Note:** `pandas-ta` was replaced by `ta` due to a numpy version conflict.
> `pandas==2.2.*` requires `numpy<2`; `pandas-ta` requires `numpy>=2.2.6`.
> The `ta` library covers the same indicators and is compatible with `numpy 1.26.4`.

---

## 15. Project Structure

```
souq-AI/
├── data/
│   ├── raw/                        # ingested parquets + _quarantine files
│   ├── features/
│   │   └── features_master.parquet # 55 columns, one row per trading day
│   ├── regimes/
│   │   └── regime_labels.parquet
│   └── feedback/
│       └── feedback.db             # SQLite
├── models/
│   ├── anomaly_scorer/
│   │   ├── rf_anomaly_v1.pkl
│   │   └── rf_anomaly_current      # symlink to latest validated model
│   ├── similarity_ranker/
│   │   └── ranker_current
│   └── regime_hmm/
│       └── hmm_current
├── scripts/
│   ├── ingest/
│   │   ├── load_raw.py             # [DONE]
│   │   └── README.md
│   ├── features/
│   │   ├── build_features.py       # [DONE]
│   │   └── README.md
│   ├── retrain/
│   │   ├── weekly_retrain.py
│   │   ├── train_anomaly.py
│   │   ├── train_ranker.py
│   │   └── train_hmm.py
│   └── models/
│       └── rollback.py
├── analytics/
│   ├── distribution.py
│   ├── trend.py
│   ├── correlation.py
│   ├── seasonality.py
│   ├── flows.py
│   ├── gcc.py
│   └── regime.py
├── ml/
│   ├── anomaly_scorer.py
│   └── similarity_ranker.py
├── feedback/
│   └── store.py
├── llm/
│   ├── interface.py
│   ├── prompts.py
│   └── router.py
├── api/
│   ├── main.py
│   ├── models.py
│   └── endpoints/
│       ├── query.py
│       ├── regime.py
│       ├── features.py
│       ├── feedback.py
│       └── models_status.py
├── ui/
│   ├── src/
│   │   ├── App.jsx
│   │   └── components/
│   │       ├── ChatWindow.jsx
│   │       ├── RegimeBadge.jsx
│   │       ├── AnomalyFeedback.jsx
│   │       ├── SimilarityCard.jsx
│   │       ├── AnalyticsPanel.jsx
│   │       └── ModelStatus.jsx
│   └── package.json
├── tests/
│   ├── make_test_data.py           # [DONE] synthetic data generator
│   ├── test_features.py
│   ├── test_analytics.py
│   ├── test_anomaly_scorer.py
│   ├── test_similarity_ranker.py
│   └── test_regime.py
├── notebooks/
│   ├── regime_exploration.ipynb
│   └── anomaly_bootstrap_labels.ipynb
├── logs/
│   ├── ingestion_errors.log
│   ├── feature_build.log
│   └── retrain_log.jsonl
├── requirements.txt
└── README.md                       # this file (merged spec + implementation)
```

---

## 16. Data Schema

### market_daily
```
date            DATE    PK
open            FLOAT
high            FLOAT
low             FLOAT
close           FLOAT
volume          BIGINT
value_traded    FLOAT   (QAR)
total_trades    INTEGER
```

### flows_daily
```
date            DATE    PK
foreign_buy     FLOAT (QAR)
foreign_sell    FLOAT (QAR)
foreign_net     FLOAT (QAR)
domestic_buy    FLOAT (QAR)
domestic_sell   FLOAT (QAR)
domestic_net    FLOAT (QAR)
```

### gcc_daily
```
date            DATE
market_name     VARCHAR
daily_change_pct FLOAT
pe_ratio        FLOAT   (optional)
dividend_yield  FLOAT   (optional)
```

### breadth_daily
```
date            DATE    PK
gainers         INTEGER
losers          INTEGER
unchanged       INTEGER
total_listed    INTEGER
total_traded    INTEGER
```

### regime_labels
```
date            DATE    PK
regime          VARCHAR (bull / bear / sideways)
regime_state_id INTEGER
prob_bull       FLOAT
prob_bear       FLOAT
prob_sideways   FLOAT
model_version   VARCHAR
```

---

## 17. Feature Definitions Reference

| Feature | Formula | Window | Notes |
|---|---|---|---|
| `return_1d` | `close.pct_change(1)` | — | |
| `return_5d/20d/60d` | `close.pct_change(N)` | — | |
| `volatility_20d` | `return_1d.rolling(20).std() x sqrt(252)` | 20d | Annualised |
| `volatility_60d` | `return_1d.rolling(60).std() x sqrt(252)` | 60d | Annualised |
| `rsi_14` | Wilder EMA RSI | 14d | 0-100 |
| `sma_20/50/200` | `close.rolling(N).mean()` | N | |
| `above_sma_20` | `close > sma_20` | — | Binary |
| `above_sma_200` | `close > sma_200` | — | Binary |
| `volume_zscore` | `(vol - mu) / sigma` | 60d rolling | NaN if std=0 |
| `value_zscore` | `(val - mu) / sigma` | 60d rolling | NaN if std=0 |
| `trades_zscore` | `(trades - mu) / sigma` | 60d rolling | NaN if std=0 |
| `breadth_ratio` | `gainers / (G+L+U)` | — | 0-1 |
| `breadth_zscore` | `rolling_zscore(breadth_ratio)` | 60d rolling | Over ratio, not net |
| `foreign_net` | `foreign_buy - foreign_sell` | — | QAR |
| `foreign_flow_zscore` | `rolling_zscore(foreign_net)` | 60d rolling | |
| `foreign_participation` | `(fbuy + fsell) / value_traded` | — | 0-1 |
| `foreign_flow_slope_10d` | OLS slope of `foreign_net` | 10d | scipy linregress |
| `qse_vs_gcc_spread` | `return_1d - gcc_avg_return_1d` | — | |
| `gcc_peer_rank` | Rank by return_1d among GCC | — | 1=best |
| `price_vs_sma20_pct` | `(close - sma_20) / sma_20` | — | |
| `day_of_week` | Remapped dayofweek | — | Sun=0, Thu=4 |
| `is_ramadan` | Hard-coded date ranges | — | Binary |
| `trading_day_of_month` | Intra-month rank | — | 1-based |

---

## 18. Acceptance Tests

The system is complete when all 18 tests pass without the LLM fabricating any value.

**Deterministic analytics (8 tests)**
1. Distribution: What percentile is today's volume in the past 60 days?
2. Distribution: How often has foreign net selling been this extreme historically?
3. Trend: Is foreign net flow trending positive or negative over the last 10 sessions?
4. Trend: How many consecutive sessions of net foreign selling have there been?
5. Correlation: Has QSE decoupled from GCC peers in the past 20 days?
6. Seasonality: Is today a typically high or low volume day of the week for QSE?
7. Flows: Who is driving the market today — foreign or domestic?
8. GCC: Is QSE outperforming or underperforming GCC peers this week?

**Trained ML layer (5 tests)**
9. Anomaly: Does the anomaly scorer identify today as anomalous, and does it name the top contributing features?
10. Anomaly: After 2 weeks of feedback, does the anomaly scorer improve its AUC-ROC vs the bootstrap baseline?
11. Similarity: Do the top-10 similar sessions show a meaningful regime distribution?
12. Similarity: After 20 ratings, does the ranker re-order results differently from pure k-NN?
13. Similarity: What happened in the 5 and 10 trading days following the top-ranked similar sessions?

**Regime (3 tests)**
14. Regime: What market regime is QSE currently in, and how long has it persisted?
15. Regime: Are the top similar sessions predominantly from the same regime as today?
16. Regime: Has the regime changed in the past month, and when did the transition occur?

**Learning loop (2 tests)**
17. Retraining: Does the weekly retraining pipeline complete successfully, log results, and update model symlinks?
18. Rollback: Does rolling back to a prior model version restore previous inference results correctly?

---

## 19. Out of Scope

Explicitly excluded — do not add without a formal scope change:

- Forecasting or price prediction of any kind
- SHAP attribution (Phase 3)
- DCC-GARCH or Granger causality modelling
- TS2Vec or any neural embedding model
- FAISS (sklearn k-NN is sufficient for this data volume)
- UMAP visualisation
- Real-time or intraday data feeds
- Arabic language output
- Multi-agent orchestration (LangGraph)
- RAG on historical PDF reports
- Fine-tuning of the local LLM
- Alert / push notification engine
- Individual stock-level analytics

---

## 20. Phase 3 Roadmap Reference

| Priority | Capability | Depends On |
|---|---|---|
| 1 | SHAP attribution for anomaly scorer | Anomaly scorer stable |
| 2 | UMAP session space visualisation | Regime labels stable |
| 3 | FAISS index for scaled similarity | Similarity ranker stable |
| 4 | Granger / VAR causality | Data quality validation |
| 5 | Individual stock analytics | Index-level system stable |
| 6 | TS2Vec embeddings | GPU infrastructure |
| 7 | RAG on historical reports | PDF corpus + Chroma |
| 8 | Alert / notification engine | Analytics layer stable |
| 9 | Arabic language support | LLM swap or fine-tuning |

---

## 21. Quick Start

```bash
# 1. Activate environment
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# 2. Ingest source data (CSV/Excel -> data/raw/)
python scripts/ingest/load_raw.py --src <your_data_dir>

# 3. Build feature master (data/features/features_master.parquet)
python scripts/features/build_features.py

# 4. Start API
uvicorn api.main:app --reload --port 8000

# 5. Ensure Ollama is running
ollama serve   # or it runs as a system service after install

# 6. Start UI
cd ui && npm run dev
```

---

## 22. Build Status

| Component | File | Status |
|---|---|---|
| Data ingestion | `scripts/ingest/load_raw.py` | Done |
| Feature engineering | `scripts/features/build_features.py` | Done |
| Test data generator | `tests/make_test_data.py` | Done |
| End-to-end test | 300 trading days, 16/16 sanity checks | Pass |
| Feature tests | `tests/test_features.py` — 59 tests | Pass |
| Distribution analytics | `analytics/distribution.py` | Done |
| Trend analytics | `analytics/trend.py` | Done |
| Correlation analytics | `analytics/correlation.py` | Done |
| Seasonality analytics | `analytics/seasonality.py` | Done |
| Flow analytics | `analytics/flows.py` | Done |
| GCC benchmarking | `analytics/gcc.py` | Done |
| Analytics acceptance tests | `tests/test_analytics.py` — 138 tests, AT-1–AT-8 | Pass |
| Regime detection | `analytics/regime.py` | Pending |
| Anomaly scorer | `ml/anomaly_scorer.py` | Pending |
| Similarity ranker | `ml/similarity_ranker.py` | Pending |
| Feedback store | `feedback/store.py` | Pending |
| LLM interface | `llm/interface.py` | Pending |
| LLM prompt builder | `llm/prompts.py` | Pending |
| Query router | `llm/router.py` | Pending |
| API main | `api/main.py` | Pending |
| API endpoints | `api/endpoints/` | Pending |
| Weekly retraining | `scripts/retrain/weekly_retrain.py` | Pending |
| Model rollback | `scripts/models/rollback.py` | Pending |
| React UI | `ui/src/components/` | Pending |
