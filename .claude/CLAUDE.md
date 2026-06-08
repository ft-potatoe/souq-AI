# QSE Market Copilot — Project Context

## What this project is
A Market Exchange Copilot for the Qatar Stock Exchange. Analysts ask natural
language questions about historical market activity. The system answers using
pre-computed analytics — never raw data directly, never LLM inference of numbers.

## Core architecture principle
Raw Data -> Features -> Analytics + ML Models -> Structured JSON -> LLM -> Answer
The LLM (qwen2.5:1.5b via Ollama at localhost:11434) receives only structured JSON.
It never performs calculations and never invents statistics.

## Tech stack
- Python 3.12, FastAPI, pandas 2.2, scikit-learn, xgboost, hmmlearn, scipy
- React 18 + Vite frontend (ui/)
- SQLite for feedback store (data/feedback/feedback.db)
- Parquet for all feature and raw data storage
- Ollama v0.30.6 running locally for LLM inference (qwen2.5:1.5b)
- ta library for technical indicators (replaces pandas-ta due to numpy<2 constraint)

## Critical constraints — never violate these
1. The similarity ranker's forward_return_5d_candidate and
   forward_return_10d_candidate must be set to None for any candidate session
   within 10 trading days of today. Use safe_forward_returns() in
   ml/similarity_ranker.py — always call it when building pairwise features.
2. The LLM payload must not exceed 3500 tokens. The query router in
   llm/router.py enforces BUCKET_PRIORITY ordering and compress_bucket()
   compression. Never bypass this.
3. This is NOT a forecasting system. No future price predictions anywhere.
4. No Unicode arrows (→, x) in log messages — Windows cp1252 console will error.
   Use ASCII -> instead.

## Data schemas (spec §16)
- market_daily:  date, open, high, low, close, volume, value_traded, total_trades
- flows_daily:   date, foreign_buy, foreign_sell, foreign_net, domestic_buy, domestic_sell, domestic_net
- gcc_daily:     date, market_name, daily_change_pct  (+ optional pe_ratio, dividend_yield)
- breadth_daily: date, gainers, losers, unchanged, total_listed, total_traded

## Layer types
- Feature engineering (scripts/features/): deterministic, never changes, never trains
- Deterministic analytics (analytics/): deterministic, never trains
- ML models (ml/): trained weekly, versioned with symlinks in models/
- LLM (llm/): explanation only, temperature 0.1

## ML models
- Anomaly scorer: RandomForestClassifier in ml/anomaly_scorer.py
  Bootstrap labels: >=2 z-scores above |2.0| -> anomalous=1
  Validation gate: precision >= 0.65, recall >= 0.60, AUC-ROC >= 0.72
- Similarity ranker: XGBRanker in ml/similarity_ranker.py
  Pipeline: k-NN top-40 candidates -> ranker re-scores -> top-10 returned
  Validation gate: NDCG@10 >= 0.70
- Regime HMM: GaussianHMM(n_components=3) in analytics/regime.py
  Minimum 250 sessions before surfacing labels
  State label assignment: sort by mean return_1d -> bear/sideways/bull

## Feature engineering notes (scripts/features/build_features.py)
- Price column is `close` (not index_value)
- RSI-14 uses Wilder EMA — built in-house, no pandas-ta needed
- breadth_zscore is computed over breadth_ratio (0-1), not breadth_net
- GCC returns come from daily_change_pct / 100 (already a pct, divide to decimal)
- GCC market column is market_name (not market)
- above_sma_20 AND above_sma_200 are both computed (binary flags)
- Zero std in rolling z-score returns NaN, never divides by zero
- forward_return_5d/10d/20d: close.pct_change(N).shift(-N) — trailing N rows are NaN by design;
  safe_forward_returns() in ml/similarity_ranker.py masks them again at inference time
- Output: 58 columns, one row per QSE trading day

## Data files
- data/raw/<dataset>.parquet          — clean ingested data
- data/raw/<dataset>_quarantine.parquet — rows that failed validation
- data/features/features_master.parquet — all 58 computed features, rebuilt nightly
- data/feedback/feedback.db           — SQLite feedback store
- models/*/[model]_current            — symlink to latest validated model artifact
- logs/ingestion_errors.log           — ingestion warnings/errors
- logs/feature_build.log              — feature build warnings/errors
- logs/retrain_log.jsonl              — weekly retraining results

## Ingestion notes (scripts/ingest/load_raw.py)
- File discovery: matches filename stem to dataset name (e.g. market_daily_2024.csv -> market_daily)
- Quarantine pattern: failing rows written to _quarantine.parquet, never silently dropped
- Gap check: halts pipeline (exit code 2) if >3 consecutive QSE trading days missing
- QSE trading week: Sun-Thu; Fri/Sat excluded from gap and validation checks

## Retraining
- Script: scripts/retrain/weekly_retrain.py
- Schedule: every Sunday at 02:00
- Models only deploy if they pass validation gates
- Rollback: python scripts/models/rollback.py --model [name] --version v1
  (all three models currently write only one versioned artifact: *_v1.pkl)

## scripts/retrain/ — implementation notes
- Feedback thresholds checked via feedback_counts(); pop "since" key before reading counts.
- Anomaly scorer feedback prep: rename target_date->date AND feedback_type->label_type
  independently (separate if-guards) before passing to build_anomaly_labels(). Both renames
  are required; guarding them together silently drops label overrides if one column is absent.
- HMM flip-rate validation uses semantic labels (bear/sideways/bull), not raw integer state IDs.
  Raw IDs are not comparable across independently-fitted models — state 0 in the old model is
  not the same state as state 0 in the new model. Decode each sequence with its own model,
  then map both through _assign_state_labels() before comparing.
- HMM deployment is atomic: load the prior (scaler, model) into memory before calling
  fit_and_save(), then call _save_model(old_scaler, old_model) to restore if the flip-rate
  gate fails. The pipeline never leaves a regime model live that failed its own gate.
- weekly_retrain.py writes status="success" only when every non-skipped model deployed.
  Any failure writes status="failed", so feedback_counts() re-counts all items since the
  last fully-successful run next week (intentionally conservative).
- API reload (step 7): set UVICORN_PID env var to the uvicorn process PID. Sends SIGHUP on
  Unix, CTRL_BREAK_EVENT on Windows. Skipped (not a failure) if UVICORN_PID is unset.
- Standalone scripts (train_anomaly.py, train_ranker.py) accept --force to bypass the
  minimum-feedback threshold check. train_hmm.py has no threshold and no --force flag.
- All _load_artifact / _load_model imports in weekly_retrain.py are at module top-level
  (not deferred inside functions). This includes _load_ranker from ml.similarity_ranker.
- rollback.py resolves the artifact as <stem>_<version>.pkl. Currently only v1 exists for
  all three models. If the file is absent the script lists what is available and exits 1.

## API endpoints (FastAPI, api/main.py)
POST /query, GET /regime/current, GET /regime/history,
GET /features/today, GET /features/{date}, GET /similarity/{date},
GET /anomaly/{date}, POST /feedback, GET /models/status, GET /health

## QSE calendar
- Trading week is Sunday-Thursday (not Monday-Friday)
- day_of_week=0 means Sunday in this codebase (Thu=4)
- Ramadan flag: hard-coded ranges in RAMADAN_RANGES (2018-2026) in build_features.py

## File naming conventions
- Analytics modules: analytics/{distribution,trend,correlation,seasonality,flows,gcc,regime}.py
- Each exposes: run(date, params) -> dict
- ML modules: ml/{anomaly_scorer,similarity_ranker}.py
- Feedback module: feedback/store.py
- LLM modules: llm/{interface,prompts,router}.py

## Build status
- [DONE] scripts/ingest/load_raw.py         — ingestion + validation, all 4 datasets
- [DONE] scripts/features/build_features.py — 58-feature master parquet (300 rows x 58 cols);
                                               includes forward_return_5d/10d/20d for similarity ranker
- [DONE] End-to-end test: 300 trading days, all sanity checks pass
- [DONE] analytics/ modules (distribution, trend, correlation, seasonality, flows, gcc, regime)
- [DONE] ml/anomaly_scorer.py               — RF anomaly scorer; tests/test_anomaly_scorer.py (44 tests)
- [DONE] ml/similarity_ranker.py            — k-NN(40) + XGBRanker re-score to top-10; safe_forward_returns leakage guard; 80/20 holdout NDCG@10 validation; tests/test_similarity_ranker.py (54 tests)
- [DONE] tests/test_features.py             — 66 tests (added TestForwardReturns, 7 tests)
- [DONE] feedback/store.py                  — SQLite store at data/feedback/feedback.db; 34 tests in tests/test_feedback_store.py
- [DONE] Full test suite: 350 tests, all passing
- [DONE] llm/interface.py  — query_llm(prompt, system) -> str; httpx POST to Ollama localhost:11434; qwen3:14b; temp=0.1, top_p=0.9, num_predict=1024, timeout=120s, stream=False
- [DONE] llm/prompts.py    — SYSTEM_PROMPT (11 rules, spec §11.2); build_prompt(question, payload) -> str (spec §11.3)
- [DONE] llm/router.py     — BUCKET_KEYWORDS (9 buckets), BUCKET_PRIORITY, PAYLOAD_TOKEN_BUDGET=3500; match_buckets(); build_llm_payload(); compress_bucket(); estimate_tokens()
- [DONE] api/ (main.py, endpoints/)               — all 10 spec endpoints; Pydantic models; CORS for localhost:5173
                                                   Start: uvicorn api.main:app --reload --port 8000
- [DONE] scripts/retrain/weekly_retrain.py  — 8-step pipeline; feedback thresholds (anomaly>=10, ranker>=20); HMM always retrains; logs/retrain_log.jsonl; UVICORN_PID reload
- [DONE] scripts/retrain/train_anomaly.py  — standalone anomaly_scorer retrainer; --force flag
- [DONE] scripts/retrain/train_ranker.py   — standalone similarity_ranker retrainer; --force flag
- [DONE] scripts/retrain/train_hmm.py      — standalone HMM refitter; semantic flip-rate gate; atomic rollback on failure
- [DONE] scripts/models/rollback.py        — updates _current symlink/ptr; logs rollback entry to retrain_log.jsonl
- [TODO] ui/ React components

## feedback/store.py — implementation notes
- feedback_counts() returns a dict with one key per feedback_type (count) PLUS a
  "since" key (ISO timestamp string of last successful retrain, or None for all-time).
  Callers must not treat "since" as a count — check/pop it before iterating type keys.
- Timestamp stored as naive UTC ISO string (no +00:00 suffix); get_since() cutoff
  comparison is string-lexicographic — safe only because all timestamps use the same
  format. Never store timezone-aware strings in the DB.
- sqlite3.Row objects are materialised to plain dicts inside the with-block in
  feedback_counts(); do not move the dict comprehension outside the connection context.
- _connect() creates data/feedback/ if it does not exist (parents=True). Safe to call
  repeatedly — uses CREATE TABLE IF NOT EXISTS.

## llm/router.py — implementation notes
- build_llm_payload() re-estimates estimate_tokens(payload) after each bucket is tentatively
  added — do NOT accumulate standalone bucket sizes; the JSON envelope overhead makes the sum
  undercount and the budget guarantee would be violated with 9 full-size buckets.
- compress_bucket() uses copy.deepcopy() for similarity and gcc — never dict() shallow copy,
  which shares nested object references and would mutate the caller's result dict.
- "sell" is NOT a flows keyword — it triggers advisory false positives ("Should I sell?").
  Use "selling", "sold", "sell volume", "sell pressure" instead.
- "buy" is NOT a flows keyword — it triggers advisory false positives ("Is now a good time to buy?").
  Use "buying", "buy volume", "buy pressure", "foreign buy", "domestic buy" instead.
- "peer" is NOT a gcc keyword — it triggers false positives ("peer review", "peer group").
  Use "peer market", "peer comparison", "gcc peer", "peer performance" instead.
- "net" is NOT a flows keyword — it hits "net asset value", "internet". Other flows keywords
  (flow, foreign, domestic, inflow, outflow) cover every genuine net-flow question.
- "range" and "spread" are NOT distribution keywords — they hit "normal range", "bid-ask spread".
  Use "return range", "historical range", "return spread" instead.
- "phase" is NOT a regime keyword — it hits generic English ("market phases over time" is fine
  via "market phase"). Use "market phase" only.
- "match" is NOT a similarity keyword — it hits "QSE matched its average". Use
  "historical match" and "has this happened" instead.
- "together" is NOT a correlation keyword — it hits "foreign and domestic together" which is
  a flows question. Use "move together" instead (co-move already handles the generic case).
- "vs" (no trailing space) is the gcc keyword — "vs " missed "vs." and end-of-string.
- Compression is only attempted when a bucket would push the payload over budget; it is not
  applied unconditionally. Callers must not assume similarity is always trimmed to 3 matches.

## api/ — implementation notes
- Entry point: api/main.py; run with uvicorn api.main:app --reload --port 8000
- Shared helpers in api/_dates.py: resolve_date() (defaults to latest features_master row),
  symlink_target(), model_versions_snapshot(), MODEL_DIRS (all public)
- POST /query pipeline: match_buckets -> analytics dispatch -> build_llm_payload ->
  build_prompt -> query_llm (run in thread executor — it is sync/blocking) -> QueryResponse
- query_llm (llm/interface.py) uses httpx.Client (sync, 120 s). All callers in async handlers
  MUST use asyncio.to_thread(query_llm, prompt, system) — never call directly.
  Similarly regime.run() and _build_regime_history() are offloaded with asyncio.to_thread().
  Do NOT use get_event_loop().run_in_executor() or functools.partial — asyncio.to_thread is
  the correct idiom for Python 3.9+ and avoids the deprecated get_event_loop() path.
- GET /regime/history uses a single HMM decode pass via regime._decode() + _assign_state_labels()
  over the full features_master. Never call regime.run() per row — O(N) model.predict() calls.
- Analytics run() and ML score/rank calls in POST /query are synchronous CPU work (pandas/
  sklearn/numpy); parquet reads are lru_cache-cached after the first request. They are called
  directly (not via to_thread) — adding thread overhead for pure CPU work on a single-user
  local tool has no benefit and introduces GIL contention.
- FeedbackRequest.rating is int | None (ge=1, le=5) to match the INTEGER column in feedback.db.
  The store.store() signature also expects Optional[int] — do not change to float.
- GET /models/status reads retrain_log.jsonl filtering status=='success' only — mirrors
  feedback/store._last_retrain_ts(). Failed retrain entries are intentionally excluded.
- GET /health probes Ollama at /api/tags with a 3 s async timeout; reports "degraded" (not 500)
  when Ollama or features are unavailable.

## ml/anomaly_scorer.py — implementation notes
- _cross_val_metrics(X, y) takes no model arg — always uses _RF_PARAMS internally
- confidence = abs(2 * p_anomaly - 1.0)  — distance from 0.5 boundary, not index arithmetic
- score() auto-trains on first call when no artifact exists; training ValueError is
  re-raised as RuntimeError("No model available and auto-training failed: ...")
- NaN z-score columns count as non-exceeding in bootstrap; early-history rows
  (before 60-session rolling window fills) are labelled 0 (normal) by default
- Feedback overrides apply even to NaN-z-score rows if provided
