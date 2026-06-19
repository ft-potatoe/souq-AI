# QSE Market Copilot — Project Context

## What this project is
A Market Exchange Copilot for the Qatar Stock Exchange. Analysts ask natural
language questions about historical market activity. The system answers using
pre-computed analytics — never raw data directly, never LLM inference of numbers.

## Core architecture principle
Raw Data -> Features -> Analytics + ML Models -> Structured JSON -> LLM -> Answer
The LLM (qwen3:8b via Ollama at localhost:11434) receives only structured JSON.
It never performs calculations and never invents statistics.

## Tech stack
- Python 3.12, FastAPI, pandas 2.2, scikit-learn, xgboost, hmmlearn, scipy
- React 18 + Vite frontend (ui/)
- SQLite for feedback store (data/feedback/feedback.db)
- Parquet for all feature and raw data storage
- Ollama v0.30.6 running locally for LLM inference (qwen3:8b)
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
5. NEVER write to, insert into, or DELETE from the live data stores during
   verification/testing. This includes data/feedback/feedback.db, data/raw/*.parquet,
   data/features/features_master.parquet, and models/*. These are untracked by git
   (feedback.db is gitignored) and have NO backup — a stray INSERT/DELETE is
   unrecoverable. To exercise store.store(), feedback_counts(), ingestion, or feature
   code, point it at a throwaway temp DB / temp dir (the test suite does this via
   pytest tmp_path — e.g. monkeypatch feedback.store._DB_PATH). If you must touch the
   real DB read-only, use SELECT only. A bad cleanup query on feedback.db already wiped
   real feedback once — do not repeat it.

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
- Volatility Regime HMM: GaussianHMM(n_components=2) in analytics/volatility_regime.py
  Completely independent of trend HMM — direction-agnostic by design
  Features: volatility_20d, volatility_60d, volume_zscore, breadth_ratio
  State label assignment: sort by mean volatility_20d -> low_vol/high_vol
  Minimum 250 sessions before surfacing labels; flip-rate gate <= 10%
  Model artifact: models/vol_hmm/vol_hmm_v1.pkl, symlink vol_hmm_current
  Additional output: volatility_20d_percentile (rank vs full history)
- Clustering (day-type discovery): sklearn.cluster.HDBSCAN in ml/clustering.py
  Unsupervised market-state discovery + noise label (-1) as a second anomaly signal
  Features (9, NO forward returns): return_1d, return_5d, volatility_20d, volume_zscore,
  breadth_ratio, foreign_flow_zscore, domestic_flow_zscore, rsi_14, price_vs_sma20_pct
  Params: min_cluster_size=15, min_samples=5, store_centers="centroid"
  Validation gate: silhouette >= 0.20 AND noise_fraction <= 0.40 AND n_clusters >= 2
  Minimum 250 sessions before surfacing labels
  Model artifact: models/clustering/hdbscan_v1.pkl (dict {scaler, model, meta}), symlink hdbscan_current
  Today is assigned by nearest centroid in scaled space (HDBSCAN has no stable per-row predict)
  Descriptive only — clusters are historical groupings, never predictions

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
- MERGE-ON-WRITE (default): process_dataset(mode="merge") reads the existing
  data/raw/<dataset>.parquet, concats new rows, dedups by _dedup_key() keeping the LAST
  (newest wins) — so incremental/overlapping slices preserve history instead of truncating.
  _merge_with_existing() runs BEFORE _check_consecutive_missing so the gap check sees the
  full combined timeline. --replace flag (mode="replace") restores the old overwrite behaviour.
  _dedup_key(): ["date","market_name"] for gcc_daily, else ["date"]. tests/test_ingest.py.
- scripts/refresh.py: one-command chain (ingest -> build_features -> weekly_retrain), stops
  on failure; flags --src/--replace/--skip-ingest/--skip-retrain. Retrain "failure" is
  surfaced but non-fatal (core-model gate auto-rollback keeps prior model).

## Retraining
- Script: scripts/retrain/weekly_retrain.py
- Schedule: every Sunday at 02:00
- Models only deploy if they pass validation gates
- Rollback: python scripts/models/rollback.py --model [name] --version v1
  (all three core models currently write only one versioned artifact: *_v1.pkl)
- Vol HMM standalone: python scripts/retrain/train_vol_hmm.py
  (no --force flag; no minimum-feedback threshold; always retrains like trend HMM)

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
- The same atomic flip-rate pattern applies to the vol HMM in _step_vol_hmm().
  vol HMM failure is NON-BLOCKING — it does not set overall_status="failed".
  Only the three core models (anomaly_scorer, similarity_ranker, regime_hmm) drive
  overall_status. The `any_failed` check explicitly filters to `core_models`.
- _step_clustering() (step 6c) follows the same NON-BLOCKING pattern as vol HMM, but its gate
  is INTERNAL to ml.clustering.train_and_save() (silhouette/noise/n_clusters), which raises
  ValueError — there is no flip-rate concept (clustering is not a sequence labeller). On gate
  failure the prior artifact (scaler, model, meta) is restored via _save_artifact(). clustering
  is NOT in core_models, so its failure never flips overall_status.
- weekly_retrain.py writes status="success" only when every non-skipped CORE model deployed.
  Any core failure writes status="failed", so feedback_counts() re-counts all items since the
  last fully-successful run next week (intentionally conservative).
- API reload (step 7): set UVICORN_PID env var to the uvicorn process PID. Sends SIGHUP on
  Unix, CTRL_BREAK_EVENT on Windows. Skipped (not a failure) if UVICORN_PID is unset.
- Standalone scripts (train_anomaly.py, train_ranker.py) accept --force to bypass the
  minimum-feedback threshold check. train_hmm.py and train_vol_hmm.py have no threshold
  and no --force flag.
- All _load_artifact / _load_model imports in weekly_retrain.py are at module top-level
  (not deferred inside functions). This includes _load_ranker from ml.similarity_ranker
  and _load_vol_hmm from analytics.volatility_regime.
- rollback.py resolves the artifact as <stem>_<version>.pkl. Currently only v1 exists for
  all three core models. Vol HMM is NOT covered by rollback.py (additive model).

## API endpoints (FastAPI, api/main.py)
POST /query, GET /regime/current, GET /regime/history,
GET /features/today, GET /features/{date}, GET /similarity/{date},
GET /anomaly/{date}, POST /feedback, GET /models/status, GET /health

## QSE calendar
- Trading week is Sunday-Thursday (not Monday-Friday)
- day_of_week=0 means Sunday in this codebase (Thu=4)
- Ramadan flag: hard-coded ranges in RAMADAN_RANGES (2018-2026) in build_features.py

## File naming conventions
- Analytics modules: analytics/{distribution,trend,correlation,seasonality,flows,gcc,regime,volatility_regime}.py
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
- [DONE] analytics/volatility_regime.py     — 2-state vol HMM; low_vol/high_vol; vol percentile output
- [DONE] analytics/relationships.py         — deterministic relationship discovery; Spearman scan over all
                                               numeric feature pairs + conditional-decile analysis; descriptive
                                               co-movement only (associations, not causation); tests/test_relationships.py
- [DONE] ml/clustering.py                   — HDBSCAN market-state discovery (sklearn.cluster.HDBSCAN, no new dep);
                                               full lifecycle (versioned artifact, gate, auto-train); descriptive
                                               day-types, never predictive; tests/test_clustering.py
- [DONE] ml/anomaly_scorer.py               — RF anomaly scorer; tests/test_anomaly_scorer.py (44 tests)
- [DONE] ml/similarity_ranker.py            — k-NN(40) + XGBRanker re-score to top-10; safe_forward_returns leakage guard; 80/20 holdout NDCG@10 validation; tests/test_similarity_ranker.py (54 tests)
- [DONE] tests/test_features.py             — 66 tests (added TestForwardReturns, 7 tests)
- [DONE] feedback/store.py                  — SQLite store at data/feedback/feedback.db; 34 tests in tests/test_feedback_store.py
- [DONE] Full test suite: 379 tests, all passing (incl. test_relationships.py, test_clustering.py, test_ingest.py)
- [DONE] llm/interface.py  — query_llm(prompt, system) -> str; httpx POST to Ollama localhost:11434; qwen3:8b; temp=0.1, top_p=0.9, num_predict=700, num_ctx=8192, timeout=600s, stream=False; _strip_thinking() removes <think> blocks
- [DONE] llm/prompts.py    — SYSTEM_PROMPT (16 rules: +14 clustering, +15 associations-not-causation, +16 threshold_count/date_range); build_prompt(question, payload, history=None) -> str; multi-turn history injection
- [DONE] llm/router.py     — BUCKET_KEYWORDS (12 buckets: +clustering, +relationships), BUCKET_PRIORITY, PAYLOAD_TOKEN_BUDGET=3500; match_buckets(); build_llm_payload(); compress_bucket(); estimate_tokens()
- [DONE] api/ (main.py, endpoints/)               — all 10 spec endpoints; Pydantic models; CORS for localhost:5173
                                                   Start: uvicorn api.main:app --reload --port 8000
- [DONE] scripts/retrain/weekly_retrain.py  — 8-step pipeline + step 6b (vol HMM) + step 6c (clustering); feedback thresholds (anomaly>=10, ranker>=20); HMM + clustering always retrain; logs/retrain_log.jsonl; UVICORN_PID reload
- [DONE] scripts/retrain/train_anomaly.py  — standalone anomaly_scorer retrainer; --force flag
- [DONE] scripts/retrain/train_ranker.py   — standalone similarity_ranker retrainer; --force flag
- [DONE] scripts/retrain/train_hmm.py      — standalone trend HMM refitter; semantic flip-rate gate; atomic rollback on failure
- [DONE] scripts/retrain/train_vol_hmm.py  — standalone vol HMM refitter; semantic flip-rate gate; atomic rollback on failure
- [DONE] scripts/retrain/train_clustering.py — standalone HDBSCAN refitter; internal gate (silhouette>=0.20, noise<=0.40, n_clusters>=2); atomic restore on failure; no --force/threshold (always retrains)
- [DONE] scripts/models/rollback.py        — updates _current symlink/ptr; logs rollback entry to retrain_log.jsonl
- [DONE] ui/ React components              — ChatWindow (date-aware queries, date chip, detected-date sync,
                                             markdown table rendering with grid lines),
                                             RegimeBadge (trend + vol pills), RegimeHistory (dual timelines),
                                             RegimeInline (vol pill in chat), AnomalyIndicator, SimilarityCard,
                                             SimilarityChart, AnalyticsPanel, ClusteringCard, RelationshipsCard
                                             ModelStatus date display uses en-GB locale (DD/MM/YYYY)

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
  undercount and the budget guarantee would be violated with 10 full-size buckets.
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
- compress_bucket() for "similarity" trims key "top_matches" (not "matches") — that is the
  key returned by similarity_ranker.rank(). Never change this key without updating both places.
- "trend" keyword removed from trend bucket — it fired on "trend of foreign flows/GCC/volatility"
  which are flows/gcc/vol questions. Trend bucket now requires specific technical terms:
  "price trend", "market trend", "index trend", "sma", "rsi", "macd", "bollinger", "atr",
  "moving average", "momentum", "overbought", "oversold", "above sma", "below sma", "technical".
- api/_daterange.py supports additional patterns: "from YYYY to Month YYYY", "from YYYY to YYYY"
- distribution bucket keywords include count/threshold phrases: "how many days", "how many sessions",
  "how many times", "fell more than", "dropped more than", "rose more than", "gained more than",
  "exceeded", "surpassed", "threshold", "fell over", "gained over", "days that", "sessions that".
  POST /query auto-infers threshold from question regex and date range from extract_date_range().
- volatility_regime bucket keywords: "volatil", "vol regime", "low vol", "high vol",
  "options", "risk environment", "vol percentile", "vol spike", "vol compressed", etc.
  Bucket sits between regime and summary in BUCKET_PRIORITY.
  _DEFAULT_BUCKETS now includes "volatility_regime" so every unmatched question gets vol context.

## llm/prompts.py — implementation notes
- build_prompt(question, payload, history=None) assembles the user-turn prompt in three parts:
  analytics JSON, optional history block, question. Parts are joined with double newlines.
- History is injected as plain text ("Conversation so far:\nUser: ...\nAssistant: ...") inside
  the single /api/generate prompt string. This is intentional — the system uses /api/generate,
  not /api/chat. The model reads prior turns as labeled text, not as structured chat messages.
- _HISTORY_TURN_LIMIT = 3: only the last 3 turns of the passed history are used. The UI sends
  at most 3 turns (slice(-3)) to match — never send more or the extras are silently discarded.
- Per-turn token budget: _HISTORY_TOKEN_LIMIT_PER_TURN = 100 tokens (not chars). Enforced via
  _truncate_to_tokens() using estimate_tokens() from llm/router.py. This correctly handles
  Arabic and other non-ASCII scripts where char-count underestimates token count. 3 turns at
  100 tokens each = ~300 tokens max history overhead on top of the 3500-token analytics budget.
- num_ctx=8192 is set in llm/interface.py to guarantee the full combined prompt
  (3500 analytics + 300 history + 200 system + question) fits within the model's context window.
  Do not lower num_ctx below 4500 or full-budget payloads will be silently truncated by Ollama.
- ConversationTurn.role is Literal["user", "assistant"] — Pydantic rejects any other value at
  the API boundary. build_prompt skips unrecognized roles defensively, but they should never
  arrive after this validation.
- Cross-date follow-ups: the analytics payload always reflects the CURRENT request's date.
  History text may reference prior dates. The LLM can compare text descriptions from history
  but cannot access structured numbers from prior dates' analytics. Rule 1 (answer only from
  JSON) still applies to the current payload; history is continuity context, not a data source.
- Rule 16: when payload contains "threshold_count", report it as an exact integer (never a
  fraction/%). When "date_range" block is present, confine the answer to that period and state
  it explicitly. Never extend to the full historical record if a date range was requested.

## api/ — implementation notes
- Entry point: api/main.py; run with uvicorn api.main:app --reload --port 8000
- Shared helpers in api/_dates.py: resolve_date() (defaults to latest features_master row),
  symlink_target(), model_versions_snapshot(), MODEL_DIRS (all public)
  MODEL_DIRS now includes "vol_hmm" -> models/vol_hmm/; SYMLINK_NAMES["vol_hmm"] = "vol_hmm_current"
  MODEL_DIRS also includes "clustering" -> models/clustering/; SYMLINK_NAMES["clustering"] = "hdbscan_current"
  model_versions_snapshot() iterates MODEL_DIRS, so clustering auto-appears in GET /models/status
- api/_daterange.py: extract_date_range(question, data_date) -> (date_from, date_to) | (None, None)
  Deterministic NL date range parser. Handles: ISO ranges, ordinal ranges (1st of June to 4th of June 2026),
  Month name ranges (June 1st to June 4th 2026), between months, single month, quarters (Q1/Q2/first quarter),
  YTD/this year/for YYYY, last N days/weeks/months, since Month YYYY, since Day Month YYYY.
  Always call this — never ask the LLM to parse dates.
- POST /query pipeline: match_buckets -> analytics dispatch -> build_llm_payload ->
  build_prompt -> query_llm (run in thread executor — it is sync/blocking) -> QueryResponse
- POST /query auto-runs volatility_regime.run() whenever "regime" bucket is matched and
  "volatility_regime" was not already in the matched buckets. Both results are merged into
  RegimeContext. Never skip the auto-run — vol regime is always useful with trend regime.
- POST /query infers distribution metric from question keywords when not supplied in params:
  return/skew/gain/loss -> return_1d; volume -> volume; volatil -> volatility_20d; foreign/inflow/outflow -> foreign_net
- POST /query infers flows date_from via extract_date_range() when "flows" bucket matched and
  no explicit params supplied. date_from is passed to flows.run() to populate range_aggregates.
- query_llm (llm/interface.py) uses httpx.Client (sync, 120 s). All callers in async handlers
  MUST use asyncio.to_thread(query_llm, prompt, system) — never call directly.
  Similarly regime.run() and _build_regime_history() are offloaded with asyncio.to_thread().
  Do NOT use get_event_loop().run_in_executor() or functools.partial — asyncio.to_thread is
  the correct idiom for Python 3.9+ and avoids the deprecated get_event_loop() path.
- GET /regime/current merges vol regime via _merge_vol() helper; vol failure is non-fatal
  (sets vol_regime=None, logs warning).
- GET /regime/history decodes BOTH HMMs in single passes (never per-row predict calls).
  Trend and vol sequences are aligned by date into per-row {regime, vol_regime} dicts.
  Vol decode failure is non-fatal — rows simply have vol_regime=None.
- RegimeContext Pydantic model now carries 8 additional optional vol fields:
  vol_regime, vol_regime_probability, vol_regime_sessions, vol_regime_start_date,
  prior_vol_regime, volatility_20d_current, volatility_20d_percentile, volatility_60d_current.
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

## analytics/volatility_regime.py — implementation notes
- Completely independent of analytics/regime.py — different model path, different features,
  different label set. Never share model artifacts between the two HMMs.
- _assign_state_labels(model) takes only the model (no feature_names arg) — the vol index
  is always index 0 of _FEATURES by definition.
- run() returns vol_regime=None with a note when < 250 clean sessions are available.
  Callers (API, UI) must handle None gracefully — the model is additive, not required.
- volatility_20d_percentile: fraction of all history rows where volatility_20d < current.
  A value of 85 means vol is higher than 85% of all historical sessions.
- Model artifact: models/vol_hmm/vol_hmm_v1.pkl  (dict with keys "scaler", "model")
  Symlink: models/vol_hmm/vol_hmm_current  (or .ptr file on Windows without dev mode)

## ui/ — implementation notes
- ChatWindow date-aware queries: parseDateFromQuestion() extracts ISO dates (YYYY-MM-DD)
  and relative terms (yesterday, last Tuesday, last week, last month) from question text.
  Detected date shown as a yellow chip above the input; ✕ dismisses it without using it.
  When send() fires, effectiveDate is used in the API call and onDateChange syncs the header
  date picker. The detected date also appears as "data: YYYY-MM-DD" in the response meta.
- ChatWindow passes onDateChange={setDate} from App.jsx — always wire this prop.
- RegimeBadge renders two pills: trend (green/red/amber) + vol (teal=low_vol, purple=high_vol).
  Vol pill shows the volatility_20d percentile rank.
- RegimeInline (inside ChatWindow message bubble) shows both regimes separated by a | divider.
- RegimeHistory shows two stacked timeline bars (Trend + Vol) labelled on the left.
  Vol timeline only renders when vol_regime data is present in the history rows.
  Segment list shows dominant vol regime per trend span (majority vote over span rows).
- Teal/purple CSS vars for vol: use var(--teal,#1abc9c) and var(--purple,#9b59b6) with
  inline fallbacks since these vars may not be defined in the global CSS.
- Multi-turn history: send() builds conversationHistory by filtering settled (non-loading)
  messages and slicing the last 3 with .slice(-3). The slice(-3) matches _HISTORY_TURN_LIMIT=3
  in llm/prompts.py exactly — keep these in sync. The role filter is not needed (all messages
  are 'user' or 'assistant' by construction). messages is the pre-send render closure, which
  correctly excludes the in-flight turn and reflects all prior completed exchanges.

## analytics/distribution.py — implementation notes
- run() returns skewness, kurtosis, and percentiles (p25/p50/p75) in addition to
  percentile_rank, historical_frequency, rolling_stats, last_comparable_date.
- Default metric is "volume"; POST /query infers metric from question keywords automatically.
- Optional date_from / date_to params (ISO strings): history is sliced to the range before ALL
  downstream computations (extremes, percentiles, rolling stats, skewness). Adds a "date_range"
  block {date_from, date_to, sessions_in_range} to the return dict when supplied.
  Follows the same pattern as analytics/flows.py range_aggregates.
- Optional threshold (float) + threshold_direction ("above"|"below", default "below") params:
  _count_threshold() counts sessions where metric >= threshold (above) or <= threshold (below).
  Adds "threshold", "threshold_direction", "threshold_count" (int) to the return dict.
  Example: threshold=-2.0, threshold_direction="below" answers "how many days fell >2%?".
- Both features compose: date_from + threshold in one call answers "how many days in 2026 fell >2%?".
- POST /query auto-infers date range (via extract_date_range) and threshold (via regex on question)
  for the distribution bucket — callers never need to set these manually for NL queries.

## analytics/relationships.py — implementation notes
- Deterministic (never trains). The machine-driven counterpart to correlation.py: correlation.py
  compares two USER-NAMED features; relationships.py SCANS all numeric pairs and finds the strong ones.
- scan_relationships(): Spearman (rank corr) over all candidate pairs; keeps |rho| >= min_strength
  (0.4) with >= min_overlap (60) paired non-NaN observations; ranked by |rho| desc.
- _candidate_columns() drops forward returns (leakage), date, calendar/id cols, and raw OHLC/SMA.
  _is_trivial_pair() additionally drops definitional duplicates (a column vs its own z-score /
  window variant, e.g. volume vs volume_zscore, volatility_20d vs volatility_60d) so the ranking
  surfaces genuine, non-obvious relationships rather than identities.
- conditional_decile(): rank-based bottom decile of `given` (nsmallest, NOT a <= value threshold,
  which would pull in tied blocks); reports how often `observed` is above its all-history median
  vs the 50% baseline. Returns None if `observed` is essentially constant or < min_overlap rows.
- run() defaults the conditional's given/observed to the strongest pair when not supplied in params.
- Output keys: primary_relationships (top 5), strongest_relationships (top 15), conditional, note.
  ALWAYS carries note="Associations only -- co-movement, not causation." Router bucket "relationships".

## ml/clustering.py — implementation notes
- Uses sklearn.cluster.HDBSCAN (sklearn >= 1.3) — NO hdbscan package / compiler dependency.
- cluster(date, params) auto-trains on first call (RuntimeError if auto-train fails), mirroring
  anomaly_scorer.score(). train_and_save() raises ValueError on the internal gate.
- _label_for_profile() builds deterministic human-readable labels from each cluster's mean feature
  z-scores (e.g. high volatility_20d + negative return_1d -> "High-vol selloff"). Stable across
  retrains because it depends only on the profile, not on HDBSCAN's arbitrary integer state IDs.
- meta.cluster_profiles precomputes per-cluster size/label/means + an internal _centroid_z used for
  nearest-centroid assignment of the queried row (stripped from the public all_clusters output).
- Today may be assigned cluster_id = -1 / is_outlier=True (far from every centroid) — surfaced as
  "Atypical / outlier day", a second unsupervised anomaly signal independent of anomaly_scorer.
- Router bucket "clustering"; keywords kept specific to avoid collision with "regime" and generic
  "pattern". Compress handler drops member_dates_sample and trims to the 3 largest clusters.

## analytics/flows.py — implementation notes
- run() accepts optional date_from param. When provided, computes range_aggregates block:
  total_foreign_buy, total_foreign_sell, total_foreign_net, total_domestic_buy,
  total_domestic_sell, total_domestic_net, trading_sessions, date_from, date_to.
- Without date_from, returns standard rolling cumulative_foreign_net/cumulative_domestic_net
  windows (5d/10d/20d) only.

## analytics/gcc.py — implementation notes
- All return and spread values in the payload are in percent (%) — keys suffixed _pct.
  Never multiply by 100 again. The "units" field in the payload makes this explicit.
- peer_returns_pct: dict of {market_display_name: return_pct} for today.
- qse_rank_among_all_markets_including_qse: rank where 1=best; total_markets_including_qse
  is the denominator. Key names are unambiguous to prevent LLM misreading.
- rolling_outperformance_interpretation_{N}d: pre-computed "outperforming"/"underperforming"
  label alongside the rate — LLM must use this label, never infer from the number alone.

## ml/anomaly_scorer.py — implementation notes
- _cross_val_metrics(X, y) takes no model arg — always uses _RF_PARAMS internally
- confidence = abs(2 * p_anomaly - 1.0)  — distance from 0.5 boundary, not index arithmetic
- score() auto-trains on first call when no artifact exists; training ValueError is
  re-raised as RuntimeError("No model available and auto-training failed: ...")
- NaN z-score columns count as non-exceeding in bootstrap; early-history rows
  (before 60-session rolling window fills) are labelled 0 (normal) by default
- Feedback overrides apply even to NaN-z-score rows if provided
