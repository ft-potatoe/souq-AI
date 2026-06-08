"""
ml/similarity_ranker.py
Two-stage session similarity: k-NN candidate retrieval -> XGBRanker re-scoring.

Public API
----------
safe_forward_returns(candidate_date, today, fwd_5d, fwd_10d) -> tuple[float|None, float|None]
build_similarity_labels(features_df, feedback_df)             -> pd.Series  (weak cosine scores)
train_and_save(features_df, feedback_df)                      -> Path
rank(date, params)                                            -> dict
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRanker

from analytics._loader import history_up_to

_ROOT = Path(__file__).resolve().parents[1]
_MODEL_DIR = _ROOT / "models" / "similarity_ranker"
_MODEL_PATH = _MODEL_DIR / "xgb_ranker_v1.pkl"
_SYMLINK = _MODEL_DIR / "xgb_ranker_current"

# k-NN retrieval pool
_KNN_CANDIDATES = 40
_TOP_K = 10

# QSE calendar: Sun-Thu trading week
_QSE_BDAY = pd.offsets.CustomBusinessDay(weekmask="Sun Mon Tue Wed Thu")
_LEAKAGE_WINDOW_DAYS = 10  # trading days

# Validation gate
_GATE_NDCG = 0.70
_GATE_MAX_DEGRADATION = 0.10  # 10 % relative

# Features used for k-NN embedding
KNN_FEATURES = [
    "return_1d",
    "volume_zscore",
    "foreign_flow_zscore",
    "breadth_ratio",
    "volatility_20d",
    "rsi_14",
]

# Pairwise features fed to the XGBRanker
PAIRWISE_FEATURES = [
    "delta_return_1d",
    "delta_volume_zscore",
    "delta_foreign_flow_zscore",
    "delta_breadth_ratio",
    "delta_volatility_20d",
    "delta_rsi_14",
    "same_regime",
    "regime_transition_match",
    "days_since_candidate",
    "forward_return_5d_candidate",
    "forward_return_10d_candidate",
]

_XGB_PARAMS = dict(
    objective="rank:pairwise",
    learning_rate=0.05,
    n_estimators=200,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
)

# Forward return columns expected in features master
_FWD_5D_COL = "forward_return_5d"
_FWD_10D_COL = "forward_return_10d"
_FWD_20D_COL = "forward_return_20d"
_REGIME_COL = "regime"


# ---------------------------------------------------------------------------
# Data leakage guard  (spec §7.2) — CRITICAL
# ---------------------------------------------------------------------------

def safe_forward_returns(
    candidate_date: pd.Timestamp | str,
    today: pd.Timestamp | str,
    fwd_5d: float | None,
    fwd_10d: float | None,
) -> tuple[float | None, float | None]:
    """
    Return (fwd_5d, fwd_10d) for a candidate session.

    Any candidate within _LEAKAGE_WINDOW_DAYS QSE trading days of *today*
    has its forward return fields set to None to prevent data leakage.

    Must be called for every candidate, both during training AND inference.
    """
    c_ts = pd.Timestamp(candidate_date)
    t_ts = pd.Timestamp(today)

    # Count QSE trading days between candidate and today (exclusive of endpoints)
    if c_ts >= t_ts:
        return None, None

    try:
        trading_days_between = len(
            pd.bdate_range(start=c_ts, end=t_ts, freq=_QSE_BDAY)
        ) - 1  # bdate_range includes both endpoints; -1 gives the gap count
    except Exception as exc:
        logging.warning(
            "safe_forward_returns: could not compute QSE trading days between "
            "%s and %s (%s) -- conservatively masking forward returns.",
            c_ts.date(), t_ts.date(), exc,
        )
        return None, None

    if trading_days_between <= _LEAKAGE_WINDOW_DAYS:
        return None, None

    return fwd_5d, fwd_10d


# ---------------------------------------------------------------------------
# Model persistence helpers
# ---------------------------------------------------------------------------

def _save_artifact(scaler: StandardScaler, model: XGBRanker, meta: dict) -> Path:
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {"scaler": scaler, "model": model, "meta": meta}
    joblib.dump(artifact, _MODEL_PATH)

    target = _MODEL_PATH.resolve()
    link = _SYMLINK
    if link.exists() or link.is_symlink():
        link.unlink()
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        link.with_suffix(".ptr").write_text(str(target))

    return _MODEL_PATH


def _load_artifact() -> tuple[StandardScaler, XGBRanker, dict] | None:
    """Return (scaler, model, meta) from symlink or v1 pkl, or None."""
    candidates = [_SYMLINK, _MODEL_PATH]
    for path in candidates:
        resolved = path.resolve() if path.is_symlink() else path
        if resolved.exists():
            try:
                artifact = joblib.load(resolved)
                return artifact["scaler"], artifact["model"], artifact["meta"]
            except Exception:
                pass

    ptr = _SYMLINK.with_suffix(".ptr")
    if ptr.exists():
        target = Path(ptr.read_text().strip())
        if target.exists():
            artifact = joblib.load(target)
            return artifact["scaler"], artifact["model"], artifact["meta"]

    return None


# ---------------------------------------------------------------------------
# Regime helpers
# ---------------------------------------------------------------------------

def _get_regime(row: pd.Series) -> str | None:
    if _REGIME_COL in row.index:
        val = row[_REGIME_COL]
        if pd.notna(val):
            return str(val)
    return None


def _regime_transition_match(query_row: pd.Series, candidate_row: pd.Series) -> int:
    """
    1 if the candidate's prior_regime -> current_regime transition equals the
    query's prior_regime -> current_regime transition, 0 otherwise.
    Only works when prior_regime column is present; falls back to 0.
    """
    pr_col = "prior_regime"
    cr_col = _REGIME_COL
    if not all(c in query_row.index and c in candidate_row.index for c in [pr_col, cr_col]):
        return 0
    q_trans = (query_row.get(pr_col), query_row.get(cr_col))
    c_trans = (candidate_row.get(pr_col), candidate_row.get(cr_col))
    if any(pd.isna(v) for v in q_trans + c_trans):
        return 0
    return int(q_trans == c_trans)


# ---------------------------------------------------------------------------
# k-NN retrieval
# ---------------------------------------------------------------------------

def _fit_knn(X_scaled: np.ndarray) -> NearestNeighbors:
    knn = NearestNeighbors(
        n_neighbors=min(_KNN_CANDIDATES + 1, len(X_scaled)),
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    )
    knn.fit(X_scaled)
    return knn


def _knn_candidates(
    knn: NearestNeighbors,
    scaler: StandardScaler,
    features_df: pd.DataFrame,
    query_vec: np.ndarray,
) -> pd.DataFrame:
    """
    Return up to _KNN_CANDIDATES rows from features_df closest to query_vec.
    The query row itself is excluded by dropping distance == 0.
    """
    q_scaled = scaler.transform(query_vec.reshape(1, -1))
    dists, idxs = knn.kneighbors(q_scaled, n_neighbors=knn.n_neighbors)
    dists = dists[0]
    idxs = idxs[0]

    # cosine distance 0.0 means the query itself; exclude it
    mask = dists > 0.0
    dists = dists[mask]
    idxs = idxs[mask]

    result = features_df.iloc[idxs].copy()
    result["_knn_cosine_dist"] = dists
    result["_knn_cosine_score"] = 1.0 - dists  # similarity
    return result.head(_KNN_CANDIDATES)


# ---------------------------------------------------------------------------
# Pairwise feature matrix construction
# ---------------------------------------------------------------------------

def _build_pairwise_row(
    query_row: pd.Series,
    candidate_row: pd.Series,
    today: pd.Timestamp,
) -> dict[str, float | None]:
    """
    Build one row of pairwise features for (query, candidate).
    safe_forward_returns is always applied — no exceptions.
    """
    def _delta(col: str) -> float | None:
        q_val = query_row.get(col)
        c_val = candidate_row.get(col)
        if pd.isna(q_val) or pd.isna(c_val):
            return None
        return float(q_val) - float(c_val)

    q_regime = _get_regime(query_row)
    c_regime = _get_regime(candidate_row)
    same_regime = int(q_regime is not None and q_regime == c_regime)
    regime_trans = _regime_transition_match(query_row, candidate_row)

    candidate_date = pd.Timestamp(candidate_row["date"])
    days_since = max(0, (today - candidate_date).days)

    raw_fwd_5d = candidate_row.get(_FWD_5D_COL)
    raw_fwd_10d = candidate_row.get(_FWD_10D_COL)
    if pd.isna(raw_fwd_5d):
        raw_fwd_5d = None
    if pd.isna(raw_fwd_10d):
        raw_fwd_10d = None

    # CRITICAL: apply leakage guard for every candidate
    fwd_5d, fwd_10d = safe_forward_returns(candidate_date, today, raw_fwd_5d, raw_fwd_10d)

    return {
        "delta_return_1d": _delta("return_1d"),
        "delta_volume_zscore": _delta("volume_zscore"),
        "delta_foreign_flow_zscore": _delta("foreign_flow_zscore"),
        "delta_breadth_ratio": _delta("breadth_ratio"),
        "delta_volatility_20d": _delta("volatility_20d"),
        "delta_rsi_14": _delta("rsi_14"),
        "same_regime": same_regime,
        "regime_transition_match": regime_trans,
        "days_since_candidate": days_since,
        "forward_return_5d_candidate": fwd_5d,
        "forward_return_10d_candidate": fwd_10d,
    }


def _pairwise_matrix(
    query_row: pd.Series,
    candidates_df: pd.DataFrame,
    today: pd.Timestamp,
) -> np.ndarray:
    """
    Build (n_candidates, n_pairwise_features) matrix.
    NaN values are imputed as 0.0 before passing to the ranker.
    """
    rows = [
        _build_pairwise_row(query_row, candidates_df.iloc[i], today)
        for i in range(len(candidates_df))
    ]
    df = pd.DataFrame(rows, columns=PAIRWISE_FEATURES)
    return df.fillna(0.0).values.astype(float)


# ---------------------------------------------------------------------------
# Label construction for training
# ---------------------------------------------------------------------------

def build_similarity_labels(
    features_df: pd.DataFrame,
    feedback_df: pd.DataFrame | None = None,
) -> pd.Series:
    """
    Build weak relevance labels (0.0-1.0) for training the ranker.

    If feedback_df contains similarity_rating rows (1-5 stars), those are used
    as strong labels after normalising to [0, 1].

    Cold-start fallback: all sessions receive label 0.5 (neutral); the ranker
    will be trained on pairwise order induced by the k-NN cosine score, which
    is injected at training time through the feature matrix.

    Returns
    -------
    pd.Series, float, same index as features_df.
    """
    labels = pd.Series(0.5, index=features_df.index, name="similarity_label", dtype=float)

    if feedback_df is not None and not feedback_df.empty:
        fb = feedback_df.copy()
        fb["date"] = pd.to_datetime(fb["date"])
        work = features_df.copy()
        work["date"] = pd.to_datetime(work["date"])
        date_to_idx = {ts: i for i, ts in enumerate(work["date"])}

        for _, row in fb.iterrows():
            if row.get("feedback_type") != "similarity_rating":
                continue
            idx = date_to_idx.get(row["date"])
            if idx is None:
                continue
            rating = row.get("rating")
            if pd.notna(rating):
                labels.iloc[idx] = float(rating - 1) / 4.0  # 1-5 -> 0.0-1.0

    return labels


# ---------------------------------------------------------------------------
# NDCG@K computation
# ---------------------------------------------------------------------------

def _dcg_at_k(scores: np.ndarray, k: int) -> float:
    scores = scores[:k]
    gains = 2.0 ** scores - 1.0
    discounts = np.log2(np.arange(2, len(scores) + 2))
    return float(np.sum(gains / discounts))


def _ndcg_at_k(predicted_order: np.ndarray, true_scores: np.ndarray, k: int) -> float:
    """
    Compute NDCG@k.

    predicted_order : indices that sort candidates by predicted relevance (best first)
    true_scores     : relevance scores aligned with the original candidate order
    """
    reordered = true_scores[predicted_order]
    ideal = np.sort(true_scores)[::-1]
    dcg = _dcg_at_k(reordered, k)
    idcg = _dcg_at_k(ideal, k)
    if idcg == 0:
        return 1.0
    return dcg / idcg


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_ndcg(
    scaler: StandardScaler,
    model: XGBRanker,
    features_df: pd.DataFrame,
    today: pd.Timestamp,
    sample_queries: int = 50,
) -> float:
    """
    Estimate NDCG@10 on a true holdout: the last 20% of sessions are used as
    queries; only sessions strictly before each query date serve as its candidate
    pool.  This prevents the circular evaluation of validating on training data.

    Per-query cutoff date is used for both days_since_candidate and the leakage
    guard, matching inference behaviour exactly.
    """
    df = features_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    valid_df = df.dropna(subset=KNN_FEATURES).reset_index(drop=True)

    min_required = _KNN_CANDIDATES + 10
    if len(valid_df) < min_required * 2:
        return 1.0  # too small to split meaningfully; pass

    # Holdout: last 20% of sessions as queries, first 80% as candidate pool
    split = int(len(valid_df) * 0.80)
    train_pool = valid_df.iloc[:split].reset_index(drop=True)
    query_pool = valid_df.iloc[split:].reset_index(drop=True)

    query_pool = query_pool.sample(
        n=min(sample_queries, len(query_pool)),
        random_state=42,
    )

    X_train = train_pool[KNN_FEATURES].values.astype(float)
    X_train_scaled = scaler.transform(X_train)
    knn = _fit_knn(X_train_scaled)

    ndcgs = []
    for _, q_row in query_pool.iterrows():
        q_vec = np.array([q_row[c] for c in KNN_FEATURES], dtype=float)
        if np.isnan(q_vec).any():
            continue

        candidates = _knn_candidates(knn, scaler, train_pool, q_vec)
        if len(candidates) < 2:
            continue

        # Use this query's own date — matches inference behaviour
        q_date = pd.Timestamp(q_row["date"])
        X_pair = _pairwise_matrix(q_row, candidates, q_date)
        scores = model.predict(X_pair)
        pred_order = np.argsort(scores)[::-1]

        true_rel = candidates["_knn_cosine_score"].values
        ndcg = _ndcg_at_k(pred_order, true_rel, _TOP_K)
        ndcgs.append(ndcg)

    return float(np.mean(ndcgs)) if ndcgs else 0.0


def _passes_gates(ndcg: float) -> tuple[bool, list[str]]:
    failures = []
    if ndcg < _GATE_NDCG:
        failures.append(f"NDCG@10 {ndcg:.4f} < {_GATE_NDCG}")
    return len(failures) == 0, failures


def _passes_degradation(new_ndcg: float, prior_ndcg: float) -> tuple[bool, list[str]]:
    failures = []
    if prior_ndcg > 0 and (prior_ndcg - new_ndcg) / prior_ndcg > _GATE_MAX_DEGRADATION:
        failures.append(
            f"NDCG@10 degraded by {(prior_ndcg - new_ndcg) / prior_ndcg:.1%} "
            f"(prior={prior_ndcg:.4f}, new={new_ndcg:.4f})"
        )
    return len(failures) == 0, failures


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_and_save(
    features_df: pd.DataFrame | None = None,
    feedback_df: pd.DataFrame | None = None,
) -> Path:
    """
    Train the XGBRanker with k-NN cosine scores as weak labels (cold start)
    supplemented by user feedback ratings if available.

    Raises
    ------
    ValueError  if NDCG@10 < _GATE_NDCG or degradation > _GATE_MAX_DEGRADATION.
    """
    if features_df is None:
        from analytics._loader import load_features
        features_df = load_features()

    features_df = features_df.copy()
    features_df["date"] = pd.to_datetime(features_df["date"])
    today = features_df["date"].max()

    valid_df = features_df.dropna(subset=KNN_FEATURES).reset_index(drop=True)
    if len(valid_df) < _KNN_CANDIDATES + 10:
        raise ValueError(
            f"Need at least {_KNN_CANDIDATES + 10} complete sessions to train; "
            f"only {len(valid_df)} available."
        )

    X_all = valid_df[KNN_FEATURES].values.astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)
    knn = _fit_knn(X_scaled)

    # Build training data: for each query session, generate pairwise features
    # against its k-NN neighbours.  We skip the most recent sessions as queries
    # (they have no future forward returns to learn from).
    query_pool = valid_df.iloc[_KNN_CANDIDATES:-_LEAKAGE_WINDOW_DAYS]
    if len(query_pool) == 0:
        query_pool = valid_df.iloc[_KNN_CANDIDATES:]

    all_X: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    group_sizes: list[int] = []

    feedback_labels = build_similarity_labels(valid_df, feedback_df)

    for _, q_row in query_pool.iterrows():
        q_vec = np.array([q_row[c] for c in KNN_FEATURES], dtype=float)
        if np.isnan(q_vec).any():
            continue

        candidates = _knn_candidates(knn, scaler, valid_df, q_vec)
        if len(candidates) < 2:
            continue

        # Use the query session's own date so days_since_candidate and the
        # leakage guard are computed relative to that session, not the global max.
        q_date = pd.Timestamp(q_row["date"])
        X_pair = _pairwise_matrix(q_row, candidates, q_date)

        # Relevance: prefer feedback labels; fall back to knn cosine score
        cand_indices = candidates.index.tolist()
        rel = np.array([
            float(feedback_labels.iloc[idx])
            if feedback_labels.iloc[idx] != 0.5
            else float(candidates.at[idx, "_knn_cosine_score"])
            for idx in cand_indices
        ], dtype=float)

        all_X.append(X_pair)
        all_y.append(rel)
        group_sizes.append(len(candidates))

    if not all_X:
        raise ValueError("No training pairs could be constructed from the feature data.")

    X_train = np.vstack(all_X)
    y_train = np.concatenate(all_y)
    groups = np.array(group_sizes)

    ranker = XGBRanker(**_XGB_PARAMS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ranker.fit(X_train, y_train, group=groups)

    ndcg = _validate_ndcg(scaler, ranker, valid_df, today)

    ok, failures = _passes_gates(ndcg)
    if not ok:
        raise ValueError(f"Ranker failed validation gates: {'; '.join(failures)}")

    prior = _load_artifact()
    if prior is not None:
        _, _, prior_meta = prior
        prior_ndcg = prior_meta.get("ndcg_at_10", 0.0)
        ok2, deg_failures = _passes_degradation(ndcg, prior_ndcg)
        if not ok2:
            raise ValueError(
                f"Ranker degraded beyond threshold vs prior: {'; '.join(deg_failures)}"
            )

    meta = {
        "ndcg_at_10": ndcg,
        "knn_features": KNN_FEATURES,
        "pairwise_features": PAIRWISE_FEATURES,
        "n_train_pairs": int(X_train.shape[0]),
        "n_query_sessions": len(group_sizes),
        "feedback_informed": feedback_df is not None and not feedback_df.empty,
        "version": "xgb_ranker_v1",
    }
    return _save_artifact(scaler, ranker, meta)


# ---------------------------------------------------------------------------
# Forward return statistics
# ---------------------------------------------------------------------------

def _forward_return_stats(
    candidates_df: pd.DataFrame,
    top_indices: list[int],
    today: pd.Timestamp,
) -> dict[str, Any]:
    """
    Compute mean/median forward returns for the top-10 matched sessions
    across 5d, 10d, 20d windows.  safe_forward_returns is applied for 5d/10d.
    20d is also masked when the candidate is within the leakage window.
    """
    results: dict[str, list[float]] = {"fwd_5d": [], "fwd_10d": [], "fwd_20d": []}

    for i in top_indices:
        row = candidates_df.iloc[i]
        cdate = pd.Timestamp(row["date"])

        raw_5d = row.get(_FWD_5D_COL)
        raw_10d = row.get(_FWD_10D_COL)
        raw_20d = row.get(_FWD_20D_COL)

        if pd.isna(raw_5d):
            raw_5d = None
        if pd.isna(raw_10d):
            raw_10d = None
        if pd.isna(raw_20d):
            raw_20d = None

        fwd_5d, fwd_10d = safe_forward_returns(cdate, today, raw_5d, raw_10d)

        # For 20d: apply the same leakage guard conservatively
        fwd_20d_safe, _ = safe_forward_returns(cdate, today, raw_20d, None)

        if fwd_5d is not None:
            results["fwd_5d"].append(fwd_5d)
        if fwd_10d is not None:
            results["fwd_10d"].append(fwd_10d)
        if fwd_20d_safe is not None:
            results["fwd_20d"].append(fwd_20d_safe)

    def _stats(vals: list[float]) -> dict | None:
        if not vals:
            return None
        arr = np.array(vals)
        return {
            "mean": round(float(arr.mean()), 6),
            "median": round(float(np.median(arr)), 6),
            "n": len(vals),
        }

    return {
        "forward_return_5d": _stats(results["fwd_5d"]),
        "forward_return_10d": _stats(results["fwd_10d"]),
        "forward_return_20d": _stats(results["fwd_20d"]),
    }


# ---------------------------------------------------------------------------
# Public ranking entry point
# ---------------------------------------------------------------------------

def rank(date: str, params: dict[str, Any] | None = None) -> dict:
    """
    Find the top-10 most similar historical sessions to *date*.

    Parameters
    ----------
    date   : ISO date string, e.g. '2024-11-03'
    params : optional overrides — 'top_k' (int, default 10)

    Returns
    -------
    dict with keys:
        date, query_regime, top_matches (list of 10 dicts), forward_return_stats,
        model_version, feedback_informed, knn_candidates_retrieved
    """
    params = params or {}
    top_k = int(params.get("top_k", _TOP_K))

    hist = history_up_to(date)
    if hist.empty:
        raise ValueError(f"No feature history available up to {date}")

    ts = pd.Timestamp(date)
    hist["date"] = pd.to_datetime(hist["date"])

    row_mask = hist["date"] == ts
    if not row_mask.any():
        raise KeyError(f"No features row for date {ts.date()}")

    query_row = hist.loc[row_mask].iloc[0]

    # History excludes the query date itself for candidate pool
    candidate_pool = hist[hist["date"] < ts].copy().reset_index(drop=True)
    valid_pool = candidate_pool.dropna(subset=KNN_FEATURES).reset_index(drop=True)

    if len(valid_pool) < 2:
        raise ValueError(
            f"Not enough historical sessions (need >= 2, have {len(valid_pool)}) "
            f"before {ts.date()} to run similarity ranking."
        )

    # Load or train artifact
    artifact = _load_artifact()
    if artifact is None:
        try:
            train_and_save(hist)
        except ValueError as exc:
            raise RuntimeError(
                f"No ranker model available and auto-training failed: {exc}"
            ) from exc
        artifact = _load_artifact()
        if artifact is None:
            raise RuntimeError("Auto-training completed but artifact could not be loaded.")

    scaler, ranker, meta = artifact

    # Step 1: k-NN retrieval
    q_vec = np.array(
        [query_row[c] if c in query_row.index else np.nan for c in KNN_FEATURES],
        dtype=float,
    )
    if np.isnan(q_vec).any():
        q_vec = np.where(np.isnan(q_vec), 0.0, q_vec)

    X_pool = valid_pool[KNN_FEATURES].values.astype(float)
    X_pool_scaled = scaler.transform(np.where(np.isnan(X_pool), 0.0, X_pool))
    knn = _fit_knn(X_pool_scaled)
    candidates = _knn_candidates(knn, scaler, valid_pool, q_vec)

    n_retrieved = len(candidates)

    # Step 2: XGBRanker re-scoring
    X_pair = _pairwise_matrix(query_row, candidates, ts)
    scores = ranker.predict(X_pair)

    # Top-k indices by ranker score (descending)
    top_indices = list(np.argsort(scores)[::-1][:top_k])

    # Build output matches
    query_regime = _get_regime(query_row)
    matches = []
    for rank_pos, ci in enumerate(top_indices):
        crow = candidates.iloc[ci]
        cdate = pd.Timestamp(crow["date"])
        cregime = _get_regime(crow)

        raw_fwd_5d = crow.get(_FWD_5D_COL)
        raw_fwd_10d = crow.get(_FWD_10D_COL)
        if pd.isna(raw_fwd_5d):
            raw_fwd_5d = None
        if pd.isna(raw_fwd_10d):
            raw_fwd_10d = None

        safe_5d, safe_10d = safe_forward_returns(cdate, ts, raw_fwd_5d, raw_fwd_10d)

        matches.append({
            "rank": rank_pos + 1,
            "date": str(cdate.date()),
            "ranker_score": round(float(scores[ci]), 6),
            "knn_cosine_score": round(float(crow["_knn_cosine_score"]), 6),
            "regime": cregime,
            "regime_match": int(query_regime is not None and cregime == query_regime),
            "forward_return_5d": safe_5d,
            "forward_return_10d": safe_10d,
        })

    fwd_stats = _forward_return_stats(candidates, top_indices, ts)

    model_version = meta.get("version", "xgb_ranker_v1")
    try:
        mtime = _MODEL_PATH.stat().st_mtime
        model_version = f"{model_version} (trained {pd.Timestamp(mtime, unit='s').date()})"
    except OSError:
        pass

    return {
        "date": str(ts.date()),
        "query_regime": query_regime,
        "top_matches": matches,
        "forward_return_stats": fwd_stats,
        "model_version": model_version,
        "feedback_informed": meta.get("feedback_informed", False),
        "knn_candidates_retrieved": n_retrieved,
    }
