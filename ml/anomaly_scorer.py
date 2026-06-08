"""
ml/anomaly_scorer.py
RandomForest-based market anomaly detection.

Public API
----------
build_anomaly_labels(features_df, feedback_df)  -> pd.Series  (index=date)
train_and_save(features_df, feedback_df)         -> Path
score(date, params)                              -> dict
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from analytics._loader import history_up_to

_ROOT = Path(__file__).resolve().parents[1]
_MODEL_DIR = _ROOT / "models" / "anomaly_scorer"
_MODEL_PATH = _MODEL_DIR / "rf_anomaly_v1.pkl"
_SYMLINK = _MODEL_DIR / "rf_anomaly_current"

# Features consumed by the classifier
FEATURES = [
    "volume_zscore",
    "value_zscore",
    "trades_zscore",
    "breadth_zscore",
    "foreign_flow_zscore",
    "domestic_flow_zscore",
    "return_1d",
    "volatility_20d",
    "rsi_14",
    "qse_vs_gcc_spread",
    "foreign_participation",
    "above_sma_200",
    "price_vs_sma20_pct",
]

# Z-score columns used for bootstrap labelling
_ZSCORE_COLS = [
    "volume_zscore",
    "value_zscore",
    "trades_zscore",
    "breadth_zscore",
    "foreign_flow_zscore",
    "domestic_flow_zscore",
]
_ZSCORE_THRESHOLD = 2.0
_ZSCORE_MIN_COUNT = 2

# Validation gates
_GATE_PRECISION = 0.65
_GATE_RECALL = 0.60
_GATE_AUC = 0.72
_GATE_MAX_DEGRADATION = 0.15  # 15 % relative degradation allowed vs prior

_RF_PARAMS = dict(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)


# ---------------------------------------------------------------------------
# Model persistence helpers (shared pattern with analytics/regime.py)
# ---------------------------------------------------------------------------

def _save_artifact(model: RandomForestClassifier, meta: dict) -> Path:
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {"model": model, "meta": meta}
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


def _load_artifact() -> tuple[RandomForestClassifier, dict] | None:
    """Return (model, meta) from rf_anomaly_current or rf_anomaly_v1.pkl, or None."""
    candidates = [_SYMLINK, _MODEL_PATH]
    for path in candidates:
        # Resolve symlink target; for non-symlinks use path directly
        target = path.resolve() if path.is_symlink() else path
        if target.exists():
            try:
                artifact = joblib.load(target)
                return artifact["model"], artifact["meta"]
            except Exception:
                pass

    ptr = _SYMLINK.with_suffix(".ptr")
    if ptr.exists():
        target = Path(ptr.read_text().strip())
        if target.exists():
            artifact = joblib.load(target)
            return artifact["model"], artifact["meta"]

    return None


# ---------------------------------------------------------------------------
# Label construction
# ---------------------------------------------------------------------------

def build_anomaly_labels(
    features_df: pd.DataFrame,
    feedback_df: pd.DataFrame | None = None,
) -> pd.Series:
    """
    Bootstrap: sessions with >= 2 z-score columns above |2.0| -> 1, else 0.
    Feedback overrides: anomaly_confirm -> 1, anomaly_reject -> 0.

    Parameters
    ----------
    features_df : DataFrame with a 'date' column and _ZSCORE_COLS present.
    feedback_df : Optional DataFrame with columns ['date', 'label_type'].
                  label_type values: 'anomaly_confirm' | 'anomaly_reject'

    Returns
    -------
    pd.Series, dtype int, index=RangeIndex aligned to features_df.

    Notes
    -----
    NaN values in z-score columns count as non-exceeding, so rows where all
    z-scores are NaN (e.g. the first ~60 sessions before rolling windows fill)
    are labelled 0 (normal) by the bootstrap. Feedback overrides still apply
    to those rows if provided.
    """
    df = features_df.copy()

    # Count z-score columns exceeding |threshold| per row
    z_present = [c for c in _ZSCORE_COLS if c in df.columns]
    if z_present:
        exceed = df[z_present].abs().gt(_ZSCORE_THRESHOLD)
        labels = (exceed.sum(axis=1) >= _ZSCORE_MIN_COUNT).astype(int)
    else:
        labels = pd.Series(0, index=df.index)

    # Apply feedback overrides keyed by date
    if feedback_df is not None and not feedback_df.empty:
        fb = feedback_df.copy()
        fb["date"] = pd.to_datetime(fb["date"])
        df["date"] = pd.to_datetime(df["date"])

        date_to_idx = {ts: i for i, ts in enumerate(df["date"])}
        for _, row in fb.iterrows():
            idx = date_to_idx.get(row["date"])
            if idx is None:
                continue
            if row["label_type"] == "anomaly_confirm":
                labels.iloc[idx] = 1
            elif row["label_type"] == "anomaly_reject":
                labels.iloc[idx] = 0

    return labels.rename("anomaly_label")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _cross_val_metrics(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
) -> dict[str, float]:
    """
    Stratified k-fold CV using _RF_PARAMS. Returns mean precision, recall, auc_roc.
    Falls back to train-on-all when positives are too sparse for CV.

    NaN values in X must be handled by the caller before this function is called.
    """
    pos_count = int(y.sum())
    n_splits = min(n_splits, pos_count) if pos_count >= 2 else 0
    if n_splits < 2:
        # Not enough positives for CV; fit/predict on full set as a floor check
        m = RandomForestClassifier(**_RF_PARAMS)
        m.fit(X, y)
        proba = m.predict_proba(X)[:, 1]
        preds = (proba >= 0.5).astype(int)
        return {
            "precision": float(precision_score(y, preds, zero_division=0)),
            "recall": float(recall_score(y, preds, zero_division=0)),
            "auc_roc": float(roc_auc_score(y, proba)) if pos_count >= 1 else 0.0,
        }

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    precisions, recalls, aucs = [], [], []

    for train_idx, val_idx in cv.split(X, y):
        m = RandomForestClassifier(**_RF_PARAMS)
        m.fit(X[train_idx], y[train_idx])
        proba = m.predict_proba(X[val_idx])[:, 1]
        preds = (proba >= 0.5).astype(int)
        precisions.append(precision_score(y[val_idx], preds, zero_division=0))
        recalls.append(recall_score(y[val_idx], preds, zero_division=0))
        # AUC requires both classes in the fold
        if len(np.unique(y[val_idx])) == 2:
            aucs.append(roc_auc_score(y[val_idx], proba))

    return {
        "precision": float(np.mean(precisions)),
        "recall": float(np.mean(recalls)),
        "auc_roc": float(np.mean(aucs)) if aucs else 0.0,
    }


def _passes_gates(metrics: dict[str, float]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if metrics["precision"] < _GATE_PRECISION:
        failures.append(
            f"precision {metrics['precision']:.3f} < {_GATE_PRECISION}"
        )
    if metrics["recall"] < _GATE_RECALL:
        failures.append(
            f"recall {metrics['recall']:.3f} < {_GATE_RECALL}"
        )
    if metrics["auc_roc"] < _GATE_AUC:
        failures.append(
            f"auc_roc {metrics['auc_roc']:.3f} < {_GATE_AUC}"
        )
    return len(failures) == 0, failures


def _passes_degradation_check(
    new_metrics: dict[str, float],
    prior_metrics: dict[str, float],
) -> tuple[bool, list[str]]:
    """New metric must not fall more than 15 % relative to prior."""
    failures: list[str] = []
    for key in ("precision", "recall", "auc_roc"):
        prior = prior_metrics.get(key, 0.0)
        new = new_metrics.get(key, 0.0)
        if prior > 0 and (prior - new) / prior > _GATE_MAX_DEGRADATION:
            failures.append(
                f"{key} degraded by "
                f"{(prior - new) / prior:.1%} (prior={prior:.3f}, new={new:.3f})"
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
    Fit the anomaly RF, validate against gates, save if it passes.

    Raises
    ------
    ValueError  if the model fails the absolute validation gates.
    ValueError  if a prior model exists and the new one degrades beyond threshold.
    """
    if features_df is None:
        from analytics._loader import load_features
        features_df = load_features()

    labels = build_anomaly_labels(features_df, feedback_df)

    # Keep only rows with complete feature set and a valid label
    work = features_df[FEATURES + ["date"]].copy()
    work["_label"] = labels.values
    work = work.dropna(subset=FEATURES)

    X = work[FEATURES].values.astype(float)
    y = work["_label"].values.astype(int)

    if len(np.unique(y)) < 2:
        raise ValueError(
            "Label vector has only one class — cannot train a classifier. "
            "Check that z-score features are present and non-trivial."
        )

    metrics = _cross_val_metrics(X, y)

    ok, gate_failures = _passes_gates(metrics)
    if not ok:
        raise ValueError(
            f"Model failed validation gates: {'; '.join(gate_failures)}"
        )

    # Degradation check against existing model
    prior = _load_artifact()
    if prior is not None:
        _, prior_meta = prior
        prior_metrics = prior_meta.get("cv_metrics", {})
        ok2, deg_failures = _passes_degradation_check(metrics, prior_metrics)
        if not ok2:
            raise ValueError(
                f"Model degraded beyond threshold vs prior: {'; '.join(deg_failures)}"
            )

    # Final fit on all data
    model = RandomForestClassifier(**_RF_PARAMS)
    model.fit(X, y)

    meta = {
        "cv_metrics": metrics,
        "feature_names": FEATURES,
        "n_train": len(X),
        "n_anomalous": int(y.sum()),
        "version": "rf_anomaly_v1",
    }
    return _save_artifact(model, meta)


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _top_features(
    model: RandomForestClassifier,
    row_values: np.ndarray,
    top_n: int = 5,
) -> list[dict]:
    """
    Return top-n features by global RF importances, with the session's own value.
    importances are the mean decrease in impurity from the fitted forest.
    """
    importances = model.feature_importances_  # shape (n_features,)
    order = np.argsort(importances)[::-1][:top_n]
    return [
        {
            "feature": FEATURES[i],
            "importance": round(float(importances[i]), 6),
            "session_value": (
                None if np.isnan(row_values[i]) else round(float(row_values[i]), 6)
            ),
        }
        for i in order
    ]


# ---------------------------------------------------------------------------
# Public scoring entry point
# ---------------------------------------------------------------------------

def score(date: str, params: dict[str, Any] | None = None) -> dict:
    """
    Score a single trading session for anomaly.

    Parameters
    ----------
    date   : ISO date string, e.g. '2024-11-03'
    params : unused; reserved for future overrides

    Returns
    -------
    dict with keys: date, anomaly_score, anomaly_label, confidence,
                    top_contributing_features, model_version,
                    bootstrap_label_used
    """
    hist = history_up_to(date)
    if hist.empty:
        raise ValueError(f"No feature history available up to {date}")

    ts = pd.Timestamp(date)
    row_mask = hist["date"] == ts
    if not row_mask.any():
        raise KeyError(f"No features row for date {ts.date()}")

    row = hist.loc[row_mask].iloc[0]

    # Load or train model
    artifact = _load_artifact()
    if artifact is None:
        try:
            train_and_save(hist)
        except ValueError as exc:
            raise RuntimeError(
                f"No model available and auto-training failed: {exc}"
            ) from exc
        artifact = _load_artifact()
        if artifact is None:
            raise RuntimeError("Auto-training completed but artifact could not be loaded.")

    model, meta = artifact

    # Build feature vector; NaN -> imputed as 0 for prediction (flag in output)
    raw_values = np.array([
        row[f] if f in row.index else np.nan for f in FEATURES
    ], dtype=float)
    has_nan = bool(np.isnan(raw_values).any())
    pred_values = np.where(np.isnan(raw_values), 0.0, raw_values)

    proba = model.predict_proba(pred_values.reshape(1, -1))[0]
    # class order: model.classes_ = [0, 1]
    anomaly_idx = list(model.classes_).index(1)
    anomaly_score = float(proba[anomaly_idx])
    anomaly_label = int(anomaly_score >= 0.5)

    # Confidence: probability margin (distance from the 0.5 decision boundary × 2)
    confidence = round(float(abs(2.0 * proba[anomaly_idx] - 1.0)), 4)

    # Bootstrap label for this session (no feedback override at score time)
    z_present = [c for c in _ZSCORE_COLS if c in row.index]
    z_exceed = sum(1 for c in z_present if not np.isnan(row[c]) and abs(row[c]) > _ZSCORE_THRESHOLD)
    bootstrap_label = int(z_exceed >= _ZSCORE_MIN_COUNT)

    top_feats = _top_features(model, raw_values)

    model_version = meta.get("version", "rf_anomaly_v1")
    try:
        mtime = _MODEL_PATH.stat().st_mtime
        model_version = f"{model_version} (trained {pd.Timestamp(mtime, unit='s').date()})"
    except OSError:
        pass

    result = {
        "date": str(ts.date()),
        "anomaly_score": round(anomaly_score, 4),
        "anomaly_label": anomaly_label,
        "confidence": confidence,
        "top_contributing_features": top_feats,
        "model_version": model_version,
        "bootstrap_label_used": bootstrap_label,
    }

    if has_nan:
        result["warning"] = "One or more input features were NaN; imputed as 0 for prediction."

    return result
