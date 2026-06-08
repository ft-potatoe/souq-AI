"""
analytics/regime.py
HMM-based market regime detection.

run(date, params) -> dict
fit_and_save()     -> path to saved model
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

from analytics._loader import history_up_to

_ROOT = Path(__file__).resolve().parents[1]
_MODEL_DIR = _ROOT / "models" / "regime_hmm"
_MODEL_PATH = _MODEL_DIR / "hmm_v1.pkl"
_SYMLINK = _MODEL_DIR / "hmm_current"

_MIN_SESSIONS = 250
_N_COMPONENTS = 3
_FEATURES = [
    "return_1d",
    "volatility_20d",
    "volume_zscore",
    "breadth_ratio",
    "foreign_flow_zscore",
    "rsi_14",
]
_STATE_LABELS = ["bear", "sideways", "bull"]  # assigned after sort by mean return_1d


# ---------------------------------------------------------------------------
# Model persistence helpers
# ---------------------------------------------------------------------------

def _save_model(scaler: StandardScaler, model: GaussianHMM) -> Path:
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {"scaler": scaler, "model": model}
    joblib.dump(artifact, _MODEL_PATH)

    # Create / update hmm_current symlink (or junction on Windows)
    target = _MODEL_PATH.resolve()
    link = _SYMLINK

    if link.exists() or link.is_symlink():
        link.unlink()

    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        # Windows without developer mode: write a small pointer file instead
        link.with_suffix(".ptr").write_text(str(target))

    return _MODEL_PATH


def _load_model() -> tuple[StandardScaler, GaussianHMM] | None:
    """Return (scaler, model) from hmm_current symlink or hmm_v1.pkl, or None."""
    candidates = [_SYMLINK, _MODEL_PATH]
    for path in candidates:
        resolved = path.resolve() if path.is_symlink() else path
        if resolved.exists():
            try:
                artifact = joblib.load(resolved)
                return artifact["scaler"], artifact["model"]
            except Exception:
                pass

    # Fallback: check for Windows pointer file
    ptr = _SYMLINK.with_suffix(".ptr")
    if ptr.exists():
        target = Path(ptr.read_text().strip())
        if target.exists():
            artifact = joblib.load(target)
            return artifact["scaler"], artifact["model"]

    return None


# ---------------------------------------------------------------------------
# State label assignment
# ---------------------------------------------------------------------------

def _assign_state_labels(model: GaussianHMM, feature_names: list[str]) -> dict[int, str]:
    """
    Sort the HMM states by their mean return_1d and assign
    bear / sideways / bull in ascending order.
    """
    return_idx = feature_names.index("return_1d")
    means = model.means_[:, return_idx]  # shape (n_components,)
    order = np.argsort(means)            # lowest -> highest return_1d
    return {int(state): _STATE_LABELS[rank] for rank, state in enumerate(order)}


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------

def fit_and_save(df: pd.DataFrame | None = None) -> Path:
    """
    Fit a GaussianHMM on all available history (or *df* if provided),
    save to models/regime_hmm/hmm_v1.pkl and update hmm_current.
    Returns the saved model path.
    Raises ValueError if fewer than _MIN_SESSIONS rows are available.
    """
    if df is None:
        from analytics._loader import load_features
        df = load_features()

    clean = df[_FEATURES].dropna()
    if len(clean) < _MIN_SESSIONS:
        raise ValueError(
            f"Need at least {_MIN_SESSIONS} sessions to fit HMM; "
            f"only {len(clean)} available."
        )

    X_raw = clean.values
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    model = GaussianHMM(
        n_components=_N_COMPONENTS,
        covariance_type="full",
        n_iter=200,
        random_state=42,
    )
    model.fit(X)

    return _save_model(scaler, model)


# ---------------------------------------------------------------------------
# Regime decoding helpers
# ---------------------------------------------------------------------------

def _decode(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, StandardScaler, GaussianHMM]:
    """
    Load model, decode hidden states for *df*, return
    (state_sequence, log_posteriors_matrix, scaler, model).
    Fits a fresh model if none exists yet.
    """
    artifact = _load_model()
    if artifact is None:
        fit_and_save(df)
        artifact = _load_model()

    scaler, model = artifact
    clean = df[_FEATURES].dropna()
    X = scaler.transform(clean.values)
    state_seq = model.predict(X)
    log_post = model.predict_proba(X)   # (n_samples, n_components)
    return state_seq, log_post, scaler, model


def _sessions_in_run(state_seq: np.ndarray, idx: int) -> int:
    """Count how many consecutive identical states end at position *idx*."""
    s = state_seq[idx]
    count = 0
    for i in range(idx, -1, -1):
        if state_seq[i] == s:
            count += 1
        else:
            break
    return count


def _prior_regime_info(
    state_seq: np.ndarray,
    current_run_start: int,
    label_map: dict[int, str],
) -> tuple[str | None, int | None]:
    """Return (prior_label, prior_duration) for the run just before current_run_start."""
    if current_run_start == 0:
        return None, None
    prior_end = current_run_start - 1
    prior_state = state_seq[prior_end]
    prior_label = label_map[prior_state]
    duration = _sessions_in_run(state_seq, prior_end)
    return prior_label, duration


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(date: str, params: dict[str, Any]) -> dict:
    """
    params: (none required; reserved for future use)

    Returns a JSON-serialisable dict with the fields specified in spec §16.
    Returns regime: null with a note if fewer than 250 sessions are available.
    """
    hist = history_up_to(date)

    # Strip rows where any feature is NaN so the count matches what HMM sees
    clean_hist = hist[_FEATURES + ["date"]].dropna()
    n_sessions = len(clean_hist)

    if n_sessions < _MIN_SESSIONS:
        return {
            "date": str(pd.Timestamp(date).date()),
            "current_regime": None,
            "regime_probability": None,
            "sessions_in_current_regime": None,
            "regime_start_date": None,
            "prior_regime": None,
            "prior_regime_duration_sessions": None,
            "regime_distribution_historical": None,
            "model_version": None,
            "note": (
                f"Regime labels require at least {_MIN_SESSIONS} sessions; "
                f"only {n_sessions} available."
            ),
        }

    state_seq, log_post, scaler, model = _decode(clean_hist)
    label_map = _assign_state_labels(model, _FEATURES)

    # Current (last) position
    last_idx = len(state_seq) - 1
    current_state = int(state_seq[last_idx])
    current_label = label_map[current_state]

    # Probability of current state at the last step
    regime_prob = round(float(log_post[last_idx, current_state]), 4)

    # How long have we been in the current regime?
    sessions_in_regime = _sessions_in_run(state_seq, last_idx)
    current_run_start = last_idx - sessions_in_regime + 1
    regime_start_date = str(clean_hist["date"].iloc[current_run_start].date())

    # Prior regime
    prior_label, prior_duration = _prior_regime_info(state_seq, current_run_start, label_map)

    # Historical distribution (fraction of sessions in each regime)
    total = len(state_seq)
    regime_distribution: dict[str, float] = {}
    for state_id, label in label_map.items():
        regime_distribution[label] = round(float((state_seq == state_id).sum() / total), 4)

    # Model version string (file mtime, or "v1")
    model_version = "hmm_v1"
    try:
        mtime = _MODEL_PATH.stat().st_mtime
        model_version = f"hmm_v1 (trained {pd.Timestamp(mtime, unit='s').date()})"
    except OSError:
        pass

    return {
        "date": str(pd.Timestamp(date).date()),
        "current_regime": current_label,
        "regime_probability": regime_prob,
        "sessions_in_current_regime": sessions_in_regime,
        "regime_start_date": regime_start_date,
        "prior_regime": prior_label,
        "prior_regime_duration_sessions": prior_duration,
        "regime_distribution_historical": regime_distribution,
        "model_version": model_version,
    }
