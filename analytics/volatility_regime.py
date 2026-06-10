"""
analytics/volatility_regime.py
2-state HMM volatility regime detector: low_vol / high_vol.

Completely independent of the trend HMM in analytics/regime.py.
Uses only volatility-characterising features so state assignment is
orthogonal to bull/bear/sideways direction.

run(date, params) -> dict
fit_and_save()     -> path to saved model
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

from analytics._loader import history_up_to

_ROOT = Path(__file__).resolve().parents[1]
_MODEL_DIR = _ROOT / "models" / "vol_hmm"
_MODEL_PATH = _MODEL_DIR / "vol_hmm_v1.pkl"
_SYMLINK = _MODEL_DIR / "vol_hmm_current"

_MIN_SESSIONS = 250
_N_COMPONENTS = 2

# Feature set: vol level (annualised), vol trend (60d), activity, participation.
# Deliberately excludes return_1d so the vol state is direction-agnostic.
_FEATURES = [
    "volatility_20d",    # short-term annualised vol — primary signal
    "volatility_60d",    # medium-term vol — persistence filter
    "volume_zscore",     # abnormal activity amplifies vol regimes
    "breadth_ratio",     # narrow market breadth -> elevated vol
]

_STATE_LABELS = ["low_vol", "high_vol"]   # assigned by mean volatility_20d (ascending)


# ---------------------------------------------------------------------------
# Model persistence helpers
# ---------------------------------------------------------------------------

def _save_model(scaler: StandardScaler, model: GaussianHMM) -> Path:
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"scaler": scaler, "model": model}, _MODEL_PATH)

    link = _SYMLINK
    if link.exists() or link.is_symlink():
        link.unlink()
    try:
        link.symlink_to(_MODEL_PATH.resolve())
    except (OSError, NotImplementedError):
        link.with_suffix(".ptr").write_text(str(_MODEL_PATH.resolve()))

    return _MODEL_PATH


def _load_model() -> tuple[StandardScaler, GaussianHMM] | None:
    """Return (scaler, model) from vol_hmm_current or vol_hmm_v1.pkl, or None."""
    for path in [_SYMLINK, _MODEL_PATH]:
        resolved = path.resolve() if path.is_symlink() else path
        if resolved.exists():
            try:
                art = joblib.load(resolved)
                return art["scaler"], art["model"]
            except Exception:
                pass

    ptr = _SYMLINK.with_suffix(".ptr")
    if ptr.exists():
        target = Path(ptr.read_text().strip())
        if target.exists():
            art = joblib.load(target)
            return art["scaler"], art["model"]

    return None


# ---------------------------------------------------------------------------
# State label assignment
# ---------------------------------------------------------------------------

def _assign_state_labels(model: GaussianHMM) -> dict[int, str]:
    """
    Sort the 2 HMM states by their mean volatility_20d and assign
    low_vol / high_vol in ascending order.
    """
    vol_idx = _FEATURES.index("volatility_20d")
    means = model.means_[:, vol_idx]
    order = np.argsort(means)   # index 0 -> lower vol state
    return {int(state): _STATE_LABELS[rank] for rank, state in enumerate(order)}


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------

def fit_and_save(df: pd.DataFrame | None = None) -> Path:
    """
    Fit a 2-state GaussianHMM on *df* (or all features_master if None).
    Raises ValueError if fewer than _MIN_SESSIONS rows are available.
    """
    if df is None:
        from analytics._loader import load_features
        df = load_features()

    clean = df[_FEATURES].dropna()
    if len(clean) < _MIN_SESSIONS:
        raise ValueError(
            f"Need at least {_MIN_SESSIONS} sessions to fit vol HMM; "
            f"only {len(clean)} available."
        )

    scaler = StandardScaler()
    X = scaler.fit_transform(clean.values)

    model = GaussianHMM(
        n_components=_N_COMPONENTS,
        covariance_type="full",
        n_iter=200,
        random_state=42,
    )
    model.fit(X)
    return _save_model(scaler, model)


# ---------------------------------------------------------------------------
# Decoding helpers
# ---------------------------------------------------------------------------

def _decode(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, StandardScaler, GaussianHMM]:
    """
    Load (or fit) the model, decode hidden states for *df*.
    Returns (state_seq, posteriors_matrix, scaler, model).
    """
    artifact = _load_model()
    if artifact is None:
        fit_and_save(df)
        artifact = _load_model()

    scaler, model = artifact
    clean = df[_FEATURES].dropna()
    X = scaler.transform(clean.values)
    state_seq = model.predict(X)
    posteriors = model.predict_proba(X)
    return state_seq, posteriors, scaler, model


def _sessions_in_run(state_seq: np.ndarray, idx: int) -> int:
    s = state_seq[idx]
    count = 0
    for i in range(idx, -1, -1):
        if state_seq[i] == s:
            count += 1
        else:
            break
    return count


# ---------------------------------------------------------------------------
# Volatility percentile helper
# ---------------------------------------------------------------------------

def _vol_percentile(hist_clean: pd.DataFrame, current_vol: float) -> float | None:
    """Percentile rank of current volatility_20d vs all history."""
    series = hist_clean["volatility_20d"].dropna()
    if series.empty or pd.isna(current_vol):
        return None
    return round(float((series < current_vol).mean() * 100), 1)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(date: str, params: dict[str, Any]) -> dict:
    """
    Returns a JSON-serialisable dict describing the volatility regime for *date*.
    Fields:
      date, vol_regime, vol_regime_probability, vol_regime_sessions,
      vol_regime_start_date, prior_vol_regime, prior_vol_regime_duration_sessions,
      volatility_20d_current, volatility_20d_percentile,
      volatility_60d_current, vol_regime_distribution, model_version, note
    """
    hist = history_up_to(date)
    clean = hist[_FEATURES + ["date"]].dropna().reset_index(drop=True)
    n = len(clean)

    if n < _MIN_SESSIONS:
        return {
            "date": str(pd.Timestamp(date).date()),
            "vol_regime": None,
            "vol_regime_probability": None,
            "vol_regime_sessions": None,
            "vol_regime_start_date": None,
            "prior_vol_regime": None,
            "prior_vol_regime_duration_sessions": None,
            "volatility_20d_current": None,
            "volatility_20d_percentile": None,
            "volatility_60d_current": None,
            "vol_regime_distribution": None,
            "model_version": None,
            "note": (
                f"Vol regime requires at least {_MIN_SESSIONS} sessions; "
                f"only {n} available."
            ),
        }

    state_seq, posteriors, scaler, model = _decode(clean)
    label_map = _assign_state_labels(model)

    last_idx = len(state_seq) - 1
    current_state = int(state_seq[last_idx])
    current_label = label_map[current_state]
    vol_prob = round(float(posteriors[last_idx, current_state]), 4)

    sessions_in_regime = _sessions_in_run(state_seq, last_idx)
    run_start = last_idx - sessions_in_regime + 1
    regime_start_date = str(clean["date"].iloc[run_start].date())

    # Prior vol regime
    prior_label: str | None = None
    prior_duration: int | None = None
    if run_start > 0:
        prior_end = run_start - 1
        prior_state = int(state_seq[prior_end])
        prior_label = label_map[prior_state]
        prior_duration = _sessions_in_run(state_seq, prior_end)

    # Historical distribution
    total = len(state_seq)
    vol_distribution = {
        label_map[sid]: round(float((state_seq == sid).sum() / total), 4)
        for sid in range(_N_COMPONENTS)
    }

    # Current vol levels
    last_row = clean.iloc[last_idx]
    vol20_current = round(float(last_row["volatility_20d"]), 6) if not pd.isna(last_row["volatility_20d"]) else None
    vol60_current = round(float(last_row["volatility_60d"]), 6) if not pd.isna(last_row["volatility_60d"]) else None
    vol20_pct = _vol_percentile(clean, last_row["volatility_20d"])

    model_version = "vol_hmm_v1"
    try:
        mtime = _MODEL_PATH.stat().st_mtime
        model_version = f"vol_hmm_v1 (trained {pd.Timestamp(mtime, unit='s').date()})"
    except OSError:
        pass

    return {
        "date": str(pd.Timestamp(date).date()),
        "vol_regime": current_label,
        "vol_regime_probability": vol_prob,
        "vol_regime_sessions": sessions_in_regime,
        "vol_regime_start_date": regime_start_date,
        "prior_vol_regime": prior_label,
        "prior_vol_regime_duration_sessions": prior_duration,
        "volatility_20d_current": vol20_current,
        "volatility_20d_percentile": vol20_pct,
        "volatility_60d_current": vol60_current,
        "vol_regime_distribution": vol_distribution,
        "model_version": model_version,
    }
