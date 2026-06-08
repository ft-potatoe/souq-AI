"""
api/_dates.py
Shared helpers: date resolution and model version snapshot.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from analytics._loader import load_features

_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIRS = {
    "anomaly_scorer":    _ROOT / "models" / "anomaly_scorer",
    "similarity_ranker": _ROOT / "models" / "similarity_ranker",
    "regime_hmm":        _ROOT / "models" / "regime_hmm",
}

SYMLINK_NAMES = {
    "anomaly_scorer":    "rf_anomaly_current",
    "similarity_ranker": "xgb_ranker_current",
    "regime_hmm":        "hmm_current",
}


def resolve_date(date_str: str | None) -> date:
    """Return the requested date, or the most recent trading day in features_master."""
    if date_str is not None:
        try:
            return date.fromisoformat(date_str)
        except ValueError:
            raise ValueError(f"Invalid date format: {date_str!r}. Use YYYY-MM-DD.")

    df = load_features()
    if df.empty:
        raise ValueError("features_master is empty — cannot resolve default date")
    return df["date"].max().date()


def symlink_target(model_key: str) -> Path | None:
    """Return the resolved path a model symlink points to, or None."""
    d = MODEL_DIRS[model_key]
    link = d / SYMLINK_NAMES[model_key]
    ptr = link.with_suffix(".ptr")

    if link.exists() or link.is_symlink():
        try:
            return link.resolve()
        except OSError:
            return None
    if ptr.exists():
        target_str = ptr.read_text().strip()
        p = Path(target_str)
        return p if p.exists() else None
    return None


def model_versions_snapshot() -> dict[str, str]:
    """Return {model_key: version_string} for all three models."""
    versions: dict[str, str] = {}
    for key in MODEL_DIRS:
        target = symlink_target(key)
        if target is not None:
            versions[key] = target.name
        else:
            versions[key] = "not_trained"
    return versions
