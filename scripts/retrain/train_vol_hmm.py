"""
scripts/retrain/train_vol_hmm.py
Standalone entry point: refit the volatility regime HMM.

The vol HMM always retrains (no minimum-feedback threshold).
Validation: label-flip rate across history must be <= 10%.

Usage
-----
python scripts/retrain/train_vol_hmm.py

Exits 0 on success, 1 on validation failure.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from analytics._loader import load_features
from analytics.volatility_regime import (
    fit_and_save,
    _load_model,
    _save_model,
    _assign_state_labels,
    _FEATURES,
)

log = logging.getLogger(__name__)

_MAX_FLIP_RATE = 0.10


def _semantic_states(scaler, model, features_df: pd.DataFrame) -> np.ndarray | None:
    """Decode *features_df* with *model* and return semantic label array (strings)."""
    clean = features_df[_FEATURES].dropna()
    if clean.empty:
        return None
    raw_ids = model.predict(scaler.transform(clean.values))
    label_map = _assign_state_labels(model)
    return np.array([label_map[s] for s in raw_ids])


def _flip_rate(old: np.ndarray, new: np.ndarray) -> float:
    n = min(len(old), len(new))
    return float((old[:n] != new[:n]).sum() / n) if n else 0.0


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [vol_hmm] %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    features_df = load_features()

    prior = _load_model()
    if prior is not None:
        old_scaler, old_model = prior
        old_labels = _semantic_states(old_scaler, old_model, features_df)
        log.info("Prior vol HMM loaded; will check label-flip rate after refit.")
    else:
        old_labels = None
        log.info("No prior vol HMM found; skipping flip-rate check.")

    log.info("Refitting vol HMM on %d sessions.", len(features_df))
    try:
        path = fit_and_save(features_df)
    except ValueError as exc:
        log.error("Vol HMM fit failed: %s", exc)
        return 1

    log.info("Vol HMM saved -> %s", path)

    if old_labels is not None:
        new_artifact = _load_model()
        if new_artifact is not None:
            new_scaler, new_model = new_artifact
            new_labels = _semantic_states(new_scaler, new_model, features_df)
            if new_labels is not None:
                rate = _flip_rate(old_labels, new_labels)
                log.info("Vol HMM label-flip rate vs prior: %.1f%%", rate * 100)
                if rate > _MAX_FLIP_RATE:
                    log.error(
                        "Vol HMM label-flip rate %.1f%% exceeds %.1f%%. Restoring prior.",
                        rate * 100, _MAX_FLIP_RATE * 100,
                    )
                    _save_model(old_scaler, old_model)
                    log.info("Prior vol HMM restored.")
                    return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
