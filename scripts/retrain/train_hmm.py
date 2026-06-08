"""
scripts/retrain/train_hmm.py
Standalone entry point: refit the regime HMM.

The HMM always retrains (no minimum-feedback threshold).
Validation: label-flip rate across history must be <= 10%.

Usage
-----
python scripts/retrain/train_hmm.py

Exits 0 on success, 1 on validation failure.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

import numpy as np

from analytics._loader import load_features
from analytics.regime import fit_and_save, _load_model, _save_model, _assign_state_labels, _FEATURES

log = logging.getLogger(__name__)

# Spec §10.1 step 6: label-flip rate <= 10% of history
_MAX_FLIP_RATE = 0.10


def _semantic_states(
    scaler, model, features_df: pd.DataFrame
) -> np.ndarray | None:
    """Decode *features_df* with *model* and return semantic label array (strings)."""
    clean = features_df[_FEATURES].dropna()
    if clean.empty:
        return None
    X = scaler.transform(clean.values)
    raw_ids = model.predict(X)
    label_map = _assign_state_labels(model, _FEATURES)
    return np.array([label_map[s] for s in raw_ids])


def _flip_rate(old_labels: np.ndarray, new_labels: np.ndarray) -> float:
    """Fraction of sessions where the semantic label (bear/sideways/bull) changed."""
    n = min(len(old_labels), len(new_labels))
    if n == 0:
        return 0.0
    return float((old_labels[:n] != new_labels[:n]).sum() / n)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [hmm] %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    features_df = load_features()

    # Capture old model + old semantic labels before overwriting the artifact.
    prior = _load_model()
    if prior is not None:
        old_scaler, old_model = prior
        old_labels = _semantic_states(old_scaler, old_model, features_df)
        log.info("Prior HMM loaded; will check label-flip rate after refit.")
    else:
        old_labels = None
        log.info("No prior HMM model found; skipping flip-rate check.")

    log.info("Refitting regime HMM on %d sessions.", len(features_df))
    try:
        path = fit_and_save(features_df)
    except ValueError as exc:
        log.error("HMM fit failed: %s", exc)
        return 1

    log.info("HMM saved -> %s", path)

    # Validate flip rate: compare old semantic labels to new semantic labels.
    # Both sequences are decoded with their respective models so state-ID
    # reordering between fits doesn't inflate the flip count.
    if old_labels is not None:
        new_artifact = _load_model()
        if new_artifact is not None:
            new_scaler, new_model = new_artifact
            new_labels = _semantic_states(new_scaler, new_model, features_df)
            if new_labels is not None:
                rate = _flip_rate(old_labels, new_labels)
                log.info("HMM label-flip rate vs prior: %.1f%%", rate * 100)
                if rate > _MAX_FLIP_RATE:
                    log.error(
                        "HMM label-flip rate %.1f%% exceeds threshold %.1f%%. "
                        "Restoring prior model.",
                        rate * 100,
                        _MAX_FLIP_RATE * 100,
                    )
                    _save_model(old_scaler, old_model)
                    log.info("Prior HMM restored.")
                    return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
