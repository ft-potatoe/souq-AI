"""
scripts/retrain/train_clustering.py
Standalone entry point: refit the HDBSCAN market-state clustering model.

Clustering always retrains (no minimum-feedback threshold), like the HMMs.
Validation is internal to train_and_save(): silhouette >= 0.20, noise_fraction
<= 0.40, n_clusters >= 2. A gate failure raises ValueError; the prior artifact is
restored atomically so a failed fit never leaves a degraded model live.

Usage
-----
python scripts/retrain/train_clustering.py

Exits 0 on success, 1 on validation failure.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from analytics._loader import load_features
from ml.clustering import train_and_save, _load_artifact, _save_artifact

log = logging.getLogger(__name__)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [clustering] %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    features_df = load_features()

    # Capture the prior artifact so we can restore it if the new fit fails its gate.
    prior = _load_artifact()
    if prior is not None:
        log.info("Prior clustering model loaded; will restore on gate failure.")
    else:
        log.info("No prior clustering model found.")

    log.info("Refitting clustering on %d sessions.", len(features_df))
    try:
        path = train_and_save(features_df)
    except ValueError as exc:
        log.error("Clustering fit/gate failed: %s", exc)
        if prior is not None:
            old_scaler, old_model, old_meta = prior
            _save_artifact(old_scaler, old_model, old_meta)
            log.info("Prior clustering model restored.")
        return 1

    log.info("Clustering model saved -> %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
