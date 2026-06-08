"""
scripts/retrain/train_anomaly.py
Standalone entry point: retrain the anomaly_scorer RF model.

Usage
-----
python scripts/retrain/train_anomaly.py [--force]

Exits 0 on success, 1 on validation gate failure.
--force  skip the minimum-feedback threshold check.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from analytics._loader import load_features
from feedback.store import get_anomaly_feedback, feedback_counts
from ml.anomaly_scorer import train_and_save

log = logging.getLogger(__name__)

_MIN_FEEDBACK_ITEMS = 10


def main(force: bool = False) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [anomaly] %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    counts = feedback_counts()
    since = counts.pop("since", None)
    anomaly_new = counts.get("anomaly_confirm", 0) + counts.get("anomaly_reject", 0)

    if not force and anomaly_new < _MIN_FEEDBACK_ITEMS:
        log.info(
            "Skipping anomaly_scorer retrain: only %d new anomaly feedback items "
            "(need %d). Pass --force to override.",
            anomaly_new,
            _MIN_FEEDBACK_ITEMS,
        )
        return 0

    log.info(
        "Retraining anomaly_scorer - %d new feedback items since %s.",
        anomaly_new,
        since or "all time",
    )

    features_df = load_features()
    feedback_df = get_anomaly_feedback()

    # build_anomaly_labels() expects 'date' (not 'target_date') and
    # 'label_type' (not 'feedback_type'). Rename each column independently
    # so a missing column on one side never silently suppresses the other.
    if not feedback_df.empty:
        if "target_date" in feedback_df.columns:
            feedback_df = feedback_df.rename(columns={"target_date": "date"})
        if "feedback_type" in feedback_df.columns:
            feedback_df = feedback_df.rename(columns={"feedback_type": "label_type"})

    try:
        path = train_and_save(features_df, feedback_df if not feedback_df.empty else None)
        log.info("anomaly_scorer saved -> %s", path)
        return 0
    except ValueError as exc:
        log.error("anomaly_scorer validation failed: %s", exc)
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrain anomaly_scorer.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip minimum-feedback threshold check.",
    )
    args = parser.parse_args()
    sys.exit(main(force=args.force))
