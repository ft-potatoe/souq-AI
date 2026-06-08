"""
scripts/retrain/train_ranker.py
Standalone entry point: retrain the similarity_ranker XGBRanker model.

Usage
-----
python scripts/retrain/train_ranker.py [--force]

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
from feedback.store import get_similarity_ratings, feedback_counts
from ml.similarity_ranker import train_and_save

log = logging.getLogger(__name__)

_MIN_FEEDBACK_ITEMS = 20


def main(force: bool = False) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [ranker] %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    counts = feedback_counts()
    since = counts.pop("since", None)
    ranker_new = counts.get("similarity_rating", 0)

    if not force and ranker_new < _MIN_FEEDBACK_ITEMS:
        log.info(
            "Skipping similarity_ranker retrain: only %d new similarity_rating items "
            "(need %d). Pass --force to override.",
            ranker_new,
            _MIN_FEEDBACK_ITEMS,
        )
        return 0

    log.info(
        "Retraining similarity_ranker -- %d new feedback items since %s.",
        ranker_new,
        since or "all time",
    )

    features_df = load_features()
    feedback_df = get_similarity_ratings()

    try:
        path = train_and_save(features_df, feedback_df if not feedback_df.empty else None)
        log.info("similarity_ranker saved -> %s", path)
        return 0
    except ValueError as exc:
        log.error("similarity_ranker validation failed: %s", exc)
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrain similarity_ranker.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip minimum-feedback threshold check.",
    )
    args = parser.parse_args()
    sys.exit(main(force=args.force))
