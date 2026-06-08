"""Feedback store backed by SQLite at data/feedback/feedback.db."""

import json
import sqlite3
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd

_DB_PATH = Path(__file__).resolve().parents[1] / "data" / "feedback" / "feedback.db"

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS feedback (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp        DATETIME NOT NULL,
    user_id          TEXT,
    query_date       DATE,
    question         TEXT,
    feedback_type    TEXT NOT NULL,
    target_date      DATE,
    rating           INTEGER,
    correction_text  TEXT,
    model_versions   TEXT
);
"""

_VALID_TYPES = {
    "thumbs_up",
    "thumbs_down",
    "anomaly_confirm",
    "anomaly_reject",
    "similarity_rating",
    "correction",
}

# Date of last scheduled retrain anchor — Sunday 02:00; used by feedback_counts.
# Retraining updates this via scripts/retrain/weekly_retrain.py writing to
# logs/retrain_log.jsonl; we read the latest entry at call time.
_RETRAIN_LOG = Path(__file__).resolve().parents[1] / "logs" / "retrain_log.jsonl"


def _connect() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(_DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute(_CREATE_TABLE)
    conn.commit()
    return conn


def _last_retrain_ts() -> Optional[datetime]:
    """Return timestamp of the most recent successful retrain, or None."""
    if not _RETRAIN_LOG.exists():
        return None
    ts = None
    with _RETRAIN_LOG.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if entry.get("status") == "success" and "timestamp" in entry:
                    ts = datetime.fromisoformat(entry["timestamp"])
            except (json.JSONDecodeError, ValueError):
                continue
    return ts


def store(
    feedback_type: str,
    query_date: Optional[date] = None,
    question: Optional[str] = None,
    user_id: Optional[str] = None,
    target_date: Optional[date] = None,
    rating: Optional[int] = None,
    correction_text: Optional[str] = None,
    model_versions: Optional[dict] = None,
) -> int:
    """Insert one feedback row and return its auto-assigned id."""
    if feedback_type not in _VALID_TYPES:
        raise ValueError(
            f"Unknown feedback_type '{feedback_type}'. "
            f"Must be one of: {sorted(_VALID_TYPES)}"
        )

    mv_json = json.dumps(model_versions) if model_versions is not None else None
    ts = datetime.now(UTC).replace(tzinfo=None).isoformat(timespec="seconds")

    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO feedback
                (timestamp, user_id, query_date, question,
                 feedback_type, target_date, rating,
                 correction_text, model_versions)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ts,
                user_id,
                query_date.isoformat() if query_date else None,
                question,
                feedback_type,
                target_date.isoformat() if target_date else None,
                rating,
                correction_text,
                mv_json,
            ),
        )
        return cur.lastrowid


def get_since(cutoff_date: date) -> pd.DataFrame:
    """Return all feedback rows with timestamp >= cutoff_date."""
    with _connect() as conn:
        df = pd.read_sql_query(
            "SELECT * FROM feedback WHERE timestamp >= ?",
            conn,
            params=(cutoff_date.isoformat(),),
        )
    return df


def get_anomaly_feedback() -> pd.DataFrame:
    """Return rows of type anomaly_confirm or anomaly_reject."""
    with _connect() as conn:
        df = pd.read_sql_query(
            "SELECT * FROM feedback WHERE feedback_type IN ('anomaly_confirm', 'anomaly_reject')",
            conn,
        )
    return df


def get_similarity_ratings() -> pd.DataFrame:
    """Return rows of type similarity_rating."""
    with _connect() as conn:
        df = pd.read_sql_query(
            "SELECT * FROM feedback WHERE feedback_type = 'similarity_rating'",
            conn,
        )
    return df


def feedback_counts() -> dict:
    """Return per-type counts plus the window used.

    Keys: one entry per feedback_type (count), plus ``"since"`` which is the
    ISO timestamp of the last successful retrain or ``None`` when no retrain
    record exists (all-time counts in that case).
    """
    cutoff = _last_retrain_ts()

    with _connect() as conn:
        if cutoff is not None:
            raw = conn.execute(
                """
                SELECT feedback_type, COUNT(*) AS cnt
                FROM feedback
                WHERE timestamp >= ?
                GROUP BY feedback_type
                """,
                (cutoff.isoformat(timespec="seconds"),),
            ).fetchall()
        else:
            raw = conn.execute(
                """
                SELECT feedback_type, COUNT(*) AS cnt
                FROM feedback
                GROUP BY feedback_type
                """
            ).fetchall()
        # Materialise inside the with-block; sqlite3.Row holds a ref to the
        # connection cursor description and must not outlive the connection.
        counts = {row["feedback_type"]: row["cnt"] for row in raw}

    counts["since"] = cutoff.isoformat(timespec="seconds") if cutoff else None
    return counts
