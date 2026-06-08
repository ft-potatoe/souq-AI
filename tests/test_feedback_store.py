"""Tests for feedback/store.py."""

import json
import sqlite3
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest

import feedback.store as store_mod
from feedback.store import (
    feedback_counts,
    get_anomaly_feedback,
    get_similarity_ratings,
    get_since,
    store,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def tmp_db(tmp_path, monkeypatch):
    """Redirect DB and retrain log to temp paths for every test."""
    db_path = tmp_path / "feedback.db"
    log_path = tmp_path / "retrain_log.jsonl"
    monkeypatch.setattr(store_mod, "_DB_PATH", db_path)
    monkeypatch.setattr(store_mod, "_RETRAIN_LOG", log_path)
    return tmp_path


# ---------------------------------------------------------------------------
# store()
# ---------------------------------------------------------------------------

class TestStore:
    def test_returns_integer_id(self):
        row_id = store("thumbs_up")
        assert isinstance(row_id, int)
        assert row_id == 1

    def test_ids_are_sequential(self):
        id1 = store("thumbs_up")
        id2 = store("thumbs_down")
        assert id2 == id1 + 1

    def test_all_valid_feedback_types(self):
        types = [
            "thumbs_up", "thumbs_down", "anomaly_confirm",
            "anomaly_reject", "similarity_rating", "correction",
        ]
        for i, ft in enumerate(types, start=1):
            row_id = store(ft)
            assert row_id == i

    def test_invalid_feedback_type_raises(self):
        with pytest.raises(ValueError, match="Unknown feedback_type"):
            store("bad_type")

    def test_optional_fields_stored(self):
        qdate = date(2024, 3, 10)
        tdate = date(2024, 3, 11)
        row_id = store(
            "correction",
            query_date=qdate,
            question="Is the market bullish?",
            user_id="analyst_1",
            target_date=tdate,
            rating=4,
            correction_text="Trend is sideways.",
            model_versions={"anomaly": "v3", "ranker": "v2"},
        )
        conn = sqlite3.connect(store_mod._DB_PATH)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM feedback WHERE id = ?", (row_id,)).fetchone()
        conn.close()

        assert row["user_id"] == "analyst_1"
        assert row["query_date"] == "2024-03-10"
        assert row["question"] == "Is the market bullish?"
        assert row["target_date"] == "2024-03-11"
        assert row["rating"] == 4
        assert row["correction_text"] == "Trend is sideways."
        assert json.loads(row["model_versions"]) == {"anomaly": "v3", "ranker": "v2"}

    def test_optional_fields_default_to_none(self):
        row_id = store("thumbs_up")
        conn = sqlite3.connect(store_mod._DB_PATH)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM feedback WHERE id = ?", (row_id,)).fetchone()
        conn.close()

        assert row["user_id"] is None
        assert row["query_date"] is None
        assert row["question"] is None
        assert row["target_date"] is None
        assert row["rating"] is None
        assert row["correction_text"] is None
        assert row["model_versions"] is None

    def test_timestamp_is_stored(self):
        row_id = store("thumbs_up")
        conn = sqlite3.connect(store_mod._DB_PATH)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT timestamp FROM feedback WHERE id = ?", (row_id,)).fetchone()
        conn.close()
        # Should parse without error
        datetime.fromisoformat(row["timestamp"])

    def test_model_versions_none_stored_as_null(self):
        row_id = store("thumbs_up", model_versions=None)
        conn = sqlite3.connect(store_mod._DB_PATH)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT model_versions FROM feedback WHERE id = ?", (row_id,)).fetchone()
        conn.close()
        assert row["model_versions"] is None

    def test_model_versions_dict_round_trips(self):
        mv = {"anomaly": "v5", "ranker": "v4", "hmm": "v2"}
        row_id = store("thumbs_up", model_versions=mv)
        conn = sqlite3.connect(store_mod._DB_PATH)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT model_versions FROM feedback WHERE id = ?", (row_id,)).fetchone()
        conn.close()
        assert json.loads(row["model_versions"]) == mv


# ---------------------------------------------------------------------------
# get_since()
# ---------------------------------------------------------------------------

class TestGetSince:
    def _insert_with_ts(self, feedback_type, ts_str):
        conn = sqlite3.connect(store_mod._DB_PATH)
        conn.execute(
            "INSERT INTO feedback (timestamp, feedback_type) VALUES (?, ?)",
            (ts_str, feedback_type),
        )
        conn.commit()
        conn.close()

    def test_returns_dataframe(self):
        store("thumbs_up")
        df = get_since(date(2000, 1, 1))
        assert isinstance(df, pd.DataFrame)

    def test_empty_when_no_rows(self):
        df = get_since(date(2000, 1, 1))
        assert len(df) == 0

    def test_filters_by_cutoff(self):
        # Ensure table exists
        store("thumbs_up")
        conn = sqlite3.connect(store_mod._DB_PATH)
        conn.execute("DELETE FROM feedback")
        conn.commit()
        conn.close()

        self._insert_with_ts("thumbs_up", "2024-01-01T10:00:00")
        self._insert_with_ts("thumbs_down", "2024-06-01T10:00:00")
        self._insert_with_ts("correction", "2025-01-01T10:00:00")

        df = get_since(date(2024, 6, 1))
        assert len(df) == 2
        assert set(df["feedback_type"]) == {"thumbs_down", "correction"}

    def test_returns_all_when_old_cutoff(self):
        store("thumbs_up")
        store("thumbs_down")
        df = get_since(date(2000, 1, 1))
        assert len(df) == 2

    def test_columns_present(self):
        store("thumbs_up", question="Q?", user_id="u1")
        df = get_since(date(2000, 1, 1))
        for col in ["id", "timestamp", "user_id", "query_date", "question",
                    "feedback_type", "target_date", "rating",
                    "correction_text", "model_versions"]:
            assert col in df.columns


# ---------------------------------------------------------------------------
# get_anomaly_feedback()
# ---------------------------------------------------------------------------

class TestGetAnomalyFeedback:
    def test_returns_only_anomaly_types(self):
        store("anomaly_confirm")
        store("anomaly_reject")
        store("thumbs_up")
        store("correction")
        df = get_anomaly_feedback()
        assert len(df) == 2
        assert set(df["feedback_type"]) == {"anomaly_confirm", "anomaly_reject"}

    def test_empty_when_no_anomaly_rows(self):
        store("thumbs_up")
        store("thumbs_down")
        df = get_anomaly_feedback()
        assert len(df) == 0

    def test_returns_dataframe(self):
        df = get_anomaly_feedback()
        assert isinstance(df, pd.DataFrame)

    def test_multiple_confirms(self):
        store("anomaly_confirm")
        store("anomaly_confirm")
        store("anomaly_reject")
        df = get_anomaly_feedback()
        assert len(df) == 3
        assert (df["feedback_type"] == "anomaly_confirm").sum() == 2

    def test_columns_present(self):
        store("anomaly_confirm")
        df = get_anomaly_feedback()
        assert "feedback_type" in df.columns
        assert "id" in df.columns


# ---------------------------------------------------------------------------
# get_similarity_ratings()
# ---------------------------------------------------------------------------

class TestGetSimilarityRatings:
    def test_returns_only_similarity_rating(self):
        store("similarity_rating", rating=5)
        store("similarity_rating", rating=3)
        store("thumbs_up")
        store("anomaly_confirm")
        df = get_similarity_ratings()
        assert len(df) == 2
        assert (df["feedback_type"] == "similarity_rating").all()

    def test_empty_when_no_similarity_rows(self):
        store("thumbs_up")
        df = get_similarity_ratings()
        assert len(df) == 0

    def test_rating_values_preserved(self):
        store("similarity_rating", rating=4)
        store("similarity_rating", rating=2)
        df = get_similarity_ratings()
        assert set(df["rating"]) == {4, 2}

    def test_returns_dataframe(self):
        df = get_similarity_ratings()
        assert isinstance(df, pd.DataFrame)


# ---------------------------------------------------------------------------
# feedback_counts()
# ---------------------------------------------------------------------------

class TestFeedbackCounts:
    def test_returns_dict(self):
        store("thumbs_up")
        result = feedback_counts()
        assert isinstance(result, dict)

    def test_counts_all_when_no_retrain_log(self):
        store("thumbs_up")
        store("thumbs_up")
        store("thumbs_down")
        counts = feedback_counts()
        assert counts.get("thumbs_up") == 2
        assert counts.get("thumbs_down") == 1

    def test_since_is_none_when_no_retrain_log(self):
        counts = feedback_counts()
        assert counts["since"] is None

    def test_empty_store_returns_only_since_key(self):
        counts = feedback_counts()
        assert set(counts.keys()) == {"since"}

    def test_counts_since_last_retrain(self, tmp_path):
        cutoff = datetime(2024, 5, 1, 2, 0, 0)
        store_mod._RETRAIN_LOG.write_text(
            json.dumps({"status": "success", "timestamp": cutoff.isoformat()}) + "\n"
        )

        store_mod._connect().close()
        conn = sqlite3.connect(store_mod._DB_PATH)
        conn.execute(_CREATE_SNIPPET, ("2024-04-01T10:00:00", "thumbs_up"))
        conn.execute(_CREATE_SNIPPET, ("2024-05-15T10:00:00", "thumbs_up"))
        conn.execute(_CREATE_SNIPPET, ("2024-06-01T10:00:00", "correction"))
        conn.commit()
        conn.close()

        counts = feedback_counts()
        assert counts.get("thumbs_up") == 1      # only the post-cutoff row
        assert counts.get("correction") == 1
        assert counts["since"] == "2024-05-01T02:00:00"

    def test_ignores_failed_retrain_entries(self):
        store_mod._RETRAIN_LOG.write_text(
            json.dumps({"status": "failed", "timestamp": "2024-01-01T02:00:00"}) + "\n"
        )
        store("thumbs_up")
        store("thumbs_up")
        counts = feedback_counts()
        assert counts.get("thumbs_up") == 2
        assert counts["since"] is None

    def test_uses_latest_successful_retrain(self):
        store_mod._RETRAIN_LOG.write_text(
            json.dumps({"status": "success", "timestamp": "2024-03-01T02:00:00"}) + "\n" +
            json.dumps({"status": "success", "timestamp": "2024-05-01T02:00:00"}) + "\n"
        )
        store_mod._connect().close()
        conn = sqlite3.connect(store_mod._DB_PATH)
        conn.execute(_CREATE_SNIPPET, ("2024-04-01T10:00:00", "thumbs_up"))  # between retrains
        conn.execute(_CREATE_SNIPPET, ("2024-06-01T10:00:00", "thumbs_down"))  # after latest
        conn.commit()
        conn.close()

        counts = feedback_counts()
        assert counts.get("thumbs_up", 0) == 0
        assert counts.get("thumbs_down") == 1
        assert counts["since"] == "2024-05-01T02:00:00"

    def test_counts_per_type(self):
        for ft in ["thumbs_up", "thumbs_up", "correction", "anomaly_confirm"]:
            store(ft)
        counts = feedback_counts()
        assert counts["thumbs_up"] == 2
        assert counts["correction"] == 1
        assert counts["anomaly_confirm"] == 1


_CREATE_SNIPPET = "INSERT INTO feedback (timestamp, feedback_type) VALUES (?, ?)"


# ---------------------------------------------------------------------------
# Database initialisation
# ---------------------------------------------------------------------------

class TestDbInit:
    def test_db_created_on_first_call(self):
        assert not store_mod._DB_PATH.exists()
        store("thumbs_up")
        assert store_mod._DB_PATH.exists()

    def test_table_schema_has_required_columns(self):
        store("thumbs_up")
        conn = sqlite3.connect(store_mod._DB_PATH)
        cols = {row[1] for row in conn.execute("PRAGMA table_info(feedback)")}
        conn.close()
        expected = {
            "id", "timestamp", "user_id", "query_date", "question",
            "feedback_type", "target_date", "rating",
            "correction_text", "model_versions",
        }
        assert expected.issubset(cols)

    def test_second_connect_reuses_table(self):
        store("thumbs_up")
        # Second call must not raise (CREATE TABLE IF NOT EXISTS)
        store("thumbs_down")
        df = get_since(date(2000, 1, 1))
        assert len(df) == 2
