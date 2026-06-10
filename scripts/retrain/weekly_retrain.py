"""
scripts/retrain/weekly_retrain.py
Weekly retraining pipeline — spec §10.1, 8 steps.

Schedule: every Sunday at 02:00
  e.g.  0 2 * * 0  python scripts/retrain/weekly_retrain.py

Log schema (spec §10.2) written to logs/retrain_log.jsonl:
  {
    "timestamp":  "<ISO UTC>",         # naive UTC, no +00:00 suffix
    "status":     "success" | "failed",
    "models": {
      "anomaly_scorer":    "deployed" | "skipped" | "failed",
      "similarity_ranker": "deployed" | "skipped" | "failed",
      "regime_hmm":        "deployed" | "failed"
    },
    "metrics": {
      "anomaly_scorer":    {cv_metrics dict} | null,
      "similarity_ranker": {ndcg_at_10: float} | null,
      "regime_hmm":        {flip_rate: float} | null
    },
    "errors": {
      "<model>": "<error message>",
      ...
    },
    "feedback_counts": { ... },
    "api_reload": "ok" | "skipped" | "failed"
  }

Exits 0 if all models that attempted training deployed successfully.
Exits 1 if any model that was attempted failed validation.

Note on log status and feedback window:
  feedback/store._last_retrain_ts() anchors the feedback counting window to the
  most recent log entry with status="success". A run where any model fails writes
  status="failed", so the feedback window stays at the prior successful run and
  all feedback items are re-counted next week. This is intentionally conservative:
  it avoids silently under-counting feedback when a partial failure leaves one or
  more models in a stale state.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
from datetime import UTC, datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from analytics._loader import load_features
from feedback.store import (
    feedback_counts,
    get_anomaly_feedback,
    get_similarity_ratings,
)
from ml.anomaly_scorer import train_and_save as anomaly_train, _load_artifact as _load_anomaly
from ml.similarity_ranker import train_and_save as ranker_train, _load_artifact as _load_ranker
from analytics.regime import (
    fit_and_save as hmm_fit,
    _load_model as _load_hmm,
    _save_model as _save_hmm,
    _assign_state_labels as _hmm_label_map,
    _FEATURES as _HMM_FEATURES,
)
from analytics.volatility_regime import (
    fit_and_save as vol_hmm_fit,
    _load_model as _load_vol_hmm,
    _save_model as _save_vol_hmm,
    _assign_state_labels as _vol_hmm_label_map,
    _FEATURES as _VOL_HMM_FEATURES,
)

_LOG_PATH = _ROOT / "logs" / "retrain_log.jsonl"
_MIN_ANOMALY_FEEDBACK = 10
_MIN_RANKER_FEEDBACK = 20
_MAX_FLIP_RATE = 0.10

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _now_utc() -> str:
    return datetime.now(UTC).replace(tzinfo=None).isoformat(timespec="seconds")


def _write_log(entry: dict) -> None:
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Flip-rate helpers for HMM
# ---------------------------------------------------------------------------

def _semantic_hmm_labels(
    scaler, model, features_df: pd.DataFrame
) -> np.ndarray | None:
    """Decode *features_df* with *model* and return semantic label array (strings)."""
    clean = features_df[_HMM_FEATURES].dropna()
    if clean.empty:
        return None
    raw_ids = model.predict(scaler.transform(clean.values))
    label_map = _hmm_label_map(model, _HMM_FEATURES)
    return np.array([label_map[s] for s in raw_ids])


def _flip_rate(old: np.ndarray, new: np.ndarray) -> float:
    n = min(len(old), len(new))
    return float((old[:n] != new[:n]).sum() / n) if n else 0.0


# ---------------------------------------------------------------------------
# API reload (step 7)
# ---------------------------------------------------------------------------

def _reload_api() -> str:
    """
    Send SIGHUP to a running uvicorn process, or restart gracefully.
    Returns "ok", "skipped", or "failed".
    On Windows SIGHUP is unavailable; we attempt to find and signal the process
    via taskkill /F (force-restart) only if UVICORN_PID is set in the environment.
    """
    pid_str = os.environ.get("UVICORN_PID", "").strip()
    if not pid_str:
        log.info("UVICORN_PID not set; skipping API reload.")
        return "skipped"

    try:
        pid = int(pid_str)
    except ValueError:
        log.warning("UVICORN_PID='%s' is not a valid integer; skipping reload.", pid_str)
        return "skipped"

    try:
        if sys.platform == "win32":
            # Windows: no SIGHUP; send CTRL_BREAK_EVENT to the process group
            os.kill(pid, signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
        else:
            os.kill(pid, signal.SIGHUP)
        log.info("Sent reload signal to uvicorn PID %d.", pid)
        return "ok"
    except (ProcessLookupError, PermissionError, OSError) as exc:
        log.warning("Could not signal uvicorn PID %d: %s", pid, exc)
        return "failed"


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def _step_anomaly(
    features_df: pd.DataFrame,
    counts: dict,
    errors: dict,
    metrics: dict,
) -> str:
    """Step 3: train anomaly_scorer. Returns deployment status string."""
    anomaly_new = counts.get("anomaly_confirm", 0) + counts.get("anomaly_reject", 0)
    if anomaly_new < _MIN_ANOMALY_FEEDBACK:
        log.info(
            "anomaly_scorer: only %d new items (need %d) -- skipping.",
            anomaly_new, _MIN_ANOMALY_FEEDBACK,
        )
        return "skipped"

    log.info("anomaly_scorer: retraining with %d new feedback items.", anomaly_new)
    feedback_df = get_anomaly_feedback()
    if not feedback_df.empty:
        if "target_date" in feedback_df.columns:
            feedback_df = feedback_df.rename(columns={"target_date": "date"})
        if "feedback_type" in feedback_df.columns:
            feedback_df = feedback_df.rename(columns={"feedback_type": "label_type"})

    try:
        train_and_save_result = anomaly_train(
            features_df,
            feedback_df if not feedback_df.empty else None,
        )
        artifact = _load_anomaly()
        if artifact is not None:
            _, meta = artifact
            metrics["anomaly_scorer"] = meta.get("cv_metrics")
        log.info("anomaly_scorer deployed -> %s", train_and_save_result)
        return "deployed"
    except ValueError as exc:
        msg = str(exc)
        log.error("anomaly_scorer failed: %s", msg)
        errors["anomaly_scorer"] = msg
        return "failed"


def _step_ranker(
    features_df: pd.DataFrame,
    counts: dict,
    errors: dict,
    metrics: dict,
) -> str:
    """Step 5: train similarity_ranker. Returns deployment status string."""
    ranker_new = counts.get("similarity_rating", 0)
    if ranker_new < _MIN_RANKER_FEEDBACK:
        log.info(
            "similarity_ranker: only %d new items (need %d) -- skipping.",
            ranker_new, _MIN_RANKER_FEEDBACK,
        )
        return "skipped"

    log.info("similarity_ranker: retraining with %d new feedback items.", ranker_new)
    feedback_df = get_similarity_ratings()

    try:
        path = ranker_train(
            features_df,
            feedback_df if not feedback_df.empty else None,
        )
        artifact = _load_ranker()
        if artifact is not None:
            _, _, meta = artifact
            metrics["similarity_ranker"] = {"ndcg_at_10": meta.get("ndcg_at_10")}
        log.info("similarity_ranker deployed -> %s", path)
        return "deployed"
    except ValueError as exc:
        msg = str(exc)
        log.error("similarity_ranker failed: %s", msg)
        errors["similarity_ranker"] = msg
        return "failed"


def _semantic_vol_labels(scaler, model, features_df: pd.DataFrame) -> np.ndarray | None:
    clean = features_df[_VOL_HMM_FEATURES].dropna()
    if clean.empty:
        return None
    raw_ids = model.predict(scaler.transform(clean.values))
    label_map = _vol_hmm_label_map(model)
    return np.array([label_map[s] for s in raw_ids])


def _step_vol_hmm(
    features_df: pd.DataFrame,
    errors: dict,
    metrics: dict,
) -> str:
    """Step 6b: refit vol HMM. Returns deployment status. Failure is non-blocking."""
    log.info("vol_hmm: refitting on %d sessions.", len(features_df))

    prior = _load_vol_hmm()
    if prior is not None:
        old_scaler, old_model = prior
        old_labels = _semantic_vol_labels(old_scaler, old_model, features_df)
    else:
        old_labels = None

    try:
        path = vol_hmm_fit(features_df)
    except ValueError as exc:
        msg = str(exc)
        log.warning("vol_hmm fit failed (non-blocking): %s", msg)
        errors["vol_hmm"] = msg
        return "failed"

    log.info("vol_hmm saved -> %s", path)

    flip = 0.0
    if old_labels is not None:
        new_artifact = _load_vol_hmm()
        if new_artifact is not None:
            new_scaler, new_model = new_artifact
            new_labels = _semantic_vol_labels(new_scaler, new_model, features_df)
            if new_labels is not None:
                flip = _flip_rate(old_labels, new_labels)
                log.info("vol_hmm label-flip rate: %.1f%%", flip * 100)
                if flip > _MAX_FLIP_RATE:
                    msg = (
                        f"vol label-flip rate {flip:.1%} > {_MAX_FLIP_RATE:.1%}; "
                        "prior model restored"
                    )
                    log.warning("vol_hmm: %s (non-blocking)", msg)
                    _save_vol_hmm(old_scaler, old_model)
                    errors["vol_hmm"] = msg
                    metrics["vol_hmm"] = {"flip_rate": round(flip, 6)}
                    return "failed"

    metrics["vol_hmm"] = {"flip_rate": round(flip, 6)}
    return "deployed"


def _step_hmm(
    features_df: pd.DataFrame,
    errors: dict,
    metrics: dict,
) -> str:
    """Step 6: refit HMM. Returns deployment status string."""
    log.info("regime_hmm: refitting on %d sessions.", len(features_df))

    # Load prior model into memory before overwriting the artifact so we can
    # restore it atomically if the flip-rate gate fails.
    prior = _load_hmm()
    if prior is not None:
        old_scaler, old_model = prior
        old_labels = _semantic_hmm_labels(old_scaler, old_model, features_df)
        log.info("Prior HMM loaded; will validate label-flip rate after refit.")
    else:
        old_labels = None
        log.info("No prior HMM model; skipping flip-rate check.")

    try:
        path = hmm_fit(features_df)
    except ValueError as exc:
        msg = str(exc)
        log.error("regime_hmm fit failed: %s", msg)
        errors["regime_hmm"] = msg
        return "failed"

    log.info("regime_hmm saved -> %s", path)

    flip = 0.0
    if old_labels is not None:
        new_artifact = _load_hmm()
        if new_artifact is not None:
            new_scaler, new_model = new_artifact
            new_labels = _semantic_hmm_labels(new_scaler, new_model, features_df)
            if new_labels is not None:
                flip = _flip_rate(old_labels, new_labels)
                log.info("regime_hmm label-flip rate: %.1f%%", flip * 100)
                if flip > _MAX_FLIP_RATE:
                    msg = (
                        f"label-flip rate {flip:.1%} > {_MAX_FLIP_RATE:.1%}; "
                        "prior model restored"
                    )
                    log.error("regime_hmm: %s", msg)
                    _save_hmm(old_scaler, old_model)
                    log.info("regime_hmm: prior model restored.")
                    errors["regime_hmm"] = msg
                    metrics["regime_hmm"] = {"flip_rate": round(flip, 6)}
                    return "failed"

    metrics["regime_hmm"] = {"flip_rate": round(flip, 6)}
    return "deployed"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [weekly_retrain] %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    log.info("=== Weekly retrain pipeline start ===")

    # Step 1: load features + feedback counts
    log.info("Step 1: loading features_master and feedback counts.")
    features_df = load_features()
    raw_counts = feedback_counts()
    since = raw_counts.pop("since", None)
    log.info("Features: %d rows. Feedback window since: %s.", len(features_df), since or "all time")

    errors: dict[str, str] = {}
    model_status: dict[str, str] = {}
    metrics: dict[str, object] = {
        "anomaly_scorer": None,
        "similarity_ranker": None,
        "regime_hmm": None,
        "vol_hmm": None,
    }

    # Step 2: build training labels (done inside each model's train_and_save)

    # Step 3: anomaly_scorer
    log.info("Step 3: anomaly_scorer.")
    model_status["anomaly_scorer"] = _step_anomaly(features_df, raw_counts, errors, metrics)

    # Step 4: build ranking pairs (done inside similarity_ranker.train_and_save)

    # Step 5: similarity_ranker
    log.info("Step 5: similarity_ranker.")
    model_status["similarity_ranker"] = _step_ranker(features_df, raw_counts, errors, metrics)

    # Step 6: regime HMM
    log.info("Step 6: regime_hmm.")
    model_status["regime_hmm"] = _step_hmm(features_df, errors, metrics)

    # Step 6b: volatility regime HMM (additive; failure does not block overall status)
    log.info("Step 6b: vol_hmm.")
    model_status["vol_hmm"] = _step_vol_hmm(features_df, errors, metrics)

    # Step 7: reload API workers
    log.info("Step 7: API reload.")
    reload_result = _reload_api()

    # Step 8: write retrain_log.jsonl
    # vol_hmm is additive — its failure does not drive overall status to "failed"
    core_models = {"anomaly_scorer", "similarity_ranker", "regime_hmm"}
    any_failed = any(v == "failed" for k, v in model_status.items() if k in core_models)
    overall_status = "failed" if any_failed else "success"

    entry = {
        "timestamp": _now_utc(),
        "status": overall_status,
        "models": model_status,
        "metrics": metrics,
        "errors": errors,
        "feedback_counts": {**raw_counts, "since": since},
        "api_reload": reload_result,
    }
    _write_log(entry)
    log.info(
        "Step 8: log written -> %s  overall=%s",
        _LOG_PATH,
        overall_status,
    )
    log.info("=== Weekly retrain pipeline end ===")

    return 1 if any_failed else 0


if __name__ == "__main__":
    sys.exit(main())
