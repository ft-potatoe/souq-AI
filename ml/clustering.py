"""
ml/clustering.py
Unsupervised market-state discovery via HDBSCAN.

Discovers recurring "day types" in the feature space that nobody defined in advance,
and labels genuine outlier sessions as noise (cluster_id = -1) -- a second, unsupervised
anomaly signal. Reports which cluster the queried date most resembles, plus each
cluster's profile.

Strictly descriptive: clusters are historical groupings, never predictions.

Public API
----------
train_and_save(features_df=None) -> Path
cluster(date, params)            -> dict
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from analytics._loader import history_up_to

log = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[1]
_MODEL_DIR = _ROOT / "models" / "clustering"
_MODEL_PATH = _MODEL_DIR / "hdbscan_v1.pkl"
_SYMLINK = _MODEL_DIR / "hdbscan_current"

_MIN_SESSIONS = 250

# HDBSCAN hyper-parameters (tuned so the QSE feature space yields >=2 stable clusters
# with low noise; see plan verification sweep).
_MIN_CLUSTER_SIZE = 15
_MIN_SAMPLES = 5

# Validation gate (raises ValueError on failure in train_and_save), parallels the
# NDCG gate (ranker) and flip-rate gate (HMMs).
_GATE_SILHOUETTE = 0.20
_GATE_MAX_NOISE = 0.40
_GATE_MIN_CLUSTERS = 2

# Stable, versioned feature allow-list (NO forward returns).
_FEATURES = [
    "return_1d",
    "return_5d",
    "volatility_20d",
    "volume_zscore",
    "breadth_ratio",
    "foreign_flow_zscore",
    "domestic_flow_zscore",
    "rsi_14",
    "price_vs_sma20_pct",
]

# Features used to derive a human-readable label for each cluster, with the
# z-score sign that drives each descriptive token.
_MEMBER_SAMPLE = 3


# ---------------------------------------------------------------------------
# Model persistence (mirror analytics/regime.py)
# ---------------------------------------------------------------------------

def _save_artifact(scaler: StandardScaler, model: HDBSCAN, meta: dict) -> Path:
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {"scaler": scaler, "model": model, "meta": meta}
    joblib.dump(artifact, _MODEL_PATH)

    target = _MODEL_PATH.resolve()
    link = _SYMLINK
    if link.exists() or link.is_symlink():
        link.unlink()
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        link.with_suffix(".ptr").write_text(str(target))

    return _MODEL_PATH


def _load_artifact() -> tuple[StandardScaler, HDBSCAN, dict] | None:
    """Return (scaler, model, meta) from symlink or versioned pkl, or None."""
    candidates = [_SYMLINK, _MODEL_PATH]
    for path in candidates:
        resolved = path.resolve() if path.is_symlink() else path
        if resolved.exists():
            try:
                artifact = joblib.load(resolved)
                return artifact["scaler"], artifact["model"], artifact["meta"]
            except Exception:
                pass

    ptr = _SYMLINK.with_suffix(".ptr")
    if ptr.exists():
        target = Path(ptr.read_text().strip())
        if target.exists():
            artifact = joblib.load(target)
            return artifact["scaler"], artifact["model"], artifact["meta"]

    return None


# ---------------------------------------------------------------------------
# Cluster labelling
# ---------------------------------------------------------------------------

def _label_for_profile(z: dict[str, float]) -> str:
    """
    Build a short, deterministic, human-readable label from a cluster's mean
    feature z-scores (relative to the full population). Stable across retrains
    because it depends only on profile signs/magnitudes, not state ordering.
    """
    parts: list[str] = []

    vol = z.get("volatility_20d", 0.0)
    if vol >= 0.6:
        parts.append("High-vol")
    elif vol <= -0.6:
        parts.append("Low-vol")

    ret = z.get("return_1d", 0.0)
    if ret >= 0.4:
        parts.append("rally")
    elif ret <= -0.4:
        parts.append("selloff")
    else:
        parts.append("drift")

    breadth = z.get("breadth_ratio", 0.0)
    if breadth >= 0.6:
        parts.append("(broad)")
    elif breadth <= -0.6:
        parts.append("(narrow)")

    label = " ".join(parts).strip()
    return label if label else "Mixed conditions"


def _build_profiles(
    clean: pd.DataFrame,
    labels: np.ndarray,
    pop_mean: pd.Series,
    pop_std: pd.Series,
) -> dict[int, dict[str, Any]]:
    """
    Per-cluster profile: size, mean of each feature, and a derived label.
    Keyed by integer cluster id (noise = -1 excluded).
    """
    profiles: dict[int, dict[str, Any]] = {}
    for cid in sorted(set(int(x) for x in labels)):
        if cid == -1:
            continue
        members = clean[labels == cid]
        means = members[_FEATURES].mean()
        z = {f: float((means[f] - pop_mean[f]) / pop_std[f]) if pop_std[f] else 0.0
             for f in _FEATURES}
        profiles[cid] = {
            "cluster_id": cid,
            "label": _label_for_profile(z),
            "size": int(len(members)),
            "characteristics": {f: round(float(means[f]), 4) for f in _FEATURES},
            "_centroid_z": z,  # internal: used for nearest-cluster assignment
        }
    return profiles


# ---------------------------------------------------------------------------
# Training + validation gate
# ---------------------------------------------------------------------------

def train_and_save(features_df: pd.DataFrame | None = None) -> Path:
    """
    Fit StandardScaler + HDBSCAN on all available history, compute per-cluster
    profiles, and persist the artifact. Raises ValueError if fewer than
    _MIN_SESSIONS rows are available or the validation gate fails.
    """
    if features_df is None:
        from analytics._loader import load_features
        features_df = load_features()

    clean = features_df[_FEATURES + ["date"]].dropna().reset_index(drop=True)
    if len(clean) < _MIN_SESSIONS:
        raise ValueError(
            f"Need at least {_MIN_SESSIONS} sessions to fit clustering; "
            f"only {len(clean)} available."
        )

    X_raw = clean[_FEATURES].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    model = HDBSCAN(
        min_cluster_size=_MIN_CLUSTER_SIZE,
        min_samples=_MIN_SAMPLES,
        store_centers="centroid",
        copy=True,
    )
    labels = model.fit_predict(X)

    n_clusters = len(set(int(x) for x in labels) - {-1})
    noise_fraction = float((labels == -1).mean())

    sil = None
    mask = labels != -1
    if n_clusters >= 2 and mask.sum() > n_clusters:
        sil = float(silhouette_score(X[mask], labels[mask]))

    # Validation gate -- parallels NDCG / flip-rate gates elsewhere.
    if n_clusters < _GATE_MIN_CLUSTERS:
        raise ValueError(
            f"Clustering gate failed: found {n_clusters} clusters "
            f"(need >= {_GATE_MIN_CLUSTERS})."
        )
    if noise_fraction > _GATE_MAX_NOISE:
        raise ValueError(
            f"Clustering gate failed: noise_fraction {noise_fraction:.2f} "
            f"> {_GATE_MAX_NOISE}."
        )
    if sil is None or sil < _GATE_SILHOUETTE:
        raise ValueError(
            f"Clustering gate failed: silhouette {sil} < {_GATE_SILHOUETTE}."
        )

    pop_mean = clean[_FEATURES].mean()
    pop_std = clean[_FEATURES].std(ddof=0)
    profiles = _build_profiles(clean, labels, pop_mean, pop_std)

    # Sample member dates per cluster (for the UI / explanation).
    member_dates: dict[int, list[str]] = {}
    for cid in profiles:
        member_rows = clean[labels == cid]
        member_dates[cid] = [
            str(d.date()) for d in member_rows["date"].head(_MEMBER_SAMPLE)
        ]

    meta = {
        "feature_names": _FEATURES,
        "n_clusters": n_clusters,
        "cluster_profiles": profiles,
        "member_dates": member_dates,
        "silhouette": round(sil, 4),
        "noise_fraction": round(noise_fraction, 4),
        "pop_mean": {f: float(pop_mean[f]) for f in _FEATURES},
        "pop_std": {f: float(pop_std[f]) for f in _FEATURES},
        "version": "hdbscan_v1",
    }

    return _save_artifact(scaler, model, meta)


# ---------------------------------------------------------------------------
# Cluster assignment for the queried date
# ---------------------------------------------------------------------------

def _assign_today(
    row_scaled: np.ndarray,
    profiles: dict[int, dict[str, Any]],
) -> tuple[int, float, bool]:
    """
    Assign the queried row to its nearest cluster centroid in z-score space.
    HDBSCAN has no stable per-row predict, so we use nearest-centroid; a row
    whose distance exceeds the largest centroid norm by a wide margin is left
    flagged via distance (the caller decides outlier status from is_outlier).

    Returns (cluster_id, distance_to_centroid, is_outlier).
    """
    if not profiles:
        return -1, float("nan"), True

    # row_scaled is already standardised by the StandardScaler, which (fit with the
    # same population) matches the z-score space used for each cluster's _centroid_z.
    row_vec = np.asarray(row_scaled, dtype=float)

    best_cid = -1
    best_dist = float("inf")
    for cid, prof in profiles.items():
        # Centroid in scaled space = z-score of cluster mean (StandardScaler with
        # ddof=0 matches population std used for profiles).
        cz = prof["_centroid_z"]
        centroid = np.array([cz[f] for f in _FEATURES])
        dist = float(np.linalg.norm(row_vec - centroid))
        if dist < best_dist:
            best_dist = dist
            best_cid = cid

    # Outlier heuristic: far from every centroid (in scaled space, sqrt(dim) ~ 3.0
    # is already a large multivariate distance for standardised data).
    is_outlier = best_dist > (np.sqrt(len(_FEATURES)) * 1.5)
    return best_cid, round(best_dist, 4), is_outlier


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def cluster(date: str, params: dict[str, Any] | None = None) -> dict:
    """
    Assign the queried date to a discovered market-state cluster and report all
    cluster profiles. Auto-trains on first call when no artifact exists.

    Returns a JSON-serialisable dict. Clusters are descriptive historical
    groupings -- never predictions of future membership or returns.
    """
    params = params or {}
    hist = history_up_to(date)
    clean = hist[_FEATURES + ["date"]].dropna()
    n_sessions = len(clean)

    if n_sessions < _MIN_SESSIONS:
        return {
            "date": str(pd.Timestamp(date).date()),
            "current_cluster_id": None,
            "current_cluster_label": None,
            "distance_to_centroid": None,
            "is_outlier": None,
            "all_clusters": [],
            "member_dates_sample": [],
            "noise_fraction": None,
            "model_version": None,
            "note": (
                f"Cluster labels require at least {_MIN_SESSIONS} sessions; "
                f"only {n_sessions} available."
            ),
        }

    artifact = _load_artifact()
    if artifact is None:
        try:
            train_and_save(hist)
        except ValueError as exc:
            raise RuntimeError(
                f"No clustering model available and auto-training failed: {exc}"
            )
        artifact = _load_artifact()

    scaler, model, meta = artifact
    profiles: dict[int, dict[str, Any]] = meta["cluster_profiles"]

    # Queried row in scaled space.
    row = clean[_FEATURES].iloc[-1].values.reshape(1, -1)
    row_scaled = scaler.transform(row)[0]

    cid, dist, is_outlier = _assign_today(row_scaled, profiles)

    current_label = "Atypical / outlier day" if (cid == -1 or is_outlier) \
        else profiles[cid]["label"]

    # Public cluster list (strip the internal _centroid_z field).
    all_clusters = []
    for c in sorted(profiles.values(), key=lambda p: p["size"], reverse=True):
        all_clusters.append(
            {
                "cluster_id": c["cluster_id"],
                "label": c["label"],
                "size": c["size"],
                "characteristics": c["characteristics"],
            }
        )

    member_sample = meta.get("member_dates", {}).get(cid, []) if cid != -1 else []

    # Model version string (with train date, like regime.py).
    model_version = meta.get("version", "hdbscan_v1")
    try:
        mtime = _MODEL_PATH.stat().st_mtime
        model_version = f"hdbscan_v1 (trained {pd.Timestamp(mtime, unit='s').date()})"
    except OSError:
        pass

    return {
        "date": str(pd.Timestamp(date).date()),
        "current_cluster_id": cid,
        "current_cluster_label": current_label,
        "distance_to_centroid": dist,
        "is_outlier": bool(is_outlier or cid == -1),
        "all_clusters": all_clusters,
        "member_dates_sample": member_sample,
        "noise_fraction": meta.get("noise_fraction"),
        "model_version": model_version,
        "note": "Descriptive grouping of historical sessions -- not predictive.",
    }
