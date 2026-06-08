"""
api/endpoints/models_status.py
GET /models/status
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter

from api._dates import MODEL_DIRS, symlink_target
from api.models import ModelInfo, ModelsStatusResponse

log = logging.getLogger(__name__)

router = APIRouter()

_ROOT = Path(__file__).resolve().parents[2]
_RETRAIN_LOG = _ROOT / "logs" / "retrain_log.jsonl"


def _last_retrain_timestamp() -> str | None:
    """Read the most recent *successful* retrain timestamp from retrain_log.jsonl.
    Mirrors feedback/store._last_retrain_ts() — only status=='success' entries count.
    """
    if not _RETRAIN_LOG.exists():
        return None
    import json
    last_ts: str | None = None
    try:
        with _RETRAIN_LOG.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get("status") == "success" and "timestamp" in entry:
                    last_ts = entry["timestamp"]
    except Exception as exc:
        log.debug("Could not read retrain log: %s", exc)
    return last_ts


def _mtime_iso(path: Path) -> str | None:
    try:
        ts = path.stat().st_mtime
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    except OSError:
        return None


@router.get("/models/status", response_model=ModelsStatusResponse)
async def get_models_status() -> ModelsStatusResponse:
    infos: list[ModelInfo] = []

    for key in MODEL_DIRS:
        target = symlink_target(key)
        if target is not None:
            infos.append(ModelInfo(
                name=key,
                version=target.name,
                artifact_path=str(target),
                last_modified=_mtime_iso(target),
            ))
        else:
            infos.append(ModelInfo(name=key, version=None, artifact_path=None, last_modified=None))

    return ModelsStatusResponse(
        models=infos,
        last_retrain_timestamp=_last_retrain_timestamp(),
    )
