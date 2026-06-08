"""
api/endpoints/feedback.py
POST /feedback
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from feedback import store
from api.models import FeedbackRequest, FeedbackResponse

log = logging.getLogger(__name__)

router = APIRouter()

_VALID_TYPES = {
    "thumbs_up", "thumbs_down",
    "anomaly_confirm", "anomaly_reject",
    "similarity_rating", "correction",
}


@router.post("/feedback", response_model=FeedbackResponse)
async def post_feedback(req: FeedbackRequest) -> FeedbackResponse:
    if req.feedback_type not in _VALID_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid feedback_type {req.feedback_type!r}. Valid: {sorted(_VALID_TYPES)}",
        )

    try:
        row_id = store.store(
            feedback_type=req.feedback_type,
            query_date=req.query_date,
            question=req.question,
            user_id=req.user_id,
            target_date=req.target_date,
            rating=req.rating,
            correction_text=req.correction_text,
            model_versions=req.model_versions,
        )
    except Exception as exc:
        log.error("Feedback store failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Could not store feedback: {exc}")

    return FeedbackResponse(id=row_id)
