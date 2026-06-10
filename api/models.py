"""
api/models.py
Pydantic request/response models — spec §12.2 and §12.3.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class ConversationTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., description="Text of the turn")


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Natural-language question about QSE market activity")
    date: str | None = Field(None, description="ISO date (YYYY-MM-DD); defaults to most recent trading day in features_master")
    params: dict[str, Any] = Field(default_factory=dict, description="Optional per-bucket override params forwarded to analytics modules")
    conversation_history: list[ConversationTurn] = Field(
        default_factory=list,
        description="Prior turns in the conversation (last N user/assistant pairs) for follow-up context",
    )


class FeedbackRequest(BaseModel):
    feedback_type: str = Field(..., description="thumbs_up | thumbs_down | anomaly_confirm | anomaly_reject | similarity_rating | correction")
    query_date: str | None = None
    question: str | None = None
    user_id: str | None = None
    target_date: str | None = None
    rating: int | None = Field(None, ge=1, le=5)
    correction_text: str | None = None
    model_versions: dict[str, str] | None = None


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class SimilarSession(BaseModel):
    rank: int
    date: str
    similarity_score: float
    regime: str | None = None
    forward_return_5d: float | None = None
    forward_return_10d: float | None = None


class AnomalyAssessment(BaseModel):
    date: str
    anomaly_score: float
    anomaly_label: int
    confidence: float
    top_contributing_features: list[str]
    model_version: str


class RegimeContext(BaseModel):
    date: str
    current_regime: str
    regime_probability: float | None = None
    sessions_in_current_regime: int | None = None
    regime_start_date: str | None = None
    prior_regime: str | None = None
    # Volatility regime (orthogonal to trend regime)
    vol_regime: str | None = None
    vol_regime_probability: float | None = None
    vol_regime_sessions: int | None = None
    vol_regime_start_date: str | None = None
    prior_vol_regime: str | None = None
    volatility_20d_current: float | None = None
    volatility_20d_percentile: float | None = None
    volatility_60d_current: float | None = None


class QueryResponse(BaseModel):
    answer: str
    analytics_used: list[str]
    regime_context: RegimeContext | None = None
    anomaly_assessment: AnomalyAssessment | None = None
    similarity_results: list[SimilarSession] = Field(default_factory=list)
    data_date: str
    model_versions: dict[str, str]
    response_time_ms: float


class FeedbackResponse(BaseModel):
    id: int
    status: str = "stored"


class ModelInfo(BaseModel):
    name: str
    version: str | None = None
    artifact_path: str | None = None
    last_modified: str | None = None


class ModelsStatusResponse(BaseModel):
    models: list[ModelInfo]
    last_retrain_timestamp: str | None = None


class HealthResponse(BaseModel):
    status: str
    features_loaded: bool
    features_rows: int
    ollama_reachable: bool
    model_versions: dict[str, str]
