"""
api/endpoints/query.py
POST /query — orchestrates the full analytics -> LLM pipeline.
"""

from __future__ import annotations

import asyncio
import time
import logging
from typing import Any

from fastapi import APIRouter, HTTPException

from analytics import distribution, trend, correlation, seasonality, flows, gcc, regime as regime_mod, summary
from ml import anomaly_scorer, similarity_ranker
from llm.router import match_buckets, build_llm_payload
from llm.prompts import SYSTEM_PROMPT, build_prompt
from llm.interface import query_llm
from api.models import (
    QueryRequest,
    QueryResponse,
    RegimeContext,
    AnomalyAssessment,
    SimilarSession,
)
from api._dates import resolve_date, model_versions_snapshot

log = logging.getLogger(__name__)

router = APIRouter()

# Maps bucket name -> analytics callable
_ANALYTICS_DISPATCH: dict[str, Any] = {
    "distribution": distribution.run,
    "trend":        trend.run,
    "correlation":  correlation.run,
    "seasonality":  seasonality.run,
    "flows":        flows.run,
    "gcc":          gcc.run,
    "regime":       regime_mod.run,
    "summary":      summary.run,
}


@router.post("/query", response_model=QueryResponse)
async def post_query(req: QueryRequest) -> QueryResponse:
    t0 = time.perf_counter()

    # Resolve date
    try:
        data_date = resolve_date(req.date)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    date_str = data_date.strftime("%Y-%m-%d")
    params = req.params or {}

    # 1. Route question to buckets
    matched = match_buckets(req.question)

    # 2. Run analytics for every matched bucket
    bucket_results: dict[str, dict] = {}
    analytics_used: list[str] = []

    for bucket in matched:
        if bucket in _ANALYTICS_DISPATCH:
            try:
                result = _ANALYTICS_DISPATCH[bucket](date_str, params.get(bucket, {}))
                bucket_results[bucket] = result
                analytics_used.append(bucket)
            except Exception as exc:
                log.warning("Analytics bucket %s failed for %s: %s", bucket, date_str, exc)

        elif bucket == "anomaly":
            try:
                result = anomaly_scorer.score(date_str, params.get("anomaly"))
                bucket_results["anomaly"] = result
                analytics_used.append("anomaly")
            except Exception as exc:
                log.warning("Anomaly scorer failed for %s: %s", date_str, exc)

        elif bucket == "similarity":
            try:
                result = similarity_ranker.rank(date_str, params.get("similarity"))
                bucket_results["similarity"] = result
                analytics_used.append("similarity")
            except Exception as exc:
                log.warning("Similarity ranker failed for %s: %s", date_str, exc)

    # 3. Build token-budgeted LLM payload
    payload = build_llm_payload(matched, bucket_results)

    # 4. Build prompt and call LLM
    # query_llm uses httpx.Client (sync, 120 s timeout) — run in a thread pool
    # so the event loop is not blocked during the Ollama round-trip.
    prompt = build_prompt(req.question, payload)
    try:
        answer = await asyncio.to_thread(query_llm, prompt, SYSTEM_PROMPT)
    except Exception as exc:
        log.error("LLM call failed: %s", exc)
        raise HTTPException(status_code=503, detail=f"LLM unavailable: {exc}")

    # 5. Build structured response fields
    regime_context: RegimeContext | None = None
    if "regime" in bucket_results:
        r = bucket_results["regime"]
        regime_context = RegimeContext(
            date=r.get("date", date_str),
            current_regime=r.get("current_regime", "unknown"),
            regime_probability=r.get("regime_probability"),
            sessions_in_current_regime=r.get("sessions_in_current_regime"),
            regime_start_date=r.get("regime_start_date"),
            prior_regime=r.get("prior_regime"),
        )

    anomaly_assessment: AnomalyAssessment | None = None
    if "anomaly" in bucket_results:
        a = bucket_results["anomaly"]
        anomaly_assessment = AnomalyAssessment(
            date=a.get("date", date_str),
            anomaly_score=float(a.get("anomaly_score", 0.0)),
            anomaly_label=int(a.get("anomaly_label", 0)),
            confidence=float(a.get("confidence", 0.0)),
            top_contributing_features=[
                f["feature"] if isinstance(f, dict) else str(f)
                for f in a.get("top_contributing_features", [])
            ],
            model_version=str(a.get("model_version", "unknown")),
        )

    similarity_results: list[SimilarSession] = []
    if "similarity" in bucket_results:
        for match in bucket_results["similarity"].get("top_matches", []):
            similarity_results.append(SimilarSession(
                rank=int(match.get("rank", 0)),
                date=str(match.get("date", "")),
                similarity_score=float(match.get("ranker_score", match.get("similarity_score", 0.0))),
                regime=match.get("regime"),
                forward_return_5d=match.get("forward_return_5d"),
                forward_return_10d=match.get("forward_return_10d"),
            ))

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return QueryResponse(
        answer=answer,
        analytics_used=analytics_used,
        regime_context=regime_context,
        anomaly_assessment=anomaly_assessment,
        similarity_results=similarity_results,
        data_date=date_str,
        model_versions=model_versions_snapshot(),
        response_time_ms=round(elapsed_ms, 1),
    )
