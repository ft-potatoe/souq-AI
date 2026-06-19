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
from analytics import volatility_regime as vol_regime_mod
from analytics import relationships as relationships_mod
from ml import anomaly_scorer, similarity_ranker, clustering as clustering_mod
from llm.router import match_buckets, build_llm_payload
from llm.prompts import SYSTEM_PROMPT, build_prompt
from llm.interface import query_llm
from api.models import (
    QueryRequest,
    QueryResponse,
    RegimeContext,
    AnomalyAssessment,
    SimilarSession,
    ClusterInfo,
    ClusteringResult,
    RelationshipItem,
    ConditionalRelationship,
    RelationshipsResult,
)
from api._dates import resolve_date, model_versions_snapshot

log = logging.getLogger(__name__)

router = APIRouter()

# Maps bucket name -> analytics callable
_ANALYTICS_DISPATCH: dict[str, Any] = {
    "distribution":      distribution.run,
    "trend":             trend.run,
    "correlation":       correlation.run,
    "seasonality":       seasonality.run,
    "flows":             flows.run,
    "gcc":               gcc.run,
    "regime":            regime_mod.run,
    "volatility_regime": vol_regime_mod.run,
    "summary":           summary.run,
    "relationships":     relationships_mod.run,
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

    # Infer date_from/date_to for flows range questions
    if "flows" in matched and "flows" not in params:
        from api._daterange import extract_date_range
        d_from, d_to = extract_date_range(req.question, data_date)
        if d_from:
            params = {**params, "flows": {"date_from": d_from}}

    # Infer correlation pair from question when not explicitly supplied
    if "correlation" in matched and "correlation" not in params:
        q_lower = req.question.lower()
        # Detect metric_b (second subject after "and"/"vs"/"with")
        if any(kw in q_lower for kw in ("turnover", "value traded", "value_traded")):
            params = {**params, "correlation": {"metric_a": "foreign_net", "metric_b": "value_traded"}}
        elif any(kw in q_lower for kw in ("volume",)):
            params = {**params, "correlation": {"metric_a": "foreign_net", "metric_b": "volume"}}
        elif any(kw in q_lower for kw in ("return", "index", "price")):
            params = {**params, "correlation": {"metric_a": "foreign_net", "metric_b": "return_1d"}}
        elif any(kw in q_lower for kw in ("domestic",)):
            params = {**params, "correlation": {"metric_a": "foreign_net", "metric_b": "domestic_net"}}
        elif any(kw in q_lower for kw in ("volatil",)):
            params = {**params, "correlation": {"metric_a": "foreign_net", "metric_b": "volatility_20d"}}

    # Infer distribution metric from question when not explicitly supplied
    if "distribution" in matched and "distribution" not in params:
        q_lower = req.question.lower()
        if any(kw in q_lower for kw in ("return", "skew", "gain", "loss", "daily change")):
            params = {**params, "distribution": {"metric": "return_1d"}}
        elif any(kw in q_lower for kw in ("volume",)):
            params = {**params, "distribution": {"metric": "volume"}}
        elif any(kw in q_lower for kw in ("volatil",)):
            params = {**params, "distribution": {"metric": "volatility_20d"}}
        elif any(kw in q_lower for kw in ("foreign", "inflow", "outflow")):
            params = {**params, "distribution": {"metric": "foreign_net"}}

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

        elif bucket == "clustering":
            try:
                result = clustering_mod.cluster(date_str, params.get("clustering"))
                bucket_results["clustering"] = result
                analytics_used.append("clustering")
            except Exception as exc:
                log.warning("Clustering failed for %s: %s", date_str, exc)

    # 3. Build token-budgeted LLM payload
    payload = build_llm_payload(matched, bucket_results)

    # 4. Build prompt and call LLM
    # query_llm uses httpx.Client (sync, 120 s timeout) — run in a thread pool
    # so the event loop is not blocked during the Ollama round-trip.
    history = [t.model_dump() for t in req.conversation_history] if req.conversation_history else None
    prompt = build_prompt(req.question, payload, history=history)
    try:
        answer = await asyncio.to_thread(query_llm, prompt, SYSTEM_PROMPT)
    except Exception as exc:
        log.error("LLM call failed: %s", exc)
        raise HTTPException(status_code=503, detail=f"LLM unavailable: {exc}")

    # 5. Build structured response fields
    # When trend regime was run, also run vol regime if not already matched
    if "regime" in bucket_results and "volatility_regime" not in bucket_results:
        try:
            bucket_results["volatility_regime"] = vol_regime_mod.run(date_str, {})
        except Exception as exc:
            log.warning("Vol regime auto-run failed for %s: %s", date_str, exc)

    regime_context: RegimeContext | None = None
    if "regime" in bucket_results:
        r = bucket_results["regime"]
        v = bucket_results.get("volatility_regime", {})
        regime_context = RegimeContext(
            date=r.get("date", date_str),
            current_regime=r.get("current_regime", "unknown"),
            regime_probability=r.get("regime_probability"),
            sessions_in_current_regime=r.get("sessions_in_current_regime"),
            regime_start_date=r.get("regime_start_date"),
            prior_regime=r.get("prior_regime"),
            vol_regime=v.get("vol_regime"),
            vol_regime_probability=v.get("vol_regime_probability"),
            vol_regime_sessions=v.get("vol_regime_sessions"),
            vol_regime_start_date=v.get("vol_regime_start_date"),
            prior_vol_regime=v.get("prior_vol_regime"),
            volatility_20d_current=v.get("volatility_20d_current"),
            volatility_20d_percentile=v.get("volatility_20d_percentile"),
            volatility_60d_current=v.get("volatility_60d_current"),
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

    clustering_result: ClusteringResult | None = None
    if "clustering" in bucket_results:
        c = bucket_results["clustering"]
        clustering_result = ClusteringResult(
            date=c.get("date", date_str),
            current_cluster_id=c.get("current_cluster_id"),
            current_cluster_label=c.get("current_cluster_label"),
            distance_to_centroid=c.get("distance_to_centroid"),
            is_outlier=c.get("is_outlier"),
            all_clusters=[
                ClusterInfo(
                    cluster_id=int(cl.get("cluster_id", 0)),
                    label=str(cl.get("label", "")),
                    size=int(cl.get("size", 0)),
                    characteristics=cl.get("characteristics", {}),
                )
                for cl in c.get("all_clusters", [])
            ],
            member_dates_sample=c.get("member_dates_sample", []),
            noise_fraction=c.get("noise_fraction"),
            model_version=c.get("model_version"),
            note=c.get("note"),
        )

    relationships_result: RelationshipsResult | None = None
    if "relationships" in bucket_results:
        rel = bucket_results["relationships"]
        cond = rel.get("conditional")
        relationships_result = RelationshipsResult(
            date=rel.get("date", date_str),
            primary_relationships=[
                RelationshipItem(
                    feature_a=str(p.get("feature_a", "")),
                    feature_b=str(p.get("feature_b", "")),
                    spearman=float(p.get("spearman", 0.0)),
                    direction=str(p.get("direction", "")),
                    sample_size=int(p.get("sample_size", 0)),
                    plain=str(p.get("plain", "")),
                )
                for p in rel.get("primary_relationships", [])
            ],
            conditional=ConditionalRelationship(**cond) if cond else None,
            note=rel.get("note"),
        )

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return QueryResponse(
        answer=answer,
        analytics_used=analytics_used,
        regime_context=regime_context,
        anomaly_assessment=anomaly_assessment,
        similarity_results=similarity_results,
        clustering_result=clustering_result,
        relationships_result=relationships_result,
        data_date=date_str,
        model_versions=model_versions_snapshot(),
        response_time_ms=round(elapsed_ms, 1),
    )
