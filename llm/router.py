"""
llm/router.py
Maps natural-language questions to analytics buckets, assembles the LLM
payload within the 3500-token budget, and compresses or drops buckets as
needed (spec §12.4).
"""

from __future__ import annotations

import copy
import json
from typing import Any

# ---------------------------------------------------------------------------
# Bucket keyword index  (spec §12.4)
# ---------------------------------------------------------------------------

BUCKET_KEYWORDS: dict[str, list[str]] = {
    "summary": [
        "all time", "all-time", "record", "highest ever", "lowest ever",
        "best day", "worst day", "biggest gain", "biggest loss", "biggest drop",
        "52 week", "52-week", "year high", "year low", "ytd", "year to date",
        "winning streak", "losing streak", "highest close", "lowest close",
        "highest volume", "peak volume", "average daily return",
        "how many up days", "how many down days",
    ],
    "anomaly": [
        "anomal", "unusual", "outlier", "spike", "abnormal", "strange",
        "weird", "extreme", "flagged", "alert",
    ],
    "regime": [
        "regime", "bull", "bear", "sideways", "market state", "trend state",
        "market phase", "hmm",
    ],
    "volatility_regime": [
        "volatil", "vol regime", "low vol", "high vol", "low volatility", "high volatility",
        "vix", "risk environment", "options", "implied vol", "realized vol",
        "vol spike", "vol elevated", "vol compressed", "vol percentile",
        "annualised vol", "annualized vol", "market risk", "vol regime",
    ],
    "flows": [
        "flow", "foreign", "domestic", "buying", "buy volume", "buy pressure",
        "foreign buy", "domestic buy", "selling", "sold", "pressure",
        "inflow", "outflow", "investor", "sell volume", "sell pressure",
        "turnover", "participation",
    ],
    "distribution": [
        "distribution", "percentile", "histogram", "percentile rank",
        "z-score", "zscore", "return range", "historical range", "return spread",
        "skew", "kurtosis", "how high", "how low", "relative to",
    ],
    "trend": [
        "price trend", "market trend", "index trend", "momentum", "sma",
        "moving average", "slope", "macd", "rsi", "bollinger", "atr",
        "above sma", "below sma", "technical", "overbought", "oversold",
    ],
    "similarity": [
        "similar", "like this before", "historical analog", "analog",
        "comparable", "past session", "resemble", "historical match", "has this happened",
    ],
    "gcc": [
        "gcc", "gulf", "regional", "peer market", "peer comparison", "gcc peer",
        "peer performance", "saudi", "tasi", "adx", "dfm",
        "kse", "msm", "bse", "tadawul", "compare to", "vs", "versus",
    ],
    "correlation": [
        "correlat", "relationship", "linked", "co-move", "move together", "rolling corr",
        "map pattern", "pattern between", "relationship between", "link between",
        "connection between", "interact", "drives", "driven by", "tied to",
        "relate to", "related to", "association", "in sync", "tracks", "track",
    ],
    "seasonality": [
        "season", "day of week", "monday effect", "weekly pattern",
        "monthly", "ramadan", "time of year", "calendar",
    ],
}

# Buckets ordered from highest to lowest priority when trimming for budget.
BUCKET_PRIORITY: list[str] = [
    "anomaly",
    "regime",
    "volatility_regime",
    "summary",
    "flows",
    "distribution",
    "trend",
    "similarity",
    "gcc",
    "correlation",
    "seasonality",
]

PAYLOAD_TOKEN_BUDGET = 3500

# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

def estimate_tokens(obj: Any) -> int:
    """Rough token count: one token per four characters of JSON."""
    return len(json.dumps(obj, default=str, separators=(",", ":"))) // 4


# ---------------------------------------------------------------------------
# Compression strategies per bucket  (spec §12.4)
# ---------------------------------------------------------------------------

def compress_bucket(bucket: str, result: dict[str, Any], token_limit: int) -> dict[str, Any] | None:
    """
    Return a compressed version of *result* that fits within *token_limit*
    tokens, or None if this bucket cannot be compressed.
    """
    if bucket == "similarity":
        compressed = copy.deepcopy(result)
        if "top_matches" in compressed:
            compressed["top_matches"] = [
                {k: v for k, v in m.items() if k != "forward_return_stats"}
                for m in compressed["top_matches"][:3]
            ]
        if estimate_tokens(compressed) <= token_limit:
            return compressed
        return None

    if bucket == "gcc":
        compressed = copy.deepcopy(result)
        compressed.pop("per_peer_breakdown", None)
        if estimate_tokens(compressed) <= token_limit:
            return compressed
        return None

    if bucket == "seasonality":
        compressed = {
            k: result[k]
            for k in ("today_day_of_week", "day_of_week_rank")
            if k in result
        }
        if estimate_tokens(compressed) <= token_limit:
            return compressed
        return None

    if bucket == "correlation":
        compressed = {
            k: result[k]
            for k in ("rolling_corr_20d", "percentile_of_current_corr")
            if k in result
        }
        if estimate_tokens(compressed) <= token_limit:
            return compressed
        return None

    if bucket == "summary":
        # Drop per-metric records for volume/return cols; keep price records + extremes
        compressed = copy.deepcopy(result)
        keep_metrics = ["close", "volume", "value_traded"]
        if "records" in compressed:
            compressed["records"] = {
                k: v for k, v in compressed["records"].items() if k in keep_metrics
            }
        if estimate_tokens(compressed) <= token_limit:
            return compressed
        # Further compress: drop 52_week and ytd, keep all_time only
        if "records" in compressed:
            for m in compressed["records"]:
                compressed["records"][m] = {"all_time": compressed["records"][m].get("all_time")}
        compressed.pop("avg_daily_return_52w", None)
        compressed.pop("avg_daily_return_ytd", None)
        compressed.pop("up_days_ytd", None)
        compressed.pop("down_days_ytd", None)
        if estimate_tokens(compressed) <= token_limit:
            return compressed
        return None

    # All other buckets: no compression strategy defined
    return None


# ---------------------------------------------------------------------------
# Payload assembly  (spec §12.4)
# ---------------------------------------------------------------------------

def build_llm_payload(
    matched_buckets: list[str],
    results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """
    Assemble a token-budget-respecting payload dict from analytics results.

    Strategy (in priority order):
    1. Try to include the bucket as-is.
    2. If it pushes over budget, try compress_bucket().
    3. If compressed result still exceeds budget, drop the bucket.
    Buckets not in matched_buckets are never included.
    """
    payload: dict[str, Any] = {}

    for bucket in BUCKET_PRIORITY:
        if bucket not in matched_buckets:
            continue
        result = results.get(bucket)
        if result is None:
            continue

        # Probe: measure tokens of the full payload with this bucket added
        payload[bucket] = result
        if estimate_tokens(payload) <= PAYLOAD_TOKEN_BUDGET:
            continue

        # Over budget with full result — try compression
        remaining = PAYLOAD_TOKEN_BUDGET - estimate_tokens({k: v for k, v in payload.items() if k != bucket})
        compressed = compress_bucket(bucket, result, remaining)
        if compressed is not None:
            payload[bucket] = compressed
        else:
            del payload[bucket]

    return payload


# ---------------------------------------------------------------------------
# Question -> bucket matching
# ---------------------------------------------------------------------------

_DEFAULT_BUCKETS: list[str] = ["trend", "distribution", "regime", "volatility_regime"]


def match_buckets(question: str) -> list[str]:
    """Return all bucket names whose keywords appear in *question* (case-insensitive).

    Falls back to _DEFAULT_BUCKETS when no keywords match, so the LLM always
    receives real data rather than an empty payload (which produces a generic greeting).
    """
    q = question.lower()
    matched = [
        bucket
        for bucket, keywords in BUCKET_KEYWORDS.items()
        if any(kw in q for kw in keywords)
    ]
    return matched if matched else _DEFAULT_BUCKETS
