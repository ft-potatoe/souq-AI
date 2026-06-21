"""
llm/prompts.py
System prompt (spec §11.2) and prompt builder (spec §11.3).
"""

from __future__ import annotations

import json
from typing import Any

from llm.router import estimate_tokens

SYSTEM_PROMPT = """\
You are the QSE Market Copilot, an analytical assistant for the Qatar Stock Exchange.

Rules you must follow without exception:
1. Answer only from the structured JSON data provided in the user message. Do not use any knowledge from your training data to fill in market statistics, prices, returns, or indicators.
2. Never perform arithmetic or infer numbers. If a value is absent from the JSON, say it is not available. Do not cite any statistic (percentile, skewness, quartile, return value, frequency, etc.) that does not appear verbatim in the JSON — not even a plausible-sounding estimate.
3. Do not forecast or predict future prices, returns, or market direction under any circumstances.
4. Present numbers with appropriate precision: percentages to two decimal places, whole counts as integers.
5. When the JSON contains a "date" field, anchor your answer to that date. Do not generalise to other time periods unless explicitly asked.
6. If the question asks about a metric that is not present in the JSON payload, state clearly that the data is not available rather than guessing.
7. Format answers as follows — pick the format that fits the content, never mix formats randomly:
   - Single metric or yes/no: one or two sentences, no list.
   - Comparing two or three values (e.g. today vs. average, gainers vs. losers): a bullet list, one item per value, bold the label.
   - Ranking or listing multiple items (top sessions, GCC peers, flow breakdown): a bullet list, sorted by relevance, bold the key figure on each line.
   - Side-by-side comparison of several attributes across two or more entities: a markdown table with a header row (| Metric | Value A | Value B |) and one row per metric.
   - Narrative explanation of a regime or pattern: short paragraphs (two to three sentences each), no list needed.
   Always use **bold** for the most important number or label in each bullet or sentence.
8. [TESTING MODE] You may reveal internal field names, JSON keys, score values, column names, and raw numeric scores (Spearman ρ, correlation coefficients, ranker_score, knn_cosine_score, etc.) to help the developer verify the data pipeline. Include the raw field name alongside any plain-language description.
9. Use plain financial language. Avoid jargon the user did not introduce.
10. Treat regime labels (bear / sideways / bull) as descriptive summaries of historical patterns only, not as predictions.
11. If a bucket is absent from the JSON payload, do not mention or speculate about it — answer only from what is present.
12. When a payload contains a "data_through" field, that is the latest date in the dataset — not a data gap. If a user asks about a full year or period and the data runs to "data_through", present the result as complete for the available period (e.g. "Jan 1 – Jun 4, the full available 2026 period"). Never say data is "not fully available" or "missing" when the period simply has not ended yet.
    Flow field definitions — never confuse these: "foreign purchases" or "foreign buys" = total_foreign_buy (gross buy-side only). "foreign sales" or "foreign sells" = total_foreign_sell. "foreign net" or "net foreign activity" = total_foreign_net (buy minus sell, can be negative). Always match the user's term to the correct field.
13. For GCC comparison questions: (a) always lead with QSE's rank among peers (e.g. "QSE ranked 3rd out of 6 markets") and the spread figure; (b) always report the rolling outperformance rate and its pre-computed interpretation label — never omit it; (c) never describe a single-day return as an "average return"; (d) all return and spread values in the gcc payload are already in percent — append % directly, never multiply by 100.
14. When the payload contains cluster or day-type data, describe each cluster's historical characteristics and which cluster the current date most resembles. Treat clusters strictly as descriptive groupings of past sessions — never as predictions of future membership or future returns. If the current day is flagged as an outlier, describe it as atypical relative to history, not as a forecast.
15. When the payload contains relationship or association data, report it as observed co-movement only — never as causation. Use the provided plain-language descriptions and state the direction (inverse or direct) and, where present, the conditional frequency and its sample size. Do not claim that one metric causes, drives, or will move another, and never extrapolate the relationship into the future.
16. When the payload contains a "threshold_count" field, report it as the exact integer count of sessions that met the stated condition — never convert it to a fraction or percentage. When a "date_range" block is present, confine the answer to that period and state it explicitly (e.g. "Over the Jan–Jun 2026 period (118 sessions), X days fell more than 2%."). Never extend the answer to the full historical record if a date range was requested.
17. When the correlation payload contains "period_spearman" and "period_pearson" fields, these are the definitive correlation figures for the requested date range — always report them first and prominently (e.g. "Over Jan–Jun 2026, the Spearman correlation was 0.42"). Rolling-window figures (rolling_corr_20d, rolling_corr_60d) reflect only the most recent window ending on the data date — mention them only as supplementary context after the period figure, and never report them as the answer to a date-scoped question. Always label the method: rolling_corr_20d and rolling_corr_60d are Pearson correlations; period_spearman is Spearman rank correlation. Never omit the method name when citing a correlation figure.
18. When the seasonality payload contains "ramadan_effect", report the "pct_difference" field verbatim as the percentage difference — never compute or re-derive it yourself. Do not describe any Ramadan vs non-Ramadan difference as "significant" or "statistically significant" — the system runs no hypothesis test. Use descriptive language only (e.g. "higher", "lower", "notably different").
19. Return values such as return_1d, forward_return_5d, forward_return_10d, best_day.value, worst_day.value, avg_daily_return_*, and any other metric expressed as a decimal fraction of 1 (e.g. 0.0999 means 9.99%) must be multiplied by 100 and presented as a percentage with two decimal places (e.g. "9.99%"). Price levels (close, open, high, low) and volume counts are not percentages — never multiply those. When in doubt, check whether the field name contains "return", "pct", or "change" — those are always decimal fractions requiring the ×100 conversion.
20. When the payload contains volatility regime data, always state the regime label (low_vol or high_vol) and its probability before citing the percentile. For example: "Volatility is currently in the high_vol regime (probability 82%). The 20-day realised volatility sits at the 81st percentile of all historical sessions." Never lead with the percentile number alone without first naming the regime label.
21. When answering a regime question, always report ALL of the following fields if present in the payload — never omit any of them:
    (a) current regime label and probability;
    (b) how many sessions the current regime has lasted (sessions_in_current_regime) and its start date (regime_start_date);
    (c) the prior regime label and how long it lasted (prior_regime_duration_sessions);
    (d) the historical distribution (regime_distribution_historical) — state what fraction of all sessions each regime has occupied, rounded to one decimal place as a percentage (e.g. bear 22%, sideways 65%, bull 13%).
    Present these as a short narrative or bullet list — never drop (b), (c), or (d) from the answer.\
"""


_HISTORY_TURN_LIMIT = 3   # max prior turns to include
_HISTORY_TOKEN_LIMIT_PER_TURN = 100  # ~400 chars for ASCII, scales correctly for Arabic/non-ASCII


def _truncate_to_tokens(text: str, token_limit: int) -> str:
    """Truncate *text* until estimate_tokens(text) fits within *token_limit*."""
    if estimate_tokens(text) <= token_limit:
        return text
    # Binary-search style: trim by ~4 chars per token overage until it fits.
    step = max(1, (estimate_tokens(text) - token_limit) * 4)
    while len(text) > 0 and estimate_tokens(text) > token_limit:
        text = text[:-step]
        step = max(1, step // 2)
    return text.rstrip() + "..."


# ---------------------------------------------------------------------------
# Per-bucket answer reminders injected into the user turn (close to the question)
# These are more reliably followed than late system-prompt rules on small models.
# ---------------------------------------------------------------------------

_CORRELATION_REMINDER = """\
Instruction for this correlation question: the primary pair being analysed is stated \
in the "pair" field. Lead with that pair's rolling correlations and what they mean \
(positive = move together historically, negative = move inversely, near zero = no \
consistent historical link). If gcc_correlations are present, present them as a \
per-market breakdown after the headline figure — not as the primary answer. Always \
state the correlation method (Pearson for rolling windows) and the window length. \
Frame all findings as historical patterns only — never say a market "is likely to" \
or "will" move in any direction. Use "has tended to", "historically", "on average".\
"""

_REGIME_REMINDER = """\
Instruction for this regime question: your answer MUST include all four of these \
(skip any that are absent from the data):
(a) current regime label + probability
(b) how many sessions it has lasted + start date
(c) prior regime label + how long it lasted
(d) historical base rates for each regime (bear / sideways / bull as percentages)
Do not stop after (a).\
"""

_VOL_REMINDER = """\
Instruction for this volatility question: lead with the vol_regime label \
(low_vol or high_vol) and its probability, then give the percentile. \
Never lead with the percentile alone.\
"""

_THRESHOLD_REMINDER = """\
Instruction for this count question: report threshold_count as an exact integer. \
State the date range explicitly. Never say the count is unavailable if \
threshold_count is present in the data.\
"""

_RETURN_REMINDER = """\
Instruction: any value whose field name contains "return", "pct", or "change" \
is a decimal fraction — multiply by 100 and show as %. \
E.g. 0.0999 → "9.99%".\
"""

# ---------------------------------------------------------------------------
# Per-bucket pre-formatters
# These replace raw JSON for the 3 buckets most prone to LLM misreading.
# The LLM receives plain English it only needs to wrap in a sentence —
# no field-name translation, no unit conversion, no rank arithmetic needed.
# ---------------------------------------------------------------------------

def _ordinal(n: int) -> str:
    """1 -> '1st', 2 -> '2nd', 3 -> '3rd', 4 -> '4th', …"""
    if 11 <= (n % 100) <= 13:
        return f"{n}th"
    return f"{n}{['th','st','nd','rd','th','th','th','th','th','th'][n % 10]}"


_COL_NAMES: dict[str, str] = {
    "return_1d":           "daily return",
    "return_5d":           "5-day return",
    "return_20d":          "20-day return",
    "volume":              "trading volume",
    "volume_zscore":       "trading volume",
    "value_traded":        "value traded",
    "value_zscore":        "value traded",
    "foreign_net":         "foreign net flow",
    "foreign_buy":         "foreign buying",
    "foreign_sell":        "foreign selling",
    "foreign_flow_zscore": "foreign flow",
    "domestic_net":        "domestic net flow",
    "domestic_buy":        "domestic buying",
    "domestic_sell":       "domestic selling",
    "breadth_ratio":       "market breadth",
    "breadth_zscore":      "market breadth",
    "breadth_net":         "market breadth",
    "volatility_20d":      "20-day volatility",
    "volatility_60d":      "60-day volatility",
    "rsi_14":              "RSI-14",
    "price_vs_sma20_pct":  "price vs 20-day SMA",
    "gcc_avg_return_1d":   "GCC average return",
    "qse_vs_gcc_spread":   "QSE vs GCC spread",
    "foreign_participation": "foreign participation rate",
}


def _col(name: str) -> str:
    return _COL_NAMES.get(name, name.replace("_", " "))


def _spearman_strength(rho: float) -> str:
    a = abs(rho)
    if a >= 0.70:
        return "strong"
    if a >= 0.50:
        return "moderate"
    return "mild"


def _pct(v: float | None, decimals: int = 2) -> str:
    if v is None:
        return "n/a"
    return f"{v:.{decimals}f}%"


def _format_gcc(g: dict[str, Any]) -> str:
    """Convert the gcc bucket dict to a plain-English summary block."""
    lines: list[str] = [f"GCC Comparison ({g.get('date', 'n/a')}):"]

    rank = g.get("qse_rank_among_all_markets_including_qse")
    total = g.get("total_markets_including_qse")
    qse_ret = g.get("qse_return_1d_pct")
    spread = g.get("qse_vs_gcc_spread_1d_pct")
    gcc_avg = g.get("gcc_avg_return_1d_pct")

    if rank is not None and total is not None:
        lines.append(
            f"- QSE ranked {_ordinal(rank)} out of {total} markets (1st = best, "
            f"{_ordinal(total)} = worst). Do not recompute this rank."
        )
    if qse_ret is not None:
        lines.append(f"- QSE return today: {_pct(qse_ret)}.")
    if gcc_avg is not None:
        lines.append(f"- GCC average return today: {_pct(gcc_avg)}.")
    if spread is not None:
        direction = "outperformed" if spread > 0 else "underperformed"
        lines.append(f"- QSE {direction} the GCC average by {_pct(abs(spread))} today.")

    # Peer returns (sorted best to worst so LLM never needs to re-rank)
    peers = g.get("peer_returns_pct", {})
    if peers:
        sorted_peers = sorted(
            [(mkt, ret) for mkt, ret in peers.items() if ret is not None],
            key=lambda x: x[1], reverse=True
        )
        peer_lines = [f"  {mkt}: {_pct(ret)}" for mkt, ret in sorted_peers]
        if peer_lines:
            lines.append("- Peer returns today (best to worst):")
            lines.extend(peer_lines)

    # Rolling outperformance — find whichever window key is present
    for key, val in g.items():
        if key.startswith("rolling_outperformance_rate_") and val is not None:
            window = key.replace("rolling_outperformance_rate_", "").replace("d", "")
            interp_key = key.replace("rate", "interpretation")
            interp = g.get(interp_key, "")
            lines.append(
                f"- Over the past {window} sessions, QSE outperformed GCC peers "
                f"{round(val * 100, 1)}% of the time ({interp})."
            )

    # Multi-horizon spreads
    for h in [5, 20]:
        k = f"qse_vs_gcc_spread_{h}d_pct"
        if g.get(k) is not None:
            lines.append(f"- Cumulative spread vs GCC over last {h} sessions: {_pct(g[k])}.")

    return "\n".join(lines)


def _format_similarity(s: dict[str, Any]) -> str:
    """Convert the similarity bucket dict to a plain-English summary block."""
    lines: list[str] = [f"Historical Similarity ({s.get('date', 'n/a')}):"]

    matches = s.get("top_matches", [])
    # Drop negative-ranker-score matches (scored below baseline)
    good = [m for m in matches if m.get("ranker_score", 0) >= 0]
    # Show top 5 at most
    shown = good[:5]

    strength_labels = ["strongest", "second closest", "third closest", "fourth closest", "fifth closest"]

    for i, m in enumerate(shown):
        label = strength_labels[i] if i < len(strength_labels) else f"match #{i+1}"
        date = m.get("date", "n/a")
        cosine = m.get("knn_cosine_score", 0)
        # Describe cosine quality
        if cosine >= 0.85:
            quality = "very close"
        elif cosine >= 0.75:
            quality = "close"
        else:
            quality = "moderate"

        fwd5 = m.get("forward_return_5d")
        fwd10 = m.get("forward_return_10d")

        fwd_str = ""
        if fwd5 is None and fwd10 is None:
            fwd_str = "too recent for outcome data"
        else:
            parts = []
            if fwd5 is not None:
                parts.append(f"5-day return after: {_pct(fwd5 * 100)}")
            if fwd10 is not None:
                parts.append(f"10-day return after: {_pct(fwd10 * 100)}")
            fwd_str = "; ".join(parts)

        lines.append(f"- {label.capitalize()} match ({quality}): {date}. {fwd_str}.")

    # Summary of outcomes across matches with data
    fwd5_vals = [m["forward_return_5d"] * 100 for m in shown if m.get("forward_return_5d") is not None]
    fwd10_vals = [m["forward_return_10d"] * 100 for m in shown if m.get("forward_return_10d") is not None]
    if fwd5_vals or fwd10_vals:
        summary_parts = []
        if fwd5_vals:
            avg5 = sum(fwd5_vals) / len(fwd5_vals)
            summary_parts.append(f"average 5-day return {_pct(avg5)}")
        if fwd10_vals:
            avg10 = sum(fwd10_vals) / len(fwd10_vals)
            summary_parts.append(f"average 10-day return {_pct(avg10)}")
        lines.append(
            f"Across the {len(fwd5_vals or fwd10_vals)} similar sessions with available outcome data: "
            + ", ".join(summary_parts)
            + ". This is historical context only — not a prediction."
        )

    return "\n".join(lines)


def _format_relationships(r: dict[str, Any]) -> str:
    """Convert the relationships bucket dict to a plain-English summary block."""
    lines: list[str] = [f"Relationship Scan ({r.get('date', 'n/a')}):"]
    lines.append("(Associations only — co-movement, not causation.)")

    primary = r.get("primary_relationships", [])
    if not primary:
        lines.append("No strong co-movements found above the |ρ| ≥ 0.40 threshold.")
    else:
        lines.append("Strongest co-movements found:")
        for p in primary[:5]:
            a = _col(p.get("feature_a", ""))
            b = _col(p.get("feature_b", ""))
            rho = p.get("spearman", 0.0)
            strength = _spearman_strength(rho)
            direction = "move together" if rho > 0 else "move inversely"
            lines.append(f"- {a.capitalize()} and {b} {direction} ({strength} association).")

    # Conditional analysis — when both low and high decile are present for the
    # same pair, emit a side-by-side comparison so the LLM can compare them
    # against each other (not just against the 50% baseline individually).
    cond = r.get("conditional")
    cond_high = r.get("conditional_high")

    both_present = (
        cond and cond_high
        and cond.get("result_pct") is not None
        and cond_high.get("result_pct") is not None
        and cond.get("given") == cond_high.get("given")
        and cond.get("observed") == cond_high.get("observed")
    )

    if both_present:
        given = _col(cond.get("given", ""))
        observed = _col(cond.get("observed", ""))
        low_pct = cond["result_pct"]
        high_pct = cond_high["result_pct"]
        low_n = cond["sample_size"]
        high_n = cond_high["sample_size"]
        # Direction of the gap — which extreme favours the observed metric
        if high_pct > low_pct:
            gap_note = (
                f"{observed.capitalize()} is MORE likely to be above its median "
                f"during HIGH {given} ({high_pct}%) than during LOW {given} ({low_pct}%). "
                f"The relevant comparison is the {round(high_pct - low_pct, 1)} percentage-point gap "
                f"between the two extremes, not just each figure vs the 50% baseline."
            )
        elif low_pct > high_pct:
            gap_note = (
                f"{observed.capitalize()} is MORE likely to be above its median "
                f"during LOW {given} ({low_pct}%) than during HIGH {given} ({high_pct}%). "
                f"The relevant comparison is the {round(low_pct - high_pct, 1)} percentage-point gap "
                f"between the two extremes, not just each figure vs the 50% baseline."
            )
        else:
            gap_note = f"{observed.capitalize()} shows no difference between low and high {given} periods."

        lines.append(
            f"Conditional analysis — {observed} vs {given} extremes:\n"
            f"  Low {given} (bottom 10%, {low_n} sessions):  {observed} above median {low_pct}% of the time\n"
            f"  High {given} (top 10%, {high_n} sessions): {observed} above median {high_pct}% of the time\n"
            f"  Baseline: 50%\n"
            f"  Interpretation: {gap_note}"
        )
    else:
        # Only one side available — fall back to individual lines
        if cond and cond.get("result_pct") is not None:
            given = _col(cond.get("given", ""))
            observed = _col(cond.get("observed", ""))
            lines.append(
                f"When {given} is in its lowest 10% ({cond['sample_size']} sessions), "
                f"{observed} is above its historical median {cond['result_pct']}% of the time "
                f"(vs 50% baseline)."
            )
        if cond_high and cond_high.get("result_pct") is not None:
            given = _col(cond_high.get("given", ""))
            observed = _col(cond_high.get("observed", ""))
            lines.append(
                f"When {given} is in its highest 10% ({cond_high['sample_size']} sessions), "
                f"{observed} is above its historical median {cond_high['result_pct']}% of the time "
                f"(vs 50% baseline)."
            )

    return "\n".join(lines)


def _bucket_reminders(payload: dict[str, Any]) -> str | None:
    """Return a short instruction block tailored to the buckets present in payload."""
    parts = []
    # gcc / similarity / relationships are now pre-formatted — no reminders needed
    if "correlation" in payload:
        parts.append(_CORRELATION_REMINDER)
    if "regime" in payload:
        parts.append(_REGIME_REMINDER)
    if "volatility_regime" in payload and "regime" not in payload:
        parts.append(_VOL_REMINDER)
    if "distribution" in payload and "threshold_count" in payload.get("distribution", {}):
        parts.append(_THRESHOLD_REMINDER)
    if "summary" in payload:
        parts.append(_RETURN_REMINDER)
    return "\n\n".join(parts) if parts else None


def build_prompt(
    question: str,
    payload: dict[str, Any],
    history: list[dict[str, str]] | None = None,
) -> str:
    """Return the user-turn prompt containing analytics payload, optional history, and the question.

    gcc, similarity, and relationships buckets are pre-formatted into plain English
    before serialisation so the LLM never sees raw field names or internal scores
    for those buckets.
    """
    # Pre-format the three complex buckets; remove them from the raw JSON payload
    # so they don't appear twice.
    formatted_blocks: list[str] = []
    raw_payload = dict(payload)  # shallow copy — we'll pop formatted keys

    if "gcc" in raw_payload:
        formatted_blocks.append(_format_gcc(raw_payload.pop("gcc")))
    if "similarity" in raw_payload:
        formatted_blocks.append(_format_similarity(raw_payload.pop("similarity")))
    if "relationships" in raw_payload:
        formatted_blocks.append(_format_relationships(raw_payload.pop("relationships")))

    parts: list[str] = []

    # Pre-formatted blocks first (plain English, easy for the LLM to read)
    if formatted_blocks:
        parts.append("\n\n".join(formatted_blocks))

    # Remaining buckets as raw JSON (distribution, trend, regime, etc.)
    if raw_payload:
        serialised = json.dumps(raw_payload, default=str, separators=(",", ":"))
        parts.append(f"Analytics data:\n{serialised}")

    if history:
        lines = []
        for turn in history[-_HISTORY_TURN_LIMIT:]:
            role = turn.get("role", "")
            content = turn.get("content", "")
            content = _truncate_to_tokens(content, _HISTORY_TOKEN_LIMIT_PER_TURN)
            if role == "user":
                lines.append(f"User: {content}")
            elif role == "assistant":
                lines.append(f"Assistant: {content}")
        if lines:
            parts.append("Conversation so far:\n" + "\n".join(lines))

    reminders = _bucket_reminders(payload)
    if reminders:
        parts.append(reminders)

    parts.append(f"Question: {question}")
    return "\n\n".join(parts)
