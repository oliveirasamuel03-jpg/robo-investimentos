from __future__ import annotations

from collections import Counter
from typing import Any


DIAGNOSTIC_MODE = "DIAGNOSTIC_ONLY"

SCORE_BELOW_MIN = "SCORE_BELOW_MIN"
MOMENTUM_WEAK = "MOMENTUM_WEAK"
SECONDARY_CONFIRMATION_WEAK = "SECONDARY_CONFIRMATION_WEAK"
RSI_OUT_OF_RANGE = "RSI_OUT_OF_RANGE"
TREND_NOT_CONFIRMED = "TREND_NOT_CONFIRMED"
VOLATILITY_FILTER = "VOLATILITY_FILTER"
CONTEXT_FILTER = "CONTEXT_FILTER"
FEED_BLOCK = "FEED_BLOCK"
GUARD_BLOCK = "GUARD_BLOCK"
UNKNOWN = "UNKNOWN"

COUNT_FIELDS = {
    SCORE_BELOW_MIN: "score_below_min_count",
    MOMENTUM_WEAK: "momentum_weak_count",
    SECONDARY_CONFIRMATION_WEAK: "secondary_confirmation_weak_count",
    RSI_OUT_OF_RANGE: "rsi_out_of_range_count",
    TREND_NOT_CONFIRMED: "trend_not_confirmed_count",
    VOLATILITY_FILTER: "volatility_filter_count",
    CONTEXT_FILTER: "context_filter_count",
    FEED_BLOCK: "feed_block_count",
    GUARD_BLOCK: "guard_block_count",
    UNKNOWN: "unknown_count",
}

STRATEGY_CATEGORIES = {
    SCORE_BELOW_MIN,
    MOMENTUM_WEAK,
    SECONDARY_CONFIRMATION_WEAK,
    RSI_OUT_OF_RANGE,
    TREND_NOT_CONFIRMED,
    VOLATILITY_FILTER,
    CONTEXT_FILTER,
}

REASON_CATEGORY_MAP = {
    "score_below_minimum": SCORE_BELOW_MIN,
    "weak_score": SCORE_BELOW_MIN,
    "confidence_too_low": MOMENTUM_WEAK,
    "momentum_filter": MOMENTUM_WEAK,
    "breakout_not_confirmed": SECONDARY_CONFIRMATION_WEAK,
    "no_setup_eligible": SECONDARY_CONFIRMATION_WEAK,
    "structure_filter": SECONDARY_CONFIRMATION_WEAK,
    "reversal_not_eligible": RSI_OUT_OF_RANGE,
    "rsi_filter": RSI_OUT_OF_RANGE,
    "trend_not_confirmed": TREND_NOT_CONFIRMED,
    "against_trend": TREND_NOT_CONFIRMED,
    "volatility_out_of_range": VOLATILITY_FILTER,
    "atr_filter": VOLATILITY_FILTER,
    "context_blocked": CONTEXT_FILTER,
    "fallback_blocked": FEED_BLOCK,
    "feed_quality_blocked": FEED_BLOCK,
    "provider_unknown": FEED_BLOCK,
    "cooldown_active": GUARD_BLOCK,
    "duplicate_signal_blocked": GUARD_BLOCK,
    "position_limit_reached": GUARD_BLOCK,
    "schedule_blocked": GUARD_BLOCK,
    "daily_loss_guard": GUARD_BLOCK,
    "macro_alert_guard": GUARD_BLOCK,
}

TEXT_CATEGORY_PATTERNS = (
    ("score ajustado", SCORE_BELOW_MIN),
    ("minimo exigido", SCORE_BELOW_MIN),
    ("momentum", MOMENTUM_WEAK),
    ("confirmacao secundaria", SECONDARY_CONFIRMATION_WEAK),
    ("breakout", SECONDARY_CONFIRMATION_WEAK),
    ("estrutura elegivel", SECONDARY_CONFIRMATION_WEAK),
    ("rsi", RSI_OUT_OF_RANGE),
    ("tendencia", TREND_NOT_CONFIRMED),
    ("volatilidade", VOLATILITY_FILTER),
    ("contexto", CONTEXT_FILTER),
    ("fallback", FEED_BLOCK),
    ("feed", FEED_BLOCK),
    ("origem do dado", FEED_BLOCK),
    ("cooldown", GUARD_BLOCK),
    ("limite de posicoes", GUARD_BLOCK),
    ("trava de perda", GUARD_BLOCK),
    ("alerta macro", GUARD_BLOCK),
)


def default_strategy_bottleneck_state(
    *,
    enabled: bool = True,
    reason: str = "No strategy bottleneck data yet.",
) -> dict[str, Any]:
    return {
        "enabled": bool(enabled),
        "mode": DIAGNOSTIC_MODE,
        "dominant_bottleneck": "",
        "dominant_setup": "",
        "dominant_asset": "",
        "total_strategy_rejections": 0,
        "score_below_min_count": 0,
        "momentum_weak_count": 0,
        "secondary_confirmation_weak_count": 0,
        "rsi_out_of_range_count": 0,
        "trend_not_confirmed_count": 0,
        "volatility_filter_count": 0,
        "context_filter_count": 0,
        "feed_block_count": 0,
        "guard_block_count": 0,
        "unknown_count": 0,
        "top_assets_blocked": [],
        "top_setups_blocked": [],
        "top_filter_reasons": [],
        "closest_candidates": [],
        "recommendation": "observe_more",
        "reason": reason,
    }


def _safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_reason_values(raw: Any) -> list[str]:
    if isinstance(raw, str):
        values = [chunk.strip() for chunk in raw.split(",")]
    elif isinstance(raw, (list, tuple, set)):
        values = [str(item).strip() for item in raw]
    else:
        values = []
    return [value for value in values if value]


def _category_for_reason(reason: str) -> str:
    normalized = str(reason or "").strip().lower()
    if not normalized:
        return UNKNOWN
    if normalized in REASON_CATEGORY_MAP:
        return REASON_CATEGORY_MAP[normalized]
    for pattern, category in TEXT_CATEGORY_PATTERNS:
        if pattern in normalized:
            return category
    return UNKNOWN


def _primary_category(categories: list[str]) -> str:
    if not categories:
        return UNKNOWN
    for category in (
        SCORE_BELOW_MIN,
        TREND_NOT_CONFIRMED,
        MOMENTUM_WEAK,
        SECONDARY_CONFIRMATION_WEAK,
        RSI_OUT_OF_RANGE,
        VOLATILITY_FILTER,
        CONTEXT_FILTER,
        FEED_BLOCK,
        GUARD_BLOCK,
        UNKNOWN,
    ):
        if category in categories:
            return category
    return categories[0]


def classify_strategy_bottleneck(signal: dict[str, Any]) -> dict[str, Any]:
    payload = dict(signal or {})
    reasons = _safe_reason_values(payload.get("rejection_reasons"))
    categories = [_category_for_reason(reason) for reason in reasons] or [UNKNOWN]
    unique_categories = list(dict.fromkeys(categories))
    primary = _primary_category(unique_categories)
    score = _safe_float(payload.get("score"))
    min_score = _safe_float(payload.get("effective_min_signal_score"))
    if min_score is None:
        min_score = _safe_float(payload.get("base_min_signal_score"))
    gap = None
    if score is not None and min_score is not None:
        gap = round(max(0.0, float(min_score) - float(score)), 6)

    return {
        "asset": str(payload.get("asset") or payload.get("symbol") or "").upper(),
        "setup": str(payload.get("strategy_name") or "trend_pullback_breakout"),
        "score": score,
        "min_score": min_score,
        "gap": gap,
        "reason": ", ".join(reasons[:3]) if reasons else "unknown",
        "bottleneck_category": primary,
        "bottleneck_categories": unique_categories,
        "feed_status": str(payload.get("data_source") or payload.get("feed_status") or ""),
        "context_status": str(payload.get("context_status") or ""),
        "macro_alert_active": bool(payload.get("macro_alert_active", False)),
        "status": DIAGNOSTIC_MODE,
    }


def _top_items(counter: Counter[str], total: int, *, limit: int = 5) -> list[dict[str, Any]]:
    if total <= 0:
        return []
    return [
        {"name": name, "count": int(count), "pct": round(float(count / total), 4)}
        for name, count in counter.most_common(limit)
        if name
    ]


def build_strategy_bottleneck_summary(
    classifications: list[dict[str, Any]],
    *,
    enabled: bool = True,
    max_candidates: int = 10,
) -> dict[str, Any]:
    if not enabled:
        return default_strategy_bottleneck_state(enabled=False, reason="Strategy bottleneck diagnostic disabled.")
    if not classifications:
        return default_strategy_bottleneck_state(enabled=True)

    category_counter: Counter[str] = Counter()
    asset_counter: Counter[str] = Counter()
    setup_counter: Counter[str] = Counter()
    strategy_related: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []

    for item in classifications:
        categories = list(item.get("bottleneck_categories", []) or [UNKNOWN])
        for category in categories:
            category_counter[str(category or UNKNOWN)] += 1
        if any(category in STRATEGY_CATEGORIES for category in categories):
            strategy_related.append(item)
            if item.get("asset"):
                asset_counter[str(item.get("asset"))] += 1
            if item.get("setup"):
                setup_counter[str(item.get("setup"))] += 1
        if item.get("score") is not None and item.get("min_score") is not None:
            candidates.append(
                {
                    "asset": item.get("asset", ""),
                    "setup": item.get("setup", ""),
                    "score": item.get("score"),
                    "min_score": item.get("min_score"),
                    "gap": item.get("gap"),
                    "reason": item.get("reason", ""),
                    "bottleneck_category": item.get("bottleneck_category", UNKNOWN),
                    "feed_status": item.get("feed_status", ""),
                    "context_status": item.get("context_status", ""),
                    "macro_alert_active": bool(item.get("macro_alert_active", False)),
                    "status": DIAGNOSTIC_MODE,
                }
            )

    strategy_total = len(strategy_related)
    top_filter_reasons = _top_items(category_counter, sum(category_counter.values()), limit=5)
    dominant_bottleneck = top_filter_reasons[0]["name"] if top_filter_reasons else ""
    closest_candidates = sorted(
        candidates,
        key=lambda item: (
            float(item.get("gap")) if item.get("gap") is not None else 999.0,
            -float(item.get("score") or 0.0),
        ),
    )[: max(0, min(int(max_candidates or 0), 10))]

    if strategy_total <= 0:
        recommendation = "observe_more"
        reason = "No strategy bottleneck sample yet; do not calibrate thresholds."
    elif strategy_total < 20:
        recommendation = "observe_more"
        reason = "Strategy bottleneck sample is still small; continue observing before calibration."
    else:
        recommendation = "review_bottleneck_distribution"
        reason = (
            f"Dominant bottleneck is {dominant_bottleneck or UNKNOWN}. "
            "Diagnostic only; no strategy or threshold change was applied."
        )

    summary = default_strategy_bottleneck_state(enabled=True, reason=reason)
    summary.update(
        {
            "dominant_bottleneck": dominant_bottleneck,
            "dominant_setup": setup_counter.most_common(1)[0][0] if setup_counter else "",
            "dominant_asset": asset_counter.most_common(1)[0][0] if asset_counter else "",
            "total_strategy_rejections": int(strategy_total),
            "top_assets_blocked": _top_items(asset_counter, strategy_total, limit=5),
            "top_setups_blocked": _top_items(setup_counter, strategy_total, limit=5),
            "top_filter_reasons": top_filter_reasons,
            "closest_candidates": closest_candidates,
            "recommendation": recommendation,
            "reason": reason,
        }
    )
    for category, field in COUNT_FIELDS.items():
        summary[field] = int(category_counter.get(category, 0))
    return summary


def summarize_strategy_bottlenecks(
    *,
    signals: list[dict[str, Any]] | None,
    enabled: bool = True,
    max_candidates: int = 10,
) -> dict[str, Any]:
    if not enabled:
        return default_strategy_bottleneck_state(enabled=False, reason="Strategy bottleneck diagnostic disabled.")
    rejected_signals = [dict(signal or {}) for signal in (signals or []) if not bool((signal or {}).get("buy", False))]
    classifications = [classify_strategy_bottleneck(signal) for signal in rejected_signals]
    return build_strategy_bottleneck_summary(
        classifications,
        enabled=True,
        max_candidates=max_candidates,
    )
