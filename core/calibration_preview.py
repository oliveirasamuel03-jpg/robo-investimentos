from __future__ import annotations

from collections import Counter
from typing import Any


PREVIEW_MODE = "PREVIEW_ONLY"
SAFE_CONTEXT_STATUSES = {"FAVORAVEL", "NEUTRO"}
STRATEGY_REJECTION_REASONS = {
    "score_below_minimum",
    "trend_not_confirmed",
    "breakout_not_confirmed",
    "reversal_not_eligible",
    "volatility_out_of_range",
    "no_setup_eligible",
    "confidence_too_low",
    "strategy_conflict",
}
RISK_CRITICAL_REJECTION_REASONS = {
    "feed_quality_blocked",
    "fallback_blocked",
    "provider_unknown",
    "context_blocked",
    "daily_loss_guard",
    "macro_alert_guard",
    "cooldown_active",
    "duplicate_signal_blocked",
    "position_limit_reached",
    "schedule_blocked",
}


def default_calibration_preview_state(
    *,
    enabled: bool = True,
    reason: str = "No calibration preview data yet.",
) -> dict[str, Any]:
    return {
        "enabled": bool(enabled),
        "mode": PREVIEW_MODE,
        "near_approved_count": 0,
        "near_approved_rate": 0.0,
        "min_score_current": None,
        "preview_score_floor": None,
        "avg_score_gap": None,
        "best_score_seen": None,
        "top_asset": "",
        "top_setup": "",
        "safe_conditions_met_count": 0,
        "unsafe_conditions_count": 0,
        "recommendation": "observe_more",
        "reason": reason,
        "near_approved_examples": [],
    }


def _safe_float(value: Any, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value or default)
    except (TypeError, ValueError):
        return default


def _normalized_list(raw: Any) -> list[str]:
    if isinstance(raw, str):
        values = [chunk.strip() for chunk in raw.split(",")]
    elif isinstance(raw, (list, tuple, set)):
        values = list(raw)
    else:
        values = []
    return [str(item).strip().upper() for item in values if str(item).strip()]


def _signal_rejection_reasons(signal: dict[str, Any]) -> set[str]:
    raw = signal.get("rejection_reasons", []) or []
    if isinstance(raw, str):
        values = [chunk.strip() for chunk in raw.split(",")]
    else:
        values = list(raw) if isinstance(raw, (list, tuple, set)) else []
    return {str(item).strip().lower() for item in values if str(item).strip()}


def _open_positions_count(raw_positions: Any) -> int:
    if isinstance(raw_positions, dict):
        return len(raw_positions)
    if isinstance(raw_positions, list):
        count = 0
        for position in raw_positions:
            payload = dict(position or {}) if isinstance(position, dict) else {}
            status = str(payload.get("status") or "OPEN").upper()
            if status == "OPEN":
                count += 1
        return count
    return 0


def _broker_is_paper(broker_state: dict[str, Any]) -> bool:
    values = [
        broker_state.get("effective_mode"),
        broker_state.get("mode"),
        broker_state.get("configured_mode"),
        broker_state.get("provider"),
    ]
    return any(str(value or "").strip().lower() == "paper" for value in values)


def _build_example(signal: dict[str, Any], *, score: float, min_score: float, score_gap: float) -> dict[str, Any]:
    reasons = sorted(_signal_rejection_reasons(signal))
    return {
        "symbol": str(signal.get("asset") or signal.get("symbol") or "").upper(),
        "strategy": str(signal.get("strategy_name") or "trend_pullback_breakout"),
        "score": round(float(score), 4),
        "min_score": round(float(min_score), 4),
        "score_gap": round(float(score_gap), 4),
        "side": "BUY_PREVIEW",
        "status": PREVIEW_MODE,
        "reason": ", ".join(reasons[:3]) if reasons else "strategy_preview",
        "timestamp": str(signal.get("signal_timestamp") or signal.get("timestamp") or ""),
    }


def classify_near_approved_signal(
    signal: dict[str, Any],
    *,
    feed_status: str,
    fallback_assets_count: int,
    macro_alert_active: bool,
    broker_is_paper: bool,
    watchlist: list[str],
    open_positions_count: int,
    max_open_positions: int,
    daily_loss_block_active: bool,
    preview_margin: float,
) -> dict[str, Any]:
    payload = dict(signal or {})
    reasons = _signal_rejection_reasons(payload)
    score = _safe_float(payload.get("score"), None)
    min_score = _safe_float(payload.get("effective_min_signal_score"), None)
    if min_score is None:
        min_score = _safe_float(payload.get("base_min_signal_score"), None)
    context_status = str(payload.get("context_status") or "NEUTRO").upper()
    asset = str(payload.get("asset") or payload.get("symbol") or "").upper()
    unsafe_reasons: list[str] = []

    if bool(payload.get("buy", False)):
        unsafe_reasons.append("already_approved_by_current_logic")
    if str(feed_status or "").upper() != "LIVE" or int(fallback_assets_count or 0) > 0:
        unsafe_reasons.append("feed_not_clean_live")
    if bool(macro_alert_active) or bool(payload.get("macro_alert_active", False)):
        unsafe_reasons.append("macro_alert_active")
    if context_status not in SAFE_CONTEXT_STATUSES:
        unsafe_reasons.append("context_not_safe")
    if not broker_is_paper:
        unsafe_reasons.append("broker_not_paper")
    if asset not in set(watchlist):
        unsafe_reasons.append("asset_outside_watchlist")
    if int(max_open_positions or 0) <= 0 or int(open_positions_count or 0) >= int(max_open_positions or 0):
        unsafe_reasons.append("position_limit_would_block")
    if bool(daily_loss_block_active):
        unsafe_reasons.append("daily_loss_lock_active")
    if reasons & RISK_CRITICAL_REJECTION_REASONS:
        unsafe_reasons.append("risk_critical_rejection")
    if not (reasons & STRATEGY_REJECTION_REASONS):
        unsafe_reasons.append("no_strategy_rejection_reason")
    if score is None or min_score is None:
        unsafe_reasons.append("missing_score")

    score_gap = None if score is None or min_score is None else round(float(min_score - score), 6)
    within_margin = bool(score_gap is not None and 0.0 < score_gap <= float(preview_margin or 0.0))
    safe_conditions_met = not unsafe_reasons
    near_approved = bool(safe_conditions_met and within_margin)

    return {
        "near_approved": near_approved,
        "safe_conditions_met": safe_conditions_met,
        "unsafe_reasons": unsafe_reasons,
        "score": score,
        "min_score": min_score,
        "score_gap": score_gap,
        "within_margin": within_margin,
        "example": _build_example(payload, score=score or 0.0, min_score=min_score or 0.0, score_gap=score_gap or 0.0)
        if near_approved
        else {},
    }


def summarize_calibration_preview(
    classifications: list[dict[str, Any]],
    *,
    enabled: bool,
    margin: float,
    max_examples: int,
) -> dict[str, Any]:
    if not enabled:
        return default_calibration_preview_state(enabled=False, reason="Calibration preview disabled by configuration.")

    rejected_count = len(classifications)
    if rejected_count <= 0:
        return default_calibration_preview_state(enabled=True, reason="No rejected signal sample available for preview yet.")

    near_items = [item for item in classifications if bool(item.get("near_approved"))]
    safe_items = [item for item in classifications if bool(item.get("safe_conditions_met"))]
    score_values = [float(item["score"]) for item in classifications if item.get("score") is not None]
    min_scores = [float(item["min_score"]) for item in classifications if item.get("min_score") is not None]
    gaps = [float(item["score_gap"]) for item in near_items if item.get("score_gap") is not None]
    examples = [dict(item.get("example") or {}) for item in near_items if item.get("example")]
    examples = examples[: max(0, int(max_examples or 0))]
    asset_counter = Counter(example.get("symbol", "") for example in examples if example.get("symbol"))
    setup_counter = Counter(example.get("strategy", "") for example in examples if example.get("strategy"))
    near_count = len(near_items)
    near_rate = round(float(near_count / rejected_count), 4) if rejected_count > 0 else 0.0
    min_score_current = max(min_scores) if min_scores else None
    preview_score_floor = None
    if min_score_current is not None:
        preview_score_floor = round(max(0.0, float(min_score_current) - float(margin or 0.0)), 4)

    if near_count <= 0:
        recommendation = "observe_more"
        reason = "No safe near-approved signals were found in this cycle. No threshold change is recommended."
    elif len(safe_items) < 5:
        recommendation = "observe_more"
        reason = "Near-approved signals exist, but the safe sample is still too small for calibration."
    elif near_rate >= 0.2:
        recommendation = "review_selectivity"
        reason = "Preview shows a material near-approved sample; study future calibration without changing thresholds now."
    else:
        recommendation = "observe_more"
        reason = "Near-approved signals exist, but more cycles are needed before any calibration decision."

    return {
        "enabled": True,
        "mode": PREVIEW_MODE,
        "near_approved_count": near_count,
        "near_approved_rate": near_rate,
        "min_score_current": None if min_score_current is None else round(float(min_score_current), 4),
        "preview_score_floor": preview_score_floor,
        "avg_score_gap": None if not gaps else round(sum(gaps) / len(gaps), 4),
        "best_score_seen": None if not score_values else round(max(score_values), 4),
        "top_asset": asset_counter.most_common(1)[0][0] if asset_counter else "",
        "top_setup": setup_counter.most_common(1)[0][0] if setup_counter else "",
        "safe_conditions_met_count": len(safe_items),
        "unsafe_conditions_count": max(0, rejected_count - len(safe_items)),
        "recommendation": recommendation,
        "reason": reason,
        "near_approved_examples": examples,
    }


def build_calibration_preview(
    *,
    signals: list[dict[str, Any]] | None,
    state: dict[str, Any] | None,
    paper_state: dict[str, Any] | None = None,
    market_data_status: dict[str, Any] | None = None,
    macro_alert: dict[str, Any] | None = None,
    enabled: bool = True,
    margin: float = 0.04,
    max_examples: int = 10,
) -> dict[str, Any]:
    if not enabled:
        return default_calibration_preview_state(enabled=False, reason="Calibration preview disabled by configuration.")

    root_state = dict(state or {})
    market_state = dict(market_data_status or root_state.get("market_data", {}) or {})
    feed_status = str(market_state.get("feed_status") or market_state.get("status") or "UNKNOWN").upper()
    fallback_assets_count = _safe_int(len(market_state.get("fallback_symbols", []) or []))
    if not fallback_assets_count:
        source_breakdown = dict(market_state.get("source_breakdown", {}) or {})
        fallback_assets_count = _safe_int(source_breakdown.get("fallback"))

    trader_state = dict(root_state.get("trader", {}) or {})
    broker_state = dict(root_state.get("broker", {}) or {})
    risk_state = dict(root_state.get("risk", {}) or {})
    macro_state = dict(macro_alert or root_state.get("macro_alert", {}) or {})
    active_paper_state = dict(paper_state or {})
    watchlist = _normalized_list(trader_state.get("watchlist", []))
    open_positions_count = _open_positions_count(
        active_paper_state.get("positions") if active_paper_state else root_state.get("positions")
    )
    max_open_positions = _safe_int(trader_state.get("max_open_positions"), 0)
    preview_margin = max(0.0, float(margin or 0.0))
    capped_examples = max(0, min(int(max_examples or 0), 20))

    rejected_signals = [dict(signal or {}) for signal in (signals or []) if not bool((signal or {}).get("buy", False))]
    classifications = [
        classify_near_approved_signal(
            signal,
            feed_status=feed_status,
            fallback_assets_count=fallback_assets_count,
            macro_alert_active=bool(macro_state.get("macro_alert_active", False)),
            broker_is_paper=_broker_is_paper(broker_state),
            watchlist=watchlist,
            open_positions_count=open_positions_count,
            max_open_positions=max_open_positions,
            daily_loss_block_active=bool(risk_state.get("daily_loss_block_active", False)),
            preview_margin=preview_margin,
        )
        for signal in rejected_signals
    ]
    return summarize_calibration_preview(
        classifications,
        enabled=True,
        margin=preview_margin,
        max_examples=capped_examples,
    )
