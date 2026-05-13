from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any

from core.feed_scope_reconciliation import build_feed_scope_reconciliation


MODE = "SHADOW_ONLY"
POLICY = "conservative_v1"
SAFE_GAP_MAX = 0.015
MARGINAL_GAP_MAX = 0.04
MAX_RECENT_CANDIDATES = 30
MAX_DISPLAY_CANDIDATES = 12
STOP_SHADOW_PCT = 0.025
TAKE_PROFIT_SHADOW_PCT = 0.04
MAX_HOLD_CYCLES_SHADOW = 24
WINDOWS = (1, 3, 6, 12, 24)

KNOWN_PROVIDERS = {"twelvedata", "yahoo", "market", "cached"}
SAFE_CONTEXTS = {"FAVORAVEL", "NEUTRO"}
SECONDARY_REASONS = {"breakout_not_confirmed", "confidence_too_low", "score_below_minimum"}
PRIMARY_BLOCKERS = {"trend_not_confirmed", "no_setup_eligible", "reversal_not_eligible", "volatility_out_of_range"}
RISK_BLOCKERS = {
    "fallback_blocked",
    "feed_quality_blocked",
    "provider_unknown",
    "context_blocked",
    "daily_loss_guard",
    "macro_alert_guard",
    "cooldown_active",
    "duplicate_signal_blocked",
    "position_limit_reached",
    "schedule_blocked",
}
ALWAYS_SHADOW_BLOCK = {
    "trend_not_confirmed",
    "no_setup_eligible",
    "reversal_not_eligible",
    "fallback_blocked",
    "feed_quality_blocked",
    "provider_unknown",
    "daily_loss_guard",
    "cooldown_active",
    "duplicate_signal_blocked",
    "position_limit_reached",
    "schedule_blocked",
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value or default)
    except (TypeError, ValueError):
        return default


def _norm_text(value: Any) -> str:
    return str(value or "").strip()


def _norm_reason_set(raw: Any) -> set[str]:
    if isinstance(raw, str):
        values = [chunk.strip() for chunk in raw.split(",")]
    elif isinstance(raw, (list, tuple, set)):
        values = list(raw)
    else:
        values = []
    return {str(item).strip().lower() for item in values if str(item).strip()}


def _broker_is_paper(state: dict[str, Any]) -> bool:
    broker_state = dict(state.get("broker", {}) or {})
    validation_state = dict(state.get("validation", {}) or {})
    values = [
        broker_state.get("effective_mode"),
        broker_state.get("mode"),
        broker_state.get("configured_mode"),
        broker_state.get("provider"),
        validation_state.get("trading_mode"),
    ]
    return any(str(value or "").strip().lower() == "paper" for value in values)


def _open_positions_count(state: dict[str, Any]) -> int:
    raw = state.get("positions")
    if isinstance(raw, dict):
        return len(raw)
    if isinstance(raw, list):
        return sum(1 for item in raw if str((dict(item or {}) if isinstance(item, dict) else {}).get("status") or "OPEN").upper() == "OPEN")
    return 0


def _max_positions(state: dict[str, Any]) -> int:
    return _as_int((state.get("trader", {}) or {}).get("max_open_positions"), 0)


def _latest_price(market_data: dict[str, Any] | None, symbol: str) -> float | None:
    frame = (market_data or {}).get(symbol)
    if frame is None:
        frame = (market_data or {}).get(str(symbol).upper())
    try:
        if frame is None or frame.empty or "close" not in frame.columns:
            return None
        return _as_float(frame["close"].iloc[-1], None)
    except Exception:
        return None


def _market_structure_for_symbol(market_structure_audit: dict[str, Any] | None, symbol: str) -> dict[str, Any]:
    for row in list((market_structure_audit or {}).get("market_structure_best_candidates", []) or []):
        payload = dict(row or {}) if isinstance(row, dict) else {}
        if str(payload.get("symbol") or "").upper() == str(symbol).upper():
            return payload
    return {}


def _fib_alignment_for_symbol(fib_alignment_audit: dict[str, Any] | None, symbol: str) -> dict[str, Any]:
    payload = dict(fib_alignment_audit or {})
    if str(payload.get("fib_alignment_top_symbol") or "").upper() == str(symbol).upper():
        return payload
    return {}


def default_shadow_decision_state(reason: str = "No shadow decision simulator data yet.") -> dict[str, Any]:
    return {
        "shadow_decision_simulator_enabled": True,
        "shadow_decision_mode": MODE,
        "shadow_entry_policy": POLICY,
        "shadow_decision_last_run_at": "",
        "preview_near_approved_count": 0,
        "shadow_candidates_received_count": 0,
        "shadow_candidates_unique_count": 0,
        "shadow_candidates_ignored_count": 0,
        "shadow_candidates_classified_count": 0,
        "shadow_candidates_analyzed_count": 0,
        "shadow_raw_near_approved_count": 0,
        "shadow_counts_scope": "current_cycle_and_accumulated_recent",
        "shadow_current_cycle_candidates_count": 0,
        "shadow_accumulated_candidates_count": 0,
        "shadow_current_cycle_received_count": 0,
        "shadow_current_cycle_new_unique_count": 0,
        "shadow_current_cycle_duplicate_count": 0,
        "shadow_current_cycle_already_analyzed_count": 0,
        "shadow_current_cycle_analyzed_count": 0,
        "shadow_current_cycle_analyzed_new_count": 0,
        "shadow_current_cycle_classified_count": 0,
        "shadow_current_cycle_classified_new_count": 0,
        "shadow_current_cycle_raw_near_approved_count": 0,
        "shadow_current_cycle_safe_near_approved_count": 0,
        "shadow_current_cycle_marginal_near_approved_count": 0,
        "shadow_current_cycle_unsafe_count": 0,
        "shadow_current_cycle_unsafe_new_count": 0,
        "shadow_current_cycle_ignored_count": 0,
        "shadow_current_cycle_primary_blocked_count": 0,
        "shadow_current_cycle_primary_blocked_new_count": 0,
        "shadow_current_cycle_secondary_blocked_count": 0,
        "shadow_current_cycle_secondary_blocked_new_count": 0,
        "shadow_accumulated_received_count": 0,
        "shadow_accumulated_raw_received_count": 0,
        "shadow_accumulated_unique_candidates_count": 0,
        "shadow_accumulated_analyzed_count": 0,
        "shadow_accumulated_analyzed_unique_count": 0,
        "shadow_accumulated_classified_unique_count": 0,
        "shadow_accumulated_raw_near_approved_count": 0,
        "shadow_accumulated_unsafe_count": 0,
        "shadow_accumulated_unsafe_unique_count": 0,
        "shadow_accumulated_primary_blocked_count": 0,
        "shadow_accumulated_secondary_blocked_count": 0,
        "shadow_raw_to_unique_ratio": 0.0,
        "shadow_duplicate_ratio": 0.0,
        "shadow_counter_health_status": "healthy",
        "shadow_near_approved_count": 0,
        "shadow_safe_near_approved_count": 0,
        "shadow_marginal_near_approved_count": 0,
        "shadow_marginal_count": 0,
        "shadow_unsafe_count": 0,
        "shadow_unsafe_rejection_count": 0,
        "shadow_primary_blocked_count": 0,
        "shadow_secondary_blocked_count": 0,
        "shadow_structure_missing_count": 0,
        "shadow_confirmation_missing_count": 0,
        "shadow_ignored_count": 0,
        "shadow_ignored_reason": "",
        "shadow_counter_warning": False,
        "shadow_counter_warning_reason": "",
        "shadow_scope_warning": False,
        "shadow_scope_warning_reason": "",
        "shadow_would_enter_count": 0,
        "shadow_pending_count": 0,
        "shadow_would_win_count": 0,
        "shadow_would_lose_count": 0,
        "shadow_best_symbol": "",
        "shadow_best_strategy": "",
        "shadow_best_candidate_score": None,
        "shadow_dominant_block_reason": "",
        "shadow_dominant_block_reason_current": "",
        "shadow_dominant_block_reason_accumulated": "",
        "dominant_exclusion_current_scope": "",
        "dominant_exclusion_accumulated_scope": "",
        "fallback_blocker_scope": "UNKNOWN",
        "fallback_current_count": 0,
        "fallback_accumulated_count": 0,
        "fallback_scope_status": "UNKNOWN_SCOPE",
        "fallback_scope_note": "",
        "shadow_policy_recommendation": "observe_more",
        "shadow_current_cycle_candidates": [],
        "shadow_accumulated_recent_candidates": [],
        "shadow_recent_candidates": [],
        "shadow_outcome_summary": {},
        "shadow_reason": reason,
        "shadow_stop_pct": STOP_SHADOW_PCT,
        "shadow_take_profit_pct": TAKE_PROFIT_SHADOW_PCT,
        "shadow_max_hold_cycles": MAX_HOLD_CYCLES_SHADOW,
    }


def _first_exclusion_reason(
    *,
    reasons: set[str],
    operational_blocks: list[str],
    primary: list[str],
    score_gap: float | None,
    fib_status: str,
    pivot: bool,
    bos: bool,
) -> tuple[str, str]:
    if "fallback" in operational_blocks:
        return "blocked_by_fallback", "feed"
    if "provider_unknown" in operational_blocks:
        return "blocked_by_provider_unknown", "feed"
    if "context_critical" in operational_blocks:
        return "blocked_by_context", "context"
    if "daily_loss_guard" in operational_blocks or "daily_loss_guard" in reasons:
        return "blocked_by_daily_loss_guard", "guard"
    if "position_limit_reached" in operational_blocks or "position_limit_reached" in reasons:
        return "blocked_by_position_limit", "guard"
    if "cooldown_active" in reasons:
        return "blocked_by_cooldown", "guard"
    if "duplicate_signal_blocked" in reasons:
        return "blocked_by_duplicate_signal", "guard"
    if len(primary) > 1:
        return "blocked_by_multiple_primary_rejections", "strategy"
    if "no_setup_eligible" in reasons:
        return "blocked_by_no_setup_eligible", "structure"
    if "trend_not_confirmed" in reasons:
        return "blocked_by_trend_not_confirmed", "strategy"
    if "reversal_not_eligible" in reasons:
        return "blocked_by_reversal_not_eligible", "strategy"
    if score_gap is None or score_gap > MARGINAL_GAP_MAX:
        return "blocked_by_score_gap", "score"
    if fib_status in {"weak_alignment", "no_sufficient_alignment"}:
        return "blocked_by_fib_weak_alignment", "structure"
    if fib_status == "partial_alignment" and not (pivot or bos):
        return "blocked_by_missing_pivot_bos", "confirmation"
    return "blocked_by_policy_conservative_v1", "policy"


def _classify_candidate(
    signal: dict[str, Any],
    *,
    state: dict[str, Any],
    market_row: dict[str, Any],
    fib_row: dict[str, Any],
) -> dict[str, Any]:
    reasons = _norm_reason_set(signal.get("rejection_reasons"))
    score = _as_float(signal.get("score"), None)
    min_score = _as_float(signal.get("effective_min_signal_score"), _as_float(signal.get("base_min_signal_score"), None))
    score_gap = None if score is None or min_score is None else round(max(0.0, float(min_score) - float(score)), 6)
    context_status = _norm_text(signal.get("context_status") or "NEUTRO").upper()
    feed_status = "LIVE" if _norm_text(signal.get("data_source")).lower() == "market" else "FALLBACK"
    provider = _norm_text(signal.get("provider_effective")).lower()
    macro_active = bool(signal.get("macro_alert_active", False))
    daily_lock = bool((state.get("risk", {}) or {}).get("daily_loss_block_active", False))
    paper = _broker_is_paper(state)
    positions_left = _open_positions_count(state) < max(1, _max_positions(state))
    fib_status = _norm_text(fib_row.get("fib_alignment_status") or "unknown")
    pivot = bool(market_row.get("pivot_detected", False))
    bos = bool(market_row.get("bos_detected", False))

    primary = sorted(reasons.intersection(PRIMARY_BLOCKERS))
    secondary = sorted(reasons.intersection(SECONDARY_REASONS))
    risk = sorted(reasons.intersection(RISK_BLOCKERS))
    operational_blocks: list[str] = []
    if feed_status != "LIVE":
        operational_blocks.append("fallback")
    if provider not in KNOWN_PROVIDERS:
        operational_blocks.append("provider_unknown")
    if context_status not in SAFE_CONTEXTS:
        operational_blocks.append("context_critical")
    if not paper:
        operational_blocks.append("broker_not_paper")
    if daily_lock:
        operational_blocks.append("daily_loss_guard")
    if macro_active and _norm_text(signal.get("macro_alert_level")).upper() in {"HIGH", "CRITICAL"}:
        operational_blocks.append("macro_alert_critical")
    if not positions_left:
        operational_blocks.append("position_limit_reached")
    if score is None or min_score is None:
        operational_blocks.append("missing_score")
    raw_near_approved = bool(score_gap is not None and 0.0 <= score_gap <= MARGINAL_GAP_MAX)

    if operational_blocks:
        candidate_class = "UNSAFE_REJECTION"
    elif "no_setup_eligible" in reasons:
        candidate_class = "STRUCTURE_MISSING"
    elif "trend_not_confirmed" in reasons or "reversal_not_eligible" in reasons:
        candidate_class = "PRIMARY_BLOCKED"
    elif score_gap is None:
        candidate_class = "INSUFFICIENT_DATA"
    elif score_gap > MARGINAL_GAP_MAX or len(primary) > 0 or reasons.intersection(RISK_BLOCKERS):
        candidate_class = "UNSAFE_REJECTION"
    elif fib_status in {"weak_alignment", "no_sufficient_alignment"}:
        candidate_class = "STRUCTURE_MISSING"
    elif fib_status == "partial_alignment" and not (pivot or bos):
        candidate_class = "CONFIRMATION_MISSING"
    elif score_gap <= SAFE_GAP_MAX and reasons.issubset(SECONDARY_REASONS):
        candidate_class = "SAFE_NEAR_APPROVED"
    elif score_gap <= MARGINAL_GAP_MAX and reasons.issubset(SECONDARY_REASONS):
        candidate_class = "MARGINAL_NEAR_APPROVED"
    else:
        candidate_class = "UNSAFE_REJECTION"
    exclusion_reason, exclusion_layer = _first_exclusion_reason(
        reasons=reasons,
        operational_blocks=operational_blocks,
        primary=primary,
        score_gap=score_gap,
        fib_status=fib_status,
        pivot=pivot,
        bos=bos,
    )
    safe_candidate = candidate_class == "SAFE_NEAR_APPROVED"
    marginal_candidate = candidate_class == "MARGINAL_NEAR_APPROVED"
    why_not_safe = "" if safe_candidate else exclusion_reason
    if marginal_candidate:
        why_not_marginal = ""
    elif not raw_near_approved:
        why_not_marginal = "not_raw_near_approved"
    elif candidate_class == "SAFE_NEAR_APPROVED":
        why_not_marginal = "candidate_is_safe"
    else:
        why_not_marginal = exclusion_reason

    return {
        "candidate_class": candidate_class,
        "raw_near_approved": raw_near_approved,
        "analyzed_by_shadow": True,
        "safe_candidate": safe_candidate,
        "score": score,
        "min_score": min_score,
        "score_gap": score_gap,
        "reasons": sorted(reasons),
        "primary_blockers": primary,
        "secondary_blockers": secondary,
        "risk_blockers": sorted(set(risk + operational_blocks)),
        "unsafe_reason_codes": sorted(set(risk + operational_blocks + primary)),
        "primary_blocker_codes": primary,
        "secondary_blocker_codes": secondary,
        "exclusion_layer": exclusion_layer if not safe_candidate else "none",
        "why_not_safe": why_not_safe,
        "why_not_marginal": why_not_marginal,
        "context_status": context_status,
        "feed_status": feed_status,
        "provider_effective": provider,
        "broker_paper": paper,
        "daily_loss_lock": daily_lock,
        "positions_left": positions_left,
        "fib_alignment_status": fib_status,
        "pivot_detected": pivot,
        "bos_detected": bos,
    }


def _shadow_decision(classification: dict[str, Any], signal: dict[str, Any], fib_row: dict[str, Any]) -> dict[str, Any]:
    reasons = set(classification.get("reasons", []) or [])
    class_name = str(classification.get("candidate_class") or "")
    score = _as_float(classification.get("score"), 0.0) or 0.0
    min_score = _as_float(classification.get("min_score"), 0.0) or 0.0
    gap = _as_float(classification.get("score_gap"), None)
    fib_status = str(classification.get("fib_alignment_status") or "unknown")
    confidence = 0.50
    block_reason = ""
    entry_reason = ""

    if class_name != "SAFE_NEAR_APPROVED":
        block_reason = f"class_{class_name.lower()}"
    elif str(signal.get("strategy_name") or "trend_pullback_breakout") != "trend_pullback_breakout":
        block_reason = "strategy_not_supported_by_policy"
    elif reasons.intersection(ALWAYS_SHADOW_BLOCK):
        block_reason = ",".join(sorted(reasons.intersection(ALWAYS_SHADOW_BLOCK)))
    elif gap is None or not (score >= min_score or gap <= SAFE_GAP_MAX):
        block_reason = "score_gap_not_small_enough"
    else:
        confidence += 0.20
        entry_reason = "secondary_only_small_gap"
        if fib_status == "strong_alignment":
            confidence += 0.12
            entry_reason = "secondary_only_small_gap_with_strong_fib_alignment"
        elif fib_status == "partial_alignment":
            confidence -= 0.06
            entry_reason = "secondary_only_small_gap_with_partial_fib_alignment"
        elif fib_status in {"weak_alignment", "no_sufficient_alignment"}:
            confidence -= 0.14
            block_reason = "fib_alignment_too_weak"

    would_enter = bool(not block_reason and class_name == "SAFE_NEAR_APPROVED")
    if not would_enter and not block_reason:
        block_reason = "policy_conservative_v1_blocked"
    if not entry_reason and would_enter:
        entry_reason = "conservative_v1_shadow_entry"

    risk_notes = []
    if fib_status == "partial_alignment":
        risk_notes.append("partial_fib_alignment")
    if not bool(classification.get("pivot_detected")):
        risk_notes.append("pivot_missing")
    if not bool(classification.get("bos_detected")):
        risk_notes.append("bos_missing")

    return {
        "shadow_would_enter": would_enter,
        "shadow_entry_policy": POLICY,
        "shadow_entry_reason": entry_reason,
        "shadow_block_reason": block_reason,
        "why_would_not_enter": "" if would_enter else block_reason,
        "shadow_expected_risk": ",".join(risk_notes) if risk_notes else "secondary_confirmation_risk",
        "shadow_confidence": round(max(0.0, min(1.0, confidence)), 4),
    }


def _candidate_key(signal: dict[str, Any]) -> str:
    return "|".join(
        [
            _norm_text(signal.get("signal_key") or signal.get("asset") or signal.get("symbol")).upper(),
            _norm_text(signal.get("signal_timestamp") or signal.get("timestamp")),
            f"{_as_float(signal.get('score'), 0.0) or 0.0:.6f}",
        ]
    )


def _build_candidate(
    signal: dict[str, Any],
    *,
    state: dict[str, Any],
    market_row: dict[str, Any],
    fib_row: dict[str, Any],
) -> dict[str, Any]:
    classification = _classify_candidate(signal, state=state, market_row=market_row, fib_row=fib_row)
    decision = _shadow_decision(classification, signal, fib_row)
    symbol = _norm_text(signal.get("asset") or signal.get("symbol")).upper()
    price = _as_float(signal.get("price"), None)
    return {
        "shadow_candidate_key": _candidate_key(signal),
        "symbol": symbol,
        "strategy": _norm_text(signal.get("strategy_name") or "trend_pullback_breakout"),
        "timestamp": _norm_text(signal.get("signal_timestamp") or signal.get("timestamp") or _utc_now_iso()),
        "duplicate_candidate": False,
        "already_seen": False,
        "analyzed_this_cycle": True,
        "analyzed_previously": False,
        "current_score": classification.get("score"),
        "min_score": classification.get("min_score"),
        "score_gap": classification.get("score_gap"),
        "reasons": classification.get("reasons", []),
        "primary_blockers": classification.get("primary_blockers", []),
        "secondary_blockers": classification.get("secondary_blockers", []),
        "risk_blockers": classification.get("risk_blockers", []),
        "raw_near_approved": bool(classification.get("raw_near_approved", False)),
        "analyzed_by_shadow": bool(classification.get("analyzed_by_shadow", True)),
        "safe_candidate": bool(classification.get("safe_candidate", False)),
        "unsafe_reason_codes": list(classification.get("unsafe_reason_codes", []) or []),
        "primary_blocker_codes": list(classification.get("primary_blocker_codes", []) or []),
        "secondary_blocker_codes": list(classification.get("secondary_blocker_codes", []) or []),
        "exclusion_layer": classification.get("exclusion_layer"),
        "why_not_safe": classification.get("why_not_safe"),
        "why_not_marginal": classification.get("why_not_marginal"),
        "context_status": classification.get("context_status"),
        "feed_status": classification.get("feed_status"),
        "provider_effective": classification.get("provider_effective"),
        "market_structure_score": market_row.get("market_structure_score"),
        "fib_zone": market_row.get("current_fib_zone", ""),
        "fib_alignment_score": fib_row.get("fib_alignment_score"),
        "fib_alignment_status": classification.get("fib_alignment_status"),
        "pivot_detected": classification.get("pivot_detected"),
        "bos_detected": classification.get("bos_detected"),
        "candidate_class": classification.get("candidate_class"),
        "shadow_trace_status": "analyzed_by_shadow",
        "classified_by_shadow": bool(classification.get("candidate_class")),
        "count_scope": "current_cycle",
        "price_at_signal": price,
        "observed_cycles": 0,
        "shadow_result_pending": bool(decision.get("shadow_would_enter")),
        "outcome_label": "STILL_PENDING" if bool(decision.get("shadow_would_enter")) else "INVALIDATED",
        "max_favorable_move": 0.0,
        "max_adverse_move": 0.0,
        "theoretical_pnl_pct": 0.0,
        "hit_take_profit_shadow": False,
        "hit_stop_loss_shadow": False,
        "after_1_cycle": None,
        "after_3_cycles": None,
        "after_6_cycles": None,
        "after_12_cycles": None,
        "after_24_cycles": None,
        **decision,
    }


def _update_outcome(candidate: dict[str, Any], market_data: dict[str, Any] | None) -> dict[str, Any]:
    payload = dict(candidate or {})
    payload["duplicate_candidate"] = False
    payload["already_seen"] = True
    payload["classified_by_shadow"] = bool(payload.get("classified_by_shadow", bool(payload.get("candidate_class"))))
    payload["analyzed_this_cycle"] = False
    payload["analyzed_previously"] = bool(payload.get("analyzed_previously", payload["classified_by_shadow"]))
    payload["analyzed_by_shadow"] = bool(payload.get("analyzed_by_shadow", payload["classified_by_shadow"]))
    payload["shadow_trace_status"] = "analyzed_previously"
    if not bool(payload.get("shadow_would_enter", False)) or str(payload.get("outcome_label") or "") not in {"STILL_PENDING", ""}:
        payload["count_scope"] = "accumulated_recent"
        return payload
    symbol = _norm_text(payload.get("symbol")).upper()
    current_price = _latest_price(market_data, symbol)
    entry_price = _as_float(payload.get("price_at_signal"), None)
    if current_price is None or entry_price is None or entry_price <= 0:
        payload["shadow_result_pending"] = True
        payload["outcome_label"] = "STILL_PENDING"
        payload["count_scope"] = "accumulated_recent"
        return payload

    cycles = _as_int(payload.get("observed_cycles"), 0) + 1
    payload["count_scope"] = "accumulated_recent"
    pnl_pct = round((float(current_price) - float(entry_price)) / float(entry_price), 6)
    payload["observed_cycles"] = cycles
    payload["price_after_window"] = round(float(current_price), 6)
    payload["theoretical_pnl_pct"] = pnl_pct
    payload["max_favorable_move"] = round(max(float(payload.get("max_favorable_move", 0.0) or 0.0), pnl_pct), 6)
    payload["max_adverse_move"] = round(min(float(payload.get("max_adverse_move", 0.0) or 0.0), pnl_pct), 6)
    if cycles in WINDOWS:
        payload[f"after_{cycles}_cycle" if cycles == 1 else f"after_{cycles}_cycles"] = pnl_pct

    if float(payload["max_favorable_move"]) >= TAKE_PROFIT_SHADOW_PCT:
        payload["hit_take_profit_shadow"] = True
        payload["outcome_label"] = "WOULD_WIN"
        payload["shadow_result_pending"] = False
    elif float(payload["max_adverse_move"]) <= -STOP_SHADOW_PCT:
        payload["hit_stop_loss_shadow"] = True
        payload["outcome_label"] = "WOULD_LOSE"
        payload["shadow_result_pending"] = False
    elif cycles >= MAX_HOLD_CYCLES_SHADOW:
        payload["shadow_result_pending"] = False
        if abs(pnl_pct) < 0.0025:
            payload["outcome_label"] = "WOULD_FLAT"
        else:
            payload["outcome_label"] = "WOULD_WIN" if pnl_pct > 0 else "WOULD_LOSE"
    else:
        payload["shadow_result_pending"] = True
        payload["outcome_label"] = "STILL_PENDING"
    return payload


def _classified(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in items if str(item.get("candidate_class") or "").strip()]


def _unsafe_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in items if item.get("candidate_class") == "UNSAFE_REJECTION"]


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(max(0.0, float(numerator) / float(denominator)), 4)


def _as_accumulated_recent(item: dict[str, Any]) -> dict[str, Any]:
    payload = dict(item or {})
    payload["count_scope"] = "accumulated_recent"
    payload["duplicate_candidate"] = bool(payload.get("duplicate_candidate", False))
    payload["already_seen"] = bool(payload.get("already_seen", False))
    payload["analyzed_this_cycle"] = bool(payload.get("analyzed_this_cycle", False))
    payload["classified_by_shadow"] = bool(payload.get("classified_by_shadow", bool(payload.get("candidate_class"))))
    payload["analyzed_previously"] = bool(payload.get("analyzed_previously", False)) or (
        not payload["analyzed_this_cycle"] and payload["classified_by_shadow"]
    )
    payload["analyzed_by_shadow"] = bool(payload.get("analyzed_by_shadow", payload["classified_by_shadow"]))
    return payload


def _counter_invariant_warning(
    *,
    received: int,
    new_unique: int,
    duplicate: int,
    analyzed_new: int,
    classified: int,
    safe: int,
    marginal: int,
    unsafe: int,
    accumulated_unique: int,
    accumulated_raw: int,
    duplicate_ratio: float,
    accumulated_table: list[dict[str, Any]],
) -> tuple[bool, str]:
    if received != new_unique + duplicate:
        return True, "current_received_differs_from_new_unique_plus_duplicate"
    if analyzed_new > new_unique:
        return True, "analyzed_new_greater_than_new_unique"
    if duplicate > received:
        return True, "duplicate_count_greater_than_received"
    if accumulated_unique > accumulated_raw:
        return True, "accumulated_unique_greater_than_raw_received"
    if duplicate_ratio < 0.0 or duplicate_ratio > 1.0:
        return True, "duplicate_ratio_out_of_range"
    if any(str(item.get("count_scope") or "") != "accumulated_recent" for item in accumulated_table):
        return True, "accumulated_table_scope_not_marked"
    if classified > 0 and analyzed_new <= 0 and new_unique > 0:
        return True, "new_classified_candidates_without_analyzed_new_count"
    if analyzed_new < safe + marginal + unsafe:
        return True, "analyzed_new_lower_than_classified_subsets"
    if classified < safe + marginal + unsafe:
        return True, "classified_lower_than_classified_subsets"
    return False, ""


def build_shadow_decision_simulator(
    *,
    signals: list[dict[str, Any]] | None,
    state: dict[str, Any] | None,
    market_data: dict[str, Any] | None,
    market_structure_audit: dict[str, Any] | None = None,
    fib_alignment_audit: dict[str, Any] | None = None,
    market_data_status: dict[str, Any] | None = None,
    enabled: bool = True,
) -> dict[str, Any]:
    if not enabled:
        return default_shadow_decision_state("Shadow decision simulator disabled.")

    root_state = dict(state or {})
    previous_state = dict(root_state.get("shadow_decision_simulator", {}) or {})
    previous_candidate_source = previous_state.get(
        "shadow_accumulated_recent_candidates",
        previous_state.get("shadow_recent_candidates", []),
    )
    previous_candidates = [
        _update_outcome(dict(item or {}), market_data)
        for item in list(previous_candidate_source or [])
        if isinstance(item, dict)
    ]
    previous_raw_received = max(
        _as_int(previous_state.get("shadow_accumulated_raw_received_count"), 0),
        _as_int(previous_state.get("shadow_accumulated_received_count"), 0),
        len(previous_candidates),
    )
    seen_keys = {str(item.get("shadow_candidate_key") or "") for item in previous_candidates if item.get("shadow_candidate_key")}

    rejected_signals = [
        dict(item or {})
        for item in list(signals or [])
        if isinstance(item, dict) and not bool(item.get("buy", False))
    ]
    preview_near_count = 0
    for signal in rejected_signals:
        score = _as_float(signal.get("score"), None)
        min_score = _as_float(signal.get("effective_min_signal_score"), _as_float(signal.get("base_min_signal_score"), None))
        gap = None if score is None or min_score is None else max(0.0, float(min_score) - float(score))
        if gap is not None and 0.0 <= gap <= MARGINAL_GAP_MAX:
            preview_near_count += 1

    new_candidates: list[dict[str, Any]] = []
    ignored_count = 0
    ignored_reason_counter: Counter[str] = Counter()
    for signal in rejected_signals:
        key = _candidate_key(signal)
        if key in seen_keys:
            ignored_count += 1
            ignored_reason_counter["duplicate_existing_shadow_candidate"] += 1
            for item in previous_candidates:
                if str(item.get("shadow_candidate_key") or "") == key:
                    item["duplicate_candidate"] = True
                    item["already_seen"] = True
                    item["analyzed_this_cycle"] = False
                    item["analyzed_previously"] = True
                    item["shadow_trace_status"] = "duplicate_existing_shadow_candidate"
                    break
            continue
        symbol = _norm_text(signal.get("asset") or signal.get("symbol")).upper()
        market_row = _market_structure_for_symbol(market_structure_audit, symbol)
        fib_row = _fib_alignment_for_symbol(fib_alignment_audit, symbol)
        candidate = _build_candidate(signal, state=root_state, market_row=market_row, fib_row=fib_row)
        seen_keys.add(key)
        new_candidates.append(candidate)

    combined = [*new_candidates, *previous_candidates]
    combined = combined[:MAX_RECENT_CANDIDATES]
    current_cycle_candidates = [dict(item) for item in new_candidates[:MAX_RECENT_CANDIDATES]]
    accumulated_recent_candidates = [_as_accumulated_recent(item) for item in combined]
    combined = accumulated_recent_candidates
    current_classified = _classified(new_candidates)
    accumulated_classified = _classified(combined)
    current_unsafe = _unsafe_items(new_candidates)
    accumulated_unsafe = _unsafe_items(combined)
    current_raw_near = [item for item in new_candidates if bool(item.get("raw_near_approved", False))]
    raw_near = [item for item in combined if bool(item.get("raw_near_approved", False))]
    safe = [item for item in combined if item.get("candidate_class") == "SAFE_NEAR_APPROVED"]
    marginal = [item for item in combined if item.get("candidate_class") == "MARGINAL_NEAR_APPROVED"]
    unsafe = accumulated_unsafe
    current_primary_blocked = [
        item
        for item in new_candidates
        if item.get("candidate_class") == "PRIMARY_BLOCKED" or item.get("primary_blocker_codes")
    ]
    current_secondary_blocked = [item for item in new_candidates if item.get("secondary_blocker_codes")]
    primary_blocked = [item for item in combined if item.get("candidate_class") == "PRIMARY_BLOCKED" or item.get("primary_blocker_codes")]
    secondary_blocked = [item for item in combined if item.get("secondary_blocker_codes")]
    structure_missing = [item for item in combined if item.get("candidate_class") == "STRUCTURE_MISSING"]
    confirmation_missing = [item for item in combined if item.get("candidate_class") == "CONFIRMATION_MISSING"]
    would_enter = [item for item in combined if bool(item.get("shadow_would_enter", False))]
    pending = [item for item in combined if bool(item.get("shadow_result_pending", False))]
    wins = [item for item in combined if item.get("outcome_label") == "WOULD_WIN"]
    losses = [item for item in combined if item.get("outcome_label") == "WOULD_LOSE"]
    block_counter = Counter(
        str(item.get("why_not_safe") or item.get("shadow_block_reason") or "none")
        for item in combined
        if not bool(item.get("shadow_would_enter", False))
    )
    current_block_counter = Counter(
        str(item.get("why_not_safe") or item.get("shadow_block_reason") or "none")
        for item in new_candidates
        if not bool(item.get("shadow_would_enter", False))
    )
    accumulated_dominant_block = block_counter.most_common(1)[0][0] if block_counter else ""
    current_dominant_block = current_block_counter.most_common(1)[0][0] if current_block_counter else ""
    best = sorted(combined, key=lambda item: float(item.get("current_score") or 0.0), reverse=True)[0] if combined else {}
    current_safe = [item for item in new_candidates if item.get("candidate_class") == "SAFE_NEAR_APPROVED"]
    current_marginal = [item for item in new_candidates if item.get("candidate_class") == "MARGINAL_NEAR_APPROVED"]
    current_received_count = len(rejected_signals)
    current_new_unique_count = len(new_candidates)
    current_duplicate_count = ignored_count
    accumulated_raw_received_count = previous_raw_received + current_received_count
    accumulated_unique_count = len(accumulated_recent_candidates)
    duplicate_ratio = _safe_ratio(current_duplicate_count, current_received_count)
    raw_to_unique_ratio = _safe_ratio(accumulated_raw_received_count, accumulated_unique_count)
    warning, warning_reason = _counter_invariant_warning(
        received=current_received_count,
        new_unique=current_new_unique_count,
        duplicate=current_duplicate_count,
        analyzed_new=len(current_classified),
        classified=len(current_classified),
        safe=len(current_safe),
        marginal=len(current_marginal),
        unsafe=len(current_unsafe),
        accumulated_unique=accumulated_unique_count,
        accumulated_raw=accumulated_raw_received_count,
        duplicate_ratio=duplicate_ratio,
        accumulated_table=accumulated_recent_candidates,
    )
    if warning:
        counter_health_status = "warning"
    elif current_received_count > 0 and current_duplicate_count == current_received_count:
        counter_health_status = "all_duplicates_current_cycle"
    elif raw_to_unique_ratio >= 10.0:
        counter_health_status = "high_raw_repetition"
    else:
        counter_health_status = "healthy"

    feed_scope = build_feed_scope_reconciliation(
        market_data_status=market_data_status or root_state.get("market_data", {}),
        validation_state=dict(root_state.get("validation", {}) or {}),
        shadow_decision_simulator={"shadow_dominant_block_reason": accumulated_dominant_block},
        signals=rejected_signals,
    )
    if bool(feed_scope.get("current_feed_is_clean")) and current_dominant_block == "blocked_by_fallback":
        current_dominant_block = next(
            (reason for reason, _count in current_block_counter.most_common() if reason != "blocked_by_fallback"),
            "current_feed_clean_no_current_fallback",
        )
    dominant_block_reason = current_dominant_block or accumulated_dominant_block
    if bool(feed_scope.get("current_feed_is_clean")) and dominant_block_reason == "blocked_by_fallback":
        dominant_block_reason = "accumulated_fallback_only"

    if wins and len(wins) >= len(losses) + 2:
        recommendation = "study_future_relaxation"
    elif pending:
        recommendation = "wait_shadow_outcomes"
    elif safe or marginal:
        recommendation = "collect_more_shadow_sample"
    else:
        recommendation = "keep_current_strategy"

    return {
        "shadow_decision_simulator_enabled": True,
        "shadow_decision_mode": MODE,
        "shadow_entry_policy": POLICY,
        "shadow_decision_last_run_at": _utc_now_iso(),
        "shadow_counts_scope": "current_cycle_and_accumulated_recent",
        "preview_near_approved_count": preview_near_count,
        "shadow_candidates_received_count": current_received_count,
        "shadow_candidates_unique_count": current_new_unique_count,
        "shadow_candidates_ignored_count": current_duplicate_count,
        "shadow_candidates_classified_count": len(accumulated_classified),
        "shadow_candidates_analyzed_count": len(accumulated_classified),
        "shadow_current_cycle_candidates_count": current_new_unique_count,
        "shadow_accumulated_candidates_count": accumulated_unique_count,
        "shadow_current_cycle_received_count": current_received_count,
        "shadow_current_cycle_new_unique_count": current_new_unique_count,
        "shadow_current_cycle_duplicate_count": current_duplicate_count,
        "shadow_current_cycle_already_analyzed_count": current_duplicate_count,
        "shadow_current_cycle_analyzed_count": len(current_classified),
        "shadow_current_cycle_analyzed_new_count": len(current_classified),
        "shadow_current_cycle_classified_count": len(current_classified),
        "shadow_current_cycle_classified_new_count": len(current_classified),
        "shadow_current_cycle_raw_near_approved_count": len(current_raw_near),
        "shadow_current_cycle_safe_near_approved_count": len(current_safe),
        "shadow_current_cycle_marginal_near_approved_count": len(current_marginal),
        "shadow_current_cycle_unsafe_count": len(current_unsafe),
        "shadow_current_cycle_unsafe_new_count": len(current_unsafe),
        "shadow_current_cycle_ignored_count": current_duplicate_count,
        "shadow_current_cycle_primary_blocked_count": len(current_primary_blocked),
        "shadow_current_cycle_primary_blocked_new_count": len(current_primary_blocked),
        "shadow_current_cycle_secondary_blocked_count": len(current_secondary_blocked),
        "shadow_current_cycle_secondary_blocked_new_count": len(current_secondary_blocked),
        "shadow_accumulated_received_count": accumulated_raw_received_count,
        "shadow_accumulated_raw_received_count": accumulated_raw_received_count,
        "shadow_accumulated_unique_candidates_count": accumulated_unique_count,
        "shadow_accumulated_analyzed_count": len(accumulated_classified),
        "shadow_accumulated_analyzed_unique_count": len(accumulated_classified),
        "shadow_accumulated_classified_unique_count": len(accumulated_classified),
        "shadow_accumulated_raw_near_approved_count": len(raw_near),
        "shadow_accumulated_unsafe_count": len(accumulated_unsafe),
        "shadow_accumulated_unsafe_unique_count": len(accumulated_unsafe),
        "shadow_accumulated_primary_blocked_count": len(primary_blocked),
        "shadow_accumulated_secondary_blocked_count": len(secondary_blocked),
        "shadow_raw_to_unique_ratio": raw_to_unique_ratio,
        "shadow_duplicate_ratio": duplicate_ratio,
        "shadow_counter_health_status": counter_health_status,
        "shadow_raw_near_approved_count": len(raw_near),
        "shadow_near_approved_count": len(raw_near),
        "shadow_safe_near_approved_count": len(safe),
        "shadow_marginal_near_approved_count": len(marginal),
        "shadow_marginal_count": len(marginal),
        "shadow_unsafe_count": len(unsafe),
        "shadow_unsafe_rejection_count": len([item for item in combined if item.get("candidate_class") == "UNSAFE_REJECTION"]),
        "shadow_primary_blocked_count": len(primary_blocked),
        "shadow_secondary_blocked_count": len(secondary_blocked),
        "shadow_structure_missing_count": len(structure_missing),
        "shadow_confirmation_missing_count": len(confirmation_missing),
        "shadow_ignored_count": ignored_count,
        "shadow_ignored_reason": ignored_reason_counter.most_common(1)[0][0] if ignored_reason_counter else "",
        "shadow_counter_warning": warning,
        "shadow_counter_warning_reason": warning_reason,
        "shadow_scope_warning": warning,
        "shadow_scope_warning_reason": warning_reason,
        "shadow_would_enter_count": len(would_enter),
        "shadow_pending_count": len(pending),
        "shadow_would_win_count": len(wins),
        "shadow_would_lose_count": len(losses),
        "shadow_best_symbol": str(best.get("symbol") or ""),
        "shadow_best_strategy": str(best.get("strategy") or ""),
        "shadow_best_candidate_score": best.get("current_score"),
        "shadow_dominant_block_reason": dominant_block_reason,
        "shadow_dominant_block_reason_current": current_dominant_block,
        "shadow_dominant_block_reason_accumulated": accumulated_dominant_block,
        "dominant_exclusion_current_scope": current_dominant_block,
        "dominant_exclusion_accumulated_scope": accumulated_dominant_block,
        "fallback_blocker_scope": str(feed_scope.get("fallback_blocker_scope") or "UNKNOWN"),
        "fallback_current_count": int(feed_scope.get("current_cycle_fallback_count", 0) or 0),
        "fallback_accumulated_count": int(feed_scope.get("accumulated_fallback_count", 0) or 0),
        "fallback_scope_status": str(feed_scope.get("fallback_scope_status") or "UNKNOWN_SCOPE"),
        "fallback_scope_note": str(feed_scope.get("notes") or ""),
        "shadow_policy_recommendation": recommendation,
        "shadow_current_cycle_candidates": current_cycle_candidates[:MAX_RECENT_CANDIDATES],
        "shadow_accumulated_recent_candidates": accumulated_recent_candidates[:MAX_RECENT_CANDIDATES],
        "shadow_recent_candidates": accumulated_recent_candidates[:MAX_RECENT_CANDIDATES],
        "shadow_outcome_summary": {
            "pending": len(pending),
            "would_win": len(wins),
            "would_lose": len(losses),
            "would_flat": len([item for item in combined if item.get("outcome_label") == "WOULD_FLAT"]),
            "invalidated": len([item for item in combined if item.get("outcome_label") == "INVALIDATED"]),
        },
        "shadow_reason": "Diagnostic-only shadow simulation. No official trade, wallet, broker, score, or PnL was changed.",
        "shadow_stop_pct": STOP_SHADOW_PCT,
        "shadow_take_profit_pct": TAKE_PROFIT_SHADOW_PCT,
        "shadow_max_hold_cycles": MAX_HOLD_CYCLES_SHADOW,
    }
