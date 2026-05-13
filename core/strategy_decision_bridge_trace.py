from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any

from core.feed_scope_reconciliation import build_feed_scope_reconciliation


MODE = "SHADOW_ONLY"
MAX_RECENT_CANDIDATES = 12
CONFIRMED_BOS_STATES = {"BOS_BY_CLOSE_CONFIRMED", "BOS_RETEST_CONFIRMED"}
CONFIRMED_PIVOT_STATES = {"PIVOT_CONFIRMED", "PIVOT_TRIGGERED"}
PARTIAL_BOS_STATES = {"BOS_BY_CLOSE_WEAK", "BOS_BY_WICK_ONLY", "BOS_RETEST_PENDING"}
PARTIAL_PIVOT_STATES = {"PIVOT_FORMING"}
SAFE_RECOMMENDATIONS = {
    "observe_more",
    "study_real_strategy_blocker",
    "reconcile_feed_scope",
    "study_future_calibration",
    "no_threshold_change",
    "keep_current_strategy",
    "observe_current_feed_clean",
    "accumulated_fallback_only",
    "historical_fallback_only",
    "candidate_old_fallback_only",
    "visual_only_fallback",
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _round(value: Any, digits: int = 4) -> float | None:
    numeric = _as_float(value, None)
    return None if numeric is None else round(float(numeric), digits)


def _normalize_feed_status(value: Any) -> str:
    raw = str(value or "").strip().upper()
    if raw in {"LIVE", "DELAYED", "FALLBACK", "UNKNOWN"}:
        return raw
    legacy = raw.lower()
    if legacy == "healthy":
        return "LIVE"
    if legacy in {"degraded", "cached"}:
        return "DELAYED"
    if legacy == "error":
        return "FALLBACK"
    return "UNKNOWN"


def default_strategy_decision_bridge_trace_state(
    reason: str = "No strategy decision bridge trace data yet.",
) -> dict[str, Any]:
    return {
        "enabled": True,
        "mode": MODE,
        "generated_at": "",
        "provider_effective": "",
        "feed_status": "UNKNOWN",
        "symbols_analyzed": 0,
        "top_symbol": "",
        "top_bridge_status": "INSUFFICIENT_TRACE_DATA",
        "top_real_blocker": "",
        "top_structure_status": "",
        "top_reconciliation_status": "UNKNOWN_MISMATCH",
        "fallback_scope_status": "UNKNOWN_SCOPE",
        "fallback_blocker_scope": "UNKNOWN",
        "current_feed_is_clean": False,
        "structure_confirmed_but_blocked_count": 0,
        "fallback_scope_mismatch_count": 0,
        "multi_tf_vs_bos_mismatch_count": 0,
        "real_strategy_authority_count": 0,
        "should_keep_blocked_count": 0,
        "recommendation": "observe_more",
        "recent_candidates": [],
        "reason": reason,
        "shadow_only": True,
    }


def _symbol_from(row: dict[str, Any]) -> str:
    return str(row.get("symbol") or row.get("asset") or "").strip().upper()


def _latest_by_symbol(rows: list[dict[str, Any]], *, symbol_keys: tuple[str, ...] = ("symbol", "asset")) -> dict[str, dict[str, Any]]:
    mapped: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        symbol = ""
        for key in symbol_keys:
            symbol = str(row.get(key) or "").strip().upper()
            if symbol:
                break
        if symbol:
            mapped[symbol] = dict(row)
    return mapped


def _by_symbol_timeframe(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    mapped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        symbol = _symbol_from(row)
        timeframe = str(row.get("timeframe") or "").strip().lower()
        if symbol and timeframe:
            mapped.setdefault(symbol, {})[timeframe] = dict(row)
    return mapped


def _reason_tokens(signal: dict[str, Any]) -> list[str]:
    raw = signal.get("rejection_reasons", [])
    if isinstance(raw, str):
        raw = [raw]
    return [str(item or "").strip().lower() for item in list(raw or []) if str(item or "").strip()]


def _has(tokens: list[str], patterns: tuple[str, ...]) -> bool:
    return any(any(pattern in token for pattern in patterns) for token in tokens)


def _bottleneck_category(tokens: list[str]) -> str:
    if _has(tokens, ("fallback", "feed", "provider_unknown")):
        return "FEED_BLOCK"
    if _has(tokens, ("daily_loss", "position_limit", "cooldown", "duplicate", "broker")):
        return "GUARD_BLOCK"
    if _has(tokens, ("no_setup",)):
        return "NO_SETUP_ELIGIBLE"
    if _has(tokens, ("reversal_not_eligible",)):
        return "REVERSAL_NOT_ELIGIBLE"
    if _has(tokens, ("trend_not_confirmed", "tendencia", "trend")):
        return "TREND_NOT_CONFIRMED"
    if _has(tokens, ("rsi",)):
        return "RSI_OUT_OF_RANGE"
    if _has(tokens, ("momentum",)):
        return "MOMENTUM_WEAK"
    if _has(tokens, ("secondary", "secundaria", "confirmacao")):
        return "SECONDARY_CONFIRMATION_WEAK"
    if _has(tokens, ("score_below", "score", "abaixo")):
        return "SCORE_BELOW_MIN"
    if not tokens:
        return "NO_REAL_REJECTION"
    return "UNKNOWN"


def _primary_secondary_blockers(tokens: list[str]) -> tuple[str, str]:
    primary_priority = [
        ("FEED_OR_PROVIDER", ("fallback", "feed", "provider_unknown")),
        ("RISK_GUARD", ("daily_loss", "position_limit", "cooldown", "duplicate", "broker")),
        ("NO_SETUP_ELIGIBLE", ("no_setup",)),
        ("REVERSAL_NOT_ELIGIBLE", ("reversal_not_eligible",)),
        ("TREND_NOT_CONFIRMED", ("trend_not_confirmed", "tendencia", "trend")),
        ("RSI_OUT_OF_RANGE", ("rsi",)),
        ("SCORE_BELOW_MIN", ("score_below", "score", "abaixo")),
    ]
    secondary_priority = [
        ("SECONDARY_CONFIRMATION_WEAK", ("secondary", "secundaria", "confirmacao")),
        ("MOMENTUM_WEAK", ("momentum",)),
        ("BREAKOUT_NOT_CONFIRMED", ("breakout", "rompimento")),
    ]
    primary = next((name for name, patterns in primary_priority if _has(tokens, patterns)), "")
    secondary = next((name for name, patterns in secondary_priority if _has(tokens, patterns)), "")
    return primary, secondary


def _candidate_feed_status(signal: dict[str, Any]) -> str:
    source = str(signal.get("data_source") or "").strip().lower()
    if source == "market":
        return "LIVE"
    if source in {"fallback", "synthetic"}:
        return "FALLBACK"
    if source in {"cached", "stale"}:
        return "DELAYED"
    return "UNKNOWN"


def _current_fallback_active(market_data_status: dict[str, Any]) -> bool:
    feed_status = _normalize_feed_status(market_data_status.get("feed_status") or market_data_status.get("status"))
    source_breakdown = dict(market_data_status.get("source_breakdown", {}) or {})
    fallback_count = int(source_breakdown.get("fallback", 0) or 0)
    return bool(feed_status == "FALLBACK" or fallback_count > 0)


def _fallback_scope(
    *,
    signal: dict[str, Any],
    market_data_status: dict[str, Any],
    validation_state: dict[str, Any],
    shadow_decision_simulator: dict[str, Any],
) -> tuple[str, bool, bool]:
    current = _current_fallback_active(market_data_status) or _candidate_feed_status(signal) == "FALLBACK"
    accumulated_count = int(validation_state.get("fallback_rejection_accumulated_count", 0) or 0)
    current_count = int(validation_state.get("fallback_rejection_current_cycle_count", 0) or 0)
    dominant_shadow = str(shadow_decision_simulator.get("shadow_dominant_block_reason") or "").lower()
    accumulated = bool(accumulated_count > 0 or "fallback" in dominant_shadow)
    if current or current_count > 0:
        return "CURRENT_CYCLE", bool(current), bool(accumulated)
    if accumulated_count > 0:
        return "ACCUMULATED", False, True
    if "fallback" in dominant_shadow:
        return "HISTORICAL", False, True
    return "NONE", False, bool(accumulated)


def _mtf_for_symbol(multi_timeframe_swing_audit: dict[str, Any], symbol: str) -> dict[str, Any]:
    for row in list(multi_timeframe_swing_audit.get("recent_candidates", []) or []):
        if isinstance(row, dict) and _symbol_from(row) == symbol:
            return dict(row)
    return {}


def _bos_rows_for_symbol(bos_pivot_trace_audit: dict[str, Any], symbol: str) -> dict[str, dict[str, Any]]:
    return _by_symbol_timeframe(list(bos_pivot_trace_audit.get("recent_candidates", []) or [])).get(symbol, {})


def _market_structure_for_symbol(market_structure_audit: dict[str, Any], symbol: str) -> dict[str, Any]:
    for row in list(market_structure_audit.get("market_structure_best_candidates", []) or []):
        if isinstance(row, dict) and _symbol_from(row) == symbol:
            return dict(row)
    return {}


def _shadow_for_symbol(shadow_decision_simulator: dict[str, Any], symbol: str) -> dict[str, Any]:
    rows = list(shadow_decision_simulator.get("shadow_current_cycle_candidates", []) or [])
    rows += list(shadow_decision_simulator.get("shadow_accumulated_recent_candidates", []) or [])
    rows += list(shadow_decision_simulator.get("shadow_recent_candidates", []) or [])
    for row in rows:
        if isinstance(row, dict) and _symbol_from(row) == symbol:
            return dict(row)
    return {}


def _structure_status(h4: dict[str, Any], h1: dict[str, Any]) -> str:
    h4_bos = str(h4.get("bos_state") or "")
    h1_bos = str(h1.get("bos_state") or "")
    h4_pivot = str(h4.get("pivot_state") or "")
    h1_pivot = str(h1.get("pivot_state") or "")
    if h4_bos in CONFIRMED_BOS_STATES and h4_pivot in CONFIRMED_PIVOT_STATES:
        return "STRUCTURE_CONFIRMED"
    if h4_bos in CONFIRMED_BOS_STATES or h1_bos in CONFIRMED_BOS_STATES:
        return "STRUCTURE_CONFIRMED"
    if h4_bos in PARTIAL_BOS_STATES or h1_bos in PARTIAL_BOS_STATES or h4_pivot in PARTIAL_PIVOT_STATES or h1_pivot in PARTIAL_PIVOT_STATES:
        return "STRUCTURE_PARTIAL"
    if not h4 and not h1:
        return "STRUCTURE_UNKNOWN"
    return "STRUCTURE_MISSING"


def _decision_bridge_status(
    *,
    structure_status: str,
    real_blocked: bool,
    fallback_scope: str,
    daily_loss_block_active: bool,
    position_limit_block_active: bool,
    trace_available: bool,
) -> str:
    if fallback_scope == "CURRENT_CYCLE":
        return "FEED_OR_DATA_BLOCKED"
    if daily_loss_block_active or position_limit_block_active:
        return "GUARD_BLOCKED"
    if not trace_available:
        return "INSUFFICIENT_TRACE_DATA"
    if real_blocked and structure_status == "STRUCTURE_CONFIRMED":
        return "STRUCTURE_CONFIRMED_BUT_REAL_BLOCKED"
    if real_blocked and structure_status == "STRUCTURE_PARTIAL":
        return "STRUCTURE_PARTIAL_AND_REAL_BLOCKED"
    if real_blocked and structure_status in {"STRUCTURE_MISSING", "STRUCTURE_UNKNOWN"}:
        return "STRUCTURE_MISSING_AND_REAL_BLOCKED"
    if real_blocked:
        return "REAL_STRATEGY_BLOCKED"
    return "REAL_STRATEGY_BLOCKED"


def _reconciliation_status(
    *,
    symbol: str,
    primary_blocker: str,
    fallback_scope: str,
    feed_scope_reconciliation: dict[str, Any],
    multi_tf: dict[str, Any],
    h4: dict[str, Any],
    h1: dict[str, Any],
    bos_pivot_trace_audit: dict[str, Any],
    multi_timeframe_swing_audit: dict[str, Any],
) -> tuple[str, str]:
    mtf_status = str(multi_tf.get("alignment_status") or multi_timeframe_swing_audit.get("top_alignment_status") or "")
    mtf_missing = [str(item) for item in list(multi_tf.get("missing_for_setup", []) or [])]
    h4_confirmed = str(h4.get("bos_state") or "") in CONFIRMED_BOS_STATES
    h1_confirmed = str(h1.get("bos_state") or "") in CONFIRMED_BOS_STATES
    top_mtf = str(multi_timeframe_swing_audit.get("top_symbol") or "").upper()
    top_bos = str(bos_pivot_trace_audit.get("top_symbol") or "").upper()
    if fallback_scope in {"ACCUMULATED", "HISTORICAL", "CANDIDATE_OLD", "VISUAL_CHART_ONLY"}:
        note = str(feed_scope_reconciliation.get("notes") or "Fallback is not a current-cycle worker blocker.")
        return "FALLBACK_SCOPE_MISMATCH", note
    if primary_blocker == "NO_SETUP_ELIGIBLE" and (h4_confirmed or h1_confirmed):
        return "BOS_CONFIRMED_BUT_REAL_SETUP_MISSING", "BOS/Pivot trace has confirmed structure, but the real setup was not eligible."
    if mtf_status == "INSUFFICIENT_DATA" and (h4_confirmed or h1_confirmed):
        return "STRUCTURE_CONFIRMED_BUT_MULTI_TF_INSUFFICIENT", "BOS/Pivot trace confirms structure while multi-timeframe aggregate remains insufficient."
    if "h4_bos_missing" in mtf_missing and h4_confirmed:
        applies = "same top symbol" if top_mtf == top_bos == symbol else "watchlist aggregate or different top symbol"
        return "MULTI_TF_ALIGNED_BUT_BOS_MISSING", f"h4_bos_missing may be from {applies}, not necessarily the BOS/Pivot top trace."
    if top_mtf and top_bos and top_mtf != top_bos:
        return "UNKNOWN_MISMATCH", "Multi-TF and BOS/Pivot top symbols differ."
    return "CONSISTENT_BLOCK", "Real strategy remains authoritative and no unsafe reconciliation was applied."


def _safe_recommendation(candidate: dict[str, Any]) -> str:
    if candidate["fallback_blocker_scope"] == "ACCUMULATED":
        return "accumulated_fallback_only"
    if candidate["fallback_blocker_scope"] == "HISTORICAL":
        return "historical_fallback_only"
    if candidate["fallback_blocker_scope"] == "CANDIDATE_OLD":
        return "candidate_old_fallback_only"
    if candidate["fallback_blocker_scope"] == "VISUAL_CHART_ONLY":
        return "visual_only_fallback"
    if candidate["decision_bridge_status"] == "STRUCTURE_CONFIRMED_BUT_REAL_BLOCKED":
        if candidate["primary_real_blocker"] in {"SCORE_BELOW_MIN", "RSI_OUT_OF_RANGE", "NO_SETUP_ELIGIBLE"}:
            return "study_real_strategy_blocker"
        return "study_future_calibration"
    if candidate["reconciliation_status"] != "CONSISTENT_BLOCK":
        return "observe_more"
    return "keep_current_strategy"


def _candidate_payload(
    *,
    symbol: str,
    signal: dict[str, Any],
    shadow: dict[str, Any],
    multi_tf: dict[str, Any],
    bos_rows: dict[str, dict[str, Any]],
    market_structure: dict[str, Any],
    market_data_status: dict[str, Any],
    validation_state: dict[str, Any],
    shadow_decision_simulator: dict[str, Any],
    bos_pivot_trace_audit: dict[str, Any],
    multi_timeframe_swing_audit: dict[str, Any],
    feed_scope_reconciliation: dict[str, Any],
    context_status: str,
    daily_loss_block_active: bool,
    position_limit_block_active: bool,
) -> dict[str, Any]:
    tokens = _reason_tokens(signal)
    primary, secondary = _primary_secondary_blockers(tokens)
    category = _bottleneck_category(tokens)
    score = _as_float(signal.get("score"), None)
    min_score = _as_float(signal.get("effective_min_signal_score") or signal.get("base_min_signal_score"), None)
    score_gap = None if score is None or min_score is None else max(0.0, float(min_score) - float(score))
    h4 = dict(bos_rows.get("4h", {}) or {})
    h1 = dict(bos_rows.get("1h", {}) or {})
    structure_status = _structure_status(h4, h1)
    if feed_scope_reconciliation:
        fallback_scope = str(feed_scope_reconciliation.get("fallback_blocker_scope") or "UNKNOWN")
        fallback_current = bool(fallback_scope == "CURRENT_CYCLE")
        fallback_accumulated = bool(fallback_scope in {"ACCUMULATED", "HISTORICAL", "CANDIDATE_OLD"})
    else:
        fallback_scope, fallback_current, fallback_accumulated = _fallback_scope(
            signal=signal,
            market_data_status=market_data_status,
            validation_state=validation_state,
            shadow_decision_simulator=shadow_decision_simulator,
        )
    reconciliation, reconciliation_reason = _reconciliation_status(
        symbol=symbol,
        primary_blocker=primary,
        fallback_scope=fallback_scope,
        feed_scope_reconciliation=feed_scope_reconciliation,
        multi_tf=multi_tf,
        h4=h4,
        h1=h1,
        bos_pivot_trace_audit=bos_pivot_trace_audit,
        multi_timeframe_swing_audit=multi_timeframe_swing_audit,
    )
    real_blocked = bool(tokens or signal.get("buy") is False)
    status = _decision_bridge_status(
        structure_status=structure_status,
        real_blocked=real_blocked,
        fallback_scope=fallback_scope,
        daily_loss_block_active=daily_loss_block_active,
        position_limit_block_active=position_limit_block_active,
        trace_available=bool(signal or h4 or h1 or multi_tf),
    )
    candidate = {
        "symbol": symbol,
        "real_strategy": str(signal.get("strategy_name") or shadow.get("strategy") or "trend_pullback_breakout"),
        "real_score": _round(score),
        "min_score": _round(min_score),
        "score_gap": _round(score_gap),
        "real_rejection_reason": ", ".join(tokens) if tokens else "",
        "real_bottleneck_category": category,
        "primary_real_blocker": primary,
        "secondary_real_blocker": secondary,
        "feed_status_current": _normalize_feed_status(market_data_status.get("feed_status") or market_data_status.get("status")),
        "feed_status_used_by_candidate": _candidate_feed_status(signal),
        "fallback_current": bool(fallback_current),
        "fallback_accumulated": bool(fallback_accumulated),
        "fallback_blocker_scope": fallback_scope,
        "fallback_scope_status": str(feed_scope_reconciliation.get("fallback_scope_status") or "UNKNOWN_SCOPE"),
        "fallback_mismatch_reason": str(feed_scope_reconciliation.get("notes") or ""),
        "current_feed_is_clean": bool(feed_scope_reconciliation.get("current_feed_is_clean", False)),
        "fallback_is_historical_only": bool(fallback_scope in {"ACCUMULATED", "HISTORICAL", "CANDIDATE_OLD"}),
        "fallback_is_current_cycle": bool(fallback_scope == "CURRENT_CYCLE"),
        "context_status": str(signal.get("context_status") or context_status or "UNKNOWN"),
        "risk_guard_status": "BLOCKED" if daily_loss_block_active or position_limit_block_active else "CLEAR",
        "daily_loss_block_active": bool(daily_loss_block_active),
        "position_limit_block_active": bool(position_limit_block_active),
        "multi_tf_alignment_status": str(multi_tf.get("alignment_status") or ""),
        "multi_tf_score": _round(multi_tf.get("alignment_score")),
        "multi_tf_missing_for_setup": list(multi_tf.get("missing_for_setup", []) or []),
        "bos_pivot_status": structure_status,
        "bos_state_4h": str(h4.get("bos_state") or ""),
        "bos_state_1h": str(h1.get("bos_state") or ""),
        "pivot_state_4h": str(h4.get("pivot_state") or ""),
        "pivot_state_1h": str(h1.get("pivot_state") or ""),
        "relationship_1h_4h": str((h4 or h1).get("relationship_to_higher_tf") or ""),
        "fibonacci_structure_score": _round(
            market_structure.get("market_structure_score") or market_structure.get("structure_score")
        ),
        "fibonacci_confluence": bool(
            market_structure.get("structure_confirms_trend_pullback")
            or market_structure.get("structure_would_improve_candidate_quality")
            or market_structure.get("current_fib_zone") in {"SHALLOW_ZONE", "MEDIUM_ZONE", "DEEP_ZONE"}
        ),
        "shadow_candidate_class": str(shadow.get("candidate_class") or ""),
        "shadow_safe_candidate": bool(shadow.get("safe_candidate", False)),
        "shadow_would_enter": bool(shadow.get("shadow_would_enter", False)),
        "real_strategy_still_authoritative": True,
        "decision_bridge_status": status,
        "reconciliation_status": reconciliation,
        "reconciliation_reason": reconciliation_reason,
        "final_bridge_reason": "",
        "recommendation": "observe_more",
        "should_keep_blocked": True,
    }
    candidate["final_bridge_reason"] = (
        f"{structure_status.lower()} with real_blocker={primary or category}; "
        f"fallback_scope={fallback_scope}; reconciliation={reconciliation}; "
        "real strategy remains authoritative."
    )
    candidate["recommendation"] = _safe_recommendation(candidate)
    if candidate["recommendation"] not in SAFE_RECOMMENDATIONS:
        candidate["recommendation"] = "observe_more"
    return candidate


def build_strategy_decision_bridge_trace(
    *,
    signals: list[dict[str, Any]] | None = None,
    shadow_decision_simulator: dict[str, Any] | None = None,
    multi_timeframe_swing_audit: dict[str, Any] | None = None,
    bos_pivot_trace_audit: dict[str, Any] | None = None,
    market_structure_audit: dict[str, Any] | None = None,
    fib_alignment_audit: dict[str, Any] | None = None,
    market_data_status: dict[str, Any] | None = None,
    validation_state: dict[str, Any] | None = None,
    paper_state: dict[str, Any] | None = None,
    feed_scope_reconciliation: dict[str, Any] | None = None,
    daily_loss_block_active: bool = False,
    slots_left: int | None = None,
    enabled: bool = True,
) -> dict[str, Any]:
    if not enabled:
        disabled = default_strategy_decision_bridge_trace_state("Strategy decision bridge trace disabled.")
        disabled["enabled"] = False
        return disabled

    signal_map = _latest_by_symbol(list(signals or []), symbol_keys=("asset", "symbol"))
    shadow_payload = dict(shadow_decision_simulator or {})
    mtf_payload = dict(multi_timeframe_swing_audit or {})
    bos_payload = dict(bos_pivot_trace_audit or {})
    market_payload = dict(market_structure_audit or {})
    status = dict(market_data_status or {})
    validation = dict(validation_state or {})
    state = dict(paper_state or {})
    feed_scope = dict(
        feed_scope_reconciliation
        or build_feed_scope_reconciliation(
            market_data_status=status,
            validation_state=validation,
            shadow_decision_simulator=shadow_payload,
            signals=list(signals or []),
        )
    )
    context_status = str((state.get("market_context", {}) or {}).get("market_context_status") or "")
    position_limit_block_active = bool(slots_left is not None and int(slots_left or 0) <= 0 and bool(state.get("positions")))
    symbols = set(signal_map.keys())
    symbols.update(
        _symbol_from(row)
        for row in list(bos_payload.get("recent_candidates", []) or [])
        if isinstance(row, dict) and _symbol_from(row)
    )
    symbols.update(
        _symbol_from(row)
        for row in list(mtf_payload.get("recent_candidates", []) or [])
        if isinstance(row, dict) and _symbol_from(row)
    )
    symbols.update(
        _symbol_from(row)
        for row in list(shadow_payload.get("shadow_recent_candidates", []) or [])
        if isinstance(row, dict) and _symbol_from(row)
    )
    bos_by_symbol = _by_symbol_timeframe(list(bos_payload.get("recent_candidates", []) or []))
    candidates = [
        _candidate_payload(
            symbol=symbol,
            signal=signal_map.get(symbol, {}),
            shadow=_shadow_for_symbol(shadow_payload, symbol),
            multi_tf=_mtf_for_symbol(mtf_payload, symbol),
            bos_rows=bos_by_symbol.get(symbol, {}),
            market_structure=_market_structure_for_symbol(market_payload, symbol),
            market_data_status=status,
            validation_state=validation,
            shadow_decision_simulator=shadow_payload,
            bos_pivot_trace_audit=bos_payload,
            multi_timeframe_swing_audit=mtf_payload,
            feed_scope_reconciliation=feed_scope,
            context_status=context_status,
            daily_loss_block_active=daily_loss_block_active,
            position_limit_block_active=position_limit_block_active,
        )
        for symbol in sorted(symbols)
    ]
    candidates.sort(
        key=lambda row: (
            1 if row.get("bos_pivot_status") == "STRUCTURE_CONFIRMED" else 0,
            float(row.get("real_score") or 0.0),
        ),
        reverse=True,
    )
    top = candidates[0] if candidates else {}
    bridge_counts = Counter(str(row.get("decision_bridge_status") or "") for row in candidates)
    reconciliation_counts = Counter(str(row.get("reconciliation_status") or "") for row in candidates)
    recommendation = str(top.get("recommendation") or "observe_more")
    if recommendation not in SAFE_RECOMMENDATIONS:
        recommendation = "observe_more"
    return {
        "enabled": True,
        "mode": MODE,
        "generated_at": _utc_now_iso(),
        "provider_effective": str(status.get("provider_effective") or status.get("provider") or ""),
        "feed_status": _normalize_feed_status(status.get("feed_status") or status.get("status")),
        "symbols_analyzed": int(len(candidates)),
        "top_symbol": str(top.get("symbol") or ""),
        "top_bridge_status": str(top.get("decision_bridge_status") or "INSUFFICIENT_TRACE_DATA"),
        "top_real_blocker": str(top.get("primary_real_blocker") or top.get("real_bottleneck_category") or ""),
        "top_structure_status": str(top.get("bos_pivot_status") or ""),
        "top_reconciliation_status": str(top.get("reconciliation_status") or "UNKNOWN_MISMATCH"),
        "fallback_scope_status": str(feed_scope.get("fallback_scope_status") or "UNKNOWN_SCOPE"),
        "fallback_blocker_scope": str(feed_scope.get("fallback_blocker_scope") or "UNKNOWN"),
        "current_feed_is_clean": bool(feed_scope.get("current_feed_is_clean", False)),
        "structure_confirmed_but_blocked_count": int(
            bridge_counts.get("STRUCTURE_CONFIRMED_BUT_REAL_BLOCKED", 0)
        ),
        "fallback_scope_mismatch_count": int(reconciliation_counts.get("FALLBACK_SCOPE_MISMATCH", 0)),
        "multi_tf_vs_bos_mismatch_count": int(
            reconciliation_counts.get("STRUCTURE_CONFIRMED_BUT_MULTI_TF_INSUFFICIENT", 0)
            + reconciliation_counts.get("MULTI_TF_ALIGNED_BUT_BOS_MISSING", 0)
        ),
        "real_strategy_authority_count": int(sum(1 for row in candidates if row.get("real_strategy_still_authoritative"))),
        "should_keep_blocked_count": int(sum(1 for row in candidates if row.get("should_keep_blocked"))),
        "recommendation": recommendation,
        "recent_candidates": candidates[:MAX_RECENT_CANDIDATES],
        "fib_alignment_status": str((fib_alignment_audit or {}).get("fib_alignment_status") or ""),
        "reason": "Strategy decision bridge reconciles real blockers with shadow structure diagnostics only.",
        "shadow_only": True,
    }
