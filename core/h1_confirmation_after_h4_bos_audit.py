"""Diagnostic-only H1 confirmation audit after confirmed H4 BOS.

This module explains why a confirmed H4 BOS/retest still lacks enough 1H
confirmation. It never changes strategy decisions, scores, thresholds, orders,
positions, broker behavior, or official paper-trading state.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


MODE = "DIAGNOSTIC_ONLY"
SAFETY_MODE = "SHADOW_ONLY"
TARGET_SETUP = "trend_pullback_breakout"
MAX_CANDIDATES = 10

H4_CONFIRMED_BOS_STATES = {
    "BOS_RETEST_CONFIRMED",
    "BOS_BY_CLOSE_CONFIRMED",
    "BOS_CONFIRMED_STRONG",
    "BOS_CONFIRMED_WEAK",
}
H1_CONFIRMED_BOS_STATES = {
    "BOS_RETEST_CONFIRMED",
    "BOS_BY_CLOSE_CONFIRMED",
    "BOS_CONFIRMED_STRONG",
}
H1_WEAK_BOS_STATES = {"BOS_BY_CLOSE_WEAK", "BOS_CONFIRMED_WEAK"}
INSUFFICIENT_BOS_STATES = {"", "INSUFFICIENT_DATA", "UNKNOWN"}
SIDEWAYS_STATES = {"SIDEWAYS", "RANGE", "NEUTRAL", "INCONCLUSIVE"}

SAFE_RECOMMENDATIONS = {
    "observe_more",
    "study_h1_confirmation",
    "study_h1_retest_quality",
    "study_h1_after_h4_bos_mapping",
    "study_multitf_confirmation",
    "study_entry_timing_risk",
    "keep_blocked_until_h1_confirms",
    "keep_blocked_until_h1_retest_confirms",
    "keep_blocked_until_h1_structure_is_clear",
    "keep_blocked_until_multitf_confirms",
    "no_threshold_change_recommended",
    "no_strategy_change_recommended",
    "insufficient_data",
}

FORBIDDEN_MESSAGE_FRAGMENTS = {
    "entrada aprovada",
    "pode comprar",
    "opere agora",
    "reduza score",
    "ignore 1h",
    "h4 e suficiente para entrada real",
    "h4 é suficiente para entrada real",
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _as_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "sim", "y"}
    return bool(value)


def _normalize_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text or default


def _upper(value: Any, default: str = "") -> str:
    return _normalize_text(value, default).upper()


def _safe_recommendation(value: Any, default: str = "observe_more") -> str:
    recommendation = _normalize_text(value, default)
    return recommendation if recommendation in SAFE_RECOMMENDATIONS else default


def _safe_message(message: str) -> str:
    text = _normalize_text(message)
    lowered = text.lower()
    if not text or any(fragment in lowered for fragment in FORBIDDEN_MESSAGE_FRAGMENTS):
        return "Confirmacao 1H segue em diagnostico: manter bloqueado ate evidencias objetivas."
    return text


def default_h1_confirmation_after_h4_bos_audit_state(
    reason: str = "No H1-after-H4 BOS confirmation audit data yet.",
) -> Dict[str, Any]:
    return {
        "enabled": True,
        "mode": MODE,
        "safety_mode": SAFETY_MODE,
        "status": "INSUFFICIENT_DATA",
        "generated_at": "",
        "target_setup": TARGET_SETUP,
        "total_candidates_checked": 0,
        "h4_bos_confirmed_count": 0,
        "h4_retest_confirmed_count": 0,
        "h1_missing_confirmation_count": 0,
        "top_symbol": "",
        "top_setup": TARGET_SETUP,
        "h4_bos_state": "INSUFFICIENT_DATA",
        "h4_retest_state": "UNKNOWN",
        "h1_bos_state": "INSUFFICIENT_DATA",
        "h1_confirmation_state": "INSUFFICIENT_DATA",
        "h1_confirmation_status": "INSUFFICIENT_DATA_FOR_H1_CONFIRMATION",
        "h1_failure_reason": "insufficient_data",
        "h1_data_quality": "missing",
        "h1_trend_direction": "INCONCLUSIVE",
        "h4_trend_direction": "INCONCLUSIVE",
        "h1_h4_alignment": "INSUFFICIENT_DATA",
        "h1_retest_state": "UNKNOWN",
        "h1_pivot_state": "INSUFFICIENT_DATA",
        "h1_entry_timing_risk": "UNKNOWN",
        "current_feed_is_clean": False,
        "fallback_blocker_scope": "UNKNOWN",
        "fallback_scope_status": "UNKNOWN_SCOPE",
        "recommendation": "insufficient_data",
        "should_keep_blocked": True,
        "safe_to_change_strategy_now": False,
        "safe_to_change_threshold_now": False,
        "notes": reason,
        "candidates": [],
        "shadow_only": True,
    }


def _feed_scope(feed_scope_reconciliation: Any, state: Any) -> Tuple[bool, str, str]:
    feed_scope = _as_dict(feed_scope_reconciliation)
    if not feed_scope:
        feed_scope = _as_dict(_as_dict(state).get("feed_scope_reconciliation"))
    current_feed_is_clean = _as_bool(feed_scope.get("current_feed_is_clean"))
    fallback_blocker_scope = _upper(feed_scope.get("fallback_blocker_scope"), "UNKNOWN")
    fallback_scope_status = _upper(feed_scope.get("fallback_scope_status"), "UNKNOWN")
    return current_feed_is_clean, fallback_blocker_scope, fallback_scope_status


def _signal_index(signals: Optional[Sequence[Mapping[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    indexed: Dict[str, Dict[str, Any]] = {}
    for signal in signals or []:
        row = _as_dict(signal)
        symbol = _normalize_text(row.get("symbol") or row.get("asset"))
        if symbol:
            indexed.setdefault(symbol.upper(), row)
    return indexed


def _rows_by_symbol(rows: Iterable[Any]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for raw in rows:
        row = _as_dict(raw)
        symbol = _normalize_text(row.get("symbol") or row.get("asset"))
        if symbol:
            grouped.setdefault(symbol.upper(), []).append(row)
    return grouped


def _find_symbol_row(rows: Iterable[Any], symbol: str) -> Dict[str, Any]:
    target = symbol.upper()
    for raw in rows:
        row = _as_dict(raw)
        if _normalize_text(row.get("symbol") or row.get("asset")).upper() == target:
            return row
    return {}


def _find_timeframe_row(rows: Sequence[Mapping[str, Any]], timeframe: str) -> Dict[str, Any]:
    target = timeframe.lower()
    for raw in rows:
        row = _as_dict(raw)
        if _normalize_text(row.get("timeframe")).lower() == target:
            return row
    return {}


def _timeframe_quality(mtf_row: Mapping[str, Any], timeframe: str, fallback: str = "missing") -> str:
    for diagnostic in _as_list(mtf_row.get("timeframe_diagnostics")):
        row = _as_dict(diagnostic)
        if _normalize_text(row.get("timeframe")).lower() == timeframe.lower():
            return _normalize_text(row.get("data_quality"), fallback)
    key = f"{timeframe.lower()}_data_quality"
    return _normalize_text(mtf_row.get(key), fallback)


def _score_fields(
    signal: Mapping[str, Any],
    taxonomy_row: Mapping[str, Any],
    no_setup_row: Mapping[str, Any],
    bridge_row: Mapping[str, Any],
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    score = _as_float(
        taxonomy_row.get("score")
        or no_setup_row.get("score")
        or bridge_row.get("real_score")
        or bridge_row.get("score")
        or signal.get("score")
        or signal.get("adjusted_score")
    )
    min_score = _as_float(
        taxonomy_row.get("min_score")
        or no_setup_row.get("min_score")
        or bridge_row.get("min_score")
        or signal.get("min_score")
        or signal.get("min_signal_score")
        or signal.get("effective_min_signal_score")
    )
    gap = _as_float(
        taxonomy_row.get("score_gap")
        or no_setup_row.get("score_gap")
        or bridge_row.get("score_gap")
        or signal.get("score_gap")
    )
    if gap is None and score is not None and min_score is not None:
        gap = round(max(0.0, min_score - score), 6)
    return score, min_score, gap


def _is_h4_confirmed(state: Any) -> bool:
    return _upper(state) in H4_CONFIRMED_BOS_STATES


def _is_h1_confirmed(state: Any) -> bool:
    return _upper(state) in H1_CONFIRMED_BOS_STATES


def _is_h1_weak(state: Any) -> bool:
    return _upper(state) in H1_WEAK_BOS_STATES


def _retest_state(row: Mapping[str, Any], bos_state: str) -> str:
    bos_state_u = _upper(bos_state, "UNKNOWN")
    if bos_state_u == "BOS_RETEST_CONFIRMED" or _as_bool(row.get("retest_hold")):
        return "RETEST_CONFIRMED"
    if bos_state_u == "BOS_RETEST_PENDING" or _as_bool(row.get("retest_detected")):
        return "RETEST_PENDING"
    if _is_h4_confirmed(bos_state_u) or _is_h1_confirmed(bos_state_u) or _is_h1_weak(bos_state_u):
        return "RETEST_UNKNOWN"
    return "RETEST_MISSING"


def _h1_h4_alignment(h1_direction: str, h4_direction: str, relationship: str) -> str:
    h1 = _upper(h1_direction, "INCONCLUSIVE")
    h4 = _upper(h4_direction, "INCONCLUSIVE")
    rel = _upper(relationship, "")
    if rel in {"H1_CONFLICTS_H4", "H1_NOISE_ONLY"}:
        return "CONFLICT"
    if not h1 or h1 in {"INSUFFICIENT", "INCONCLUSIVE", "UNKNOWN"}:
        return "INSUFFICIENT_DATA"
    if h1 in SIDEWAYS_STATES:
        return "SIDEWAYS"
    if h4 in {"INSUFFICIENT", "INCONCLUSIVE", "UNKNOWN"}:
        return "H4_INCONCLUSIVE"
    if h1 == h4:
        return "ALIGNED"
    return "CONFLICT"


def _entry_timing_risk(status: str) -> str:
    if status in {
        "H1_INSUFFICIENT_DATA_AFTER_H4_BOS",
        "H1_NO_BOS_AFTER_H4_BOS",
        "H1_PIVOT_FORMING_AFTER_H4_BOS",
        "H1_RETEST_PENDING_AFTER_H4_BOS",
        "H4_BOS_CONFIRMED_BUT_H1_NOT_READY",
    }:
        return "ENTRY_TOO_EARLY_RISK"
    if status in {"H1_CONFLICTS_WITH_H4_BOS", "H1_SIDEWAYS_AFTER_H4_BOS"}:
        return "ENTRY_QUALITY_RISK"
    if status == "H1_CONFIRMED_AFTER_H4_BOS":
        return "TIMING_CONFIRMED_SHADOW_ONLY"
    if status == "H1_WEAK_CONFIRMATION_AFTER_H4_BOS":
        return "ENTRY_TOO_LATE_OR_WEAK_RISK"
    return "UNKNOWN"


def _message_for_status(status: str) -> str:
    status_u = _upper(status)
    if status_u == "H1_INSUFFICIENT_DATA_AFTER_H4_BOS":
        return "4H confirmou BOS/reteste, mas o 1H ainda nao tem dados suficientes para confirmacao."
    if status_u == "H1_NO_BOS_AFTER_H4_BOS":
        return "4H confirmou estrutura, mas o 1H ainda nao confirmou BOS."
    if status_u == "H1_CONFLICTS_WITH_H4_BOS":
        return "4H confirmou estrutura, mas o 1H esta contra a direcao do 4H; manter bloqueado."
    if status_u == "H1_SIDEWAYS_AFTER_H4_BOS":
        return "4H confirmou estrutura, mas o 1H esta lateral ou sem forca; manter bloqueado."
    if status_u == "H1_PIVOT_FORMING_AFTER_H4_BOS":
        return "1H esta formando pivo apos BOS 4H, mas ainda falta acionamento."
    if status_u == "H1_RETEST_PENDING_AFTER_H4_BOS":
        return "1H confirmou parcialmente, mas ainda falta reteste ou continuidade."
    if status_u == "H1_CONFIRMED_AFTER_H4_BOS":
        return "1H confirmou continuidade em shadow, mas a decisao oficial continua bloqueada pela estrategia real."
    if status_u == "H4_BOS_NOT_CONFIRMED":
        return "4H ainda nao confirmou BOS; a verificacao 1H apos BOS 4H permanece inconclusiva."
    return "Confirmacao 1H apos BOS 4H ainda inconclusiva; manter bloqueado."


def _classify_h1_confirmation(
    *,
    h4_bos_state: str,
    h4_retest_state: str,
    h1_bos_state: str,
    h1_retest_state: str,
    h1_pivot_state: str,
    h1_data_quality: str,
    h1_trend_direction: str,
    h4_trend_direction: str,
    h1_h4_alignment: str,
) -> Tuple[str, str, str, str]:
    h4_bos = _upper(h4_bos_state, "UNKNOWN")
    h1_bos = _upper(h1_bos_state, "UNKNOWN")
    h1_pivot = _upper(h1_pivot_state, "UNKNOWN")
    quality = _upper(h1_data_quality, "MISSING")
    h1_dir = _upper(h1_trend_direction, "INCONCLUSIVE")
    alignment = _upper(h1_h4_alignment, "UNKNOWN")

    if not _is_h4_confirmed(h4_bos):
        return "H4_BOS_NOT_CONFIRMED", "h4_bos_missing", "INSUFFICIENT_DATA", "observe_more"
    if h1_bos in INSUFFICIENT_BOS_STATES or quality in {"MISSING", "INSUFFICIENT", "INSUFFICIENT_DATA"}:
        return (
            "H1_INSUFFICIENT_DATA_AFTER_H4_BOS",
            "h1_insufficient_data",
            "INSUFFICIENT_DATA",
            "keep_blocked_until_h1_confirms",
        )
    if alignment == "CONFLICT":
        return "H1_CONFLICTS_WITH_H4_BOS", "h1_trend_conflict", "CONFLICT", "keep_blocked_until_multitf_confirms"
    if h1_dir in SIDEWAYS_STATES or alignment == "SIDEWAYS":
        return "H1_SIDEWAYS_AFTER_H4_BOS", "h1_sideways", "SIDEWAYS", "study_multitf_confirmation"
    if h1_pivot == "PIVOT_FORMING":
        return "H1_PIVOT_FORMING_AFTER_H4_BOS", "h1_pivot_forming_only", "PIVOT_FORMING", "study_h1_after_h4_bos_mapping"
    if h1_bos == "NO_BOS":
        return "H1_NO_BOS_AFTER_H4_BOS", "h1_no_bos", "NO_BOS", "keep_blocked_until_h1_confirms"
    if _is_h1_weak(h1_bos):
        return "H1_WEAK_CONFIRMATION_AFTER_H4_BOS", "h1_confirmation_candle_missing", "WEAK_CONFIRMATION", "study_h1_confirmation"
    if _is_h1_confirmed(h1_bos) and h1_retest_state != "RETEST_CONFIRMED":
        return "H1_RETEST_PENDING_AFTER_H4_BOS", "h1_retest_pending", "BOS_CONFIRMED_RETEST_PENDING", "keep_blocked_until_h1_retest_confirms"
    if _is_h1_confirmed(h1_bos) and h1_retest_state == "RETEST_CONFIRMED":
        return "H1_CONFIRMED_AFTER_H4_BOS", "unknown", "CONFIRMED", "no_strategy_change_recommended"
    return "H4_BOS_CONFIRMED_BUT_H1_NOT_READY", "h1_confirmation_candle_missing", "NOT_READY", "study_h1_confirmation"


def _build_candidate(
    *,
    symbol: str,
    h4_row: Mapping[str, Any],
    h1_row: Mapping[str, Any],
    signal: Mapping[str, Any],
    taxonomy_row: Mapping[str, Any],
    no_setup_row: Mapping[str, Any],
    bridge_row: Mapping[str, Any],
    mtf_row: Mapping[str, Any],
    bos_quality_row: Mapping[str, Any],
    market_row: Mapping[str, Any],
    current_feed_is_clean: bool,
    fallback_blocker_scope: str,
) -> Dict[str, Any]:
    score, min_score, score_gap = _score_fields(signal, taxonomy_row, no_setup_row, bridge_row)
    setup = _normalize_text(
        taxonomy_row.get("setup")
        or no_setup_row.get("setup")
        or bridge_row.get("real_strategy")
        or signal.get("strategy_name")
        or signal.get("strategy")
        or TARGET_SETUP,
        TARGET_SETUP,
    )
    official_primary_blocker = _upper(
        taxonomy_row.get("official_primary_blocker")
        or no_setup_row.get("primary_real_blocker")
        or bridge_row.get("primary_real_blocker")
        or signal.get("primary_real_blocker")
        or signal.get("rejection_reason"),
        "UNKNOWN",
    )
    normalized_primary_reason = _upper(
        taxonomy_row.get("normalized_primary_reason")
        or no_setup_row.get("reason_bucket")
        or bridge_row.get("primary_real_blocker")
        or signal.get("primary_real_blocker"),
        "UNKNOWN",
    )
    h4_bos_state = _upper(h4_row.get("bos_state") or bridge_row.get("bos_state_4h") or bos_quality_row.get("h4_bos_state"), "UNKNOWN")
    h1_bos_state = _upper(h1_row.get("bos_state") or bridge_row.get("bos_state_1h") or bos_quality_row.get("h1_bos_state"), "UNKNOWN")
    h4_retest_state = _retest_state(h4_row, h4_bos_state)
    h1_retest_state = _retest_state(h1_row, h1_bos_state)
    h1_pivot_state = _upper(h1_row.get("pivot_state") or bridge_row.get("pivot_state_1h"), "UNKNOWN")
    h4_trend_direction = _upper(mtf_row.get("h4_structure") or bridge_row.get("h4_structure"), "INCONCLUSIVE")
    h1_trend_direction = _upper(mtf_row.get("h1_confirmation") or bridge_row.get("h1_confirmation"), "INCONCLUSIVE")
    relationship = _upper((h4_row or h1_row).get("relationship_to_higher_tf") or bridge_row.get("relationship_1h_4h"), "UNKNOWN")
    h1_h4_alignment = _h1_h4_alignment(h1_trend_direction, h4_trend_direction, relationship)
    h1_data_quality = _timeframe_quality(mtf_row, "1h", "missing")
    if h1_bos_state not in INSUFFICIENT_BOS_STATES and h1_data_quality == "missing":
        h1_data_quality = "ok"

    status, failure_reason, confirmation_state, recommendation = _classify_h1_confirmation(
        h4_bos_state=h4_bos_state,
        h4_retest_state=h4_retest_state,
        h1_bos_state=h1_bos_state,
        h1_retest_state=h1_retest_state,
        h1_pivot_state=h1_pivot_state,
        h1_data_quality=h1_data_quality,
        h1_trend_direction=h1_trend_direction,
        h4_trend_direction=h4_trend_direction,
        h1_h4_alignment=h1_h4_alignment,
    )
    recommendation = _safe_recommendation(recommendation)
    timing_risk = _entry_timing_risk(status)
    message = _safe_message(_message_for_status(status))

    return {
        "symbol": symbol,
        "setup": setup,
        "score": score,
        "min_score": min_score,
        "score_gap": score_gap,
        "official_primary_blocker": official_primary_blocker,
        "normalized_primary_reason": normalized_primary_reason,
        "h4_bos_state": h4_bos_state,
        "h4_retest_state": h4_retest_state,
        "h1_bos_state": h1_bos_state,
        "h1_confirmation_state": confirmation_state,
        "h1_confirmation_status": status,
        "h1_failure_reason": failure_reason,
        "h1_data_quality": h1_data_quality,
        "h1_trend_direction": h1_trend_direction,
        "h4_trend_direction": h4_trend_direction,
        "h1_h4_alignment": h1_h4_alignment,
        "h1_retest_state": h1_retest_state,
        "h1_pivot_state": h1_pivot_state,
        "h1_entry_timing_risk": timing_risk,
        "multi_tf_alignment_status": _upper(mtf_row.get("alignment_status") or bridge_row.get("multi_tf_alignment_status"), "UNKNOWN"),
        "fib_zone": _upper(market_row.get("current_fib_zone") or taxonomy_row.get("fib_zone"), "UNKNOWN"),
        "current_feed_is_clean": current_feed_is_clean,
        "fallback_blocker_scope": fallback_blocker_scope,
        "recommendation": recommendation,
        "should_keep_blocked": True,
        "safe_to_change_strategy_now": False,
        "safe_to_change_threshold_now": False,
        "suggested_ui_message": message,
        "suggested_future_study": recommendation,
        "shadow_only": True,
    }


def _candidate_priority(candidate: Mapping[str, Any]) -> Tuple[int, float]:
    status = _upper(candidate.get("h1_confirmation_status"))
    priority = {
        "H1_INSUFFICIENT_DATA_AFTER_H4_BOS": 0,
        "H1_NO_BOS_AFTER_H4_BOS": 1,
        "H1_CONFLICTS_WITH_H4_BOS": 2,
        "H1_SIDEWAYS_AFTER_H4_BOS": 3,
        "H1_PIVOT_FORMING_AFTER_H4_BOS": 4,
        "H1_RETEST_PENDING_AFTER_H4_BOS": 5,
        "H1_WEAK_CONFIRMATION_AFTER_H4_BOS": 6,
        "H1_CONFIRMED_AFTER_H4_BOS": 7,
        "H4_BOS_NOT_CONFIRMED": 8,
        "INSUFFICIENT_DATA_FOR_H1_CONFIRMATION": 9,
    }.get(status, 20)
    score_gap = _as_float(candidate.get("score_gap"), 999.0) or 999.0
    return priority, score_gap


def build_h1_confirmation_after_h4_bos_audit(
    *,
    signals: Optional[Sequence[Mapping[str, Any]]] = None,
    bos_confirmation_quality_audit: Any = None,
    bos_pivot_trace_audit: Any = None,
    multi_timeframe_swing_audit: Any = None,
    setup_blocker_taxonomy_audit: Any = None,
    no_setup_eligible_decomposition: Any = None,
    strategy_decision_bridge_trace: Any = None,
    market_structure_audit: Any = None,
    fibonacci_alignment_audit: Any = None,
    feed_scope_reconciliation: Any = None,
    strategy_bottleneck: Any = None,
    state: Any = None,
    enabled: bool = True,
) -> Dict[str, Any]:
    """Build a shadow-only explanation of H1 confirmation after H4 BOS."""
    if not enabled:
        result = default_h1_confirmation_after_h4_bos_audit_state("H1-after-H4 BOS audit disabled.")
        result.update({"enabled": False, "status": "DISABLED", "recommendation": "observe_more"})
        return result

    bos_quality = _as_dict(bos_confirmation_quality_audit)
    bos_audit = _as_dict(bos_pivot_trace_audit)
    mtf_audit = _as_dict(multi_timeframe_swing_audit)
    taxonomy_audit = _as_dict(setup_blocker_taxonomy_audit)
    no_setup_audit = _as_dict(no_setup_eligible_decomposition)
    bridge_audit = _as_dict(strategy_decision_bridge_trace)
    market_audit = _as_dict(market_structure_audit)

    bos_rows = _as_list(bos_audit.get("recent_candidates"))
    bos_quality_rows = _as_list(bos_quality.get("candidates"))
    mtf_rows = _as_list(mtf_audit.get("recent_candidates"))
    taxonomy_rows = _as_list(taxonomy_audit.get("candidates"))
    no_setup_rows = _as_list(no_setup_audit.get("candidates"))
    bridge_rows = _as_list(bridge_audit.get("recent_candidates"))
    market_rows = _as_list(market_audit.get("market_structure_best_candidates") or market_audit.get("recent_candidates"))
    signal_by_symbol = _signal_index(signals)

    current_feed_is_clean, fallback_blocker_scope, fallback_scope_status = _feed_scope(
        feed_scope_reconciliation,
        state,
    )

    candidate_symbols: List[str] = []
    for source_rows in (bos_rows, bos_quality_rows, mtf_rows, taxonomy_rows, no_setup_rows, bridge_rows, signals or []):
        for raw in source_rows:
            row = _as_dict(raw)
            symbol = _normalize_text(row.get("symbol") or row.get("asset"))
            if symbol and symbol.upper() not in candidate_symbols:
                candidate_symbols.append(symbol.upper())

    bos_by_symbol = _rows_by_symbol(bos_rows)
    candidates: List[Dict[str, Any]] = []
    for symbol in candidate_symbols:
        rows_for_symbol = bos_by_symbol.get(symbol, [])
        h4_row = _find_timeframe_row(rows_for_symbol, "4h")
        h1_row = _find_timeframe_row(rows_for_symbol, "1h")
        taxonomy_row = _find_symbol_row(taxonomy_rows, symbol)
        no_setup_row = _find_symbol_row(no_setup_rows, symbol)
        bridge_row = _find_symbol_row(bridge_rows, symbol)
        mtf_row = _find_symbol_row(mtf_rows, symbol)
        bos_quality_row = _find_symbol_row(bos_quality_rows, symbol)
        market_row = _find_symbol_row(market_rows, symbol)
        signal = signal_by_symbol.get(symbol, {})
        candidate = _build_candidate(
            symbol=symbol,
            h4_row=h4_row,
            h1_row=h1_row,
            signal=signal,
            taxonomy_row=taxonomy_row,
            no_setup_row=no_setup_row,
            bridge_row=bridge_row,
            mtf_row=mtf_row,
            bos_quality_row=bos_quality_row,
            market_row=market_row,
            current_feed_is_clean=current_feed_is_clean,
            fallback_blocker_scope=fallback_blocker_scope,
        )
        candidates.append(candidate)

    if not candidates:
        result = default_h1_confirmation_after_h4_bos_audit_state()
        result.update(
            {
                "generated_at": _utc_now_iso(),
                "current_feed_is_clean": current_feed_is_clean,
                "fallback_blocker_scope": fallback_blocker_scope,
                "fallback_scope_status": fallback_scope_status,
            }
        )
        return result

    candidates = sorted(candidates, key=_candidate_priority)[:MAX_CANDIDATES]
    top = candidates[0]
    h4_confirmed_count = sum(1 for row in candidates if _is_h4_confirmed(row.get("h4_bos_state")))
    h4_retest_confirmed_count = sum(1 for row in candidates if row.get("h4_retest_state") == "RETEST_CONFIRMED")
    h1_missing_count = sum(
        1
        for row in candidates
        if row.get("h1_confirmation_status")
        in {
            "H1_INSUFFICIENT_DATA_AFTER_H4_BOS",
            "H1_NO_BOS_AFTER_H4_BOS",
            "H1_CONFLICTS_WITH_H4_BOS",
            "H1_SIDEWAYS_AFTER_H4_BOS",
            "H1_PIVOT_FORMING_AFTER_H4_BOS",
            "H1_RETEST_PENDING_AFTER_H4_BOS",
            "H4_BOS_CONFIRMED_BUT_H1_NOT_READY",
        }
    )
    recommendation = _safe_recommendation(top.get("recommendation"), "study_h1_confirmation")
    return {
        "enabled": True,
        "mode": MODE,
        "safety_mode": SAFETY_MODE,
        "status": "READY",
        "generated_at": _utc_now_iso(),
        "target_setup": TARGET_SETUP,
        "total_candidates_checked": len(candidate_symbols),
        "h4_bos_confirmed_count": h4_confirmed_count,
        "h4_retest_confirmed_count": h4_retest_confirmed_count,
        "h1_missing_confirmation_count": h1_missing_count,
        "top_symbol": top.get("symbol") or "",
        "top_setup": top.get("setup") or TARGET_SETUP,
        "h4_bos_state": top.get("h4_bos_state") or "UNKNOWN",
        "h4_retest_state": top.get("h4_retest_state") or "UNKNOWN",
        "h1_bos_state": top.get("h1_bos_state") or "UNKNOWN",
        "h1_confirmation_state": top.get("h1_confirmation_state") or "UNKNOWN",
        "h1_confirmation_status": top.get("h1_confirmation_status") or "UNKNOWN_H1_CONFIRMATION_STATUS",
        "h1_failure_reason": top.get("h1_failure_reason") or "unknown",
        "h1_data_quality": top.get("h1_data_quality") or "missing",
        "h1_trend_direction": top.get("h1_trend_direction") or "INCONCLUSIVE",
        "h4_trend_direction": top.get("h4_trend_direction") or "INCONCLUSIVE",
        "h1_h4_alignment": top.get("h1_h4_alignment") or "UNKNOWN",
        "h1_retest_state": top.get("h1_retest_state") or "UNKNOWN",
        "h1_pivot_state": top.get("h1_pivot_state") or "UNKNOWN",
        "h1_entry_timing_risk": top.get("h1_entry_timing_risk") or "UNKNOWN",
        "current_feed_is_clean": current_feed_is_clean,
        "fallback_blocker_scope": fallback_blocker_scope,
        "fallback_scope_status": fallback_scope_status,
        "recommendation": recommendation,
        "should_keep_blocked": True,
        "safe_to_change_strategy_now": False,
        "safe_to_change_threshold_now": False,
        "notes": "Diagnostic-only H1 confirmation after H4 BOS audit. No trade decision changed.",
        "candidates": candidates,
        "shadow_only": True,
    }


__all__ = [
    "build_h1_confirmation_after_h4_bos_audit",
    "default_h1_confirmation_after_h4_bos_audit_state",
]
