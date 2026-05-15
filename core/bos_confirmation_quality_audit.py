"""Diagnostic-only BOS confirmation quality audit.

This module explains why a BOS signal is missing, weak, failed, or pending.
It never changes strategy decisions, scores, thresholds, orders, positions, or
official paper-trading state.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


MODE = "DIAGNOSTIC_ONLY"
SAFETY_MODE = "SHADOW_ONLY"
TARGET_SETUP = "trend_pullback_breakout"
MAX_CANDIDATES = 10
WEAK_CLOSE_BUFFER_PCT = 0.0015

CONFIRMED_BOS_STATES = {"BOS_BY_CLOSE_CONFIRMED", "BOS_RETEST_CONFIRMED"}
MISSING_BOS_STATES = {"", "NO_BOS", "INSUFFICIENT_DATA", "UNKNOWN"}

SAFE_RECOMMENDATIONS = {
    "observe_more",
    "study_bos_confirmation",
    "study_bos_close_quality",
    "study_bos_retest_quality",
    "study_multitf_bos_confirmation",
    "study_pivot_to_bos_mapping",
    "study_false_breakout_risk",
    "keep_blocked_until_bos_confirms",
    "keep_blocked_until_retest_confirms",
    "keep_blocked_until_h4_confirms",
    "keep_blocked_until_h1_confirms",
    "no_threshold_change_recommended",
    "no_strategy_change_recommended",
    "insufficient_data",
}

FORBIDDEN_MESSAGE_FRAGMENTS = {
    "entrada aprovada",
    "pode comprar",
    "opere agora",
    "reduza score",
    "ignore bos",
    "bos suficiente para entrada real",
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
        return "BOS ainda em diagnostico: manter bloqueado ate confirmacao objetiva."
    return text


def default_bos_confirmation_quality_audit_state(
    reason: str = "No BOS confirmation quality audit data yet.",
) -> Dict[str, Any]:
    return {
        "enabled": True,
        "mode": MODE,
        "safety_mode": SAFETY_MODE,
        "status": "INSUFFICIENT_DATA",
        "generated_at": "",
        "target_setup": TARGET_SETUP,
        "total_candidates_checked": 0,
        "bos_missing_count": 0,
        "bos_quality_cases_count": 0,
        "top_symbol": "",
        "top_setup": TARGET_SETUP,
        "top_timeframe": "",
        "bos_quality_status": "INSUFFICIENT_DATA_FOR_BOS_QUALITY",
        "bos_failure_reason": "insufficient_data",
        "bos_level": None,
        "last_close": None,
        "close_distance_to_bos_pct": None,
        "close_confirmed_beyond_level": False,
        "wick_crossed_level": False,
        "close_confirmed_level": False,
        "weak_close_detected": False,
        "wick_only_detected": False,
        "failed_breakout_detected": False,
        "retest_pending": False,
        "retest_confirmed": False,
        "h1_bos_state": "INSUFFICIENT_DATA",
        "h4_bos_state": "INSUFFICIENT_DATA",
        "h1_h4_relationship": "INSUFFICIENT_DATA",
        "pivot_state": "INSUFFICIENT_DATA",
        "multi_tf_alignment_status": "INSUFFICIENT_DATA",
        "current_feed_is_clean": False,
        "fallback_blocker_scope": "UNKNOWN",
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
            indexed.setdefault(symbol, row)
    return indexed


def _rows_by_symbol(rows: Iterable[Any]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for raw in rows:
        row = _as_dict(raw)
        symbol = _normalize_text(row.get("symbol"))
        if symbol:
            grouped.setdefault(symbol, []).append(row)
    return grouped


def _find_symbol_row(rows: Iterable[Any], symbol: str) -> Dict[str, Any]:
    for raw in rows:
        row = _as_dict(raw)
        if _normalize_text(row.get("symbol")) == symbol:
            return row
    return {}


def _find_timeframe_row(rows: Sequence[Mapping[str, Any]], timeframe: str) -> Dict[str, Any]:
    target = timeframe.lower()
    for raw in rows:
        row = _as_dict(raw)
        if _normalize_text(row.get("timeframe")).lower() == target:
            return row
    return {}


def _confirmed_bos(state: Any) -> bool:
    return _upper(state) in CONFIRMED_BOS_STATES


def _missing_bos(state: Any) -> bool:
    return _upper(state, "UNKNOWN") in MISSING_BOS_STATES


def _pivot_triggered_or_confirmed(state: Any) -> bool:
    return _upper(state) in {"PIVOT_TRIGGERED", "PIVOT_CONFIRMED"}


def _pivot_forming(state: Any) -> bool:
    return _upper(state) == "PIVOT_FORMING"


def _score_fields(
    symbol: str,
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


def _classify_bos_quality(
    *,
    pivot_state: str,
    bos_state: str,
    h1_bos_state: str,
    h4_bos_state: str,
    wick_crossed_level: bool,
    close_confirmed_beyond_level: bool,
    close_confirmed_level: bool,
    close_distance_to_bos_pct: Optional[float],
    retest_detected: bool,
    retest_confirmed: bool,
    false_breakout_risk: str,
    multi_tf_alignment_status: str,
    normalized_primary_reason: str,
    why_bos_not_confirmed: str,
    bos_level: Optional[float],
) -> Tuple[str, str, str]:
    bos_state_u = _upper(bos_state, "UNKNOWN")
    pivot_state_u = _upper(pivot_state, "UNKNOWN")
    h1_state_u = _upper(h1_bos_state, "UNKNOWN")
    h4_state_u = _upper(h4_bos_state, "UNKNOWN")
    mtf_u = _upper(multi_tf_alignment_status, "UNKNOWN")
    normalized_u = _upper(normalized_primary_reason, "UNKNOWN")
    reason_text = _normalize_text(why_bos_not_confirmed).lower()
    distance = abs(close_distance_to_bos_pct or 0.0)

    if (not bos_state_u or bos_state_u == "UNKNOWN") and normalized_u == "BOS_MISSING":
        if bos_level is None:
            return "STRUCTURE_LEVEL_NOT_CLEAR", "structure_level_unknown", "study_bos_confirmation"
        return "BOS_MISSING_CLOSE_CONFIRMATION", "no_close_above_structure", "study_bos_confirmation"

    if not bos_state_u or bos_state_u == "UNKNOWN":
        return (
            "INSUFFICIENT_DATA_FOR_BOS_QUALITY",
            "insufficient_data",
            "insufficient_data",
        )

    if bos_state_u == "BOS_FAILED" or "returned_inside" in reason_text or false_breakout_risk == "HIGH":
        return "BOS_FAILED", "close_back_inside_structure", "study_false_breakout_risk"

    if bos_state_u == "BOS_BY_WICK_ONLY" or (wick_crossed_level and not close_confirmed_beyond_level):
        return "BOS_BY_WICK_ONLY", "wick_cross_without_close", "study_bos_close_quality"

    if bos_state_u == "BOS_BY_CLOSE_WEAK" or (
        close_confirmed_beyond_level
        and not close_confirmed_level
        and distance < WEAK_CLOSE_BUFFER_PCT
    ):
        return "BOS_BY_CLOSE_WEAK", "close_distance_too_small", "study_bos_close_quality"

    if bos_state_u == "BOS_RETEST_PENDING" or (retest_detected and not retest_confirmed):
        return "BOS_RETEST_PENDING", "retest_not_done", "study_bos_retest_quality"

    if _confirmed_bos(h1_state_u) and not _confirmed_bos(h4_state_u):
        return "BOS_MISSING_H4_CONFIRMATION", "h1_only_without_h4", "keep_blocked_until_h4_confirms"

    if _confirmed_bos(h4_state_u) and not _confirmed_bos(h1_state_u):
        return "BOS_MISSING_H1_CONFIRMATION", "h4_only_without_h1_confirmation", "keep_blocked_until_h1_confirms"

    if bos_state_u == "BOS_RETEST_CONFIRMED":
        return "BOS_RETEST_CONFIRMED", "unknown", "no_strategy_change_recommended"

    if bos_state_u == "BOS_BY_CLOSE_CONFIRMED":
        if distance and distance < (WEAK_CLOSE_BUFFER_PCT * 2):
            return "BOS_CONFIRMED_WEAK", "weak_close_confirmation", "study_bos_close_quality"
        return "BOS_CONFIRMED_STRONG", "unknown", "no_strategy_change_recommended"

    if _pivot_triggered_or_confirmed(pivot_state_u) and _missing_bos(h1_state_u) and _missing_bos(h4_state_u):
        return "PIVOT_TRIGGERED_BUT_BOS_MISSING", "h4_bos_missing", "study_pivot_to_bos_mapping"

    if _pivot_forming(pivot_state_u) and _missing_bos(h1_state_u) and _missing_bos(h4_state_u):
        return "PIVOT_FORMING_BUT_BOS_MISSING", "h1_bos_missing", "study_bos_confirmation"

    if mtf_u in {"WEAK_ALIGNMENT", "CONFLICT"} and _missing_bos(bos_state_u):
        return "BOS_MISSING_MULTITF_CONFIRMATION", "h4_bos_missing", "study_multitf_bos_confirmation"

    if normalized_u == "BOS_MISSING" and _missing_bos(bos_state_u):
        if bos_level is None:
            return "STRUCTURE_LEVEL_NOT_CLEAR", "structure_level_unknown", "study_bos_confirmation"
        return "BOS_MISSING_CLOSE_CONFIRMATION", "no_close_above_structure", "study_bos_confirmation"

    if _missing_bos(bos_state_u):
        if bos_level is None:
            return "STRUCTURE_LEVEL_NOT_CLEAR", "structure_level_unknown", "study_bos_confirmation"
        return "BOS_MISSING_CLOSE_CONFIRMATION", "no_close_above_structure", "study_bos_confirmation"

    return "UNKNOWN_BOS_QUALITY", "unknown", "observe_more"


def _message_for_status(status: str, reason: str) -> str:
    status_u = _upper(status)
    if status_u == "BOS_BY_WICK_ONLY":
        return "BOS nao confirmado: movimento cruzou o nivel por pavio, mas nao fechou alem da estrutura."
    if status_u == "BOS_BY_CLOSE_WEAK":
        return "BOS fraco: fechamento ficou perto demais do nivel estrutural."
    if status_u == "BOS_FAILED":
        return "BOS falhou: rompimento voltou para dentro da estrutura."
    if status_u == "BOS_RETEST_PENDING":
        return "BOS pendente: estrutura sugere rompimento, mas ainda falta reteste."
    if status_u in {"PIVOT_TRIGGERED_BUT_BOS_MISSING", "PIVOT_FORMING_BUT_BOS_MISSING"}:
        return "BOS nao confirmado: houve pivo, mas faltou fechamento estrutural alem do nivel."
    if status_u in {"BOS_MISSING_H4_CONFIRMATION", "BOS_MISSING_H1_CONFIRMATION"}:
        return "BOS nao confirmado: um timeframe sinalizou estrutura, mas falta confirmacao no par 1H/4H."
    if status_u == "BOS_MISSING_MULTITF_CONFIRMATION":
        return "BOS bloqueado por conflito multi-timeframe: 4H/1H ainda nao confirmaram juntos."
    if status_u == "BOS_MISSING_CLOSE_CONFIRMATION":
        return "BOS nao confirmado: faltou fechamento objetivo alem do nivel estrutural."
    if status_u == "STRUCTURE_LEVEL_NOT_CLEAR":
        return "BOS inconclusivo: nivel estrutural ainda nao esta claro para auditoria objetiva."
    if status_u in {"BOS_CONFIRMED_STRONG", "BOS_RETEST_CONFIRMED", "BOS_CONFIRMED_WEAK"}:
        return "BOS estrutural detectado em shadow, mas a decisao oficial continua bloqueada pela estrategia real."
    if reason == "insufficient_data":
        return "BOS inconclusivo: dados insuficientes para qualidade de confirmacao."
    return "BOS ainda em diagnostico: manter bloqueado ate confirmacao objetiva."


def _build_candidate(
    *,
    symbol: str,
    row: Mapping[str, Any],
    h1_row: Mapping[str, Any],
    h4_row: Mapping[str, Any],
    signal: Mapping[str, Any],
    taxonomy_row: Mapping[str, Any],
    no_setup_row: Mapping[str, Any],
    bridge_row: Mapping[str, Any],
    mtf_row: Mapping[str, Any],
    market_row: Mapping[str, Any],
    current_feed_is_clean: bool,
    fallback_blocker_scope: str,
) -> Dict[str, Any]:
    timeframe = _normalize_text(row.get("timeframe"), "unknown")
    score, min_score, score_gap = _score_fields(symbol, signal, taxonomy_row, no_setup_row, bridge_row)
    normalized_primary_reason = _upper(
        taxonomy_row.get("normalized_primary_reason")
        or no_setup_row.get("reason_bucket")
        or bridge_row.get("primary_real_blocker")
        or signal.get("primary_real_blocker"),
        "UNKNOWN",
    )
    official_primary_blocker = _upper(
        taxonomy_row.get("official_primary_blocker")
        or no_setup_row.get("primary_real_blocker")
        or bridge_row.get("primary_real_blocker")
        or signal.get("primary_real_blocker")
        or signal.get("rejection_reason"),
        "UNKNOWN",
    )
    setup = _normalize_text(
        taxonomy_row.get("setup")
        or no_setup_row.get("setup")
        or bridge_row.get("real_strategy")
        or row.get("setup")
        or signal.get("strategy")
        or TARGET_SETUP,
        TARGET_SETUP,
    )
    bos_state = _upper(row.get("bos_state"), "UNKNOWN")
    h1_bos_state = _upper(h1_row.get("bos_state") or bridge_row.get("bos_state_1h"), "UNKNOWN")
    h4_bos_state = _upper(h4_row.get("bos_state") or bridge_row.get("bos_state_4h"), "UNKNOWN")
    if timeframe.lower() == "1h" and h1_bos_state == "UNKNOWN":
        h1_bos_state = bos_state
    if timeframe.lower() == "4h" and h4_bos_state == "UNKNOWN":
        h4_bos_state = bos_state
    pivot_state = _upper(row.get("pivot_state") or bridge_row.get("pivot_state_4h") or bridge_row.get("pivot_state_1h"), "UNKNOWN")
    h1_h4_relationship = _upper(
        row.get("relationship_to_higher_tf")
        or bridge_row.get("relationship_1h_4h")
        or bridge_row.get("timeframe_bos_pivot_relationship"),
        "UNKNOWN",
    )
    multi_tf_alignment_status = _upper(
        mtf_row.get("alignment_status")
        or taxonomy_row.get("multi_tf_alignment_status")
        or no_setup_row.get("multi_tf_alignment_status")
        or bridge_row.get("multi_tf_alignment_status"),
        "UNKNOWN",
    )
    close_confirmed_level = _as_bool(row.get("close_confirmed_level"))
    close_confirmed_beyond_level = _as_bool(
        row.get("close_confirmed_beyond_level")
        if "close_confirmed_beyond_level" in row
        else row.get("close_above_or_below_level")
    )
    if close_confirmed_level:
        close_confirmed_beyond_level = True
    wick_crossed_level = _as_bool(row.get("wick_crossed_level"))
    retest_detected = _as_bool(row.get("retest_detected"))
    retest_confirmed = _as_bool(row.get("retest_hold")) or bos_state == "BOS_RETEST_CONFIRMED"
    close_distance = _as_float(row.get("close_distance_to_bos_pct") or row.get("bos_close_distance_pct"))
    bos_level = _as_float(row.get("bos_level"))
    last_close = _as_float(row.get("last_close") or row.get("bos_close_price"))
    false_breakout_risk = _upper(row.get("false_breakout_risk"), "UNKNOWN")

    status, failure_reason, recommendation = _classify_bos_quality(
        pivot_state=pivot_state,
        bos_state=bos_state,
        h1_bos_state=h1_bos_state,
        h4_bos_state=h4_bos_state,
        wick_crossed_level=wick_crossed_level,
        close_confirmed_beyond_level=close_confirmed_beyond_level,
        close_confirmed_level=close_confirmed_level,
        close_distance_to_bos_pct=close_distance,
        retest_detected=retest_detected,
        retest_confirmed=retest_confirmed,
        false_breakout_risk=false_breakout_risk,
        multi_tf_alignment_status=multi_tf_alignment_status,
        normalized_primary_reason=normalized_primary_reason,
        why_bos_not_confirmed=_normalize_text(row.get("why_bos_not_confirmed")),
        bos_level=bos_level,
    )
    recommendation = _safe_recommendation(recommendation)
    message = _safe_message(_message_for_status(status, failure_reason))

    return {
        "symbol": symbol,
        "setup": setup,
        "score": score,
        "min_score": min_score,
        "score_gap": score_gap,
        "official_primary_blocker": official_primary_blocker,
        "normalized_primary_reason": normalized_primary_reason,
        "bos_quality_status": status,
        "bos_failure_reason": failure_reason,
        "timeframe": timeframe,
        "h1_bos_state": h1_bos_state,
        "h4_bos_state": h4_bos_state,
        "h1_h4_relationship": h1_h4_relationship,
        "pivot_state": pivot_state,
        "bos_level": bos_level,
        "last_close": last_close,
        "close_distance_to_bos_pct": close_distance,
        "wick_crossed_level": wick_crossed_level,
        "close_confirmed_beyond_level": close_confirmed_beyond_level,
        "close_confirmed_level": close_confirmed_level,
        "weak_close_detected": status == "BOS_BY_CLOSE_WEAK",
        "wick_only_detected": status == "BOS_BY_WICK_ONLY",
        "failed_breakout_detected": status == "BOS_FAILED",
        "retest_pending": status == "BOS_RETEST_PENDING",
        "retest_confirmed": status == "BOS_RETEST_CONFIRMED" or retest_confirmed,
        "multi_tf_alignment_status": multi_tf_alignment_status,
        "daily_bias": _upper(mtf_row.get("daily_bias") or bridge_row.get("daily_bias"), "UNKNOWN"),
        "h4_structure": _upper(mtf_row.get("h4_structure") or bridge_row.get("h4_structure"), "UNKNOWN"),
        "h1_confirmation": _upper(mtf_row.get("h1_confirmation") or bridge_row.get("h1_confirmation"), "UNKNOWN"),
        "fib_zone": _upper(market_row.get("current_fib_zone") or taxonomy_row.get("fib_zone"), "UNKNOWN"),
        "current_feed_is_clean": current_feed_is_clean,
        "fallback_blocker_scope": fallback_blocker_scope,
        "recommendation": recommendation,
        "should_keep_blocked": True,
        "safe_to_change_strategy_now": False,
        "safe_to_change_threshold_now": False,
        "suggested_future_study": recommendation,
        "suggested_ui_message": message,
        "shadow_only": True,
    }


def _candidate_priority(candidate: Mapping[str, Any]) -> Tuple[int, float]:
    status = _upper(candidate.get("bos_quality_status"))
    priority = {
        "PIVOT_TRIGGERED_BUT_BOS_MISSING": 0,
        "PIVOT_FORMING_BUT_BOS_MISSING": 1,
        "BOS_BY_WICK_ONLY": 2,
        "BOS_BY_CLOSE_WEAK": 3,
        "BOS_FAILED": 4,
        "BOS_MISSING_H4_CONFIRMATION": 5,
        "BOS_MISSING_H1_CONFIRMATION": 6,
        "BOS_MISSING_MULTITF_CONFIRMATION": 7,
        "BOS_MISSING_CLOSE_CONFIRMATION": 8,
        "BOS_RETEST_PENDING": 9,
        "BOS_CONFIRMED_WEAK": 10,
        "BOS_CONFIRMED_STRONG": 11,
        "BOS_RETEST_CONFIRMED": 12,
        "STRUCTURE_LEVEL_NOT_CLEAR": 13,
        "INSUFFICIENT_DATA_FOR_BOS_QUALITY": 14,
    }.get(status, 20)
    score_gap = _as_float(candidate.get("score_gap"), 999.0) or 999.0
    return priority, score_gap


def build_bos_confirmation_quality_audit(
    *,
    signals: Optional[Sequence[Mapping[str, Any]]] = None,
    bos_pivot_trace_audit: Any = None,
    multi_timeframe_swing_audit: Any = None,
    setup_blocker_taxonomy_audit: Any = None,
    no_setup_eligible_decomposition: Any = None,
    reversal_blocker_routing_audit: Any = None,
    strategy_decision_bridge_trace: Any = None,
    market_structure_audit: Any = None,
    fibonacci_alignment_audit: Any = None,
    feed_scope_reconciliation: Any = None,
    strategy_bottleneck: Any = None,
    state: Any = None,
    enabled: bool = True,
) -> Dict[str, Any]:
    """Build a shadow-only explanation of BOS confirmation quality."""
    if not enabled:
        result = default_bos_confirmation_quality_audit_state("BOS confirmation quality audit disabled.")
        result.update({"enabled": False, "status": "DISABLED", "recommendation": "observe_more"})
        return result

    bos_audit = _as_dict(bos_pivot_trace_audit)
    mtf_audit = _as_dict(multi_timeframe_swing_audit)
    taxonomy_audit = _as_dict(setup_blocker_taxonomy_audit)
    no_setup_audit = _as_dict(no_setup_eligible_decomposition)
    bridge_audit = _as_dict(strategy_decision_bridge_trace)
    market_audit = _as_dict(market_structure_audit)

    bos_rows = _as_list(bos_audit.get("recent_candidates"))
    taxonomy_rows = _as_list(taxonomy_audit.get("candidates"))
    no_setup_rows = _as_list(no_setup_audit.get("candidates"))
    bridge_rows = _as_list(bridge_audit.get("recent_candidates"))
    mtf_rows = _as_list(mtf_audit.get("recent_candidates"))
    market_rows = _as_list(market_audit.get("market_structure_best_candidates") or market_audit.get("recent_candidates"))
    signal_by_symbol = _signal_index(signals)

    current_feed_is_clean, fallback_blocker_scope, fallback_scope_status = _feed_scope(
        feed_scope_reconciliation,
        state,
    )

    bos_by_symbol = _rows_by_symbol(bos_rows)
    candidate_symbols: List[str] = []
    for source_rows in (bos_rows, taxonomy_rows, no_setup_rows, bridge_rows, mtf_rows, signals or []):
        for raw in source_rows:
            row = _as_dict(raw)
            symbol = _normalize_text(row.get("symbol") or row.get("asset"))
            if symbol and symbol not in candidate_symbols:
                candidate_symbols.append(symbol)

    candidates: List[Dict[str, Any]] = []
    for symbol in candidate_symbols:
        rows_for_symbol = bos_by_symbol.get(symbol, [])
        h1_row = _find_timeframe_row(rows_for_symbol, "1h")
        h4_row = _find_timeframe_row(rows_for_symbol, "4h")
        taxonomy_row = _find_symbol_row(taxonomy_rows, symbol)
        no_setup_row = _find_symbol_row(no_setup_rows, symbol)
        bridge_row = _find_symbol_row(bridge_rows, symbol)
        mtf_row = _find_symbol_row(mtf_rows, symbol)
        market_row = _find_symbol_row(market_rows, symbol)
        signal = signal_by_symbol.get(symbol, {})

        rows_to_emit = rows_for_symbol or [taxonomy_row or no_setup_row or bridge_row or mtf_row or signal]
        for row in rows_to_emit:
            row_dict = _as_dict(row)
            if not row_dict:
                continue
            candidate = _build_candidate(
                symbol=symbol,
                row=row_dict,
                h1_row=h1_row,
                h4_row=h4_row,
                signal=signal,
                taxonomy_row=taxonomy_row,
                no_setup_row=no_setup_row,
                bridge_row=bridge_row,
                mtf_row=mtf_row,
                market_row=market_row,
                current_feed_is_clean=current_feed_is_clean,
                fallback_blocker_scope=fallback_blocker_scope,
            )
            candidates.append(candidate)

    if not candidates:
        result = default_bos_confirmation_quality_audit_state()
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
    bos_missing_count = sum(
        1
        for candidate in candidates
        if "MISSING" in _upper(candidate.get("bos_quality_status"))
        or _upper(candidate.get("normalized_primary_reason")) == "BOS_MISSING"
    )
    top_recommendation = _safe_recommendation(top.get("recommendation"), "study_bos_confirmation")
    result = {
        "enabled": True,
        "mode": MODE,
        "safety_mode": SAFETY_MODE,
        "status": "READY",
        "generated_at": _utc_now_iso(),
        "target_setup": TARGET_SETUP,
        "total_candidates_checked": len(candidate_symbols),
        "bos_missing_count": bos_missing_count,
        "bos_quality_cases_count": len(candidates),
        "top_symbol": top.get("symbol") or "",
        "top_setup": top.get("setup") or TARGET_SETUP,
        "top_timeframe": top.get("timeframe") or "",
        "bos_quality_status": top.get("bos_quality_status") or "UNKNOWN_BOS_QUALITY",
        "bos_failure_reason": top.get("bos_failure_reason") or "unknown",
        "bos_level": top.get("bos_level"),
        "last_close": top.get("last_close"),
        "close_distance_to_bos_pct": top.get("close_distance_to_bos_pct"),
        "close_confirmed_beyond_level": bool(top.get("close_confirmed_beyond_level")),
        "wick_crossed_level": bool(top.get("wick_crossed_level")),
        "close_confirmed_level": bool(top.get("close_confirmed_level")),
        "weak_close_detected": bool(top.get("weak_close_detected")),
        "wick_only_detected": bool(top.get("wick_only_detected")),
        "failed_breakout_detected": bool(top.get("failed_breakout_detected")),
        "retest_pending": bool(top.get("retest_pending")),
        "retest_confirmed": bool(top.get("retest_confirmed")),
        "h1_bos_state": top.get("h1_bos_state") or "UNKNOWN",
        "h4_bos_state": top.get("h4_bos_state") or "UNKNOWN",
        "h1_h4_relationship": top.get("h1_h4_relationship") or "UNKNOWN",
        "pivot_state": top.get("pivot_state") or "UNKNOWN",
        "multi_tf_alignment_status": top.get("multi_tf_alignment_status") or "UNKNOWN",
        "current_feed_is_clean": current_feed_is_clean,
        "fallback_blocker_scope": fallback_blocker_scope,
        "fallback_scope_status": fallback_scope_status,
        "recommendation": top_recommendation,
        "should_keep_blocked": True,
        "safe_to_change_strategy_now": False,
        "safe_to_change_threshold_now": False,
        "notes": "Diagnostic-only BOS confirmation quality audit. No trade decision changed.",
        "candidates": candidates,
        "shadow_only": True,
    }
    return result


__all__ = [
    "build_bos_confirmation_quality_audit",
    "default_bos_confirmation_quality_audit_state",
]
