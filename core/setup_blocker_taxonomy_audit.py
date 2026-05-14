from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any


MODE = "DIAGNOSTIC_ONLY"
SAFETY_MODE = "SHADOW_ONLY"
TARGET_SETUP = "trend_pullback_breakout"
MAX_CANDIDATES = 10
SCORE_LARGE_GAP = 0.04

CONFIRMED_BOS_STATES = {"BOS_BY_CLOSE_CONFIRMED", "BOS_RETEST_CONFIRMED"}
MISSING_BOS_STATES = {"", "NO_BOS", "BOS_FAILED", "BOS_BY_WICK_ONLY", "BOS_RETEST_PENDING", "INSUFFICIENT_DATA"}
PIVOT_PRESENT_STATES = {"PIVOT_TRIGGERED", "PIVOT_CONFIRMED", "PIVOT_FORMING"}
FIB_GOOD_ZONES = {"BREAKOUT_ZONE", "DEEP_ZONE", "MEDIUM_ZONE", "SHALLOW_ZONE"}
SAFE_RECOMMENDATIONS = {
    "observe_more",
    "study_taxonomy_mapping",
    "study_real_rule_mapping",
    "study_bos_confirmation",
    "study_pullback_quality",
    "study_breakout_confirmation",
    "study_multitf_conflict",
    "keep_blocked_until_bos_confirms",
    "keep_blocked_until_pullback_reaction",
    "keep_blocked_until_real_setup_maps_structure",
    "no_threshold_change_recommended",
    "no_strategy_change_recommended",
    "insufficient_data",
}

FORBIDDEN_MESSAGE_FRAGMENTS = (
    "entrada aprovada",
    "pode comprar",
    "reduza score",
    "ignore blocker",
    "remova reversal",
    "opere agora",
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _norm_text(value: Any) -> str:
    return str(value or "").strip()


def _upper(value: Any) -> str:
    return _norm_text(value).upper()


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


def _symbol_from(row: dict[str, Any]) -> str:
    return _norm_text(row.get("symbol") or row.get("asset")).upper()


def _reason_tokens(row: dict[str, Any]) -> list[str]:
    raw = row.get("rejection_reasons") or row.get("official_rejection_reason") or row.get("real_rejection_reason") or []
    if isinstance(raw, str):
        raw = [part.strip() for part in raw.replace(";", ",").split(",")]
    return [str(item or "").strip().lower() for item in list(raw or []) if str(item or "").strip()]


def _has(tokens: list[str], patterns: tuple[str, ...]) -> bool:
    return any(any(pattern in token for pattern in patterns) for token in tokens)


def _latest_by_symbol(rows: list[dict[str, Any]], *, symbol_keys: tuple[str, ...] = ("symbol", "asset")) -> dict[str, dict[str, Any]]:
    mapped: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        symbol = ""
        for key in symbol_keys:
            symbol = _norm_text(row.get(key)).upper()
            if symbol:
                break
        if symbol:
            mapped[symbol] = dict(row)
    return mapped


def _first_by_symbol(rows: list[dict[str, Any]], symbol: str) -> dict[str, Any]:
    target = symbol.upper()
    for row in rows:
        if isinstance(row, dict) and _symbol_from(row) == target:
            return dict(row)
    return {}


def _bos_rows_for_symbol(rows: list[dict[str, Any]], symbol: str) -> dict[str, dict[str, Any]]:
    target = symbol.upper()
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or _symbol_from(row) != target:
            continue
        timeframe = _norm_text(row.get("timeframe")).lower()
        if timeframe:
            result[timeframe] = dict(row)
    return result


def _field(*rows: dict[str, Any], keys: tuple[str, ...], default: Any = "") -> Any:
    for row in rows:
        if not isinstance(row, dict):
            continue
        for key in keys:
            value = row.get(key)
            if value not in (None, ""):
                return value
    return default


def _primary_from_tokens(tokens: list[str]) -> str:
    if _has(tokens, ("no_setup",)):
        return "NO_SETUP_ELIGIBLE"
    if _has(tokens, ("reversal_not_eligible",)):
        return "REVERSAL_NOT_ELIGIBLE"
    if _has(tokens, ("score_below", "score")):
        return "SCORE_BELOW_MIN"
    if _has(tokens, ("trend_not_confirmed", "trend")):
        return "TREND_NOT_CONFIRMED"
    if _has(tokens, ("fallback", "feed", "provider_unknown")):
        return "FEED_OR_PROVIDER"
    return ""


def _secondary_from_tokens(tokens: list[str]) -> str:
    if _has(tokens, ("breakout_not_confirmed", "breakout", "rompimento")):
        return "BREAKOUT_NOT_CONFIRMED"
    if _has(tokens, ("secondary", "secundaria", "confirmacao", "confirmation")):
        return "SECONDARY_CONFIRMATION_WEAK"
    if _has(tokens, ("momentum", "confidence_too_low")):
        return "MOMENTUM_WEAK"
    if _has(tokens, ("rsi",)):
        return "RSI_OUT_OF_RANGE"
    return ""


def _bos_state(bridge: dict[str, Any], bos_rows: dict[str, dict[str, Any]], no_setup: dict[str, Any], reversal: dict[str, Any]) -> str:
    h4 = dict(bos_rows.get("4h", {}) or {})
    h1 = dict(bos_rows.get("1h", {}) or {})
    return _upper(
        _field(
            bridge,
            no_setup,
            reversal,
            h4,
            h1,
            keys=("bos_state_4h", "bos_state", "bos_state_1h", "top_bos_state"),
        )
    )


def _pivot_state(bridge: dict[str, Any], bos_rows: dict[str, dict[str, Any]], no_setup: dict[str, Any], reversal: dict[str, Any]) -> str:
    h4 = dict(bos_rows.get("4h", {}) or {})
    h1 = dict(bos_rows.get("1h", {}) or {})
    return _upper(
        _field(
            bridge,
            no_setup,
            reversal,
            h4,
            h1,
            keys=("pivot_state_4h", "pivot_state", "pivot_state_1h", "top_pivot_state"),
        )
    )


def _market_structure_for_symbol(market_payload: dict[str, Any], market_rows: list[dict[str, Any]], symbol: str) -> dict[str, Any]:
    market_structure = _first_by_symbol(market_rows, symbol)
    if market_structure:
        return market_structure
    if _upper(market_payload.get("market_structure_top_symbol")) == symbol.upper():
        return {
            "market_structure_top_zone": market_payload.get("market_structure_top_zone"),
            "market_structure_score": market_payload.get("market_structure_top_score"),
            "current_fib_zone": market_payload.get("market_structure_top_zone"),
        }
    return {}


def _is_feed_clean(feed_scope: dict[str, Any]) -> bool:
    scope = _upper(feed_scope.get("fallback_blocker_scope"))
    return bool(feed_scope.get("current_feed_is_clean", False) and scope != "CURRENT_CYCLE")


def _recommendation_for(status: str, primary_reason: str) -> str:
    mapping = {
        "NO_SETUP_WITH_PIVOT_BUT_NO_BOS": "study_bos_confirmation",
        "NO_SETUP_WITH_BOS_MISSING": "study_bos_confirmation",
        "NO_SETUP_WITH_PULLBACK_REACTION_MISSING": "study_pullback_quality",
        "NO_SETUP_WITH_BREAKOUT_NOT_CONFIRMED": "study_breakout_confirmation",
        "SCORE_BELOW_MIN_PRIMARY": "no_threshold_change_recommended",
        "SECONDARY_CONFIRMATION_PRIMARY": "study_real_rule_mapping",
        "MULTITF_CONFLICT_PRIMARY": "study_multitf_conflict",
        "FIB_STRUCTURE_NOT_OPERATIONAL": "study_pullback_quality",
        "REVERSAL_CONTEXT_ON_TREND_SETUP": "study_taxonomy_mapping",
        "MIXED_TREND_REVERSAL_TAXONOMY": "study_taxonomy_mapping",
        "CLEAR_REVERSAL_BLOCKER": "no_strategy_change_recommended",
        "CLEAR_TREND_BLOCKER": "study_real_rule_mapping",
        "FEED_CLEAN_NOT_BLOCKER": "observe_more",
        "INSUFFICIENT_DATA_FOR_TAXONOMY": "insufficient_data",
    }
    recommendation = mapping.get(status, "")
    if not recommendation and primary_reason == "BOS_MISSING":
        recommendation = "keep_blocked_until_bos_confirms"
    if not recommendation:
        recommendation = "observe_more"
    return recommendation if recommendation in SAFE_RECOMMENDATIONS else "observe_more"


def _message_for(status: str, primary_reason: str, secondary_reason: str) -> str:
    messages = {
        "NO_SETUP_WITH_PIVOT_BUT_NO_BOS": "Trend pullback bloqueado: pivo detectado, mas BOS/fechamento estrutural ainda nao confirmou.",
        "NO_SETUP_WITH_BOS_MISSING": "Trend pullback bloqueado: falta BOS confirmado para transformar estrutura em setup operacional.",
        "NO_SETUP_WITH_PULLBACK_REACTION_MISSING": "Trend pullback bloqueado: estrutura existe, mas falta reacao objetiva de pullback.",
        "NO_SETUP_WITH_BREAKOUT_NOT_CONFIRMED": "Trend pullback bloqueado: breakout minimo ainda nao foi confirmado.",
        "SCORE_BELOW_MIN_PRIMARY": "Trend pullback bloqueado: score abaixo do minimo; sem mudanca de threshold recomendada.",
        "SECONDARY_CONFIRMATION_PRIMARY": "Trend pullback bloqueado: confirmacao secundaria ainda esta fraca.",
        "MULTITF_CONFLICT_PRIMARY": "Trend pullback bloqueado: timeframes ainda apontam conflito estrutural.",
        "FIB_STRUCTURE_NOT_OPERATIONAL": "Estrutura shadow favoravel, mas sem BOS/pullback/reacao operacional suficiente.",
        "REVERSAL_CONTEXT_ON_TREND_SETUP": "Reversal aparece apenas como contexto; o bloqueio principal continua na estrutura operacional.",
        "MIXED_TREND_REVERSAL_TAXONOMY": "Taxonomia mista: reversao aparece no roteamento, mas o setup real segue bloqueado por estrutura/score.",
        "CLEAR_REVERSAL_BLOCKER": "Reversao bloqueada corretamente pelo filtro de elegibilidade.",
        "CLEAR_TREND_BLOCKER": "Trend pullback bloqueado por regra real da estrategia; decisao oficial preservada.",
        "FEED_CLEAN_NOT_BLOCKER": "Feed atual limpo; rejeicao nao foi causada por fallback do ciclo.",
        "INSUFFICIENT_DATA_FOR_TAXONOMY": "Dados insuficientes para clarificar a taxonomia do bloqueio.",
    }
    message = messages.get(status) or f"Bloqueio explicado como {primary_reason or 'UNKNOWN'}; contexto secundario {secondary_reason or 'UNKNOWN'}."
    lower_message = message.lower()
    if any(fragment in lower_message for fragment in FORBIDDEN_MESSAGE_FRAGMENTS):
        return "Diagnostico mantido em modo shadow; decisao oficial permanece bloqueada."
    return message


def default_setup_blocker_taxonomy_audit_state(
    reason: str = "No setup/blocker taxonomy audit data yet.",
) -> dict[str, Any]:
    return {
        "enabled": True,
        "mode": MODE,
        "safety_mode": SAFETY_MODE,
        "status": "INSUFFICIENT_DATA",
        "generated_at": "",
        "target_setup": TARGET_SETUP,
        "total_candidates_checked": 0,
        "taxonomy_cases_count": 0,
        "top_symbol": "",
        "top_setup": TARGET_SETUP,
        "official_primary_blocker": "",
        "official_secondary_blocker": "",
        "normalized_primary_reason": "INSUFFICIENT_DATA",
        "normalized_secondary_reason": "INSUFFICIENT_DATA",
        "taxonomy_status": "INSUFFICIENT_DATA_FOR_TAXONOMY",
        "taxonomy_confidence": 0.0,
        "mixed_taxonomy_count": 0,
        "reversal_as_primary_count": 0,
        "reversal_as_context_count": 0,
        "no_setup_taxonomy_count": 0,
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


def _classify_taxonomy(
    *,
    setup: str,
    tokens: list[str],
    official_primary: str,
    official_secondary: str,
    score_gap: float | None,
    no_setup_bucket: str,
    route_status: str,
    route_alternative: str,
    mtf_status: str,
    bos_state: str,
    pivot_state: str,
    fib_zone: str,
    structure_score: float | None,
    feed_clean: bool,
) -> tuple[str, str, str, float]:
    if not tokens and not official_primary and not no_setup_bucket and not route_status and not mtf_status and not bos_state and not pivot_state:
        return ("INSUFFICIENT_DATA_FOR_TAXONOMY", "INSUFFICIENT_DATA", "INSUFFICIENT_DATA", 0.0)

    setup_is_trend = _norm_text(setup) == TARGET_SETUP
    no_setup = official_primary == "NO_SETUP_ELIGIBLE" or "NO_SETUP" in no_setup_bucket or _has(tokens, ("no_setup",))
    score_block = official_primary == "SCORE_BELOW_MIN" or _has(tokens, ("score_below", "score"))
    breakout_missing = official_secondary == "BREAKOUT_NOT_CONFIRMED" or _has(tokens, ("breakout_not_confirmed", "breakout"))
    secondary_weak = official_secondary in {"SECONDARY_CONFIRMATION_WEAK", "MOMENTUM_WEAK", "RSI_OUT_OF_RANGE"} or _has(
        tokens, ("secondary", "momentum", "confidence_too_low", "rsi")
    )
    bos_missing = bos_state in MISSING_BOS_STATES or route_alternative == "SHOULD_BE_MULTITF_CONFLICT"
    pivot_present = pivot_state in PIVOT_PRESENT_STATES or "PIVOT_TRIGGERED" in no_setup_bucket
    fib_good = bool(_upper(fib_zone) in FIB_GOOD_ZONES or (structure_score is not None and float(structure_score) >= 0.65))
    reversal_context = (
        official_primary == "REVERSAL_NOT_ELIGIBLE"
        or route_status in {"REVERSAL_PATTERN_NOT_ACTIVE_BUT_BLOCKER_PRESENT", "MIXED_TREND_REVERSAL_ROUTING"}
        or route_alternative == "MIXED_TREND_REVERSAL_ROUTING"
        or _has(tokens, ("reversal_not_eligible",))
    )

    if setup_is_trend and no_setup and pivot_present and bos_missing:
        return ("NO_SETUP_WITH_PIVOT_BUT_NO_BOS", "BOS_MISSING", "PIVOT_WITHOUT_BOS", 0.92)
    if setup_is_trend and no_setup and bos_missing:
        return ("NO_SETUP_WITH_BOS_MISSING", "BOS_MISSING", "NO_SETUP_ELIGIBLE", 0.88)
    if setup_is_trend and _upper(mtf_status) in {"CONFLICT", "WEAK_ALIGNMENT"}:
        return ("MULTITF_CONFLICT_PRIMARY", "MULTITF_CONFLICT", "NO_SETUP_ELIGIBLE" if no_setup else "UNKNOWN", 0.86)
    if setup_is_trend and reversal_context and route_status == "REVERSAL_PATTERN_NOT_ACTIVE_BUT_BLOCKER_PRESENT":
        preferred = "BOS_MISSING" if bos_missing else "SCORE_BELOW_MIN" if score_block else "BREAKOUT_NOT_CONFIRMED" if breakout_missing else "NO_SETUP_ELIGIBLE"
        return ("REVERSAL_CONTEXT_ON_TREND_SETUP", preferred, "REVERSAL_RISK_CONTEXT", 0.84)
    if setup_is_trend and reversal_context and route_alternative == "MIXED_TREND_REVERSAL_ROUTING":
        preferred = "BOS_MISSING" if bos_missing else "MULTITF_CONFLICT" if _upper(mtf_status) == "CONFLICT" else "NO_SETUP_ELIGIBLE"
        return ("MIXED_TREND_REVERSAL_TAXONOMY", preferred, "REVERSAL_RISK_CONTEXT", 0.8)
    if setup_is_trend and breakout_missing:
        return ("NO_SETUP_WITH_BREAKOUT_NOT_CONFIRMED", "BREAKOUT_NOT_CONFIRMED", "NO_SETUP_ELIGIBLE" if no_setup else "UNKNOWN", 0.78)
    if score_block and score_gap is not None and float(score_gap) > SCORE_LARGE_GAP:
        return ("SCORE_BELOW_MIN_PRIMARY", "SCORE_BELOW_MIN", "NO_SETUP_ELIGIBLE" if no_setup else "UNKNOWN", 0.9)
    if fib_good and bos_missing:
        return ("FIB_STRUCTURE_NOT_OPERATIONAL", "BOS_MISSING", "FIB_STRUCTURE_NOT_ENOUGH", 0.78)
    if setup_is_trend and no_setup and _has(tokens, ("pullback", "reaction", "reacao")):
        return ("NO_SETUP_WITH_PULLBACK_REACTION_MISSING", "PULLBACK_REACTION_MISSING", "NO_SETUP_ELIGIBLE", 0.76)
    if secondary_weak:
        return ("SECONDARY_CONFIRMATION_PRIMARY", "RSI_MOMENTUM_BLOCKER", "CONFIDENCE_TOO_LOW", 0.72)
    if "reversal" in _norm_text(setup).lower() and official_primary == "REVERSAL_NOT_ELIGIBLE":
        return ("CLEAR_REVERSAL_BLOCKER", "REVERSAL_RISK_CONTEXT", "NO_SETUP_ELIGIBLE", 0.82)
    if setup_is_trend and (no_setup or score_block or breakout_missing):
        primary = "SCORE_BELOW_MIN" if score_block else "BREAKOUT_NOT_CONFIRMED" if breakout_missing else "NO_SETUP_ELIGIBLE"
        return ("CLEAR_TREND_BLOCKER", primary, "UNKNOWN", 0.65)
    if feed_clean:
        return ("FEED_CLEAN_NOT_BLOCKER", "FEED_NOT_CURRENT_BLOCKER", "UNKNOWN", 0.6)
    return ("UNKNOWN_TAXONOMY", "UNKNOWN", "UNKNOWN", 0.3)


def _candidate_payload(
    *,
    symbol: str,
    signal: dict[str, Any],
    bridge: dict[str, Any],
    no_setup: dict[str, Any],
    reversal: dict[str, Any],
    mtf: dict[str, Any],
    bos_rows: dict[str, dict[str, Any]],
    market_structure: dict[str, Any],
    feed_scope: dict[str, Any],
) -> dict[str, Any]:
    tokens = _reason_tokens(signal) or _reason_tokens(bridge) or _reason_tokens(no_setup) or _reason_tokens(reversal)
    official_primary = (
        _upper(bridge.get("primary_real_blocker"))
        or _upper(no_setup.get("primary_real_blocker"))
        or _upper(no_setup.get("top_real_blocker"))
        or _upper(reversal.get("primary_real_blocker"))
        or _primary_from_tokens(tokens)
    )
    official_secondary = (
        _upper(bridge.get("secondary_real_blocker"))
        or _upper(no_setup.get("secondary_real_blocker"))
        or _upper(no_setup.get("top_secondary_blocker"))
        or _upper(reversal.get("secondary_real_blocker"))
        or _secondary_from_tokens(tokens)
    )
    setup = _norm_text(
        signal.get("strategy_name")
        or bridge.get("real_strategy")
        or no_setup.get("setup")
        or reversal.get("setup")
        or TARGET_SETUP
    )
    score = _as_float(
        bridge.get("real_score"),
        _as_float(no_setup.get("score"), _as_float(reversal.get("score"), _as_float(signal.get("score")))),
    )
    min_score = _as_float(
        bridge.get("min_score"),
        _as_float(
            no_setup.get("min_score"),
            _as_float(reversal.get("min_score"), _as_float(signal.get("effective_min_signal_score"), _as_float(signal.get("base_min_signal_score")))),
        ),
    )
    score_gap = None if score is None or min_score is None else max(0.0, float(min_score) - float(score))
    no_setup_bucket = _upper(no_setup.get("reason_bucket") or no_setup.get("top_reason_bucket"))
    route_status = _upper(reversal.get("route_status") or reversal.get("top_route_status"))
    route_alternative = _upper(reversal.get("alternative_bucket") or reversal.get("top_alternative_bucket"))
    mtf_status = _upper(mtf.get("alignment_status") or bridge.get("multi_tf_alignment_status"))
    bos_state = _bos_state(bridge, bos_rows, no_setup, reversal)
    pivot_state = _pivot_state(bridge, bos_rows, no_setup, reversal)
    fib_zone = _upper(
        market_structure.get("current_fib_zone")
        or market_structure.get("fib_zone")
        or market_structure.get("market_structure_top_zone")
    )
    structure_score = _as_float(market_structure.get("market_structure_score") or market_structure.get("structure_score"), None)
    feed_clean = _is_feed_clean(feed_scope)
    taxonomy_status, normalized_primary, normalized_secondary, confidence = _classify_taxonomy(
        setup=setup,
        tokens=tokens,
        official_primary=official_primary,
        official_secondary=official_secondary,
        score_gap=score_gap,
        no_setup_bucket=no_setup_bucket,
        route_status=route_status,
        route_alternative=route_alternative,
        mtf_status=mtf_status,
        bos_state=bos_state,
        pivot_state=pivot_state,
        fib_zone=fib_zone,
        structure_score=structure_score,
        feed_clean=feed_clean,
    )
    recommendation = _recommendation_for(taxonomy_status, normalized_primary)
    suggested_message = _message_for(taxonomy_status, normalized_primary, normalized_secondary)
    fallback_scope = _upper(feed_scope.get("fallback_blocker_scope") or bridge.get("fallback_blocker_scope") or "UNKNOWN")
    if feed_clean and fallback_scope == "CURRENT_CYCLE":
        fallback_scope = "NONE"
    return {
        "symbol": symbol,
        "setup": setup,
        "score": _round(score),
        "min_score": _round(min_score),
        "score_gap": _round(score_gap),
        "official_rejection_reason": ", ".join(tokens),
        "official_primary_blocker": official_primary,
        "official_secondary_blocker": official_secondary,
        "normalized_primary_reason": normalized_primary,
        "normalized_secondary_reason": normalized_secondary,
        "taxonomy_status": taxonomy_status,
        "taxonomy_confidence": _round(confidence),
        "taxonomy_detail": (
            f"official={official_primary or 'none'}; secondary={official_secondary or 'none'}; "
            f"route={route_status or 'none'}; no_setup_bucket={no_setup_bucket or 'none'}; "
            f"bos={bos_state or 'none'}; pivot={pivot_state or 'none'}"
        ),
        "route_status": route_status,
        "no_setup_bucket": no_setup_bucket,
        "multi_tf_alignment_status": mtf_status,
        "daily_bias": _norm_text(mtf.get("daily_bias") or ""),
        "h4_structure": _norm_text(mtf.get("h4_structure") or ""),
        "h1_confirmation": _norm_text(mtf.get("h1_confirmation") or ""),
        "bos_state": bos_state,
        "pivot_state": pivot_state,
        "fib_zone": fib_zone,
        "structure_score": _round(structure_score),
        "pullback_evidence": _norm_text(no_setup.get("pullback_evidence") or market_structure.get("pullback_state") or ""),
        "breakout_evidence": _norm_text(no_setup.get("breakout_evidence") or bridge.get("breakout_evidence") or ""),
        "reversal_context": normalized_secondary == "REVERSAL_RISK_CONTEXT" or official_primary == "REVERSAL_NOT_ELIGIBLE",
        "current_feed_is_clean": feed_clean,
        "fallback_blocker_scope": fallback_scope,
        "should_keep_blocked": True,
        "safe_to_change_strategy_now": False,
        "safe_to_change_threshold_now": False,
        "suggested_ui_message": suggested_message,
        "suggested_future_study": recommendation,
    }


def build_setup_blocker_taxonomy_audit(
    *,
    signals: list[dict[str, Any]] | None = None,
    strategy_bottleneck: dict[str, Any] | None = None,
    strategy_structure_audit: dict[str, Any] | None = None,
    market_structure_audit: dict[str, Any] | None = None,
    fib_alignment_audit: dict[str, Any] | None = None,
    multi_timeframe_swing_audit: dict[str, Any] | None = None,
    bos_pivot_trace_audit: dict[str, Any] | None = None,
    strategy_decision_bridge_trace: dict[str, Any] | None = None,
    feed_scope_reconciliation: dict[str, Any] | None = None,
    no_setup_eligible_decomposition: dict[str, Any] | None = None,
    reversal_blocker_routing_audit: dict[str, Any] | None = None,
    state: dict[str, Any] | None = None,
    enabled: bool = True,
) -> dict[str, Any]:
    if not enabled:
        disabled = default_setup_blocker_taxonomy_audit_state("Setup/blocker taxonomy audit disabled.")
        disabled["enabled"] = False
        return disabled

    signal_rows = [dict(item) for item in list(signals or []) if isinstance(item, dict)]
    bridge_payload = dict(strategy_decision_bridge_trace or {})
    no_setup_payload = dict(no_setup_eligible_decomposition or {})
    reversal_payload = dict(reversal_blocker_routing_audit or {})
    mtf_payload = dict(multi_timeframe_swing_audit or {})
    bos_payload = dict(bos_pivot_trace_audit or {})
    market_payload = dict(market_structure_audit or {})
    feed_scope = dict(feed_scope_reconciliation or {})

    signal_map = _latest_by_symbol(signal_rows, symbol_keys=("asset", "symbol"))
    bridge_rows = [dict(item) for item in list(bridge_payload.get("recent_candidates", []) or []) if isinstance(item, dict)]
    no_setup_rows = [dict(item) for item in list(no_setup_payload.get("candidates", []) or []) if isinstance(item, dict)]
    reversal_rows = [dict(item) for item in list(reversal_payload.get("candidates", []) or []) if isinstance(item, dict)]
    mtf_rows = [dict(item) for item in list(mtf_payload.get("recent_candidates", []) or []) if isinstance(item, dict)]
    bos_rows = [dict(item) for item in list(bos_payload.get("recent_candidates", []) or []) if isinstance(item, dict)]
    market_rows = [
        dict(item)
        for item in list(market_payload.get("market_structure_best_candidates", []) or [])
        if isinstance(item, dict)
    ]

    symbols = set(signal_map.keys())
    for rows in (bridge_rows, no_setup_rows, reversal_rows, mtf_rows, bos_rows, market_rows):
        symbols.update(_symbol_from(row) for row in rows if _symbol_from(row))
    for payload in (bridge_payload, no_setup_payload, reversal_payload, mtf_payload, market_payload):
        top_symbol = _upper(payload.get("top_symbol") or payload.get("market_structure_top_symbol"))
        if top_symbol:
            symbols.add(top_symbol)
    symbols = {symbol for symbol in symbols if symbol}
    if not symbols:
        return default_setup_blocker_taxonomy_audit_state("Insufficient data for setup/blocker taxonomy audit.")

    candidates: list[dict[str, Any]] = []
    for symbol in sorted(symbols):
        signal = signal_map.get(symbol, {})
        bridge = _first_by_symbol(bridge_rows, symbol)
        no_setup = _first_by_symbol(no_setup_rows, symbol)
        if not no_setup and _upper(no_setup_payload.get("top_symbol")) == symbol:
            no_setup = {
                "symbol": symbol,
                "setup": no_setup_payload.get("top_setup"),
                "score": no_setup_payload.get("top_score"),
                "min_score": no_setup_payload.get("top_min_score"),
                "score_gap": no_setup_payload.get("top_score_gap"),
                "reason_bucket": no_setup_payload.get("top_reason_bucket"),
                "primary_real_blocker": no_setup_payload.get("top_real_blocker"),
                "secondary_real_blocker": no_setup_payload.get("top_secondary_blocker"),
            }
        reversal = _first_by_symbol(reversal_rows, symbol)
        if not reversal and _upper(reversal_payload.get("top_symbol")) == symbol:
            reversal = {
                "symbol": symbol,
                "setup": reversal_payload.get("top_setup"),
                "score": reversal_payload.get("top_score"),
                "min_score": reversal_payload.get("top_min_score"),
                "score_gap": reversal_payload.get("top_score_gap"),
                "route_status": reversal_payload.get("top_route_status"),
                "alternative_bucket": reversal_payload.get("top_alternative_bucket"),
                "primary_real_blocker": reversal_payload.get("observed_blocker"),
            }
        mtf = _first_by_symbol(mtf_rows, symbol)
        per_symbol_bos = _bos_rows_for_symbol(bos_rows, symbol)
        market_structure = _market_structure_for_symbol(market_payload, market_rows, symbol)
        candidate = _candidate_payload(
            symbol=symbol,
            signal=signal,
            bridge=bridge,
            no_setup=no_setup,
            reversal=reversal,
            mtf=mtf,
            bos_rows=per_symbol_bos,
            market_structure=market_structure,
            feed_scope=feed_scope,
        )
        if candidate["taxonomy_status"] != "INSUFFICIENT_DATA_FOR_TAXONOMY" or candidate["official_primary_blocker"]:
            candidates.append(candidate)

    candidates.sort(
        key=lambda row: (
            1 if row.get("taxonomy_status") == "NO_SETUP_WITH_PIVOT_BUT_NO_BOS" else 0,
            1 if row.get("taxonomy_status") in {"REVERSAL_CONTEXT_ON_TREND_SETUP", "MIXED_TREND_REVERSAL_TAXONOMY"} else 0,
            float(row.get("taxonomy_confidence") or 0.0),
            float(row.get("score") or 0.0),
        ),
        reverse=True,
    )
    limited = candidates[:MAX_CANDIDATES]
    if not limited:
        empty = default_setup_blocker_taxonomy_audit_state("No setup/blocker taxonomy sample found in this cycle.")
        empty["generated_at"] = _utc_now_iso()
        empty["status"] = "NO_TAXONOMY_SAMPLE"
        empty["current_feed_is_clean"] = _is_feed_clean(feed_scope)
        empty["fallback_blocker_scope"] = _upper(feed_scope.get("fallback_blocker_scope") or "UNKNOWN")
        empty["recommendation"] = "observe_more"
        return empty

    top = limited[0]
    status_counts = Counter(str(row.get("taxonomy_status") or "") for row in limited)
    recommendation = str(top.get("suggested_future_study") or "observe_more")
    if recommendation not in SAFE_RECOMMENDATIONS:
        recommendation = "observe_more"
    current_feed_is_clean = _is_feed_clean(feed_scope)
    fallback_scope = _upper(feed_scope.get("fallback_blocker_scope") or top.get("fallback_blocker_scope") or "UNKNOWN")
    if current_feed_is_clean and fallback_scope == "CURRENT_CYCLE":
        fallback_scope = "NONE"
    return {
        "enabled": True,
        "mode": MODE,
        "safety_mode": SAFETY_MODE,
        "status": "READY",
        "generated_at": _utc_now_iso(),
        "target_setup": TARGET_SETUP,
        "total_candidates_checked": int(len(symbols)),
        "taxonomy_cases_count": int(len(limited)),
        "top_symbol": str(top.get("symbol") or ""),
        "top_setup": str(top.get("setup") or TARGET_SETUP),
        "official_primary_blocker": str(top.get("official_primary_blocker") or ""),
        "official_secondary_blocker": str(top.get("official_secondary_blocker") or ""),
        "normalized_primary_reason": str(top.get("normalized_primary_reason") or "UNKNOWN"),
        "normalized_secondary_reason": str(top.get("normalized_secondary_reason") or "UNKNOWN"),
        "taxonomy_status": str(top.get("taxonomy_status") or "UNKNOWN_TAXONOMY"),
        "taxonomy_confidence": top.get("taxonomy_confidence"),
        "mixed_taxonomy_count": int(
            status_counts.get("MIXED_TREND_REVERSAL_TAXONOMY", 0)
            + status_counts.get("REVERSAL_CONTEXT_ON_TREND_SETUP", 0)
        ),
        "reversal_as_primary_count": int(
            sum(1 for row in limited if str(row.get("official_primary_blocker") or "") == "REVERSAL_NOT_ELIGIBLE")
        ),
        "reversal_as_context_count": int(
            sum(1 for row in limited if str(row.get("normalized_secondary_reason") or "") == "REVERSAL_RISK_CONTEXT")
        ),
        "no_setup_taxonomy_count": int(
            sum(1 for row in limited if str(row.get("official_primary_blocker") or "") == "NO_SETUP_ELIGIBLE" or "NO_SETUP" in str(row.get("taxonomy_status") or ""))
        ),
        "current_feed_is_clean": current_feed_is_clean,
        "fallback_blocker_scope": fallback_scope,
        "recommendation": recommendation,
        "should_keep_blocked": True,
        "safe_to_change_strategy_now": False,
        "safe_to_change_threshold_now": False,
        "notes": (
            "Setup/blocker taxonomy audit is diagnostic only; official blockers remain unchanged "
            "and shadow structure is not converted into operational authority."
        ),
        "candidates": limited,
        "strategy_bottleneck_dominant": _norm_text((strategy_bottleneck or {}).get("dominant_bottleneck")),
        "structural_audit_top_setup": _norm_text((strategy_structure_audit or {}).get("structural_audit_top_setup")),
        "fib_alignment_status": _norm_text((fib_alignment_audit or {}).get("fib_alignment_status")),
        "shadow_only": True,
        "state_seen": bool(state is not None),
    }
