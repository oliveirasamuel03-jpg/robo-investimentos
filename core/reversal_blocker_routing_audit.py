from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any


MODE = "DIAGNOSTIC_ONLY"
SAFETY_MODE = "SHADOW_ONLY"
TARGET_SETUP = "trend_pullback_breakout"
OBSERVED_BLOCKER = "REVERSAL_NOT_ELIGIBLE"
MAX_CANDIDATES = 10
SCORE_PRIMARY_GAP = 0.02

SAFE_RECOMMENDATIONS = {
    "observe_more",
    "study_routing_map",
    "study_real_rule_mapping",
    "study_pullback_quality",
    "study_breakout_confirmation",
    "study_multitf_conflict",
    "keep_blocked_until_structure_confirms",
    "keep_blocked_until_pullback_reaction",
    "keep_blocked_until_real_setup_maps_structure",
    "no_strategy_change_recommended",
    "insufficient_data",
}

CONFIRMED_BOS_STATES = {"BOS_BY_CLOSE_CONFIRMED", "BOS_RETEST_CONFIRMED"}
WEAK_OR_MISSING_BOS_STATES = {"", "NO_BOS", "BOS_BY_WICK_ONLY", "BOS_FAILED", "INSUFFICIENT_DATA"}
WEAK_OR_INVALID_PIVOT_STATES = {"", "NO_PIVOT", "PIVOT_FORMING", "PIVOT_INVALIDATED", "INSUFFICIENT_DATA"}
UP_DIRECTIONS = {"UP", "BULLISH"}
DOWN_DIRECTIONS = {"DOWN", "BEARISH"}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _norm_text(value: Any) -> str:
    return str(value or "").strip()


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
    raw = row.get("rejection_reasons") or row.get("real_rejection_reason") or []
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


def _primary_from_tokens(tokens: list[str]) -> str:
    if _has(tokens, ("reversal_not_eligible", "reversao_nao_elegivel", "reversal not eligible")):
        return OBSERVED_BLOCKER
    if _has(tokens, ("no_setup",)):
        return "NO_SETUP_ELIGIBLE"
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


def _bool_or_none(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        return None
    text = _norm_text(value).lower()
    if text in {"true", "1", "yes", "sim"}:
        return True
    if text in {"false", "0", "no", "nao", "não"}:
        return False
    return None


def _field(*rows: dict[str, Any], keys: tuple[str, ...], default: Any = "") -> Any:
    for row in rows:
        if not isinstance(row, dict):
            continue
        for key in keys:
            value = row.get(key)
            if value not in (None, ""):
                return value
    return default


def _direction_conflict(daily: str, h4: str, h1: str) -> bool:
    directions = {_norm_text(item).upper() for item in (daily, h4, h1) if _norm_text(item)}
    has_up = bool(directions & UP_DIRECTIONS)
    has_down = bool(directions & DOWN_DIRECTIONS)
    return bool(has_up and has_down)


def _mtf_conflict(bridge: dict[str, Any], mtf: dict[str, Any]) -> bool:
    status = _norm_text(_field(mtf, bridge, keys=("alignment_status", "multi_tf_alignment_status", "top_alignment_status"))).upper()
    if status == "CONFLICT":
        return True
    daily = _norm_text(_field(mtf, bridge, keys=("daily_bias",))).upper()
    h4 = _norm_text(_field(mtf, bridge, keys=("h4_structure",))).upper()
    h1 = _norm_text(_field(mtf, bridge, keys=("h1_confirmation",))).upper()
    return _direction_conflict(daily, h4, h1)


def _bos_state(bridge: dict[str, Any], bos_rows: dict[str, dict[str, Any]]) -> str:
    h4 = dict(bos_rows.get("4h", {}) or {})
    h1 = dict(bos_rows.get("1h", {}) or {})
    return _norm_text(
        _field(
            bridge,
            h4,
            h1,
            keys=("bos_state_4h", "bos_state", "bos_state_1h", "top_bos_state"),
        )
    ).upper()


def _pivot_state(bridge: dict[str, Any], bos_rows: dict[str, dict[str, Any]]) -> str:
    h4 = dict(bos_rows.get("4h", {}) or {})
    h1 = dict(bos_rows.get("1h", {}) or {})
    return _norm_text(
        _field(
            bridge,
            h4,
            h1,
            keys=("pivot_state_4h", "pivot_state", "pivot_state_1h", "top_pivot_state"),
        )
    ).upper()


def _has_reversal_blocker(
    *,
    tokens: list[str],
    primary: str,
    no_setup: dict[str, Any],
    bridge: dict[str, Any],
) -> bool:
    bucket = _norm_text(no_setup.get("reason_bucket") or no_setup.get("top_reason_bucket")).upper()
    bridge_blocker = _norm_text(bridge.get("primary_real_blocker")).upper()
    return bool(
        _norm_text(primary).upper() == OBSERVED_BLOCKER
        or bridge_blocker == OBSERVED_BLOCKER
        or "REVERSAL_NOT_ELIGIBLE" in bucket
        or bucket == "REVERSAL_BLOCKER_ON_TREND_SETUP"
        or _has(tokens, ("reversal_not_eligible", "reversao_nao_elegivel", "reversal not eligible"))
    )


def _recommendation_for(route_status: str, alternative_bucket: str) -> str:
    if route_status == "INSUFFICIENT_DATA_FOR_ROUTING":
        return "insufficient_data"
    if alternative_bucket == "SHOULD_BE_MULTITF_CONFLICT":
        return "study_multitf_conflict"
    if alternative_bucket == "SHOULD_BE_BREAKOUT_NOT_CONFIRMED":
        return "study_breakout_confirmation"
    if alternative_bucket in {"SHOULD_BE_PULLBACK_REACTION_MISSING", "SHOULD_BE_PULLBACK_NOT_CONFIRMED"}:
        return "study_pullback_quality"
    if alternative_bucket == "SHOULD_BE_SCORE_BELOW_MIN_PRIMARY":
        return "no_strategy_change_recommended"
    if route_status in {"MIXED_TREND_REVERSAL_ROUTING", "REVERSAL_PATTERN_NOT_ACTIVE_BUT_BLOCKER_PRESENT"}:
        return "study_routing_map"
    if route_status == "LEGITIMATE_REVERSAL_RISK_BLOCK":
        return "keep_blocked_until_structure_confirms"
    if route_status == "REVERSAL_BLOCKER_ON_TREND_SETUP":
        return "study_routing_map"
    return "observe_more"


def default_reversal_blocker_routing_audit_state(
    reason: str = "No reversal blocker routing audit data yet.",
) -> dict[str, Any]:
    return {
        "enabled": True,
        "mode": MODE,
        "safety_mode": SAFETY_MODE,
        "status": "INSUFFICIENT_DATA",
        "generated_at": "",
        "target_setup": TARGET_SETUP,
        "observed_blocker": OBSERVED_BLOCKER,
        "total_candidates_checked": 0,
        "reversal_blocker_count": 0,
        "trend_candidates_with_reversal_blocker": 0,
        "reversal_candidates_with_reversal_blocker": 0,
        "mixed_routing_count": 0,
        "top_symbol": "",
        "top_setup": TARGET_SETUP,
        "top_route_status": "INSUFFICIENT_DATA_FOR_ROUTING",
        "top_reason": "",
        "top_alternative_bucket": "INSUFFICIENT_DATA_FOR_ROUTING",
        "current_feed_is_clean": False,
        "fallback_blocker_scope": "UNKNOWN",
        "recommendation": "insufficient_data",
        "should_keep_blocked": True,
        "safe_to_change_strategy_now": False,
        "notes": reason,
        "candidates": [],
        "shadow_only": True,
    }


def _classify_route(
    *,
    setup: str,
    tokens: list[str],
    primary: str,
    secondary: str,
    score_gap: float | None,
    bridge: dict[str, Any],
    no_setup: dict[str, Any],
    mtf: dict[str, Any],
    bos_rows: dict[str, dict[str, Any]],
    market_structure: dict[str, Any],
) -> tuple[str, str, str]:
    if not tokens and not primary and not bridge and not no_setup and not mtf and not bos_rows:
        return (
            "INSUFFICIENT_DATA_FOR_ROUTING",
            "INSUFFICIENT_DATA_FOR_ROUTING",
            "insufficient_data_for_routing",
        )

    setup_norm = _norm_text(setup).lower()
    trend_setup = setup_norm == TARGET_SETUP
    reversal_setup = "reversal" in setup_norm
    has_reversal = _has_reversal_blocker(tokens=tokens, primary=primary, no_setup=no_setup, bridge=bridge)
    if not has_reversal:
        return ("UNKNOWN_ROUTING_REASON", "UNKNOWN_ROUTING_REASON", "reversal_blocker_not_observed")

    multi_tf_conflict = _mtf_conflict(bridge, mtf)
    bos_state = _bos_state(bridge, bos_rows)
    pivot_state = _pivot_state(bridge, bos_rows)
    no_confirmed_bos = bos_state in WEAK_OR_MISSING_BOS_STATES
    weak_or_invalid_pivot = pivot_state in WEAK_OR_INVALID_PIVOT_STATES
    explicit_reversal_active = _bool_or_none(
        _field(bridge, no_setup, market_structure, keys=("reversal_pattern_active", "structure_confirms_reversal"))
    )
    breakout_missing = (
        _norm_text(secondary).upper() == "BREAKOUT_NOT_CONFIRMED"
        or _has(tokens, ("breakout_not_confirmed", "breakout", "rompimento"))
    )
    score_primary = _norm_text(primary).upper() == "SCORE_BELOW_MIN" or _has(tokens, ("score_below", "score"))
    score_gap_large = score_gap is not None and float(score_gap) > SCORE_PRIMARY_GAP
    pullback_missing = _has(tokens, ("pullback", "reaction", "reacao", "reação"))
    no_setup_bucket = _norm_text(no_setup.get("reason_bucket") or no_setup.get("top_reason_bucket")).upper()

    if trend_setup and multi_tf_conflict and (no_confirmed_bos or weak_or_invalid_pivot):
        return (
            "LEGITIMATE_REVERSAL_RISK_BLOCK",
            "SHOULD_BE_MULTITF_CONFLICT",
            "multi_timeframe_conflict_with_missing_or_invalid_bos_pivot",
        )
    if trend_setup and explicit_reversal_active is False:
        return (
            "REVERSAL_PATTERN_NOT_ACTIVE_BUT_BLOCKER_PRESENT",
            "MIXED_TREND_REVERSAL_ROUTING",
            "trend_setup_received_reversal_blocker_while_reversal_pattern_inactive",
        )
    if reversal_setup:
        return (
            "LEGITIMATE_REVERSAL_RISK_BLOCK",
            "LEGITIMATE_REVERSAL_RISK_BLOCK",
            "reversal_setup_failed_reversal_eligibility_guard",
        )
    if trend_setup and breakout_missing:
        return (
            "REVERSAL_BLOCKER_ON_TREND_SETUP",
            "SHOULD_BE_BREAKOUT_NOT_CONFIRMED",
            "trend_setup_reversal_blocker_coexists_with_breakout_not_confirmed",
        )
    if trend_setup and (score_primary or score_gap_large):
        return (
            "REVERSAL_BLOCKER_ON_TREND_SETUP",
            "SHOULD_BE_SCORE_BELOW_MIN_PRIMARY",
            "trend_setup_reversal_blocker_coexists_with_score_below_minimum",
        )
    if trend_setup and pullback_missing:
        return (
            "REVERSAL_BLOCKER_ON_TREND_SETUP",
            "SHOULD_BE_PULLBACK_REACTION_MISSING",
            "trend_setup_reversal_blocker_coexists_with_pullback_reaction_missing",
        )
    if trend_setup and no_setup_bucket == "REVERSAL_BLOCKER_ON_TREND_SETUP":
        return (
            "REVERSAL_BLOCKER_ON_TREND_SETUP",
            "REVERSAL_BLOCKER_ON_TREND_SETUP",
            "no_setup_decomposition_already_flagged_reversal_blocker_on_trend_setup",
        )
    if trend_setup:
        return (
            "REVERSAL_BLOCKER_ON_TREND_SETUP",
            "UNKNOWN_ROUTING_REASON",
            "trend_setup_received_reversal_not_eligible_blocker",
        )
    return (
        "MIXED_TREND_REVERSAL_ROUTING",
        "UNKNOWN_ROUTING_REASON",
        "reversal_blocker_observed_outside_clear_reversal_setup",
    )


def _candidate_payload(
    *,
    symbol: str,
    signal: dict[str, Any],
    bridge: dict[str, Any],
    no_setup: dict[str, Any],
    mtf: dict[str, Any],
    bos_rows: dict[str, dict[str, Any]],
    market_structure: dict[str, Any],
    feed_scope: dict[str, Any],
) -> dict[str, Any] | None:
    tokens = _reason_tokens(signal) or _reason_tokens(bridge) or _reason_tokens(no_setup)
    primary = (
        _norm_text(bridge.get("primary_real_blocker"))
        or _norm_text(no_setup.get("primary_real_blocker"))
        or _norm_text(no_setup.get("top_real_blocker"))
        or _primary_from_tokens(tokens)
    )
    secondary = (
        _norm_text(bridge.get("secondary_real_blocker"))
        or _norm_text(no_setup.get("secondary_real_blocker"))
        or _norm_text(no_setup.get("top_secondary_blocker"))
        or _secondary_from_tokens(tokens)
    )
    has_reversal = _has_reversal_blocker(tokens=tokens, primary=primary, no_setup=no_setup, bridge=bridge)
    if not has_reversal:
        return None

    setup = _norm_text(signal.get("strategy_name") or bridge.get("real_strategy") or no_setup.get("setup") or TARGET_SETUP)
    score = _as_float(
        bridge.get("real_score"),
        _as_float(no_setup.get("score"), _as_float(signal.get("score"))),
    )
    min_score = _as_float(
        bridge.get("min_score"),
        _as_float(
            no_setup.get("min_score"),
            _as_float(signal.get("effective_min_signal_score"), _as_float(signal.get("base_min_signal_score"))),
        ),
    )
    score_gap = None if score is None or min_score is None else max(0.0, float(min_score) - float(score))
    route_status, alternative_bucket, route_detail = _classify_route(
        setup=setup,
        tokens=tokens,
        primary=primary,
        secondary=secondary,
        score_gap=score_gap,
        bridge=bridge,
        no_setup=no_setup,
        mtf=mtf,
        bos_rows=bos_rows,
        market_structure=market_structure,
    )
    h4 = dict(bos_rows.get("4h", {}) or {})
    h1 = dict(bos_rows.get("1h", {}) or {})
    bos_state = _norm_text(
        bridge.get("bos_state_4h")
        or h4.get("bos_state")
        or bridge.get("bos_state_1h")
        or h1.get("bos_state")
    )
    pivot_state = _norm_text(
        bridge.get("pivot_state_4h")
        or h4.get("pivot_state")
        or bridge.get("pivot_state_1h")
        or h1.get("pivot_state")
    )
    recommendation = _recommendation_for(route_status, alternative_bucket)
    if recommendation not in SAFE_RECOMMENDATIONS:
        recommendation = "observe_more"
    current_feed_is_clean = bool(feed_scope.get("current_feed_is_clean", False))
    fallback_scope = _norm_text(feed_scope.get("fallback_blocker_scope") or bridge.get("fallback_blocker_scope") or "UNKNOWN")
    if current_feed_is_clean and fallback_scope == "CURRENT_CYCLE":
        fallback_scope = "NONE"
    return {
        "symbol": symbol,
        "setup": setup or TARGET_SETUP,
        "score": _round(score),
        "min_score": _round(min_score),
        "score_gap": _round(score_gap),
        "primary_real_blocker": primary,
        "secondary_real_blocker": secondary,
        "real_rejection_reason": ", ".join(tokens),
        "route_status": route_status,
        "alternative_bucket": alternative_bucket,
        "route_detail": route_detail,
        "reversal_pattern_active": bool(_bool_or_none(_field(bridge, no_setup, keys=("reversal_pattern_active",))) is True),
        "trend_setup_active": setup == TARGET_SETUP,
        "multi_tf_alignment_status": _norm_text(
            mtf.get("alignment_status") or bridge.get("multi_tf_alignment_status") or ""
        ),
        "daily_bias": _norm_text(mtf.get("daily_bias") or ""),
        "h4_structure": _norm_text(mtf.get("h4_structure") or ""),
        "h1_confirmation": _norm_text(mtf.get("h1_confirmation") or ""),
        "bos_state": bos_state,
        "pivot_state": pivot_state,
        "fib_zone": _norm_text(
            market_structure.get("current_fib_zone")
            or market_structure.get("fib_zone")
            or market_structure.get("market_structure_top_zone")
            or ""
        ),
        "pullback_evidence": _norm_text(no_setup.get("pullback_evidence") or market_structure.get("pullback_state") or ""),
        "breakout_evidence": _norm_text(no_setup.get("breakout_evidence") or bridge.get("breakout_evidence") or ""),
        "current_feed_is_clean": current_feed_is_clean,
        "fallback_blocker_scope": fallback_scope,
        "should_keep_blocked": True,
        "safe_to_change_strategy_now": False,
        "suggested_future_study": recommendation,
    }


def build_reversal_blocker_routing_audit(
    *,
    signals: list[dict[str, Any]] | None = None,
    strategy_decision_bridge_trace: dict[str, Any] | None = None,
    no_setup_eligible_decomposition: dict[str, Any] | None = None,
    multi_timeframe_swing_audit: dict[str, Any] | None = None,
    bos_pivot_trace_audit: dict[str, Any] | None = None,
    market_structure_audit: dict[str, Any] | None = None,
    fib_alignment_audit: dict[str, Any] | None = None,
    feed_scope_reconciliation: dict[str, Any] | None = None,
    enabled: bool = True,
) -> dict[str, Any]:
    if not enabled:
        disabled = default_reversal_blocker_routing_audit_state("Reversal blocker routing audit disabled.")
        disabled["enabled"] = False
        return disabled

    signal_rows = [dict(item) for item in list(signals or []) if isinstance(item, dict)]
    bridge_payload = dict(strategy_decision_bridge_trace or {})
    no_setup_payload = dict(no_setup_eligible_decomposition or {})
    mtf_payload = dict(multi_timeframe_swing_audit or {})
    bos_payload = dict(bos_pivot_trace_audit or {})
    market_payload = dict(market_structure_audit or {})
    feed_scope = dict(feed_scope_reconciliation or {})

    signal_map = _latest_by_symbol(signal_rows, symbol_keys=("asset", "symbol"))
    bridge_rows = [dict(item) for item in list(bridge_payload.get("recent_candidates", []) or []) if isinstance(item, dict)]
    no_setup_rows = [dict(item) for item in list(no_setup_payload.get("candidates", []) or []) if isinstance(item, dict)]
    mtf_rows = [dict(item) for item in list(mtf_payload.get("recent_candidates", []) or []) if isinstance(item, dict)]
    bos_rows = [dict(item) for item in list(bos_payload.get("recent_candidates", []) or []) if isinstance(item, dict)]
    market_rows = [
        dict(item)
        for item in list(market_payload.get("market_structure_best_candidates", []) or [])
        if isinstance(item, dict)
    ]

    symbols = set(signal_map.keys())
    symbols.update(_symbol_from(row) for row in bridge_rows if _symbol_from(row))
    symbols.update(_symbol_from(row) for row in no_setup_rows if _symbol_from(row))
    symbols.update(_symbol_from(row) for row in mtf_rows if _symbol_from(row))
    symbols.update(_symbol_from(row) for row in bos_rows if _symbol_from(row))
    if _norm_text(no_setup_payload.get("top_symbol")):
        symbols.add(_norm_text(no_setup_payload.get("top_symbol")).upper())
    if _norm_text(bridge_payload.get("top_symbol")):
        symbols.add(_norm_text(bridge_payload.get("top_symbol")).upper())
    symbols = {symbol for symbol in symbols if symbol}
    if not symbols:
        return default_reversal_blocker_routing_audit_state("Insufficient data for reversal blocker routing audit.")

    candidates: list[dict[str, Any]] = []
    for symbol in sorted(symbols):
        signal = signal_map.get(symbol, {})
        bridge = _first_by_symbol(bridge_rows, symbol)
        no_setup = _first_by_symbol(no_setup_rows, symbol)
        if not no_setup and _norm_text(no_setup_payload.get("top_symbol")).upper() == symbol:
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
        mtf = _first_by_symbol(mtf_rows, symbol)
        per_symbol_bos = _bos_rows_for_symbol(bos_rows, symbol)
        market_structure = _first_by_symbol(market_rows, symbol)
        if not market_structure and _norm_text(market_payload.get("market_structure_top_symbol")).upper() == symbol:
            market_structure = {
                "market_structure_top_zone": market_payload.get("market_structure_top_zone"),
                "market_structure_score": market_payload.get("market_structure_top_score"),
            }
        candidate = _candidate_payload(
            symbol=symbol,
            signal=signal,
            bridge=bridge,
            no_setup=no_setup,
            mtf=mtf,
            bos_rows=per_symbol_bos,
            market_structure=market_structure,
            feed_scope=feed_scope,
        )
        if candidate:
            candidates.append(candidate)

    candidates.sort(
        key=lambda row: (
            1 if row.get("route_status") == "LEGITIMATE_REVERSAL_RISK_BLOCK" else 0,
            1 if row.get("alternative_bucket") == "SHOULD_BE_MULTITF_CONFLICT" else 0,
            1 if row.get("route_status") == "REVERSAL_BLOCKER_ON_TREND_SETUP" else 0,
            float(row.get("score") or 0.0),
        ),
        reverse=True,
    )
    limited = candidates[:MAX_CANDIDATES]
    if not limited:
        empty = default_reversal_blocker_routing_audit_state("No REVERSAL_NOT_ELIGIBLE routing sample found in this cycle.")
        empty["generated_at"] = _utc_now_iso()
        empty["status"] = "NO_REVERSAL_BLOCKER_SAMPLE"
        empty["current_feed_is_clean"] = bool(feed_scope.get("current_feed_is_clean", False))
        empty["fallback_blocker_scope"] = _norm_text(feed_scope.get("fallback_blocker_scope") or "UNKNOWN")
        empty["recommendation"] = "observe_more"
        return empty

    top = limited[0]
    route_counts = Counter(str(row.get("route_status") or "") for row in limited)
    mixed_count = sum(
        int(route_counts.get(status, 0))
        for status in (
            "MIXED_TREND_REVERSAL_ROUTING",
            "REVERSAL_PATTERN_NOT_ACTIVE_BUT_BLOCKER_PRESENT",
            "TREND_SETUP_BLOCKED_BY_REVERSAL_FILTER",
        )
    )
    recommendation = str(top.get("suggested_future_study") or "observe_more")
    if recommendation not in SAFE_RECOMMENDATIONS:
        recommendation = "observe_more"
    fallback_scope = _norm_text(feed_scope.get("fallback_blocker_scope") or top.get("fallback_blocker_scope") or "UNKNOWN")
    current_feed_is_clean = bool(feed_scope.get("current_feed_is_clean", False))
    if current_feed_is_clean and fallback_scope == "CURRENT_CYCLE":
        fallback_scope = "NONE"
    return {
        "enabled": True,
        "mode": MODE,
        "safety_mode": SAFETY_MODE,
        "status": "READY",
        "generated_at": _utc_now_iso(),
        "target_setup": TARGET_SETUP,
        "observed_blocker": OBSERVED_BLOCKER,
        "total_candidates_checked": int(len(symbols)),
        "reversal_blocker_count": int(len(limited)),
        "trend_candidates_with_reversal_blocker": int(
            sum(1 for row in limited if str(row.get("setup") or "") == TARGET_SETUP)
        ),
        "reversal_candidates_with_reversal_blocker": int(
            sum(1 for row in limited if "reversal" in str(row.get("setup") or "").lower())
        ),
        "mixed_routing_count": int(mixed_count),
        "top_symbol": str(top.get("symbol") or ""),
        "top_setup": str(top.get("setup") or TARGET_SETUP),
        "top_route_status": str(top.get("route_status") or "UNKNOWN_ROUTING_REASON"),
        "top_reason": str(top.get("route_detail") or ""),
        "top_alternative_bucket": str(top.get("alternative_bucket") or "UNKNOWN_ROUTING_REASON"),
        "top_score": top.get("score"),
        "top_min_score": top.get("min_score"),
        "top_score_gap": top.get("score_gap"),
        "current_feed_is_clean": current_feed_is_clean,
        "fallback_blocker_scope": fallback_scope,
        "recommendation": recommendation,
        "should_keep_blocked": True,
        "safe_to_change_strategy_now": False,
        "notes": (
            "Reversal blocker routing audit is diagnostic only; real strategy remains authoritative "
            "and REVERSAL_NOT_ELIGIBLE is not removed or bypassed."
        ),
        "candidates": limited,
        "fib_alignment_status": _norm_text((fib_alignment_audit or {}).get("fib_alignment_status")),
        "shadow_only": True,
    }
