from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any


MODE = "DIAGNOSTIC_ONLY"
SAFETY_MODE = "SHADOW_ONLY"
TARGET_SETUP = "trend_pullback_breakout"
MAX_CANDIDATES = 10
NEAR_SCORE_GAP = 0.01

CONFIRMED_BOS_STATES = {"BOS_BY_CLOSE_CONFIRMED", "BOS_RETEST_CONFIRMED"}
CONFIRMED_PIVOT_STATES = {"PIVOT_CONFIRMED", "PIVOT_TRIGGERED"}
SAFE_RECOMMENDATIONS = {
    "observe_more",
    "study_real_rule_mapping",
    "study_pullback_quality",
    "study_breakout_confirmation",
    "study_trend_pullback_selectivity",
    "candidate_for_future_shadow_calibration",
    "keep_blocked_until_pullback_reaction",
    "keep_blocked_until_real_setup_maps_structure",
    "no_threshold_change_recommended",
    "insufficient_data",
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


def _norm_text(value: Any) -> str:
    return str(value or "").strip()


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
    if _has(tokens, ("fallback", "feed", "provider_unknown")):
        return "FEED_OR_PROVIDER"
    if _has(tokens, ("daily_loss", "position_limit", "cooldown", "duplicate", "broker")):
        return "RISK_GUARD"
    if _has(tokens, ("no_setup",)):
        return "NO_SETUP_ELIGIBLE"
    if _has(tokens, ("reversal_not_eligible",)):
        return "REVERSAL_NOT_ELIGIBLE"
    if _has(tokens, ("trend_not_confirmed", "trend")):
        return "TREND_NOT_CONFIRMED"
    if _has(tokens, ("rsi",)):
        return "RSI_OUT_OF_RANGE"
    if _has(tokens, ("score_below", "score")):
        return "SCORE_BELOW_MIN"
    return ""


def _secondary_from_tokens(tokens: list[str]) -> str:
    if _has(tokens, ("breakout", "rompimento")):
        return "BREAKOUT_NOT_CONFIRMED"
    if _has(tokens, ("momentum", "confidence_too_low")):
        return "MOMENTUM_WEAK"
    if _has(tokens, ("secondary", "secundaria", "confirmacao")):
        return "SECONDARY_CONFIRMATION_WEAK"
    return ""


def _structure_confirmed(*, bridge: dict[str, Any], mtf: dict[str, Any], bos_rows: dict[str, dict[str, Any]]) -> bool:
    bridge_status = _norm_text(bridge.get("top_structure_status") or bridge.get("bos_pivot_status")).upper()
    if bridge_status == "STRUCTURE_CONFIRMED":
        return True
    h4 = dict(bos_rows.get("4h", {}) or {})
    h1 = dict(bos_rows.get("1h", {}) or {})
    bos_confirmed = _norm_text(h4.get("bos_state") or bridge.get("bos_state_4h")).upper() in CONFIRMED_BOS_STATES
    bos_confirmed = bos_confirmed or _norm_text(h1.get("bos_state") or bridge.get("bos_state_1h")).upper() in CONFIRMED_BOS_STATES
    pivot_confirmed = _norm_text(h4.get("pivot_state") or bridge.get("pivot_state_4h")).upper() in CONFIRMED_PIVOT_STATES
    pivot_confirmed = pivot_confirmed or _norm_text(h1.get("pivot_state") or bridge.get("pivot_state_1h")).upper() in CONFIRMED_PIVOT_STATES
    mtf_status = _norm_text(mtf.get("alignment_status") or bridge.get("multi_tf_alignment_status")).upper()
    return bool((bos_confirmed and pivot_confirmed) or (mtf_status == "STRONG_ALIGNMENT" and (bos_confirmed or pivot_confirmed)))


def _mtf_strong(mtf: dict[str, Any], bridge: dict[str, Any]) -> bool:
    status = _norm_text(mtf.get("alignment_status") or bridge.get("multi_tf_alignment_status")).upper()
    return status == "STRONG_ALIGNMENT"


def _bos_confirmed(bridge: dict[str, Any], bos_rows: dict[str, dict[str, Any]]) -> bool:
    states = [
        bridge.get("bos_state_4h"),
        bridge.get("bos_state_1h"),
        (bos_rows.get("4h", {}) or {}).get("bos_state"),
        (bos_rows.get("1h", {}) or {}).get("bos_state"),
    ]
    return any(_norm_text(item).upper() in CONFIRMED_BOS_STATES for item in states)


def _pivot_triggered(bridge: dict[str, Any], bos_rows: dict[str, dict[str, Any]]) -> bool:
    states = [
        bridge.get("pivot_state_4h"),
        bridge.get("pivot_state_1h"),
        (bos_rows.get("4h", {}) or {}).get("pivot_state"),
        (bos_rows.get("1h", {}) or {}).get("pivot_state"),
    ]
    return any(_norm_text(item).upper() in CONFIRMED_PIVOT_STATES for item in states)


def _classify_reason_bucket(
    *,
    tokens: list[str],
    primary: str,
    secondary: str,
    score_gap: float | None,
    bridge: dict[str, Any],
    mtf: dict[str, Any],
    bos_rows: dict[str, dict[str, Any]],
    market_structure: dict[str, Any],
    feed_scope: dict[str, Any],
) -> str:
    if not tokens and not bridge and not mtf and not bos_rows and not market_structure:
        return "INSUFFICIENT_DATA_FOR_DECOMPOSITION"

    feed_clean = bool(feed_scope.get("current_feed_is_clean", False))
    no_setup = primary == "NO_SETUP_ELIGIBLE" or _has(tokens, ("no_setup",))
    breakout_missing = secondary == "BREAKOUT_NOT_CONFIRMED" or _has(tokens, ("breakout_not_confirmed", "breakout"))
    structure_confirmed = _structure_confirmed(bridge=bridge, mtf=mtf, bos_rows=bos_rows)
    mtf_strong = _mtf_strong(mtf, bridge)
    bos_confirmed = _bos_confirmed(bridge, bos_rows)
    pivot_triggered = _pivot_triggered(bridge, bos_rows)
    fib_zone = _norm_text(market_structure.get("current_fib_zone") or market_structure.get("fib_zone") or market_structure.get("market_structure_top_zone")).upper()
    fib_score = _as_float(market_structure.get("market_structure_score") or market_structure.get("structure_score"), 0.0) or 0.0
    fib_good = bool(fib_score >= 0.65 or fib_zone in {"BREAKOUT_ZONE", "SHALLOW_ZONE", "MEDIUM_ZONE", "DEEP_ZONE"})

    if no_setup and structure_confirmed:
        return "STRUCTURE_CONFIRMED_BUT_SETUP_NOT_ELIGIBLE"
    if score_gap is not None and float(score_gap) <= NEAR_SCORE_GAP and breakout_missing:
        return "SCORE_NEAR_MIN_BUT_BREAKOUT_MISSING"
    if no_setup and mtf_strong:
        return "MULTITF_STRONG_BUT_REAL_RULE_MISSING"
    if no_setup and bos_confirmed:
        return "BOS_CONFIRMED_BUT_REAL_SETUP_MISSING"
    if no_setup and pivot_triggered:
        return "PIVOT_TRIGGERED_BUT_REAL_SETUP_MISSING"
    if no_setup and fib_good:
        return "FIB_STRUCTURE_GOOD_BUT_PULLBACK_MISSING"
    if _has(tokens, ("reversal_not_eligible",)):
        return "REVERSAL_BLOCKER_ON_TREND_SETUP"
    if breakout_missing:
        return "BREAKOUT_NOT_CONFIRMED"
    if no_setup:
        return "PULLBACK_NOT_CONFIRMED"
    if _has(tokens, ("pullback",)) and _has(tokens, ("reaction",)):
        return "PULLBACK_REACTION_MISSING"
    if _has(tokens, ("score_below", "score")):
        return "SCORE_BELOW_MIN_PRIMARY"
    if _has(tokens, ("rsi", "momentum", "confidence_too_low", "secondary")):
        return "RSI_MOMENTUM_SECONDARY_BLOCKER"
    if feed_clean:
        return "FEED_CLEAN_NOT_BLOCKER"
    return "UNKNOWN_NO_SETUP_REASON"


def _recommendation_for_bucket(bucket: str) -> str:
    mapping = {
        "STRUCTURE_CONFIRMED_BUT_SETUP_NOT_ELIGIBLE": "study_real_rule_mapping",
        "SCORE_NEAR_MIN_BUT_BREAKOUT_MISSING": "study_breakout_confirmation",
        "BREAKOUT_NOT_CONFIRMED": "study_breakout_confirmation",
        "PULLBACK_NOT_CONFIRMED": "study_pullback_quality",
        "PULLBACK_REACTION_MISSING": "keep_blocked_until_pullback_reaction",
        "PULLBACK_TOO_SHALLOW_OR_NOT_VISIBLE": "study_pullback_quality",
        "PULLBACK_TOO_DEEP_OR_REVERSAL_RISK": "study_pullback_quality",
        "ENTRY_CONFIRMATION_MISSING": "keep_blocked_until_real_setup_maps_structure",
        "MULTITF_STRONG_BUT_REAL_RULE_MISSING": "study_real_rule_mapping",
        "BOS_CONFIRMED_BUT_REAL_SETUP_MISSING": "study_real_rule_mapping",
        "PIVOT_TRIGGERED_BUT_REAL_SETUP_MISSING": "study_real_rule_mapping",
        "FIB_STRUCTURE_GOOD_BUT_PULLBACK_MISSING": "study_pullback_quality",
        "RSI_MOMENTUM_SECONDARY_BLOCKER": "study_trend_pullback_selectivity",
        "REVERSAL_BLOCKER_ON_TREND_SETUP": "keep_blocked_until_real_setup_maps_structure",
        "SCORE_BELOW_MIN_PRIMARY": "no_threshold_change_recommended",
        "FEED_CLEAN_NOT_BLOCKER": "observe_more",
        "INSUFFICIENT_DATA_FOR_DECOMPOSITION": "insufficient_data",
    }
    recommendation = mapping.get(bucket, "observe_more")
    return recommendation if recommendation in SAFE_RECOMMENDATIONS else "observe_more"


def default_no_setup_eligible_decomposition_state(
    reason: str = "No NO_SETUP_ELIGIBLE decomposition data yet.",
) -> dict[str, Any]:
    return {
        "enabled": True,
        "mode": MODE,
        "safety_mode": SAFETY_MODE,
        "status": "INSUFFICIENT_DATA",
        "generated_at": "",
        "target_setup": TARGET_SETUP,
        "total_candidates_checked": 0,
        "no_setup_eligible_count": 0,
        "top_symbol": "",
        "top_setup": TARGET_SETUP,
        "top_reason_bucket": "INSUFFICIENT_DATA_FOR_DECOMPOSITION",
        "top_real_blocker": "",
        "top_secondary_blocker": "",
        "top_score": None,
        "top_min_score": None,
        "top_score_gap": None,
        "current_feed_is_clean": False,
        "fallback_blocker_scope": "UNKNOWN",
        "structure_confirmed_count": 0,
        "structure_confirmed_but_no_setup_count": 0,
        "near_approved_no_setup_count": 0,
        "recommendation": "insufficient_data",
        "should_keep_blocked": True,
        "notes": reason,
        "candidates": [],
        "shadow_only": True,
    }


def _candidate_payload(
    *,
    symbol: str,
    signal: dict[str, Any],
    bridge: dict[str, Any],
    mtf: dict[str, Any],
    bos_rows: dict[str, dict[str, Any]],
    market_structure: dict[str, Any],
    feed_scope: dict[str, Any],
) -> dict[str, Any]:
    tokens = _reason_tokens(signal) or _reason_tokens(bridge)
    primary = _norm_text(bridge.get("primary_real_blocker")) or _primary_from_tokens(tokens)
    secondary = _norm_text(bridge.get("secondary_real_blocker")) or _secondary_from_tokens(tokens)
    score = _as_float(bridge.get("real_score"), _as_float(signal.get("score")))
    min_score = _as_float(
        bridge.get("min_score"),
        _as_float(signal.get("effective_min_signal_score"), _as_float(signal.get("base_min_signal_score"))),
    )
    score_gap = None if score is None or min_score is None else max(0.0, float(min_score) - float(score))
    bucket = _classify_reason_bucket(
        tokens=tokens,
        primary=primary,
        secondary=secondary,
        score_gap=score_gap,
        bridge=bridge,
        mtf=mtf,
        bos_rows=bos_rows,
        market_structure=market_structure,
        feed_scope=feed_scope,
    )
    h4 = dict(bos_rows.get("4h", {}) or {})
    h1 = dict(bos_rows.get("1h", {}) or {})
    fib_zone = (
        market_structure.get("current_fib_zone")
        or market_structure.get("fib_zone")
        or market_structure.get("market_structure_top_zone")
        or ""
    )
    structure_confirmed = _structure_confirmed(bridge=bridge, mtf=mtf, bos_rows=bos_rows)
    feed_clean = bool(feed_scope.get("current_feed_is_clean", False))
    suggestion = _recommendation_for_bucket(bucket)
    return {
        "symbol": symbol,
        "setup": _norm_text(signal.get("strategy_name") or bridge.get("real_strategy") or TARGET_SETUP),
        "score": _round(score),
        "min_score": _round(min_score),
        "score_gap": _round(score_gap),
        "real_rejection_reason": ", ".join(tokens),
        "primary_real_blocker": primary,
        "secondary_real_blocker": secondary,
        "reason_bucket": bucket,
        "reason_detail": (
            f"bucket={bucket}; primary={primary or 'none'}; secondary={secondary or 'none'}; "
            f"structure_confirmed={int(structure_confirmed)}; feed_clean={int(feed_clean)}"
        ),
        "multi_tf_alignment_status": _norm_text(mtf.get("alignment_status") or bridge.get("multi_tf_alignment_status")),
        "daily_bias": _norm_text(mtf.get("daily_bias")),
        "h4_structure": _norm_text(mtf.get("h4_structure")),
        "h1_confirmation": _norm_text(mtf.get("h1_confirmation")),
        "bos_state": _norm_text(bridge.get("bos_state_4h") or h4.get("bos_state") or bridge.get("bos_state_1h") or h1.get("bos_state")),
        "pivot_state": _norm_text(bridge.get("pivot_state_4h") or h4.get("pivot_state") or bridge.get("pivot_state_1h") or h1.get("pivot_state")),
        "fib_zone": _norm_text(fib_zone),
        "structure_score": _round(market_structure.get("market_structure_score") or market_structure.get("structure_score")),
        "feed_scope_status": _norm_text(feed_scope.get("fallback_scope_status") or "UNKNOWN_SCOPE"),
        "current_feed_is_clean": feed_clean,
        "feed_is_not_current_blocker": bool(feed_clean and _norm_text(feed_scope.get("fallback_blocker_scope")) != "CURRENT_CYCLE"),
        "should_keep_blocked": True,
        "safe_to_change_threshold_now": False,
        "suggested_future_study": suggestion,
    }


def build_no_setup_eligible_decomposition(
    *,
    signals: list[dict[str, Any]] | None = None,
    strategy_decision_bridge_trace: dict[str, Any] | None = None,
    multi_timeframe_swing_audit: dict[str, Any] | None = None,
    bos_pivot_trace_audit: dict[str, Any] | None = None,
    market_structure_audit: dict[str, Any] | None = None,
    fib_alignment_audit: dict[str, Any] | None = None,
    feed_scope_reconciliation: dict[str, Any] | None = None,
    enabled: bool = True,
) -> dict[str, Any]:
    if not enabled:
        disabled = default_no_setup_eligible_decomposition_state("NO_SETUP_ELIGIBLE decomposition disabled.")
        disabled["enabled"] = False
        disabled["recommendation"] = "insufficient_data"
        return disabled

    signal_rows = [dict(item) for item in list(signals or []) if isinstance(item, dict)]
    bridge_payload = dict(strategy_decision_bridge_trace or {})
    mtf_payload = dict(multi_timeframe_swing_audit or {})
    bos_payload = dict(bos_pivot_trace_audit or {})
    market_payload = dict(market_structure_audit or {})
    feed_scope = dict(feed_scope_reconciliation or {})

    signal_map = _latest_by_symbol(signal_rows, symbol_keys=("asset", "symbol"))
    bridge_rows = [dict(item) for item in list(bridge_payload.get("recent_candidates", []) or []) if isinstance(item, dict)]
    mtf_rows = [dict(item) for item in list(mtf_payload.get("recent_candidates", []) or []) if isinstance(item, dict)]
    bos_rows = [dict(item) for item in list(bos_payload.get("recent_candidates", []) or []) if isinstance(item, dict)]
    market_rows = [
        dict(item)
        for item in list(market_payload.get("market_structure_best_candidates", []) or [])
        if isinstance(item, dict)
    ]

    symbols = set(signal_map.keys())
    symbols.update(_symbol_from(row) for row in bridge_rows if _symbol_from(row))
    symbols.update(_symbol_from(row) for row in mtf_rows if _symbol_from(row))
    symbols.update(_symbol_from(row) for row in bos_rows if _symbol_from(row))
    if not symbols:
        return default_no_setup_eligible_decomposition_state("Insufficient data for NO_SETUP_ELIGIBLE decomposition.")

    candidates = []
    for symbol in sorted(symbols):
        signal = signal_map.get(symbol, {})
        bridge = _first_by_symbol(bridge_rows, symbol)
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
            mtf=mtf,
            bos_rows=per_symbol_bos,
            market_structure=market_structure,
            feed_scope=feed_scope,
        )
        primary = str(candidate.get("primary_real_blocker") or "")
        bucket = str(candidate.get("reason_bucket") or "")
        tokens = _reason_tokens(signal) or _reason_tokens(bridge)
        if (
            primary == "NO_SETUP_ELIGIBLE"
            or "NO_SETUP" in bucket
            or bucket == "SCORE_NEAR_MIN_BUT_BREAKOUT_MISSING"
            or _has(tokens, ("no_setup", "breakout_not_confirmed"))
        ):
            candidates.append(candidate)

    candidates.sort(
        key=lambda row: (
            1 if row.get("reason_bucket") == "STRUCTURE_CONFIRMED_BUT_SETUP_NOT_ELIGIBLE" else 0,
            1 if bool(row.get("current_feed_is_clean", False)) else 0,
            float(row.get("score") or 0.0),
        ),
        reverse=True,
    )
    limited = candidates[:MAX_CANDIDATES]
    if not limited:
        empty = default_no_setup_eligible_decomposition_state("No NO_SETUP_ELIGIBLE candidates found in this cycle.")
        empty["generated_at"] = _utc_now_iso()
        empty["status"] = "NO_NO_SETUP_SAMPLE"
        empty["current_feed_is_clean"] = bool(feed_scope.get("current_feed_is_clean", False))
        empty["fallback_blocker_scope"] = _norm_text(feed_scope.get("fallback_blocker_scope") or "UNKNOWN")
        empty["recommendation"] = "observe_more"
        return empty

    top = limited[0]
    bucket_counts = Counter(str(row.get("reason_bucket") or "") for row in limited)
    structure_confirmed_count = sum(
        1
        for row in limited
        if str(row.get("reason_bucket") or "") in {
            "STRUCTURE_CONFIRMED_BUT_SETUP_NOT_ELIGIBLE",
            "BOS_CONFIRMED_BUT_REAL_SETUP_MISSING",
            "PIVOT_TRIGGERED_BUT_REAL_SETUP_MISSING",
            "MULTITF_STRONG_BUT_REAL_RULE_MISSING",
        }
    )
    near_count = sum(
        1
        for row in limited
        if row.get("score_gap") is not None and float(row.get("score_gap") or 0.0) <= NEAR_SCORE_GAP
    )
    recommendation = str(top.get("suggested_future_study") or "observe_more")
    if recommendation not in SAFE_RECOMMENDATIONS:
        recommendation = "observe_more"
    return {
        "enabled": True,
        "mode": MODE,
        "safety_mode": SAFETY_MODE,
        "status": "READY",
        "generated_at": _utc_now_iso(),
        "target_setup": TARGET_SETUP,
        "total_candidates_checked": int(len(symbols)),
        "no_setup_eligible_count": int(len(limited)),
        "top_symbol": str(top.get("symbol") or ""),
        "top_setup": str(top.get("setup") or TARGET_SETUP),
        "top_reason_bucket": str(top.get("reason_bucket") or "UNKNOWN_NO_SETUP_REASON"),
        "top_real_blocker": str(top.get("primary_real_blocker") or ""),
        "top_secondary_blocker": str(top.get("secondary_real_blocker") or ""),
        "top_score": top.get("score"),
        "top_min_score": top.get("min_score"),
        "top_score_gap": top.get("score_gap"),
        "current_feed_is_clean": bool(feed_scope.get("current_feed_is_clean", False)),
        "fallback_blocker_scope": _norm_text(feed_scope.get("fallback_blocker_scope") or "UNKNOWN"),
        "structure_confirmed_count": int(structure_confirmed_count),
        "structure_confirmed_but_no_setup_count": int(
            bucket_counts.get("STRUCTURE_CONFIRMED_BUT_SETUP_NOT_ELIGIBLE", 0)
        ),
        "near_approved_no_setup_count": int(near_count),
        "recommendation": recommendation,
        "should_keep_blocked": True,
        "notes": (
            "NO_SETUP_ELIGIBLE decomposition is diagnostic only; real strategy remains authoritative "
            "and no threshold, score, broker, order, PnL, history, or position was changed."
        ),
        "candidates": limited,
        "fib_alignment_status": _norm_text((fib_alignment_audit or {}).get("fib_alignment_status")),
        "shadow_only": True,
    }
