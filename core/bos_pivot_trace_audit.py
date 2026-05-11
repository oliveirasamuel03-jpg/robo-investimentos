from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any

import pandas as pd


MODE = "SHADOW_ONLY"
DEFAULT_TIMEFRAMES = ("4h", "1h")
MAX_RECENT_CANDIDATES = 12
MIN_BARS_BY_TIMEFRAME = {"4h": 30, "1h": 40}
SWING_WINDOW_BY_TIMEFRAME = {"4h": 3, "1h": 4}
REQUIRED_CLOSE_BUFFER_PCT = 0.0015
WEAK_CLOSE_BUFFER_PCT = 0.0001


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _round(value: Any, digits: int = 6) -> float | None:
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


def _empty_default(reason: str) -> dict[str, Any]:
    return {
        "enabled": True,
        "mode": MODE,
        "generated_at": "",
        "provider_effective": "",
        "feed_status": "UNKNOWN",
        "uses_real_intraday_data": False,
        "symbols_analyzed": 0,
        "timeframes_analyzed": list(DEFAULT_TIMEFRAMES),
        "top_symbol": "",
        "top_timeframe": "",
        "top_pivot_state": "INSUFFICIENT_DATA",
        "top_bos_state": "INSUFFICIENT_DATA",
        "top_relationship": "INSUFFICIENT_DATA",
        "top_recommendation": "insufficient_data",
        "top_primary_missing_piece": reason,
        "h4_bos_missing_count": 0,
        "h1_bos_only_count": 0,
        "wick_only_bos_count": 0,
        "weak_close_bos_count": 0,
        "confirmed_bos_count": 0,
        "retest_pending_count": 0,
        "pivot_forming_count": 0,
        "pivot_confirmed_count": 0,
        "pivot_triggered_count": 0,
        "insufficient_data_count": 0,
        "should_keep_blocked_count": 0,
        "recent_candidates": [],
        "reason": reason,
        "shadow_only": True,
    }


def default_bos_pivot_trace_audit_state(reason: str = "No BOS/Pivot trace audit data yet.") -> dict[str, Any]:
    return _empty_default(reason)


def _clean_frame(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    required = ["open", "high", "low", "close"]
    if any(column not in frame.columns for column in required):
        return pd.DataFrame()
    data = frame.copy()
    for column in ("open", "high", "low", "close"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["open", "high", "low", "close"])
    if data.empty:
        return pd.DataFrame()
    if "datetime" in data.columns:
        dt_index = pd.to_datetime(data["datetime"], errors="coerce", utc=True)
        data = data.loc[dt_index.notna()].copy()
        data.index = dt_index[dt_index.notna()]
    elif isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index, errors="coerce", utc=True)
        data = data.loc[data.index.notna()].copy()
    else:
        data.index = pd.date_range(end=pd.Timestamp.utcnow(), periods=len(data), freq="1H")
    return data.sort_index()


def _with_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    data = frame.copy()
    close = pd.to_numeric(data["close"], errors="coerce")
    data["ma20"] = close.rolling(20, min_periods=5).mean()
    data["ma50"] = close.rolling(50, min_periods=10).mean()
    data["momentum_3"] = close.pct_change(3)
    return data


def _trend_direction(frame: pd.DataFrame) -> str:
    if frame.empty or len(frame) < 12:
        return "INCONCLUSIVE"
    close = _as_float(frame["close"].iloc[-1], None)
    ma20 = _as_float(frame["ma20"].iloc[-1], None)
    ma50 = _as_float(frame["ma50"].iloc[-1], None)
    ma20_prev = _as_float(frame["ma20"].iloc[-min(len(frame), 6)], None)
    if close is None or ma20 is None or ma50 is None or ma20_prev is None:
        return "INCONCLUSIVE"
    if close >= ma20 >= ma50 and ma20 >= ma20_prev:
        return "UP"
    if close <= ma20 <= ma50 and ma20 <= ma20_prev:
        return "DOWN"
    if abs(ma20 - ma50) / max(abs(close), 1e-9) <= 0.006:
        return "SIDEWAYS"
    return "INCONCLUSIVE"


def _swing_points(frame: pd.DataFrame, window: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    highs: list[dict[str, Any]] = []
    lows: list[dict[str, Any]] = []
    if len(frame) < (window * 2) + 3:
        return highs, lows
    for idx in range(window, len(frame) - window):
        high = float(frame["high"].iloc[idx])
        low = float(frame["low"].iloc[idx])
        high_window = frame["high"].iloc[idx - window : idx + window + 1]
        low_window = frame["low"].iloc[idx - window : idx + window + 1]
        high_neighbors = high_window.drop(high_window.index[window])
        low_neighbors = low_window.drop(low_window.index[window])
        if high >= float(high_window.max()) and high > float(high_neighbors.max()):
            highs.append({"idx": idx, "price": high, "ts": frame.index[idx].isoformat()})
        if low <= float(low_window.min()) and low < float(low_neighbors.min()):
            lows.append({"idx": idx, "price": low, "ts": frame.index[idx].isoformat()})
    return highs, lows


def _swing_context(frame: pd.DataFrame, timeframe: str) -> dict[str, Any]:
    window = int(SWING_WINDOW_BY_TIMEFRAME.get(str(timeframe).lower(), 3))
    highs, lows = _swing_points(frame, window)
    latest_high = highs[-1] if highs else {}
    previous_high = highs[-2] if len(highs) >= 2 else {}
    latest_low = lows[-1] if lows else {}
    previous_low = lows[-2] if len(lows) >= 2 else {}
    return {
        "highs": highs,
        "lows": lows,
        "latest_high": latest_high,
        "previous_high": previous_high,
        "latest_low": latest_low,
        "previous_low": previous_low,
    }


def _cross_payload(
    *,
    direction: str,
    level: float | None,
    close: float,
    high: float,
    low: float,
) -> dict[str, Any]:
    if level is None or level <= 0:
        return {
            "close_distance_pct": None,
            "wick_distance_pct": None,
            "close_crossed": False,
            "wick_crossed": False,
        }
    if direction == "DOWN":
        close_distance = (float(level) - float(close)) / max(abs(float(level)), 1e-9)
        wick_distance = (float(level) - float(low)) / max(abs(float(level)), 1e-9)
    else:
        close_distance = (float(close) - float(level)) / max(abs(float(level)), 1e-9)
        wick_distance = (float(high) - float(level)) / max(abs(float(level)), 1e-9)
    return {
        "close_distance_pct": close_distance,
        "wick_distance_pct": wick_distance,
        "close_crossed": close_distance > 0,
        "wick_crossed": wick_distance > 0,
    }


def _recent_failed_break(frame: pd.DataFrame, *, direction: str, level: float | None) -> bool:
    if level is None or level <= 0 or len(frame) < 4:
        return False
    recent = frame.tail(4)
    previous = recent.iloc[:-1]
    latest_close = float(recent["close"].iloc[-1])
    if direction == "DOWN":
        prior_break = (previous["close"] < float(level) * (1.0 - REQUIRED_CLOSE_BUFFER_PCT)).any()
        returned = latest_close >= float(level)
    else:
        prior_break = (previous["close"] > float(level) * (1.0 + REQUIRED_CLOSE_BUFFER_PCT)).any()
        returned = latest_close <= float(level)
    return bool(prior_break and returned)


def _retest_state(frame: pd.DataFrame, *, direction: str, level: float | None) -> tuple[bool, bool]:
    if level is None or level <= 0 or len(frame) < 5:
        return False, False
    recent = frame.tail(5)
    tolerance = max(abs(float(level)) * 0.004, 1e-9)
    latest_close = float(recent["close"].iloc[-1])
    if direction == "DOWN":
        touched = bool((recent["high"] >= float(level) - tolerance).any())
        held = latest_close < float(level)
    else:
        touched = bool((recent["low"] <= float(level) + tolerance).any())
        held = latest_close > float(level)
    return touched, bool(touched and held)


def _build_bos_trace(frame: pd.DataFrame, context: dict[str, Any], trend: str) -> dict[str, Any]:
    latest = frame.iloc[-1]
    close = float(latest["close"])
    high = float(latest["high"])
    low = float(latest["low"])
    direction = trend if trend in {"UP", "DOWN"} else "NONE"
    if direction == "DOWN":
        lows = [float(item["price"]) for item in list(context.get("lows", []) or [])[-4:] if item.get("price") is not None]
        level = min(lows) if lows else _as_float(context.get("latest_low", {}).get("price"), None)
    elif direction == "UP":
        highs = [float(item["price"]) for item in list(context.get("highs", []) or [])[-4:] if item.get("price") is not None]
        level = max(highs) if highs else _as_float(context.get("latest_high", {}).get("price"), None)
    else:
        level = None

    if level is None:
        return {
            "bos_state": "NO_BOS",
            "bos_direction": "NONE",
            "bos_level": None,
            "bos_close_price": _round(close),
            "bos_close_distance_pct": None,
            "bos_wick_distance_pct": None,
            "required_close_buffer_pct": REQUIRED_CLOSE_BUFFER_PCT,
            "close_above_or_below_level": False,
            "wick_crossed_level": False,
            "retest_detected": False,
            "retest_hold": False,
            "false_breakout_risk": "UNKNOWN",
            "bos_confidence_score": 0.0,
            "why_bos_not_confirmed": "no_structural_level",
        }

    crossed = _cross_payload(direction=direction, level=level, close=close, high=high, low=low)
    close_distance = _as_float(crossed["close_distance_pct"], 0.0) or 0.0
    wick_distance = _as_float(crossed["wick_distance_pct"], 0.0) or 0.0
    retest_detected, retest_hold = _retest_state(frame, direction=direction, level=level)
    failed_break = _recent_failed_break(frame, direction=direction, level=level)

    if failed_break:
        state = "BOS_FAILED"
        confidence = 0.05
        false_risk = "HIGH"
        reason = "breakout_returned_inside_structure"
    elif crossed["close_crossed"] and close_distance >= REQUIRED_CLOSE_BUFFER_PCT:
        state = "BOS_RETEST_CONFIRMED" if retest_hold else "BOS_BY_CLOSE_CONFIRMED"
        confidence = 0.78 if retest_hold else 0.68
        false_risk = "LOW" if close_distance >= REQUIRED_CLOSE_BUFFER_PCT * 2 else "MEDIUM"
        reason = ""
    elif crossed["close_crossed"] and close_distance >= WEAK_CLOSE_BUFFER_PCT:
        state = "BOS_BY_CLOSE_WEAK"
        confidence = 0.42
        false_risk = "MEDIUM"
        reason = "close_buffer_too_small"
    elif crossed["wick_crossed"]:
        state = "BOS_BY_WICK_ONLY"
        confidence = 0.18
        false_risk = "HIGH"
        reason = "wick_crossed_without_close_confirmation"
    else:
        state = "NO_BOS"
        confidence = 0.0
        false_risk = "UNKNOWN"
        reason = "close_did_not_cross_structure_level"

    if state == "BOS_BY_CLOSE_CONFIRMED" and retest_detected and not retest_hold:
        state = "BOS_RETEST_PENDING"
        confidence = min(confidence, 0.58)
        reason = "retest_pending_or_not_held"

    return {
        "bos_state": state,
        "bos_direction": direction,
        "bos_level": _round(level),
        "bos_close_price": _round(close),
        "bos_close_distance_pct": _round(close_distance, 6),
        "bos_wick_distance_pct": _round(wick_distance, 6),
        "required_close_buffer_pct": REQUIRED_CLOSE_BUFFER_PCT,
        "close_above_or_below_level": bool(crossed["close_crossed"]),
        "wick_crossed_level": bool(crossed["wick_crossed"]),
        "retest_detected": bool(retest_detected),
        "retest_hold": bool(retest_hold),
        "false_breakout_risk": false_risk,
        "bos_confidence_score": round(float(confidence), 4),
        "why_bos_not_confirmed": reason,
    }


def _build_pivot_trace(frame: pd.DataFrame, context: dict[str, Any], trend: str) -> dict[str, Any]:
    latest = frame.iloc[-1]
    close = float(latest["close"])
    highs = list(context.get("highs", []) or [])
    lows = list(context.get("lows", []) or [])
    latest_high = dict(context.get("latest_high", {}) or {})
    previous_high = dict(context.get("previous_high", {}) or {})
    latest_low = dict(context.get("latest_low", {}) or {})
    previous_low = dict(context.get("previous_low", {}) or {})

    if len(highs) < 2 or len(lows) < 2:
        return {
            "pivot_state": "NO_PIVOT",
            "pivot_direction": "NONE",
            "pivot_confirmation_close": None,
            "pivot_activation_level": None,
            "pivot_invalidated_reason": "missing_two_confirmed_swings",
            "candles_since_pivot": None,
            "pivot_confidence_score": 0.0,
            "why_pivot_not_confirmed": "insufficient_swing_structure",
        }

    latest_high_price = float(latest_high["price"])
    previous_high_price = float(previous_high["price"])
    latest_low_price = float(latest_low["price"])
    previous_low_price = float(previous_low["price"])
    relevant_high = max(float(item["price"]) for item in highs[-4:])
    relevant_low = min(float(item["price"]) for item in lows[-4:])
    higher_low = latest_low_price > previous_low_price * (1.0 + WEAK_CLOSE_BUFFER_PCT)
    lower_high = latest_high_price < previous_high_price * (1.0 - WEAK_CLOSE_BUFFER_PCT)
    candles_since_pivot = int(len(frame) - 1 - int(latest_low.get("idx", len(frame) - 1)))

    if trend == "DOWN":
        activation = relevant_low
        close_distance = (activation - close) / max(abs(activation), 1e-9)
        invalidated = close > latest_high_price * (1.0 + REQUIRED_CLOSE_BUFFER_PCT)
        if invalidated:
            state = "PIVOT_INVALIDATED"
            reason = "close_above_reference_high"
            confidence = 0.05
        elif lower_high and close_distance >= REQUIRED_CLOSE_BUFFER_PCT:
            state = "PIVOT_TRIGGERED"
            reason = ""
            confidence = 0.80
        elif lower_high and close_distance > 0:
            state = "PIVOT_CONFIRMED"
            reason = "close_confirmed_but_buffer_is_weak"
            confidence = 0.62
        elif lower_high:
            state = "PIVOT_FORMING"
            reason = "lower_high_present_but_activation_missing"
            confidence = 0.38
        else:
            state = "NO_PIVOT"
            reason = "lower_high_not_formed"
            confidence = 0.0
        direction = "DOWN" if state != "NO_PIVOT" else "NONE"
    else:
        activation = relevant_high
        close_distance = (close - activation) / max(abs(activation), 1e-9)
        invalidated = close < latest_low_price * (1.0 - REQUIRED_CLOSE_BUFFER_PCT)
        if invalidated:
            state = "PIVOT_INVALIDATED"
            reason = "close_below_reference_low"
            confidence = 0.05
        elif higher_low and close_distance >= REQUIRED_CLOSE_BUFFER_PCT:
            state = "PIVOT_TRIGGERED"
            reason = ""
            confidence = 0.80
        elif higher_low and close_distance > 0:
            state = "PIVOT_CONFIRMED"
            reason = "close_confirmed_but_buffer_is_weak"
            confidence = 0.62
        elif higher_low:
            state = "PIVOT_FORMING"
            reason = "higher_low_present_but_activation_missing"
            confidence = 0.38
        else:
            state = "NO_PIVOT"
            reason = "higher_low_not_formed"
            confidence = 0.0
        direction = "UP" if state != "NO_PIVOT" else "NONE"

    return {
        "pivot_state": state,
        "pivot_direction": direction,
        "pivot_confirmation_close": _round(close),
        "pivot_activation_level": _round(activation),
        "pivot_invalidated_reason": reason if state == "PIVOT_INVALIDATED" else "",
        "candles_since_pivot": candles_since_pivot,
        "pivot_confidence_score": round(float(confidence), 4),
        "why_pivot_not_confirmed": reason,
    }


def _signal_for_symbol(signals: list[dict[str, Any]], symbol: str) -> dict[str, Any]:
    target = str(symbol or "").upper()
    for signal in reversed(signals):
        if str(signal.get("asset") or signal.get("symbol") or "").upper() == target:
            return dict(signal)
    return {}


def _reason_tokens(signal: dict[str, Any]) -> list[str]:
    reasons = signal.get("rejection_reasons", [])
    if isinstance(reasons, str):
        reasons = [reasons]
    return [str(reason or "").strip().lower() for reason in list(reasons or []) if str(reason or "").strip()]


def _has_any(tokens: list[str], patterns: tuple[str, ...]) -> bool:
    return any(any(pattern in token for pattern in patterns) for token in tokens)


def _setup_block_trace(symbol: str, signals: list[dict[str, Any]], bos: dict[str, Any], pivot: dict[str, Any], timeframe: str) -> dict[str, Any]:
    signal = _signal_for_symbol(signals, symbol)
    tokens = _reason_tokens(signal)
    score_blocker = _has_any(tokens, ("score_below", "abaixo", "score_adjusted"))
    rsi_blocker = _has_any(tokens, ("rsi",))
    momentum_blocker = _has_any(tokens, ("momentum",))
    secondary_blocker = _has_any(tokens, ("secondary", "secundaria", "confirmacao"))
    trend_blocker = _has_any(tokens, ("trend_not_confirmed", "tendencia", "trend"))
    no_setup = _has_any(tokens, ("no_setup", "setup"))

    bos_state = str(bos.get("bos_state") or "")
    pivot_state = str(pivot.get("pivot_state") or "")
    if timeframe == "4h" and bos_state not in {"BOS_BY_CLOSE_CONFIRMED", "BOS_RETEST_CONFIRMED"}:
        primary_missing = "h4_close_above_structure_missing"
    elif timeframe == "1h" and bos_state not in {"BOS_BY_CLOSE_CONFIRMED", "BOS_RETEST_CONFIRMED"}:
        primary_missing = "h1_close_above_structure_missing"
    elif pivot_state not in {"PIVOT_CONFIRMED", "PIVOT_TRIGGERED"}:
        primary_missing = "pivot_confirmation_missing"
    else:
        primary_missing = "real_strategy_still_authoritative"

    supports_setup = (
        timeframe in {"4h", "1h"}
        and bos_state in {"BOS_BY_CLOSE_CONFIRMED", "BOS_RETEST_CONFIRMED", "BOS_BY_CLOSE_WEAK"}
        and pivot_state in {"PIVOT_CONFIRMED", "PIVOT_TRIGGERED", "PIVOT_FORMING"}
    )
    return {
        "supports_trend_pullback_breakout": bool(supports_setup),
        "structure_blocker": primary_missing,
        "primary_missing_piece": primary_missing,
        "secondary_missing_piece": "secondary_confirmation_or_momentum" if secondary_blocker or momentum_blocker else "",
        "score_blocker_present": bool(score_blocker),
        "rsi_blocker_present": bool(rsi_blocker),
        "momentum_blocker_present": bool(momentum_blocker),
        "secondary_confirmation_blocker_present": bool(secondary_blocker),
        "trend_blocker_present": bool(trend_blocker),
        "no_setup_eligible_present": bool(no_setup),
        "would_structure_help_if_confirmed": bool(supports_setup and not trend_blocker and not no_setup),
        "should_keep_blocked": True,
    }


def _fib_context(symbol: str, market_structure_audit: dict[str, Any] | None, pivot: dict[str, Any], bos: dict[str, Any]) -> dict[str, Any]:
    candidates = list((market_structure_audit or {}).get("market_structure_best_candidates", []) or [])
    row = {}
    for candidate in candidates:
        if str(candidate.get("symbol") or candidate.get("asset") or "").upper() == str(symbol).upper():
            row = dict(candidate)
            break
    zone = str(row.get("current_fib_zone") or row.get("fib_zone") or "").upper()
    useful_zone = zone in {"SHALLOW_ZONE", "MEDIUM_ZONE", "DEEP_ZONE"}
    pivot_ok = str(pivot.get("pivot_state") or "") in {"PIVOT_CONFIRMED", "PIVOT_TRIGGERED"}
    bos_ok = str(bos.get("bos_state") or "") in {"BOS_BY_CLOSE_CONFIRMED", "BOS_RETEST_CONFIRMED"}
    return {
        "fib_zone_context": zone or "UNKNOWN",
        "fib_zone_supports_pivot": bool(useful_zone and pivot_ok),
        "fib_zone_supports_bos": bool(useful_zone and bos_ok),
        "fib_conflict_reason": "" if useful_zone else "fib_zone_not_useful_or_missing",
        "fib_projection_ready_candidate": False,
    }


def _candidate_for_timeframe(
    *,
    symbol: str,
    timeframe: str,
    frame: pd.DataFrame,
    signals: list[dict[str, Any]],
    market_structure_audit: dict[str, Any] | None,
) -> dict[str, Any]:
    data = _with_indicators(_clean_frame(frame))
    min_bars = int(MIN_BARS_BY_TIMEFRAME.get(str(timeframe).lower(), 30))
    if len(data) < min_bars:
        return {
            "symbol": str(symbol).upper(),
            "timeframe": str(timeframe).lower(),
            "trend_direction": "INCONCLUSIVE",
            "pivot_state": "INSUFFICIENT_DATA",
            "pivot_direction": "NONE",
            "pivot_confidence_score": 0.0,
            "bos_state": "INSUFFICIENT_DATA",
            "bos_direction": "NONE",
            "bos_confidence_score": 0.0,
            "relationship_to_higher_tf": "INSUFFICIENT_DATA",
            "swing_high_reference": None,
            "swing_low_reference": None,
            "previous_swing_high": None,
            "previous_swing_low": None,
            "last_close": None,
            "bos_level": None,
            "close_distance_to_bos_pct": None,
            "wick_crossed_level": False,
            "close_confirmed_level": False,
            "retest_detected": False,
            "false_breakout_risk": "UNKNOWN",
            "primary_missing_piece": "insufficient_intraday_candles",
            "why_pivot_not_confirmed": "insufficient_intraday_candles",
            "why_bos_not_confirmed": "insufficient_intraday_candles",
            "supports_trend_pullback_breakout": False,
            "should_keep_blocked": True,
            "recommendation": "insufficient_data",
            "shadow_only": True,
        }

    trend = _trend_direction(data)
    context = _swing_context(data, str(timeframe).lower())
    pivot = _build_pivot_trace(data, context, trend if trend in {"UP", "DOWN"} else "UP")
    bos = _build_bos_trace(data, context, trend)
    setup = _setup_block_trace(str(symbol), signals, bos, pivot, str(timeframe).lower())
    fib = _fib_context(str(symbol), market_structure_audit, pivot, bos)
    latest_high = dict(context.get("latest_high", {}) or {})
    previous_high = dict(context.get("previous_high", {}) or {})
    latest_low = dict(context.get("latest_low", {}) or {})
    previous_low = dict(context.get("previous_low", {}) or {})
    recommendation = "observe_more"
    if bos["bos_state"] == "BOS_BY_WICK_ONLY":
        recommendation = "wait_close_confirmation"
    elif bos["bos_state"] == "BOS_BY_CLOSE_WEAK":
        recommendation = "wait_stronger_close_or_retest"
    elif bos["bos_state"] in {"BOS_BY_CLOSE_CONFIRMED", "BOS_RETEST_CONFIRMED"}:
        recommendation = "structure_confirmed_keep_shadow_only"
    elif pivot["pivot_state"] == "PIVOT_FORMING":
        recommendation = "watch_pivot_activation"
    elif trend == "INCONCLUSIVE":
        recommendation = "insufficient_structure_direction"

    return {
        "symbol": str(symbol).upper(),
        "timeframe": str(timeframe).lower(),
        "trend_direction": trend,
        "pivot_state": pivot["pivot_state"],
        "pivot_direction": pivot["pivot_direction"],
        "pivot_confidence_score": pivot["pivot_confidence_score"],
        "bos_state": bos["bos_state"],
        "bos_direction": bos["bos_direction"],
        "bos_confidence_score": bos["bos_confidence_score"],
        "relationship_to_higher_tf": "INCONCLUSIVE",
        "swing_high_reference": _round(latest_high.get("price")),
        "swing_low_reference": _round(latest_low.get("price")),
        "previous_swing_high": _round(previous_high.get("price")),
        "previous_swing_low": _round(previous_low.get("price")),
        "last_close": _round(data["close"].iloc[-1]),
        "pivot_confirmation_close": pivot["pivot_confirmation_close"],
        "pivot_activation_level": pivot["pivot_activation_level"],
        "pivot_invalidated_reason": pivot["pivot_invalidated_reason"],
        "candles_since_pivot": pivot["candles_since_pivot"],
        "why_pivot_not_confirmed": pivot["why_pivot_not_confirmed"],
        "bos_level": bos["bos_level"],
        "bos_close_price": bos["bos_close_price"],
        "bos_close_distance_pct": bos["bos_close_distance_pct"],
        "close_distance_to_bos_pct": bos["bos_close_distance_pct"],
        "bos_wick_distance_pct": bos["bos_wick_distance_pct"],
        "required_close_buffer_pct": bos["required_close_buffer_pct"],
        "close_above_or_below_level": bos["close_above_or_below_level"],
        "wick_crossed_level": bos["wick_crossed_level"],
        "close_confirmed_level": bos["bos_state"] in {"BOS_BY_CLOSE_CONFIRMED", "BOS_RETEST_CONFIRMED"},
        "retest_detected": bos["retest_detected"],
        "retest_hold": bos["retest_hold"],
        "false_breakout_risk": bos["false_breakout_risk"],
        "why_bos_not_confirmed": bos["why_bos_not_confirmed"],
        "recommendation": recommendation,
        "shadow_rank_score": round(
            float(pivot["pivot_confidence_score"]) * 0.45 + float(bos["bos_confidence_score"]) * 0.55,
            4,
        ),
        "shadow_only": True,
        **setup,
        **fib,
    }


def _confirmed_bos(row: dict[str, Any]) -> bool:
    return str(row.get("bos_state") or "") in {"BOS_BY_CLOSE_CONFIRMED", "BOS_RETEST_CONFIRMED"}


def _confirmed_pivot(row: dict[str, Any]) -> bool:
    return str(row.get("pivot_state") or "") in {"PIVOT_CONFIRMED", "PIVOT_TRIGGERED"}


def _relationship_for_symbol(h4: dict[str, Any], h1: dict[str, Any]) -> dict[str, Any]:
    if not h4 or not h1:
        return {"relationship": "INSUFFICIENT_DATA", "score": 0.0, "reason": "missing_timeframe"}
    if "INSUFFICIENT_DATA" in {str(h4.get("bos_state")), str(h1.get("bos_state"))}:
        return {"relationship": "INSUFFICIENT_DATA", "score": 0.0, "reason": "insufficient_timeframe_data"}
    h4_dir = str(h4.get("trend_direction") or "INCONCLUSIVE")
    h1_dir = str(h1.get("trend_direction") or "INCONCLUSIVE")
    h4_confirmed = _confirmed_bos(h4) or _confirmed_pivot(h4)
    h1_confirmed = _confirmed_bos(h1) or _confirmed_pivot(h1)
    if h4_dir in {"UP", "DOWN"} and h1_dir in {"UP", "DOWN"} and h4_dir != h1_dir:
        return {"relationship": "H1_CONFLICTS_H4", "score": 0.10, "reason": "timeframe_direction_conflict"}
    if h4_confirmed and h1_confirmed:
        return {"relationship": "BOTH_CONFIRMED", "score": 0.82, "reason": "h4_and_h1_confirm_structure"}
    if h1_confirmed and not h4_confirmed and h4_dir == h1_dir and h4_dir in {"UP", "DOWN"}:
        return {"relationship": "H1_LEADS_H4", "score": 0.55, "reason": "h1_structure_leads_without_h4_close_confirmation"}
    if h4_confirmed and not h1_confirmed and h4_dir == h1_dir and h4_dir in {"UP", "DOWN"}:
        return {"relationship": "H4_CONFIRMS_H1", "score": 0.52, "reason": "h4_structure_present_h1_not_confirming"}
    if h1_confirmed and h4_dir not in {"UP", "DOWN"}:
        return {"relationship": "H4_STRUCTURE_MISSING", "score": 0.32, "reason": "h1_signal_without_h4_structure"}
    if h1_confirmed and h4_dir != h1_dir:
        return {"relationship": "H1_NOISE_ONLY", "score": 0.20, "reason": "h1_confirmation_not_supported_by_h4"}
    if not h4_confirmed and not h1_confirmed:
        return {"relationship": "BOTH_MISSING", "score": 0.0, "reason": "no_confirmed_bos_or_pivot"}
    return {"relationship": "INCONCLUSIVE", "score": 0.0, "reason": "relationship_inconclusive"}


def _enrich_relationships(candidates: list[dict[str, Any]]) -> None:
    by_symbol: dict[str, dict[str, dict[str, Any]]] = {}
    for candidate in candidates:
        by_symbol.setdefault(str(candidate.get("symbol") or ""), {})[str(candidate.get("timeframe") or "")] = candidate
    for frames in by_symbol.values():
        relationship = _relationship_for_symbol(frames.get("4h", {}), frames.get("1h", {}))
        for candidate in frames.values():
            candidate["relationship_to_higher_tf"] = relationship["relationship"]
            candidate["timeframe_bos_pivot_relationship"] = relationship["relationship"]
            candidate["relationship_score"] = relationship["score"]
            candidate["relationship_reason"] = relationship["reason"]
            candidate["h1_supports_h4"] = relationship["relationship"] in {"H1_LEADS_H4", "BOTH_CONFIRMED"}
            candidate["h4_confirms_h1"] = relationship["relationship"] in {"H4_CONFIRMS_H1", "BOTH_CONFIRMED"}
            candidate["shadow_rank_score"] = round(
                float(candidate.get("shadow_rank_score", 0.0) or 0.0) + float(relationship["score"]) * 0.15,
                4,
            )


def _top_recommendation(top: dict[str, Any]) -> str:
    if not top:
        return "insufficient_data"
    if str(top.get("bos_state") or "") == "BOS_BY_WICK_ONLY":
        return "wait_close_confirmation"
    if str(top.get("bos_state") or "") == "BOS_BY_CLOSE_WEAK":
        return "wait_stronger_close_or_retest"
    if str(top.get("relationship_to_higher_tf") or "") == "H1_LEADS_H4":
        return "observe_h4_confirmation"
    if str(top.get("relationship_to_higher_tf") or "") == "BOTH_CONFIRMED":
        return "structure_confirmed_keep_shadow_only"
    return str(top.get("recommendation") or "observe_more")


def build_bos_pivot_trace_audit(
    *,
    intraday_market_data: dict[str, dict[str, pd.DataFrame]] | None = None,
    market_data_status: dict[str, Any] | None = None,
    signals: list[dict[str, Any]] | None = None,
    market_structure_audit: dict[str, Any] | None = None,
    multi_timeframe_swing_audit: dict[str, Any] | None = None,
    enabled: bool = True,
    timeframes: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    if not enabled:
        disabled = _empty_default("BOS/Pivot trace audit disabled.")
        disabled["enabled"] = False
        return disabled

    status = dict(market_data_status or {})
    feed_status = _normalize_feed_status(status.get("feed_status") or status.get("status"))
    provider = str(status.get("provider_effective") or status.get("provider") or "")
    requested_timeframes = [
        str(item).strip().lower()
        for item in list(timeframes or DEFAULT_TIMEFRAMES)
        if str(item).strip().lower() in {"4h", "1h"}
    ] or list(DEFAULT_TIMEFRAMES)
    frames_by_symbol = {
        str(symbol).upper(): dict(frames or {})
        for symbol, frames in dict(intraday_market_data or {}).items()
        if isinstance(frames, dict)
    }
    if not frames_by_symbol:
        empty = _empty_default("No real 4H/1H intraday frames available for BOS/Pivot trace audit.")
        empty.update(
            {
                "generated_at": _utc_now_iso(),
                "provider_effective": provider,
                "feed_status": feed_status,
                "timeframes_analyzed": requested_timeframes,
            }
        )
        return empty

    candidates: list[dict[str, Any]] = []
    for symbol, frames in frames_by_symbol.items():
        for timeframe in requested_timeframes:
            candidates.append(
                _candidate_for_timeframe(
                    symbol=symbol,
                    timeframe=timeframe,
                    frame=frames.get(timeframe),
                    signals=list(signals or []),
                    market_structure_audit=market_structure_audit,
                )
            )

    _enrich_relationships(candidates)
    candidates.sort(key=lambda row: float(row.get("shadow_rank_score", 0.0) or 0.0), reverse=True)
    top = candidates[0] if candidates else {}
    bos_counts = Counter(str(row.get("bos_state") or "") for row in candidates)
    pivot_counts = Counter(str(row.get("pivot_state") or "") for row in candidates)
    h4_missing = sum(
        1
        for row in candidates
        if row.get("timeframe") == "4h"
        and str(row.get("bos_state") or "") not in {"BOS_BY_CLOSE_CONFIRMED", "BOS_RETEST_CONFIRMED"}
    )
    h1_bos_only = 0
    by_symbol: dict[str, dict[str, dict[str, Any]]] = {}
    for row in candidates:
        by_symbol.setdefault(str(row.get("symbol") or ""), {})[str(row.get("timeframe") or "")] = row
    for frames in by_symbol.values():
        if _confirmed_bos(frames.get("1h", {})) and not _confirmed_bos(frames.get("4h", {})):
            h1_bos_only += 1

    recent = candidates[:MAX_RECENT_CANDIDATES]
    top_h4 = by_symbol.get(str(top.get("symbol") or ""), {}).get("4h", {})
    top_h1 = by_symbol.get(str(top.get("symbol") or ""), {}).get("1h", {})
    return {
        "enabled": True,
        "mode": MODE,
        "generated_at": _utc_now_iso(),
        "provider_effective": provider,
        "feed_status": feed_status,
        "uses_real_intraday_data": True,
        "symbols_analyzed": int(len(by_symbol)),
        "timeframes_analyzed": requested_timeframes,
        "top_symbol": str(top.get("symbol") or ""),
        "top_timeframe": str(top.get("timeframe") or ""),
        "top_pivot_state": str(top.get("pivot_state") or "INSUFFICIENT_DATA"),
        "top_bos_state": str(top.get("bos_state") or "INSUFFICIENT_DATA"),
        "top_h4_bos_state": str(top_h4.get("bos_state") or "INSUFFICIENT_DATA"),
        "top_h1_bos_state": str(top_h1.get("bos_state") or "INSUFFICIENT_DATA"),
        "top_relationship": str(top.get("relationship_to_higher_tf") or "INSUFFICIENT_DATA"),
        "top_recommendation": _top_recommendation(top),
        "top_primary_missing_piece": str(top.get("primary_missing_piece") or ""),
        "dominant_missing_piece": str(top.get("primary_missing_piece") or ""),
        "h4_bos_missing_count": int(h4_missing),
        "h1_bos_only_count": int(h1_bos_only),
        "wick_only_bos_count": int(bos_counts.get("BOS_BY_WICK_ONLY", 0)),
        "weak_close_bos_count": int(bos_counts.get("BOS_BY_CLOSE_WEAK", 0)),
        "confirmed_bos_count": int(bos_counts.get("BOS_BY_CLOSE_CONFIRMED", 0) + bos_counts.get("BOS_RETEST_CONFIRMED", 0)),
        "retest_pending_count": int(bos_counts.get("BOS_RETEST_PENDING", 0)),
        "pivot_forming_count": int(pivot_counts.get("PIVOT_FORMING", 0)),
        "pivot_confirmed_count": int(pivot_counts.get("PIVOT_CONFIRMED", 0)),
        "pivot_triggered_count": int(pivot_counts.get("PIVOT_TRIGGERED", 0)),
        "insufficient_data_count": int(
            bos_counts.get("INSUFFICIENT_DATA", 0) + pivot_counts.get("INSUFFICIENT_DATA", 0)
        ),
        "should_keep_blocked_count": int(sum(1 for row in candidates if bool(row.get("should_keep_blocked", True)))),
        "recent_candidates": recent,
        "multi_timeframe_context": {
            "top_symbol": str((multi_timeframe_swing_audit or {}).get("top_symbol") or ""),
            "top_alignment_status": str((multi_timeframe_swing_audit or {}).get("top_alignment_status") or ""),
            "top_missing_confirmation": str((multi_timeframe_swing_audit or {}).get("top_missing_confirmation") or ""),
        },
        "reason": "BOS/Pivot trace audit uses real cached 4H/1H candles for diagnostics only.",
        "shadow_only": True,
    }
