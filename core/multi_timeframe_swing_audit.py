from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any

import pandas as pd


MODE = "SHADOW_ONLY"
DEFAULT_TIMEFRAMES = ("1d", "4h", "1h")
DEFAULT_CACHE_TTL_SECONDS = 900
DEFAULT_MAX_SYMBOLS = 5
MAX_RECENT_CANDIDATES = 10
MIN_BARS_BY_TIMEFRAME = {"1d": 18, "4h": 30, "1h": 40}
TIMEFRAME_RULES = {"1d": "1D", "4h": "4H", "1h": "1H"}
KNOWN_PROVIDERS = {"twelvedata", "yahoo", "mixed"}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_float(value: Any, default: float | None = 0.0) -> float | None:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _round(value: Any, digits: int = 4) -> float | None:
    numeric = _as_float(value, None)
    return None if numeric is None else round(float(numeric), digits)


def _safe_bool(value: Any) -> bool:
    return bool(value) if value is not None else False


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


def _last_text(frame: pd.DataFrame, column: str, fallback: str = "") -> str:
    if frame.empty or column not in frame.columns:
        return fallback
    values = [str(item).strip() for item in frame[column].dropna().tolist() if str(item).strip()]
    return values[-1] if values else fallback


def _empty_default(reason: str) -> dict[str, Any]:
    return {
        "enabled": True,
        "mode": MODE,
        "generated_at": "",
        "provider_effective": "",
        "feed_status": "UNKNOWN",
        "timeframes_used": list(DEFAULT_TIMEFRAMES),
        "timeframe_source": "operational_cycle_resample",
        "timeframe_fallbacks": [],
        "symbols_analyzed": 0,
        "top_symbol": "",
        "top_alignment_score": None,
        "top_alignment_status": "INSUFFICIENT_DATA",
        "top_missing_confirmation": "",
        "top_recommendation": "insufficient_data",
        "dominant_conflict_reason": reason,
        "candidates_count": 0,
        "strong_alignment_count": 0,
        "partial_alignment_count": 0,
        "conflict_count": 0,
        "insufficient_data_count": 0,
        "setup_support_count": 0,
        "recent_candidates": [],
        "estimated_provider_calls": 0,
        "cache_ttl_seconds": DEFAULT_CACHE_TTL_SECONDS,
        "cache_status": "cycle_data_resample_only",
        "provider_guard": "not_evaluated",
        "shadow_only": True,
        "uses_real_intraday_data": False,
        "intraday_timeframes_available": [],
        "intraday_top_symbol": "",
        "intraday_missing_reason": "",
        "h4_data_quality": "missing",
        "h1_data_quality": "missing",
        "bos_pivot_trace_relationship": "INSUFFICIENT_DATA",
        "bos_pivot_top_pivot_state": "INSUFFICIENT_DATA",
        "bos_pivot_top_bos_state": "INSUFFICIENT_DATA",
        "bos_pivot_dominant_missing_piece": "",
        "reason": reason,
    }


def default_multi_timeframe_swing_audit_state(reason: str = "No multi-timeframe swing audit data yet.") -> dict[str, Any]:
    return _empty_default(reason)


def _clean_frame(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    required = ["high", "low", "close"]
    if any(column not in frame.columns for column in required):
        return pd.DataFrame()

    data = frame.copy()
    if "open" not in data.columns:
        data["open"] = data["close"]
    if "volume" not in data.columns:
        data["volume"] = 0.0

    for column in ("open", "high", "low", "close", "volume"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["open", "high", "low", "close"])
    if data.empty:
        return pd.DataFrame()

    if "datetime" in data.columns:
        dt_index = pd.to_datetime(data["datetime"], errors="coerce", utc=True)
        data = data.loc[dt_index.notna()].copy()
        data.index = dt_index[dt_index.notna()]
    elif not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.date_range(end=pd.Timestamp.utcnow(), periods=len(data), freq="1H")
    else:
        data.index = pd.to_datetime(data.index, errors="coerce", utc=True)
        data = data.loc[data.index.notna()].copy()

    if data.empty:
        return pd.DataFrame()
    return data.sort_index()


def _with_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    data = frame.copy()
    close = pd.to_numeric(data["close"], errors="coerce")
    data["ma20"] = close.rolling(20, min_periods=5).mean()
    data["ma50"] = close.rolling(50, min_periods=10).mean()
    data["momentum_3"] = close.pct_change(3)
    data["momentum_8"] = close.pct_change(8)
    candle_range = (pd.to_numeric(data["high"], errors="coerce") - pd.to_numeric(data["low"], errors="coerce")) / close.replace(0, pd.NA)
    data["range_pct"] = candle_range.rolling(14, min_periods=5).mean()
    return data


def _resample_frame(frame: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    data = _clean_frame(frame)
    if data.empty:
        return pd.DataFrame()

    normalized = str(timeframe).lower()
    if normalized == "1h":
        return _with_indicators(data.tail(240))

    rule = TIMEFRAME_RULES.get(normalized)
    if not rule:
        return pd.DataFrame()

    agg: dict[str, str] = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    if "data_source" in data.columns:
        agg["data_source"] = "last"
    if "provider_name" in data.columns:
        agg["provider_name"] = "last"
    resampled = data.resample(rule).agg(agg).dropna(subset=["open", "high", "low", "close"])
    return _with_indicators(resampled.tail(160))


def _base_interval_allows_intraday_resample(base_interval: str) -> bool:
    normalized = str(base_interval or "").strip().lower()
    if not normalized:
        return True
    return normalized in {"1h", "60m", "60min", "30m", "30min", "15m", "15min", "5m", "5min", "1m", "1min"}


def _resolve_timeframe_frames(payload: Any, timeframes: list[str], *, base_interval: str = "") -> tuple[dict[str, pd.DataFrame], list[str]]:
    fallbacks: list[str] = []
    frames: dict[str, pd.DataFrame] = {}
    if isinstance(payload, dict):
        for timeframe in timeframes:
            direct = payload.get(timeframe)
            if direct is None:
                direct = payload.get(timeframe.upper())
            frames[timeframe] = _with_indicators(_clean_frame(direct))
        return frames, fallbacks

    intraday_resample_allowed = _base_interval_allows_intraday_resample(base_interval)
    for timeframe in timeframes:
        if timeframe in {"1h", "4h"} and not intraday_resample_allowed:
            frames[timeframe] = pd.DataFrame()
            fallbacks.append(f"{timeframe}=blocked_base_interval_{str(base_interval or 'unknown').lower()}")
            continue
        frames[timeframe] = _resample_frame(payload, timeframe)
        if timeframe in {"1d", "4h"}:
            fallbacks.append(f"{timeframe}=resampled_from_operational_cycle")
        elif timeframe == "1h":
            fallbacks.append("1h=operational_cycle_frame")
    return frames, fallbacks


def _merge_intraday_frames(
    frames: dict[str, pd.DataFrame],
    intraday_frames: dict[str, pd.DataFrame] | None,
    timeframes: list[str],
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    merged = dict(frames)
    available: list[str] = []
    for timeframe in timeframes:
        if timeframe not in {"4h", "1h"}:
            continue
        frame = (intraday_frames or {}).get(timeframe)
        cleaned = _clean_frame(frame if isinstance(frame, pd.DataFrame) else None)
        if not cleaned.empty and _last_text(cleaned, "data_source", "").lower() == "market":
            merged[timeframe] = _with_indicators(cleaned)
            available.append(timeframe)
    return merged, available


def _prior_range(frame: pd.DataFrame, bars: int = 20) -> tuple[float | None, float | None]:
    if len(frame) < 3:
        return None, None
    prior = frame.iloc[max(0, len(frame) - bars - 1) : max(1, len(frame) - 1)]
    if prior.empty:
        return None, None
    return float(prior["high"].max()), float(prior["low"].min())


def _swing_points(frame: pd.DataFrame, window: int = 3) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    highs: list[dict[str, Any]] = []
    lows: list[dict[str, Any]] = []
    if len(frame) < (window * 2) + 1:
        return highs, lows
    for idx in range(window, len(frame) - window):
        high = float(frame["high"].iloc[idx])
        low = float(frame["low"].iloc[idx])
        high_window = frame["high"].iloc[idx - window : idx + window + 1]
        low_window = frame["low"].iloc[idx - window : idx + window + 1]
        if high >= float(high_window.max()) and high > float(high_window.drop(high_window.index[window]).max()):
            highs.append({"idx": idx, "price": high})
        if low <= float(low_window.min()) and low < float(low_window.drop(low_window.index[window]).min()):
            lows.append({"idx": idx, "price": low})
    return highs, lows


def _trend_direction(frame: pd.DataFrame) -> tuple[str, str]:
    if frame.empty or len(frame) < 12:
        return "INCONCLUSIVE", "insufficient"
    close = float(frame["close"].iloc[-1])
    ma20 = _as_float(frame["ma20"].iloc[-1], None)
    ma50 = _as_float(frame["ma50"].iloc[-1], None)
    ma20_prev = _as_float(frame["ma20"].iloc[-min(len(frame), 6)], None)
    if ma20 is None or ma50 is None or ma20_prev is None:
        return "INCONCLUSIVE", "insufficient"
    slope_up = float(ma20) >= float(ma20_prev)
    slope_down = float(ma20) <= float(ma20_prev)
    if close >= float(ma20) >= float(ma50) and slope_up:
        return "UP", "bullish"
    if close <= float(ma20) <= float(ma50) and slope_down:
        return "DOWN", "bearish"
    if abs(float(ma20) - float(ma50)) / max(abs(close), 1e-9) <= 0.006:
        return "SIDEWAYS", "mixed"
    return "INCONCLUSIVE", "mixed"


def _momentum_state(frame: pd.DataFrame) -> str:
    if frame.empty or len(frame) < 10:
        return "insufficient"
    mom3 = _as_float(frame["momentum_3"].iloc[-1], None)
    mom8 = _as_float(frame["momentum_8"].iloc[-1], None)
    if mom3 is None or mom8 is None:
        return "insufficient"
    if float(mom3) > 0.003 and float(mom8) > -0.005:
        return "rising"
    if float(mom3) < -0.003 and float(mom8) < 0.005:
        return "falling"
    if abs(float(mom3)) <= 0.003 and abs(float(mom8)) <= 0.006:
        return "neutral"
    return "weak"


def _volatility_state(frame: pd.DataFrame) -> str:
    if frame.empty or "range_pct" not in frame.columns:
        return "insufficient"
    range_pct = _as_float(frame["range_pct"].iloc[-1], None)
    if range_pct is None:
        return "insufficient"
    if float(range_pct) >= 0.055:
        return "high"
    if float(range_pct) <= 0.006:
        return "low"
    return "normal"


def build_timeframe_diagnostic(frame: pd.DataFrame | None, timeframe: str) -> dict[str, Any]:
    data = _with_indicators(_clean_frame(frame))
    min_bars = int(MIN_BARS_BY_TIMEFRAME.get(str(timeframe).lower(), 30))
    if len(data) < min_bars:
        return {
            "timeframe": str(timeframe).lower(),
            "trend_direction": "INCONCLUSIVE",
            "ma_alignment": "insufficient",
            "momentum_state": "insufficient",
            "volatility_state": "insufficient",
            "recent_swing_high": None,
            "recent_swing_low": None,
            "pivot_confirmed": False,
            "bos_confirmed": False,
            "choch_detected": False,
            "pullback_state": "insufficient",
            "structure_quality_score": 0.0,
            "why_not_confirmed": ["insufficient_data"],
            "bars": int(len(data)),
        }

    trend, ma_alignment = _trend_direction(data)
    momentum = _momentum_state(data)
    volatility = _volatility_state(data)
    highs, lows = _swing_points(data, window=3)
    recent_high = float(highs[-1]["price"]) if highs else None
    recent_low = float(lows[-1]["price"]) if lows else None
    close = float(data["close"].iloc[-1])
    open_ = float(data["open"].iloc[-1])
    previous = data.iloc[-2]
    prior_high, prior_low = _prior_range(data)
    buffer = 0.001

    pivot = False
    bos = False
    choch = False
    if trend == "DOWN":
        pivot = close < float(previous["low"]) and close < open_
        bos = prior_low is not None and close < float(prior_low) * (1.0 - buffer)
        choch = prior_high is not None and close > float(prior_high) * (1.0 + buffer)
    elif trend == "UP":
        pivot = close > float(previous["high"]) and close > open_
        bos = prior_high is not None and close > float(prior_high) * (1.0 + buffer)
        choch = prior_low is not None and close < float(prior_low) * (1.0 - buffer)

    ma20 = _as_float(data["ma20"].iloc[-1], None)
    ma50 = _as_float(data["ma50"].iloc[-1], None)
    pullback_state = "insufficient"
    if ma20 is not None and ma50 is not None:
        if trend == "UP":
            if close < float(ma50):
                pullback_state = "invalidated"
            elif close < float(ma20):
                pullback_state = "healthy"
            else:
                pullback_state = "shallow"
        elif trend == "DOWN":
            if close > float(ma50):
                pullback_state = "invalidated"
            elif close > float(ma20):
                pullback_state = "healthy"
            else:
                pullback_state = "shallow"
        else:
            pullback_state = "insufficient"

    missing: list[str] = []
    if trend not in {"UP", "DOWN"}:
        missing.append("trend_inconclusive")
    if momentum not in {"rising", "falling", "neutral"}:
        missing.append("momentum_weak")
    if not pivot:
        missing.append("pivot_missing")
    if not bos:
        missing.append("bos_missing")
    if pullback_state in {"invalidated", "insufficient"}:
        missing.append(f"pullback_{pullback_state}")
    if volatility in {"high", "insufficient"}:
        missing.append(f"volatility_{volatility}")

    score = 0.0
    score += 0.25 if trend in {"UP", "DOWN"} else 0.0
    score += 0.15 if ma_alignment in {"bullish", "bearish"} else 0.05 if ma_alignment == "mixed" else 0.0
    score += 0.15 if momentum in {"rising", "falling"} else 0.07 if momentum == "neutral" else 0.0
    score += 0.15 if pivot else 0.0
    score += 0.20 if bos else 0.0
    score += 0.10 if pullback_state in {"healthy", "shallow"} else 0.0
    if volatility == "high":
        score -= 0.08
    score = max(0.0, min(1.0, score))

    return {
        "timeframe": str(timeframe).lower(),
        "trend_direction": trend,
        "ma_alignment": ma_alignment,
        "momentum_state": momentum,
        "volatility_state": volatility,
        "recent_swing_high": _round(recent_high, 6),
        "recent_swing_low": _round(recent_low, 6),
        "pivot_confirmed": bool(pivot),
        "bos_confirmed": bool(bos),
        "choch_detected": bool(choch),
        "pullback_state": pullback_state,
        "structure_quality_score": round(float(score), 4),
        "why_not_confirmed": missing,
        "bars": int(len(data)),
    }


def _alignment_status(daily: dict[str, Any], h4: dict[str, Any], h1: dict[str, Any], *, feed_blocked: bool) -> str:
    if feed_blocked:
        return "INSUFFICIENT_DATA"
    directions = [daily.get("trend_direction"), h4.get("trend_direction"), h1.get("trend_direction")]
    if any(direction == "INCONCLUSIVE" for direction in directions):
        return "INSUFFICIENT_DATA"
    if daily.get("trend_direction") in {"UP", "DOWN"} and h4.get("trend_direction") in {"UP", "DOWN"}:
        if daily.get("trend_direction") != h4.get("trend_direction"):
            return "CONFLICT"
    if daily.get("trend_direction") == h4.get("trend_direction") == h1.get("trend_direction"):
        h4_confirms = _safe_bool(h4.get("pivot_confirmed")) or _safe_bool(h4.get("bos_confirmed"))
        h1_confirms = _safe_bool(h1.get("pivot_confirmed")) or _safe_bool(h1.get("bos_confirmed")) or h1.get("momentum_state") == "rising"
        if h4_confirms and h1_confirms:
            return "STRONG_ALIGNMENT"
    if daily.get("trend_direction") == h4.get("trend_direction") and daily.get("trend_direction") in {"UP", "DOWN"}:
        return "PARTIAL_ALIGNMENT"
    if len({str(item) for item in directions if item in {"UP", "DOWN"}}) > 1:
        return "CONFLICT"
    return "WEAK_ALIGNMENT"


def _missing_for_setup(daily: dict[str, Any], h4: dict[str, Any], h1: dict[str, Any], status: str, feed_blocked: bool) -> list[str]:
    missing: list[str] = []
    if feed_blocked:
        missing.append("feed_not_live")
    if status == "CONFLICT":
        missing.append("timeframe_conflict")
    if status == "INSUFFICIENT_DATA":
        missing.append("insufficient_data")
    if daily.get("trend_direction") != "UP":
        missing.append("daily_trend_not_confirmed")
    if not _safe_bool(h4.get("bos_confirmed")):
        missing.append("h4_bos_missing")
    if not _safe_bool(h4.get("pivot_confirmed")):
        missing.append("h4_pivot_missing")
    if not (_safe_bool(h1.get("pivot_confirmed")) or _safe_bool(h1.get("bos_confirmed")) or h1.get("momentum_state") == "rising"):
        missing.append("h1_confirmation_missing")
    if h4.get("momentum_state") not in {"rising", "neutral"} or h1.get("momentum_state") not in {"rising", "neutral"}:
        missing.append("momentum_weak")
    return list(dict.fromkeys(missing))


def build_alignment_candidate(
    symbol: str,
    diagnostics: dict[str, dict[str, Any]],
    *,
    feed_blocked: bool = False,
    provider_effective: str = "",
    fib_candidate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    daily = diagnostics.get("1d", {})
    h4 = diagnostics.get("4h", {})
    h1 = diagnostics.get("1h", {})
    status = _alignment_status(daily, h4, h1, feed_blocked=feed_blocked)
    quality_avg = (
        float(daily.get("structure_quality_score", 0.0) or 0.0) * 0.40
        + float(h4.get("structure_quality_score", 0.0) or 0.0) * 0.38
        + float(h1.get("structure_quality_score", 0.0) or 0.0) * 0.22
    )
    direction_bonus = 0.12 if daily.get("trend_direction") == h4.get("trend_direction") == h1.get("trend_direction") == "UP" else 0.0
    h4_bonus = 0.08 if _safe_bool(h4.get("pivot_confirmed")) or _safe_bool(h4.get("bos_confirmed")) else 0.0
    h1_bonus = 0.05 if _safe_bool(h1.get("pivot_confirmed")) or _safe_bool(h1.get("bos_confirmed")) or h1.get("momentum_state") == "rising" else 0.0
    alignment_score = max(0.0, min(1.0, quality_avg + direction_bonus + h4_bonus + h1_bonus))
    if status == "CONFLICT":
        alignment_score = min(alignment_score, 0.42)
    if status == "INSUFFICIENT_DATA":
        alignment_score = min(alignment_score, 0.30)
    missing = _missing_for_setup(daily, h4, h1, status, feed_blocked)

    pivot_timeframes = [tf for tf, payload in diagnostics.items() if _safe_bool(payload.get("pivot_confirmed"))]
    bos_timeframes = [tf for tf, payload in diagnostics.items() if _safe_bool(payload.get("bos_confirmed"))]
    pivot_source = pivot_timeframes[0] if pivot_timeframes else "none"
    bos_source = bos_timeframes[0] if bos_timeframes else "none"

    fib_zone = str((fib_candidate or {}).get("current_fib_zone") or (fib_candidate or {}).get("fib_zone") or "").upper()
    fib_score = _as_float((fib_candidate or {}).get("structure_score"), None)
    fib_confluence = bool(fib_zone in {"MEDIUM_ZONE", "DEEP_ZONE", "SHALLOW_ZONE"} or (fib_score is not None and float(fib_score) >= 0.55))
    fib_missing = "" if fib_confluence else "fib_no_useful_zone_or_score"

    supports_setup = (
        not feed_blocked
        and status in {"STRONG_ALIGNMENT", "PARTIAL_ALIGNMENT"}
        and daily.get("trend_direction") == "UP"
        and h4.get("trend_direction") == "UP"
    )
    would_improve = bool(supports_setup and alignment_score >= 0.55 and status != "CONFLICT")
    should_keep_blocked = bool(missing)
    if not missing:
        should_keep_blocked = True  # This audit is never approval authority.

    recommendation = "observe_more"
    if feed_blocked:
        recommendation = "feed_guard_blocks_shadow_diagnostic"
    elif status == "STRONG_ALIGNMENT" and fib_confluence:
        recommendation = "study_future_filter_confluence"
    elif status == "PARTIAL_ALIGNMENT":
        recommendation = "wait_h1_confirmation"
    elif status == "CONFLICT":
        recommendation = "keep_blocked_timeframe_conflict"
    elif status == "INSUFFICIENT_DATA":
        recommendation = "insufficient_data"
    elif status == "WEAK_ALIGNMENT":
        recommendation = "keep_current_strategy"

    return {
        "symbol": str(symbol).upper(),
        "daily_bias": str(daily.get("trend_direction") or "INCONCLUSIVE"),
        "h4_structure": str(h4.get("trend_direction") or "INCONCLUSIVE"),
        "h1_confirmation": str(h1.get("trend_direction") or "INCONCLUSIVE"),
        "alignment_score": round(float(alignment_score), 4),
        "alignment_status": status,
        "pivot_confirmed_timeframes": pivot_timeframes,
        "bos_confirmed_timeframes": bos_timeframes,
        "pivot_timeframes": pivot_timeframes,
        "bos_timeframes": bos_timeframes,
        "supports_trend_pullback_breakout": bool(supports_setup),
        "missing_for_setup": missing,
        "would_improve_signal_quality": bool(would_improve),
        "should_keep_blocked": bool(should_keep_blocked),
        "recommendation": recommendation,
        "fib_confluence_with_h4": bool(fib_confluence and h4.get("trend_direction") == "UP"),
        "fib_confluence_with_h1": bool(fib_confluence and h1.get("trend_direction") == "UP"),
        "fib_missing_confirmation_reason": fib_missing,
        "bos_source_timeframe": bos_source,
        "pivot_source_timeframe": pivot_source,
        "provider_effective": str(provider_effective or ""),
        "timeframe_diagnostics": diagnostics,
        "shadow_only": True,
    }


def _feed_guard_for_symbol(payload: Any, market_data_status: dict[str, Any], require_live_feed: bool) -> tuple[bool, str, str]:
    frame = payload.get("1h") if isinstance(payload, dict) else payload
    frame = _clean_frame(frame if isinstance(frame, pd.DataFrame) else None)
    frame_source = _last_text(frame, "data_source", "") if not frame.empty else ""
    frame_provider = _last_text(frame, "provider_name", "") if not frame.empty else ""
    feed_status = _normalize_feed_status(market_data_status.get("feed_status") or market_data_status.get("status"))
    provider = str(market_data_status.get("provider_effective") or market_data_status.get("provider") or frame_provider or "").lower()
    if not require_live_feed:
        return False, provider, "live_not_required"
    if feed_status != "LIVE":
        return True, provider, f"feed_status_{feed_status.lower()}"
    if frame_source and frame_source.lower() != "market":
        return True, provider, f"frame_source_{frame_source.lower()}"
    if provider and provider not in KNOWN_PROVIDERS:
        return True, provider, "provider_unknown"
    return False, provider, "feed_live_provider_known"


def _candidate_for_fib(symbol: str, market_structure_audit: dict[str, Any] | None) -> dict[str, Any]:
    audit = dict(market_structure_audit or {})
    candidates = list(audit.get("market_structure_best_candidates", []) or [])
    for candidate in candidates:
        if str(candidate.get("symbol") or candidate.get("asset") or "").upper() == str(symbol).upper():
            return dict(candidate)
    return {}


def _trace_candidates_by_symbol(bos_pivot_trace_audit: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    trace = dict(bos_pivot_trace_audit or {})
    grouped: dict[str, dict[str, Any]] = {}
    for row in list(trace.get("recent_candidates", []) or []):
        if not isinstance(row, dict):
            continue
        symbol = str(row.get("symbol") or "").upper()
        timeframe = str(row.get("timeframe") or "").lower()
        if not symbol or timeframe not in {"4h", "1h"}:
            continue
        grouped.setdefault(symbol, {})[timeframe] = dict(row)
    return grouped


def _apply_bos_pivot_trace(candidate: dict[str, Any], trace_rows: dict[str, Any]) -> dict[str, Any]:
    if not trace_rows:
        return candidate
    h4 = dict(trace_rows.get("4h", {}) or {})
    h1 = dict(trace_rows.get("1h", {}) or {})
    relationship = str((h4 or h1).get("relationship_to_higher_tf") or "INSUFFICIENT_DATA")
    missing = list(candidate.get("missing_for_setup", []) or [])
    for piece in (
        h4.get("primary_missing_piece") if h4 else "",
        h1.get("primary_missing_piece") if h1 else "",
    ):
        text = str(piece or "").strip()
        if text and text not in missing and text != "real_strategy_still_authoritative":
            missing.append(text)
    candidate.update(
        {
            "missing_for_setup": missing,
            "bos_pivot_trace_relationship": relationship,
            "h4_bos_state": str(h4.get("bos_state") or "INSUFFICIENT_DATA"),
            "h1_bos_state": str(h1.get("bos_state") or "INSUFFICIENT_DATA"),
            "h4_pivot_state": str(h4.get("pivot_state") or "INSUFFICIENT_DATA"),
            "h1_pivot_state": str(h1.get("pivot_state") or "INSUFFICIENT_DATA"),
            "bos_pivot_primary_missing_piece": str(
                h4.get("primary_missing_piece") or h1.get("primary_missing_piece") or ""
            ),
        }
    )
    return candidate


def build_multi_timeframe_swing_audit(
    *,
    market_data: dict[str, Any] | None,
    market_data_status: dict[str, Any] | None = None,
    market_structure_audit: dict[str, Any] | None = None,
    fib_alignment_audit: dict[str, Any] | None = None,
    intraday_market_data: dict[str, dict[str, pd.DataFrame]] | None = None,
    intraday_fetch_summary: dict[str, Any] | None = None,
    bos_pivot_trace_audit: dict[str, Any] | None = None,
    enabled: bool = True,
    timeframes: list[str] | tuple[str, ...] | None = None,
    max_symbols: int = DEFAULT_MAX_SYMBOLS,
    require_live_feed: bool = True,
    cache_ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS,
) -> dict[str, Any]:
    if not enabled:
        disabled = _empty_default("Multi-timeframe swing audit disabled.")
        disabled["enabled"] = False
        disabled["provider_guard"] = "disabled"
        return disabled

    status = dict(market_data_status or {})
    requested_timeframes = [str(item).strip().lower() for item in (timeframes or DEFAULT_TIMEFRAMES) if str(item).strip()]
    if not requested_timeframes:
        requested_timeframes = list(DEFAULT_TIMEFRAMES)
    requested_timeframes = [tf for tf in requested_timeframes if tf in DEFAULT_TIMEFRAMES] or list(DEFAULT_TIMEFRAMES)

    data = dict(market_data or {})
    if not data:
        empty = _empty_default("No market data available for multi-timeframe swing audit.")
        empty["generated_at"] = _utc_now_iso()
        empty["timeframes_used"] = requested_timeframes
        empty["feed_status"] = _normalize_feed_status(status.get("feed_status") or status.get("status"))
        empty["provider_effective"] = str(status.get("provider_effective") or status.get("provider") or "")
        empty["cache_ttl_seconds"] = int(cache_ttl_seconds)
        empty["provider_guard"] = "no_market_data"
        return empty

    candidates: list[dict[str, Any]] = []
    fallback_notes: list[str] = []
    provider_guards: Counter[str] = Counter()
    intraday_available_by_symbol: dict[str, list[str]] = {}
    trace_by_symbol = _trace_candidates_by_symbol(bos_pivot_trace_audit)
    base_interval = str(status.get("effective_interval") or status.get("requested_interval") or "").strip().lower()
    for symbol, payload in list(data.items())[: max(1, int(max_symbols or DEFAULT_MAX_SYMBOLS))]:
        feed_blocked, provider, provider_guard = _feed_guard_for_symbol(payload, status, require_live_feed)
        provider_guards[provider_guard] += 1
        frames, notes = _resolve_timeframe_frames(payload, requested_timeframes, base_interval=base_interval)
        frames, available_intraday = _merge_intraday_frames(
            frames,
            dict((intraday_market_data or {}).get(str(symbol).upper(), {}) or {}),
            requested_timeframes,
        )
        if available_intraday:
            intraday_available_by_symbol[str(symbol).upper()] = available_intraday
            notes.extend([f"{tf}=real_intraday_fetcher" for tf in available_intraday])
        fallback_notes.extend(notes)
        diagnostics = {tf: build_timeframe_diagnostic(frames.get(tf), tf) for tf in requested_timeframes}
        for tf in DEFAULT_TIMEFRAMES:
            diagnostics.setdefault(tf, build_timeframe_diagnostic(pd.DataFrame(), tf))
        candidate = build_alignment_candidate(
            str(symbol),
            diagnostics,
            feed_blocked=feed_blocked,
            provider_effective=provider,
            fib_candidate=_candidate_for_fib(str(symbol), market_structure_audit),
        )
        candidate = _apply_bos_pivot_trace(candidate, trace_by_symbol.get(str(symbol).upper(), {}))
        candidates.append(candidate)

    candidates.sort(key=lambda item: float(item.get("alignment_score", 0.0) or 0.0), reverse=True)
    top = candidates[0] if candidates else {}
    status_counts = Counter(str(item.get("alignment_status") or "") for item in candidates)
    missing_counter: Counter[str] = Counter()
    for item in candidates:
        missing_counter.update([str(reason) for reason in list(item.get("missing_for_setup", []) or []) if str(reason)])
    dominant_missing = missing_counter.most_common(1)[0][0] if missing_counter else ""
    provider_guard = provider_guards.most_common(1)[0][0] if provider_guards else "not_evaluated"
    feed_status = _normalize_feed_status(status.get("feed_status") or status.get("status"))
    provider_effective = str(status.get("provider_effective") or status.get("provider") or top.get("provider_effective") or "")
    intraday_summary = dict(intraday_fetch_summary or {})
    uses_real_intraday = bool(intraday_available_by_symbol)
    intraday_timeframes_available = sorted({tf for values in intraday_available_by_symbol.values() for tf in values})
    intraday_top_symbol = next((symbol for symbol, values in intraday_available_by_symbol.items() if values), "")
    diagnostics = [row for row in list(intraday_summary.get("diagnostics", []) or []) if isinstance(row, dict)]
    h4_quality = next((str(row.get("data_quality") or "") for row in diagnostics if row.get("timeframe") == "4h"), "")
    h1_quality = next((str(row.get("data_quality") or "") for row in diagnostics if row.get("timeframe") == "1h"), "")
    intraday_missing_reason = ""
    if not uses_real_intraday:
        intraday_missing_reason = str(
            intraday_summary.get("provider_guard_reason")
            or intraday_summary.get("last_error")
            or "insufficient_intraday_candles"
        )
    reason = (
        "Multi-timeframe swing audit generated from operational cycle data. "
        "1D/4H are resampled locally; no extra provider call is required."
    )
    if uses_real_intraday:
        reason = "Multi-timeframe swing audit used real intraday 4H/1H data from the SHADOW_ONLY fetcher."
    if provider_guard != "feed_live_provider_known":
        reason = f"Multi-timeframe swing audit degraded by provider guard: {provider_guard}."

    return {
        "enabled": True,
        "mode": MODE,
        "generated_at": _utc_now_iso(),
        "provider_effective": provider_effective,
        "feed_status": feed_status,
        "timeframes_used": requested_timeframes,
        "timeframe_source": "real_intraday_fetcher_plus_operational_resample" if uses_real_intraday else "operational_cycle_resample",
        "timeframe_fallbacks": list(dict.fromkeys(fallback_notes))[:12],
        "symbols_analyzed": int(len(candidates)),
        "top_symbol": str(top.get("symbol") or ""),
        "top_alignment_score": top.get("alignment_score"),
        "top_alignment_status": str(top.get("alignment_status") or "INSUFFICIENT_DATA"),
        "top_missing_confirmation": dominant_missing,
        "top_recommendation": str(top.get("recommendation") or "insufficient_data"),
        "dominant_conflict_reason": dominant_missing,
        "candidates_count": int(len(candidates)),
        "strong_alignment_count": int(status_counts.get("STRONG_ALIGNMENT", 0)),
        "partial_alignment_count": int(status_counts.get("PARTIAL_ALIGNMENT", 0)),
        "conflict_count": int(status_counts.get("CONFLICT", 0)),
        "insufficient_data_count": int(status_counts.get("INSUFFICIENT_DATA", 0)),
        "setup_support_count": int(sum(1 for item in candidates if bool(item.get("supports_trend_pullback_breakout")))),
        "recent_candidates": candidates[:MAX_RECENT_CANDIDATES],
        "estimated_provider_calls": int(intraday_summary.get("estimated_provider_calls", 0) or 0),
        "cache_ttl_seconds": int(cache_ttl_seconds),
        "cache_status": "intraday_fetcher_with_cache" if intraday_summary else "cycle_data_resample_only",
        "provider_guard": provider_guard,
        "shadow_only": True,
        "fib_alignment_status": str((fib_alignment_audit or {}).get("fib_alignment_status") or ""),
        "uses_real_intraday_data": bool(uses_real_intraday),
        "intraday_timeframes_available": intraday_timeframes_available,
        "intraday_top_symbol": intraday_top_symbol,
        "intraday_missing_reason": intraday_missing_reason,
        "h4_data_quality": h4_quality or "missing",
        "h1_data_quality": h1_quality or "missing",
        "bos_pivot_trace_relationship": str((bos_pivot_trace_audit or {}).get("top_relationship") or "INSUFFICIENT_DATA"),
        "bos_pivot_top_pivot_state": str((bos_pivot_trace_audit or {}).get("top_pivot_state") or "INSUFFICIENT_DATA"),
        "bos_pivot_top_bos_state": str((bos_pivot_trace_audit or {}).get("top_bos_state") or "INSUFFICIENT_DATA"),
        "bos_pivot_dominant_missing_piece": str(
            (bos_pivot_trace_audit or {}).get("dominant_missing_piece")
            or (bos_pivot_trace_audit or {}).get("top_primary_missing_piece")
            or ""
        ),
        "reason": reason,
    }
