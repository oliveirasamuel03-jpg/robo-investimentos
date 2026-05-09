from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

import pandas as pd

from core.market_data import fetch_market_data_frame, frame_data_source, frame_provider_name


MODE = "SHADOW_ONLY"
DEFAULT_INTRADAY_TIMEFRAMES = ("4h", "1h")
DEFAULT_MAX_SYMBOLS = 5
DEFAULT_MAX_CALLS_PER_CYCLE = 3
DEFAULT_CACHE_TTL_SECONDS = 1800
DEFAULT_HISTORY_LIMIT = 240
DEFAULT_PERIOD = "3mo"
WATCHLIST_FALLBACK = ("BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "LINK-USD")
MIN_CANDLES = {"4h": 40, "1h": 60}
_INTRADAY_CACHE: dict[str, dict[str, Any]] = {}


FetchFrameFunc = Callable[..., tuple[pd.DataFrame, dict[str, Any]]]


@dataclass
class MultiTimeframeIntradayFetchResult:
    frames: dict[str, dict[str, pd.DataFrame]]
    summary: dict[str, Any]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().isoformat()


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


def normalize_intraday_timeframe(value: str | None) -> str:
    raw = str(value or "").strip().lower()
    aliases = {
        "1d": "1day",
        "1day": "1day",
        "4h": "4h",
        "240m": "4h",
        "240min": "4h",
        "1h": "1h",
        "60m": "1h",
        "60min": "1h",
    }
    return aliases.get(raw, raw)


def _canonical_timeframe(value: str | None) -> str:
    normalized = normalize_intraday_timeframe(value)
    if normalized == "1day":
        return "1d"
    return normalized


def _cache_key(symbol: str, timeframe: str, provider: str) -> str:
    return "|".join([str(provider or "default").lower(), str(symbol).upper(), _canonical_timeframe(timeframe)])


def clear_multi_timeframe_intraday_cache() -> None:
    _INTRADAY_CACHE.clear()


def _safe_frame(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    data = frame.copy()
    if "datetime" in data.columns:
        data["datetime"] = pd.to_datetime(data["datetime"], utc=True, errors="coerce")
        data = data.dropna(subset=["datetime"])
    for column in ("open", "high", "low", "close"):
        if column not in data.columns:
            return pd.DataFrame()
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["open", "high", "low", "close"])
    return data


def _frame_quality(frame: pd.DataFrame, timeframe: str) -> tuple[str, str]:
    data = _safe_frame(frame)
    if data.empty:
        return "missing", "no_real_intraday_frame"
    source = frame_data_source(data)
    if source != "market":
        return "provider_blocked", f"source_{source or 'unknown'}"
    candles = len(data)
    minimum = int(MIN_CANDLES.get(_canonical_timeframe(timeframe), 40))
    if candles < minimum:
        return "insufficient", "insufficient_intraday_candles"
    return "ok", "ok"


def _cached_frame(symbol: str, timeframe: str, provider: str, *, ttl_seconds: int, now: datetime) -> pd.DataFrame:
    entry = _INTRADAY_CACHE.get(_cache_key(symbol, timeframe, provider))
    if not entry:
        return pd.DataFrame()
    fetched_at = entry.get("fetched_at")
    frame = entry.get("frame")
    if not isinstance(fetched_at, datetime) or not isinstance(frame, pd.DataFrame):
        return pd.DataFrame()
    if (now - fetched_at).total_seconds() > int(ttl_seconds):
        return pd.DataFrame()
    return frame.copy()


def _store_cache(symbol: str, timeframe: str, provider: str, frame: pd.DataFrame, *, now: datetime) -> None:
    data = _safe_frame(frame)
    if data.empty:
        return
    _INTRADAY_CACHE[_cache_key(symbol, timeframe, provider)] = {
        "fetched_at": now,
        "frame": data.copy(),
    }


def _push_unique(items: list[str], value: Any) -> None:
    text = str(value or "").strip().upper()
    if text and text not in items:
        items.append(text)


def build_intraday_symbol_priority(
    *,
    base_symbols: list[str] | tuple[str, ...] | None = None,
    strategy_structure_audit: dict[str, Any] | None = None,
    calibration_preview: dict[str, Any] | None = None,
    market_structure_audit: dict[str, Any] | None = None,
    max_symbols: int = DEFAULT_MAX_SYMBOLS,
) -> list[str]:
    symbols: list[str] = []
    _push_unique(symbols, (strategy_structure_audit or {}).get("structural_audit_top_symbol"))
    _push_unique(symbols, (calibration_preview or {}).get("top_asset"))
    _push_unique(symbols, (market_structure_audit or {}).get("market_structure_top_symbol"))
    _push_unique(symbols, "BTC-USD")
    for symbol in list(base_symbols or []) + list(WATCHLIST_FALLBACK):
        _push_unique(symbols, symbol)
    return symbols[: max(1, int(max_symbols or DEFAULT_MAX_SYMBOLS))]


def default_multi_timeframe_intraday_fetcher_state(reason: str = "No multi-timeframe intraday fetch data yet.") -> dict[str, Any]:
    return {
        "enabled": True,
        "mode": MODE,
        "generated_at": "",
        "provider_effective": "",
        "feed_status": "UNKNOWN",
        "timeframes_requested": list(DEFAULT_INTRADAY_TIMEFRAMES),
        "timeframes_available": [],
        "symbols_requested": [],
        "symbols_fetched": [],
        "cache_hits": 0,
        "cache_misses": 0,
        "provider_calls_attempted": 0,
        "provider_calls_skipped": 0,
        "provider_budget_guard_active": False,
        "provider_guard_reason": reason,
        "estimated_provider_calls": 0,
        "last_success_at": "",
        "last_error": "",
        "intraday_data_quality": "NO_DATA",
        "intraday_fetch_recommendation": "observe_more",
        "diagnostics": [],
        "shadow_only": True,
    }


def _diagnostic_row(
    *,
    symbol: str,
    timeframe: str,
    frame: pd.DataFrame | None,
    cache_status: str,
    provider_call_attempted: bool,
    provider_call_skipped: bool = False,
    reason: str = "",
) -> dict[str, Any]:
    data = _safe_frame(frame)
    quality, quality_reason = _frame_quality(data, timeframe)
    return {
        "symbol": str(symbol).upper(),
        "timeframe": _canonical_timeframe(timeframe),
        "provider_interval": normalize_intraday_timeframe(timeframe),
        "candles_available": int(len(data)),
        "data_quality": quality,
        "quality_reason": reason or quality_reason,
        "cache_status": cache_status,
        "provider_call_attempted": bool(provider_call_attempted),
        "provider_call_skipped": bool(provider_call_skipped),
        "data_source": frame_data_source(data) if not data.empty else "unknown",
        "provider_name": frame_provider_name(data) if not data.empty else "unknown",
    }


def _summarize_recommendation(rows: list[dict[str, Any]], guard_reason: str) -> tuple[str, str]:
    if guard_reason and guard_reason not in {"ok", "partial"}:
        return "PROVIDER_BLOCKED", "wait_provider_budget_or_feed"
    ok_count = sum(1 for row in rows if row.get("data_quality") == "ok")
    if ok_count <= 0:
        return "INSUFFICIENT", "collect_more_intraday_data"
    if ok_count < len(rows):
        return "PARTIAL", "observe_more"
    return "OK", "use_for_shadow_diagnostic"


def fetch_multi_timeframe_intraday_data(
    *,
    symbols: list[str] | tuple[str, ...] | None,
    market_data_status: dict[str, Any] | None = None,
    enabled: bool = True,
    timeframes: list[str] | tuple[str, ...] | None = None,
    ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS,
    max_symbols: int = DEFAULT_MAX_SYMBOLS,
    max_calls_per_cycle: int = DEFAULT_MAX_CALLS_PER_CYCLE,
    require_live_feed: bool = True,
    provider_budget_mode: str = "conservative",
    provider: str | None = None,
    history_limit: int = DEFAULT_HISTORY_LIMIT,
    fetch_frame_func: FetchFrameFunc | None = None,
    now: datetime | None = None,
) -> MultiTimeframeIntradayFetchResult:
    current_time = now.astimezone(timezone.utc) if isinstance(now, datetime) else _utc_now()
    status = dict(market_data_status or {})
    feed_status = _normalize_feed_status(status.get("feed_status") or status.get("status"))
    provider_effective = str(provider or status.get("provider_effective") or status.get("provider") or "twelvedata").lower()
    requested_timeframes = [
        _canonical_timeframe(item)
        for item in list(timeframes or DEFAULT_INTRADAY_TIMEFRAMES)
        if _canonical_timeframe(item) in {"4h", "1h", "1d"}
    ]
    requested_timeframes = requested_timeframes or list(DEFAULT_INTRADAY_TIMEFRAMES)
    requested_symbols = [str(item).upper() for item in list(symbols or WATCHLIST_FALLBACK) if str(item).strip()]
    requested_symbols = requested_symbols[: max(1, int(max_symbols or DEFAULT_MAX_SYMBOLS))]

    summary = default_multi_timeframe_intraday_fetcher_state()
    summary.update(
        {
            "enabled": bool(enabled),
            "mode": MODE,
            "generated_at": current_time.isoformat(),
            "provider_effective": provider_effective,
            "feed_status": feed_status,
            "timeframes_requested": requested_timeframes,
            "symbols_requested": requested_symbols,
            "cache_ttl_seconds": int(ttl_seconds),
            "max_symbols": int(max_symbols),
            "max_calls_per_cycle": int(max_calls_per_cycle),
            "provider_budget_mode": str(provider_budget_mode or "conservative"),
            "shadow_only": True,
        }
    )
    if not enabled:
        summary["provider_guard_reason"] = "intraday_fetch_disabled"
        summary["intraday_data_quality"] = "DISABLED"
        summary["intraday_fetch_recommendation"] = "keep_shadow_only"
        return MultiTimeframeIntradayFetchResult(frames={}, summary=summary)

    if require_live_feed and feed_status != "LIVE":
        summary["provider_guard_reason"] = "feed_not_live"
        summary["provider_budget_guard_active"] = True
        summary["intraday_data_quality"] = "PROVIDER_BLOCKED"
        summary["intraday_fetch_recommendation"] = "wait_live_feed"
        return MultiTimeframeIntradayFetchResult(frames={}, summary=summary)

    frames: dict[str, dict[str, pd.DataFrame]] = {}
    diagnostics: list[dict[str, Any]] = []
    available_timeframes: set[str] = set()
    fetched_symbols: set[str] = set()
    calls_attempted = 0
    calls_skipped = 0
    cache_hits = 0
    cache_misses = 0
    guard_reason = "ok"
    last_error = ""
    fetcher = fetch_frame_func or fetch_market_data_frame

    for symbol in requested_symbols:
        for timeframe in requested_timeframes:
            cached = _cached_frame(symbol, timeframe, provider_effective, ttl_seconds=int(ttl_seconds), now=current_time)
            if not cached.empty:
                cache_hits += 1
                frames.setdefault(symbol, {})[timeframe] = cached
                diagnostics.append(
                    _diagnostic_row(
                        symbol=symbol,
                        timeframe=timeframe,
                        frame=cached,
                        cache_status="hit",
                        provider_call_attempted=False,
                    )
                )
                quality, _ = _frame_quality(cached, timeframe)
                if quality == "ok":
                    available_timeframes.add(timeframe)
                    fetched_symbols.add(symbol)
                continue

            cache_misses += 1
            if calls_attempted >= int(max_calls_per_cycle):
                calls_skipped += 1
                guard_reason = "provider_budget_guard_blocked"
                diagnostics.append(
                    _diagnostic_row(
                        symbol=symbol,
                        timeframe=timeframe,
                        frame=pd.DataFrame(),
                        cache_status="miss_budget_blocked",
                        provider_call_attempted=False,
                        provider_call_skipped=True,
                        reason="cache_miss_budget_blocked",
                    )
                )
                continue

            calls_attempted += 1
            try:
                frame, frame_status = fetcher(
                    symbol,
                    period=DEFAULT_PERIOD,
                    interval=normalize_intraday_timeframe(timeframe),
                    history_limit=int(history_limit),
                    provider=provider_effective,
                    allow_stale=False,
                    requested_by="multi_tf_intraday_fetcher",
                )
            except Exception as exc:
                frame = pd.DataFrame()
                frame_status = {"last_error": str(exc)}
                last_error = str(exc)

            data = _safe_frame(frame)
            source = frame_data_source(data) if not data.empty else "unknown"
            if not data.empty and source == "market":
                _store_cache(symbol, timeframe, provider_effective, data, now=current_time)
                frames.setdefault(symbol, {})[timeframe] = data
            else:
                last_error = str((frame_status or {}).get("last_error") or last_error or "provider_failed")

            row = _diagnostic_row(
                symbol=symbol,
                timeframe=timeframe,
                frame=data,
                cache_status="miss_fetched" if source == "market" else "miss_provider_failed",
                provider_call_attempted=True,
                reason="" if source == "market" else "provider_failed",
            )
            diagnostics.append(row)
            if row["data_quality"] == "ok":
                available_timeframes.add(timeframe)
                fetched_symbols.add(symbol)

    quality, recommendation = _summarize_recommendation(diagnostics, guard_reason)
    summary.update(
        {
            "timeframes_available": sorted(available_timeframes),
            "symbols_fetched": sorted(fetched_symbols),
            "cache_hits": int(cache_hits),
            "cache_misses": int(cache_misses),
            "provider_calls_attempted": int(calls_attempted),
            "provider_calls_skipped": int(calls_skipped),
            "provider_budget_guard_active": bool(calls_skipped > 0),
            "provider_guard_reason": guard_reason,
            "estimated_provider_calls": int(calls_attempted),
            "last_success_at": current_time.isoformat() if fetched_symbols else "",
            "last_error": last_error,
            "intraday_data_quality": quality,
            "intraday_fetch_recommendation": recommendation,
            "diagnostics": diagnostics[:30],
        }
    )
    return MultiTimeframeIntradayFetchResult(frames=frames, summary=summary)


def summary_without_frames(result: MultiTimeframeIntradayFetchResult | dict[str, Any] | None) -> dict[str, Any]:
    if isinstance(result, MultiTimeframeIntradayFetchResult):
        return deepcopy(result.summary)
    return deepcopy(dict(result or {}))
