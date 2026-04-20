from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime, timezone
import io
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from core.config import MARKET_DATA_CACHE_TTL_SECONDS, MARKET_DATA_PROVIDER


_MARKET_DATA_CACHE: dict[str, dict[str, Any]] = {}


@dataclass
class MarketDataBatchResult:
    frames: dict[str, pd.DataFrame]
    status: dict[str, Any]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().isoformat()


def frame_data_source(frame: pd.DataFrame) -> str:
    if frame is None or frame.empty or "data_source" not in frame.columns:
        return "unknown"

    values = [str(value).strip().lower() for value in frame["data_source"].dropna().tolist() if str(value).strip()]
    if not values:
        return "unknown"
    return values[-1]


def is_live_market_source(source: str | None) -> bool:
    return str(source or "").strip().lower() == "market"


def fallback_data(symbol: str, rows: int = 240) -> pd.DataFrame:
    seed = sum(ord(ch) for ch in str(symbol).upper()) % 17
    idx = np.arange(rows, dtype=float)
    base = 100 + seed + idx * 0.12 + np.sin(idx / 8 + seed) * 2.4
    close = np.maximum(base, 1.0)
    open_ = close + np.sin(idx / 5 + seed) * 0.35
    high = np.maximum(open_, close) + 0.6
    low = np.minimum(open_, close) - 0.6
    volume = 1000 + ((idx + seed) % 30) * 25

    return pd.DataFrame(
        {
            "datetime": pd.date_range(end=pd.Timestamp.utcnow(), periods=rows, freq="h"),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "data_source": "fallback",
        }
    )


def _normalize_downloaded_prices(df: pd.DataFrame, history_limit: int) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [column[0] for column in df.columns]

    df = df.reset_index().rename(
        columns={
            "Date": "datetime",
            "Datetime": "datetime",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )

    if "close" not in df.columns:
        return pd.DataFrame()

    if "volume" not in df.columns:
        df["volume"] = 0.0
    df["data_source"] = "market"

    keep = [column for column in ["datetime", "open", "high", "low", "close", "volume", "data_source"] if column in df.columns]
    return df[keep].tail(int(history_limit)).copy()


def _normalize_symbols(symbols: list[str]) -> list[str]:
    normalized = [str(symbol).strip().upper() for symbol in (symbols or []) if str(symbol).strip()]
    return normalized or ["AAPL"]


def _cache_key(symbols: list[str], period: str, interval: str, history_limit: int, provider: str) -> str:
    return "|".join([provider, period, interval, str(int(history_limit)), *symbols])


def _clone_frames(frames: dict[str, pd.DataFrame], source_override: str | None = None) -> dict[str, pd.DataFrame]:
    cloned: dict[str, pd.DataFrame] = {}
    for symbol, frame in (frames or {}).items():
        if frame is None or frame.empty:
            continue
        copy = frame.copy()
        if source_override:
            copy["data_source"] = source_override
        cloned[str(symbol).upper()] = copy
    return cloned


def _cached_frames(
    symbols: list[str],
    *,
    period: str,
    interval: str,
    history_limit: int,
    provider: str,
    allow_stale: bool,
) -> dict[str, pd.DataFrame]:
    cache_entry = _MARKET_DATA_CACHE.get(_cache_key(symbols, period, interval, history_limit, provider))
    if not cache_entry:
        return {}

    fetched_at = cache_entry.get("fetched_at")
    if not isinstance(fetched_at, datetime):
        return {}

    age_seconds = (_utc_now() - fetched_at).total_seconds()
    if not allow_stale and age_seconds > MARKET_DATA_CACHE_TTL_SECONDS:
        return {}

    source = "cached" if allow_stale and age_seconds > MARKET_DATA_CACHE_TTL_SECONDS else "market"
    return _clone_frames(cache_entry.get("frames", {}), source_override=source)


def _extract_symbol_download(batch_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if batch_df is None or batch_df.empty:
        return pd.DataFrame()

    if isinstance(batch_df.columns, pd.MultiIndex):
        top_level = {str(value).upper() for value in batch_df.columns.get_level_values(0)}
        if str(symbol).upper() not in top_level:
            return pd.DataFrame()
        return batch_df.xs(str(symbol).upper(), axis=1, level=0, drop_level=True).copy()

    return batch_df.copy()


def _status_from_frames(
    frames: dict[str, pd.DataFrame],
    *,
    provider: str,
    symbols: list[str],
    requested_by: str,
    last_error: str = "",
) -> dict[str, Any]:
    source_breakdown = {"market": 0, "cached": 0, "fallback": 0, "unknown": 0}
    last_source = "unknown"

    for symbol in symbols:
        source = frame_data_source(frames.get(symbol, pd.DataFrame()))
        last_source = source
        source_breakdown[source if source in source_breakdown else "unknown"] += 1

    if source_breakdown["market"] == len(symbols):
        status = "healthy"
        last_source = "market"
    elif source_breakdown["market"] > 0 and source_breakdown["fallback"] == 0:
        status = "degraded"
        last_source = "mixed"
    elif source_breakdown["cached"] > 0 and source_breakdown["fallback"] == 0:
        status = "cached"
        last_source = "cached"
    elif source_breakdown["fallback"] > 0 and source_breakdown["market"] == 0 and source_breakdown["cached"] == 0:
        status = "error"
        last_source = "fallback"
    else:
        status = "degraded"
        last_source = "mixed"

    return {
        "provider": provider,
        "status": status,
        "last_sync_at": _utc_now_iso(),
        "last_error": last_error,
        "last_source": last_source,
        "source_breakdown": source_breakdown,
        "symbols": symbols,
        "requested_by": requested_by,
    }


def fetch_market_data_map(
    symbols: list[str],
    *,
    period: str,
    interval: str,
    history_limit: int,
    provider: str | None = None,
    allow_stale: bool = True,
    requested_by: str = "runtime",
) -> MarketDataBatchResult:
    provider_name = str(provider or MARKET_DATA_PROVIDER or "yahoo").strip().lower()
    normalized_symbols = _normalize_symbols(symbols)

    fresh_cache = _cached_frames(
        normalized_symbols,
        period=period,
        interval=interval,
        history_limit=history_limit,
        provider=provider_name,
        allow_stale=False,
    )
    if fresh_cache:
        return MarketDataBatchResult(
            frames=fresh_cache,
            status=_status_from_frames(
                fresh_cache,
                provider=provider_name,
                symbols=normalized_symbols,
                requested_by=requested_by,
            ),
        )

    batch_df = pd.DataFrame()
    error_messages: list[str] = []

    if provider_name == "yahoo":
        try:
            stdout_buffer = io.StringIO()
            stderr_buffer = io.StringIO()
            target: str | list[str] = normalized_symbols if len(normalized_symbols) > 1 else normalized_symbols[0]
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                batch_df = yf.download(
                    tickers=target,
                    period=period,
                    interval=interval,
                    auto_adjust=False,
                    progress=False,
                    threads=False,
                    group_by="ticker",
                )
            suppressed_output = "\n".join(
                item.strip()
                for item in [stdout_buffer.getvalue(), stderr_buffer.getvalue()]
                if item and item.strip()
            )
            if suppressed_output:
                error_messages.append(suppressed_output)
        except Exception as exc:
            batch_df = pd.DataFrame()
            error_messages.append(str(exc))

    frames: dict[str, pd.DataFrame] = {}
    for symbol in normalized_symbols:
        normalized_frame = _normalize_downloaded_prices(_extract_symbol_download(batch_df, symbol), history_limit)
        if not normalized_frame.empty:
            frames[symbol] = normalized_frame

    if frames:
        _MARKET_DATA_CACHE[_cache_key(normalized_symbols, period, interval, history_limit, provider_name)] = {
            "fetched_at": _utc_now(),
            "frames": _clone_frames(frames),
        }

    stale_frames = (
        _cached_frames(
            normalized_symbols,
            period=period,
            interval=interval,
            history_limit=history_limit,
            provider=provider_name,
            allow_stale=True,
        )
        if allow_stale
        else {}
    )

    for symbol in normalized_symbols:
        if symbol in frames:
            continue
        if symbol in stale_frames:
            frames[symbol] = stale_frames[symbol]
        else:
            frames[symbol] = fallback_data(symbol, rows=max(180, int(history_limit)))

    last_error = " | ".join(message for message in error_messages if message).strip()
    return MarketDataBatchResult(
        frames=frames,
        status=_status_from_frames(
            frames,
            provider=provider_name,
            symbols=normalized_symbols,
            requested_by=requested_by,
            last_error=last_error,
        ),
    )


def fetch_market_data_frame(
    symbol: str,
    *,
    period: str,
    interval: str,
    history_limit: int,
    provider: str | None = None,
    allow_stale: bool = True,
    requested_by: str = "runtime",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    result = fetch_market_data_map(
        [symbol],
        period=period,
        interval=interval,
        history_limit=history_limit,
        provider=provider,
        allow_stale=allow_stale,
        requested_by=requested_by,
    )
    normalized_symbol = _normalize_symbols([symbol])[0]
    return result.frames.get(normalized_symbol, pd.DataFrame()), result.status
