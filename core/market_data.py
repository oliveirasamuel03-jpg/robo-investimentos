from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime, timezone
import io
import json
from typing import Any
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

import numpy as np
import pandas as pd
import yfinance as yf

from core.config import (
    APP_SOURCE_COMMIT_SHA,
    BUILD_TIMESTAMP,
    CACHE_DIR,
    MARKET_DATA_CACHE_TTL_SECONDS,
    MARKET_DATA_BUILD_LABEL,
    MARKET_DATA_DIAGNOSTIC_VERSION,
    MARKET_DATA_FALLBACK_PROVIDER,
    MARKET_DATA_PROVIDER,
    RAILWAY_SERVICE_NAME,
    RAILWAY_GIT_COMMIT_SHA,
    SERVICE_NAME,
    SWING_VALIDATION_RECOMMENDED_WATCHLIST,
    TWELVEDATA_API_BASE,
    TWELVEDATA_API_KEY,
    TWELVEDATA_MIN_CACHE_TTL_SECONDS,
)


_MARKET_DATA_CACHE: dict[str, dict[str, Any]] = {}
_YFINANCE_CACHE_DIR = CACHE_DIR / "yfinance"
_SUPPORTED_MARKET_PROVIDERS = {"twelvedata", "yahoo"}


@dataclass
class MarketDataBatchResult:
    frames: dict[str, pd.DataFrame]
    status: dict[str, Any]


FEED_STATUS_LIVE = "LIVE"
FEED_STATUS_DELAYED = "DELAYED"
FEED_STATUS_FALLBACK = "FALLBACK"
FEED_STATUS_UNKNOWN = "UNKNOWN"


def _configure_yfinance_cache() -> None:
    # yfinance uses sqlite-backed caches internally. Point them to our writable
    # project storage so the feed does not drop into fallback because of an
    # unwritable default user cache directory.
    try:
        _YFINANCE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        if hasattr(yf, "set_tz_cache_location"):
            yf.set_tz_cache_location(str(_YFINANCE_CACHE_DIR))
    except Exception:
        # The live/cached/fallback classification should keep working even if the
        # cache reconfiguration itself fails for some reason.
        return


_configure_yfinance_cache()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().isoformat()


def parse_market_timestamp(value: object) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        normalized = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def format_market_timestamp(value: object) -> str:
    parsed = parse_market_timestamp(value)
    if parsed is None:
        return "Sem registro"
    return parsed.strftime("%Y-%m-%d %H:%M:%S UTC")


def frame_data_source(frame: pd.DataFrame) -> str:
    if frame is None or frame.empty or "data_source" not in frame.columns:
        return "unknown"

    values = [str(value).strip().lower() for value in frame["data_source"].dropna().tolist() if str(value).strip()]
    if not values:
        return "unknown"
    return values[-1]


def frame_provider_name(frame: pd.DataFrame) -> str:
    if frame is None or frame.empty or "provider_name" not in frame.columns:
        return "unknown"

    values = [str(value).strip().lower() for value in frame["provider_name"].dropna().tolist() if str(value).strip()]
    if not values:
        return "unknown"
    return values[-1]


def is_live_market_source(source: str | None) -> bool:
    return str(source or "").strip().lower() == "market"


def classify_feed_status(
    *,
    status: str | None = None,
    last_source: str | None = None,
    source_breakdown: dict[str, Any] | None = None,
) -> str:
    normalized_status = str(status or "").strip().upper()
    if normalized_status in {
        FEED_STATUS_LIVE,
        FEED_STATUS_DELAYED,
        FEED_STATUS_FALLBACK,
        FEED_STATUS_UNKNOWN,
    }:
        return normalized_status

    legacy_status = str(status or "").strip().lower()
    source = str(last_source or "").strip().lower()
    breakdown = dict(source_breakdown or {})
    market_count = int(breakdown.get("market", 0) or 0)
    cached_count = int(breakdown.get("cached", 0) or 0)
    fallback_count = int(breakdown.get("fallback", 0) or 0)
    known_total = market_count + cached_count + fallback_count + int(breakdown.get("unknown", 0) or 0)

    if fallback_count > 0 and market_count == 0 and cached_count == 0:
        return FEED_STATUS_FALLBACK
    if market_count > 0 and cached_count == 0 and fallback_count == 0:
        return FEED_STATUS_LIVE
    if cached_count > 0 or fallback_count > 0:
        return FEED_STATUS_DELAYED

    if source == "fallback" or legacy_status == "error":
        return FEED_STATUS_FALLBACK
    if source in {"cached", "mixed"} or legacy_status in {"cached", "degraded"}:
        return FEED_STATUS_DELAYED
    if source == "market" or legacy_status == "healthy":
        return FEED_STATUS_LIVE
    if known_total == 0:
        return FEED_STATUS_UNKNOWN
    return FEED_STATUS_DELAYED


def legacy_market_status(
    *,
    status: str | None = None,
    last_source: str | None = None,
    source_breakdown: dict[str, Any] | None = None,
) -> str:
    raw = str(status or "").strip().lower()
    if raw in {"healthy", "degraded", "cached", "error", "unknown"}:
        return raw

    feed_status = classify_feed_status(
        status=status,
        last_source=last_source,
        source_breakdown=source_breakdown,
    )
    source = str(last_source or "").strip().lower()

    if feed_status == FEED_STATUS_LIVE:
        return "healthy"
    if feed_status == FEED_STATUS_FALLBACK:
        return "error"
    if feed_status == FEED_STATUS_DELAYED:
        return "cached" if source == "cached" else "degraded"
    return "unknown"


def build_feed_quality_snapshot(status: dict[str, Any] | None) -> dict[str, Any]:
    payload = dict(status or {})
    diagnostics = dict(payload.get("provider_diagnostics", {}) or {})
    twelvedata_diag = dict(diagnostics.get("twelvedata", {}) or {})
    requested_symbols = [str(item).upper() for item in (payload.get("requested_symbols") or payload.get("symbols") or []) if str(item)]
    live_symbols = [str(item).upper() for item in (payload.get("live_symbols") or []) if str(item)]
    cached_symbols = [str(item).upper() for item in (payload.get("cached_symbols") or []) if str(item)]
    fallback_symbols = [str(item).upper() for item in (payload.get("fallback_symbols") or []) if str(item)]
    unknown_symbols = [str(item).upper() for item in (payload.get("unknown_symbols") or []) if str(item)]
    total_symbols = len(requested_symbols)
    success_total = int(twelvedata_diag.get("symbols_total") or total_symbols or 0)
    success_count = int(twelvedata_diag.get("success_count") or len(live_symbols) or 0)
    success_rate = None if success_total <= 0 else round(float(success_count / success_total), 4)
    feed_status = classify_feed_status(
        status=payload.get("feed_status") or payload.get("status"),
        last_source=payload.get("last_source"),
        source_breakdown=payload.get("source_breakdown"),
    )
    fallback_reason = str(payload.get("last_error") or twelvedata_diag.get("last_error") or "").strip()

    if feed_status == FEED_STATUS_LIVE and len(fallback_symbols) == 0:
        quality_label = "Boa"
        quality_message = "O worker conseguiu dado live para os ativos monitorados neste ciclo."
    elif feed_status == FEED_STATUS_DELAYED or len(cached_symbols) > 0:
        quality_label = "Moderada"
        quality_message = "Parte do ciclo usou cache ou mistura de fontes; a leitura operacional segue util, mas pede atencao."
    elif len(fallback_symbols) > 0 or feed_status == FEED_STATUS_FALLBACK:
        quality_label = "Baixa"
        quality_message = "O ciclo ainda dependeu de fallback e nao sustenta uma leitura estrategica confiavel."
    else:
        quality_label = "Baixa"
        quality_message = "Ainda nao ha leitura suficiente do feed para classificar a qualidade desta janela."

    return {
        "feed_status": feed_status,
        "provider_effective": str(payload.get("provider_effective") or payload.get("provider") or "unknown"),
        "requested_symbols": requested_symbols,
        "live_symbols": live_symbols,
        "cached_symbols": cached_symbols,
        "fallback_symbols": fallback_symbols,
        "unknown_symbols": unknown_symbols,
        "live_count": len(live_symbols),
        "cached_count": len(cached_symbols),
        "fallback_count": len(fallback_symbols),
        "unknown_count": len(unknown_symbols),
        "total_symbols": total_symbols,
        "twelvedata_success_rate": success_rate,
        "last_success_at": str(payload.get("last_success_at") or ""),
        "last_sync_at": str(payload.get("last_sync_at") or ""),
        "fallback_reason": fallback_reason,
        "quality_label": quality_label,
        "quality_message": quality_message,
    }


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
            "provider_name": "synthetic",
        }
    )


def _normalize_downloaded_prices(df: pd.DataFrame, history_limit: int, provider_name: str = "yahoo") -> pd.DataFrame:
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
    df["datetime"] = pd.to_datetime(df.get("datetime"), utc=True, errors="coerce")
    df = df.dropna(subset=["datetime", "close"])
    if df.empty:
        return pd.DataFrame()

    for column in ("open", "high", "low", "close", "volume"):
        if column not in df.columns:
            df[column] = 0.0
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["open", "high", "low", "close"])
    if df.empty:
        return pd.DataFrame()

    df["data_source"] = "market"
    df["provider_name"] = provider_name

    keep = [
        column
        for column in ["datetime", "open", "high", "low", "close", "volume", "data_source", "provider_name"]
        if column in df.columns
    ]
    return df[keep].tail(int(history_limit)).copy()


def _normalize_twelvedata_prices(payload: dict[str, Any], history_limit: int) -> pd.DataFrame:
    if str(payload.get("status") or "").strip().lower() != "ok":
        return pd.DataFrame()

    values = payload.get("values")
    if not isinstance(values, list) or not values:
        return pd.DataFrame()

    df = pd.DataFrame(values)
    if df.empty or "close" not in df.columns or "datetime" not in df.columns:
        return pd.DataFrame()

    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    for column in ("open", "high", "low", "close", "volume"):
        if column not in df.columns:
            df[column] = 0.0
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["datetime", "open", "high", "low", "close"])
    if df.empty:
        return pd.DataFrame()

    df = df.sort_values("datetime")
    df["data_source"] = "market"
    df["provider_name"] = "twelvedata"

    keep = ["datetime", "open", "high", "low", "close", "volume", "data_source", "provider_name"]
    return df[keep].tail(int(history_limit)).copy()


def _normalize_symbols(symbols: list[str]) -> list[str]:
    normalized = [str(symbol).strip().upper() for symbol in (symbols or []) if str(symbol).strip()]
    return normalized or list(SWING_VALIDATION_RECOMMENDED_WATCHLIST)


def _normalize_provider_name(value: str | None, *, default: str = "yahoo") -> str:
    raw = str(value or "").strip().lower()
    aliases = {
        "twelvedata": "twelvedata",
        "twelve_data": "twelvedata",
        "twelve-data": "twelvedata",
        "yahoo": "yahoo",
        "synthetic": "synthetic",
        "fallback": "synthetic",
    }
    return aliases.get(raw, default)


def _provider_chain(provider: str | None = None) -> list[str]:
    primary = _normalize_provider_name(provider or MARKET_DATA_PROVIDER, default="twelvedata")
    fallback = _normalize_provider_name(MARKET_DATA_FALLBACK_PROVIDER, default="yahoo")
    chain: list[str] = []
    for candidate in (primary, fallback):
        if candidate in _SUPPORTED_MARKET_PROVIDERS and candidate not in chain:
            chain.append(candidate)
    if not chain:
        chain.append("yahoo")
    return chain


def _cache_key(symbols: list[str], period: str, interval: str, history_limit: int, provider: str) -> str:
    return "|".join([provider, period, interval, str(int(history_limit)), *symbols])


def _effective_cache_ttl_seconds(provider: str) -> int:
    normalized_provider = _normalize_provider_name(provider)
    if normalized_provider == "twelvedata":
        # Twelve Data Basic (free) offers 800 daily credits. With our 5-symbol
        # crypto watchlist, polling every 5 minutes would consume 1,440 credits/day.
        # A 15-minute fresh cache keeps the worker inside the free-plan budget
        # while preserving a live enough cadence for the current 1h swing cycle.
        return max(MARKET_DATA_CACHE_TTL_SECONDS, TWELVEDATA_MIN_CACHE_TTL_SECONDS)
    return MARKET_DATA_CACHE_TTL_SECONDS


def _safe_diagnostic_text(value: Any, limit: int = 160) -> str:
    text = str(value or "").replace("\r", " ").replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return f"{text[: max(0, limit - 3)]}..."


def _build_twelvedata_diagnostic(symbol: str, *, interval: str, history_limit: int) -> dict[str, Any]:
    api_key = str(TWELVEDATA_API_KEY or "").strip()
    normalized_base = str(TWELVEDATA_API_BASE or "").strip() or "https://api.twelvedata.com"
    parsed_base = urllib_parse.urlparse(normalized_base)
    base_host = parsed_base.netloc or ""
    normalized_interval = _normalize_twelvedata_interval(interval)
    return {
        "diagnostic_version": MARKET_DATA_DIAGNOSTIC_VERSION,
        "build_label": MARKET_DATA_BUILD_LABEL,
        "service_name": str(RAILWAY_SERVICE_NAME or "").strip() or "unknown",
        "api_key_present": bool(api_key),
        "api_key_length": len(api_key),
        "api_base": normalized_base.rstrip("/"),
        "api_base_host": base_host,
        "api_base_valid": bool(parsed_base.scheme and base_host),
        "endpoint_path": "/time_series",
        "symbol": str(symbol or "").strip().upper(),
        "normalized_symbol": _normalize_twelvedata_symbol(symbol),
        "interval_raw": str(interval or "1h"),
        "interval": normalized_interval,
        "outputsize": min(max(int(history_limit), 30), 5000),
        "request_built": False,
        "request_attempted": False,
        "response_received": False,
        "http_status": None,
        "payload_status": "",
        "payload_code": "",
        "values_count": 0,
        "last_stage": "init",
        "last_error": "",
        "error_class": "",
        "used_live_data": False,
    }


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

    ttl_seconds = _effective_cache_ttl_seconds(provider)
    age_seconds = (_utc_now() - fetched_at).total_seconds()
    if not allow_stale and age_seconds > ttl_seconds:
        return {}

    source = "cached" if allow_stale and age_seconds > ttl_seconds else "market"
    return _clone_frames(cache_entry.get("frames", {}), source_override=source)


def _store_cached_frames(
    symbols: list[str],
    *,
    period: str,
    interval: str,
    history_limit: int,
    provider: str,
    frames: dict[str, pd.DataFrame],
) -> None:
    if not frames:
        return
    _MARKET_DATA_CACHE[_cache_key(symbols, period, interval, history_limit, provider)] = {
        "fetched_at": _utc_now(),
        "frames": _clone_frames(frames),
    }


def _extract_symbol_download(batch_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if batch_df is None or batch_df.empty:
        return pd.DataFrame()

    if isinstance(batch_df.columns, pd.MultiIndex):
        top_level = {str(value).upper() for value in batch_df.columns.get_level_values(0)}
        if str(symbol).upper() not in top_level:
            return pd.DataFrame()
        return batch_df.xs(str(symbol).upper(), axis=1, level=0, drop_level=True).copy()

    return batch_df.copy()


def _normalize_twelvedata_symbol(symbol: str) -> str:
    normalized = str(symbol or "").strip().upper()
    if not normalized or "/" in normalized or "." in normalized or "=" in normalized:
        return normalized

    parts = normalized.split("-")
    if len(parts) == 2 and all(parts):
        return f"{parts[0]}/{parts[1]}"
    return normalized


def _normalize_twelvedata_interval(interval: str | None) -> str:
    raw = str(interval or "").strip().lower()
    if not raw:
        return "1h"

    aliases = {
        "1m": "1min",
        "1min": "1min",
        "5m": "5min",
        "5min": "5min",
        "15m": "15min",
        "15min": "15min",
        "30m": "30min",
        "30min": "30min",
        "45m": "45min",
        "45min": "45min",
        "60m": "1h",
        "60min": "1h",
        "1d": "1day",
        "1day": "1day",
        "1w": "1week",
        "1wk": "1week",
        "1week": "1week",
        "1mo": "1month",
        "1month": "1month",
    }
    return aliases.get(raw, raw)


def _twelvedata_error_message(payload: dict[str, Any] | None, fallback_message: str) -> str:
    body = payload or {}
    status = str(body.get("status") or "").strip()
    code = str(body.get("code") or "").strip()
    message = str(body.get("message") or "").strip()
    normalized_code = code.strip()
    if normalized_code == "401":
        message = message or "API key invalida ou nao autorizada no Twelve Data"
    elif normalized_code == "403":
        message = message or "Acesso negado pelo Twelve Data"
    elif normalized_code == "429":
        message = message or "Limite de creditos/requisicoes do Twelve Data atingido"
    details = [part for part in [status, code, message] if part]
    if details:
        return " | ".join(details)
    return fallback_message


def _summarize_twelvedata_diagnostics(entries: list[dict[str, Any]], *, symbols: list[str]) -> dict[str, Any]:
    if not entries:
        return {}

    first = dict(entries[0] or {})
    errors = [_safe_diagnostic_text(item.get("last_error")) for item in entries if item.get("last_error")]
    http_statuses = [
        int(item.get("http_status"))
        for item in entries
        if isinstance(item.get("http_status"), int)
    ]
    payload_statuses = [str(item.get("payload_status") or "").strip() for item in entries if item.get("payload_status")]
    payload_codes = [str(item.get("payload_code") or "").strip() for item in entries if item.get("payload_code")]
    return {
        "diagnostic_version": first.get("diagnostic_version") or MARKET_DATA_DIAGNOSTIC_VERSION,
        "build_label": first.get("build_label") or MARKET_DATA_BUILD_LABEL,
        "service_name": first.get("service_name") or (str(RAILWAY_SERVICE_NAME or "").strip() or "unknown"),
        "api_key_present": bool(first.get("api_key_present")),
        "api_key_length": int(first.get("api_key_length") or 0),
        "api_base": str(first.get("api_base") or ""),
        "api_base_host": str(first.get("api_base_host") or ""),
        "api_base_valid": bool(first.get("api_base_valid")),
        "endpoint_path": str(first.get("endpoint_path") or "/time_series"),
        "symbols_total": len(symbols),
        "symbols_requested": [str(symbol).upper() for symbol in symbols[:5]],
        "request_built_count": sum(1 for item in entries if bool(item.get("request_built"))),
        "request_attempted_count": sum(1 for item in entries if bool(item.get("request_attempted"))),
        "response_received_count": sum(1 for item in entries if bool(item.get("response_received"))),
        "success_count": sum(1 for item in entries if bool(item.get("used_live_data"))),
        "request_built": any(bool(item.get("request_built")) for item in entries),
        "request_attempted": any(bool(item.get("request_attempted")) for item in entries),
        "response_received": any(bool(item.get("response_received")) for item in entries),
        "used_live_data": any(bool(item.get("used_live_data")) for item in entries),
        "sample_symbol": str(first.get("symbol") or ""),
        "sample_normalized_symbol": str(first.get("normalized_symbol") or ""),
        "interval_raw": str(first.get("interval_raw") or ""),
        "normalized_interval": str(first.get("interval") or ""),
        "last_stage": str(
            next((item.get("last_stage") for item in entries if item.get("last_stage")), first.get("last_stage") or "unknown")
        ),
        "last_error": next((error for error in errors if error), ""),
        "http_statuses": http_statuses[:5],
        "payload_statuses": payload_statuses[:5],
        "payload_codes": payload_codes[:5],
    }


def _summarize_provider_diagnostics(
    provider: str,
    symbols: list[str],
    errors: list[str],
    *,
    interval: str = "",
) -> dict[str, Any]:
    normalized_provider = _normalize_provider_name(provider)
    return {
        normalized_provider: {
            "symbols_total": len(symbols),
            "symbols_requested": [str(symbol).upper() for symbol in symbols[:5]],
            "interval": str(interval or ""),
            "last_error": _safe_diagnostic_text(errors[0]) if errors else "",
        }
    }


def _fetch_twelvedata_frame(symbol: str, *, interval: str, history_limit: int) -> tuple[pd.DataFrame, str, dict[str, Any]]:
    diagnostic = _build_twelvedata_diagnostic(symbol, interval=interval, history_limit=history_limit)
    normalized_interval = str(diagnostic.get("interval") or _normalize_twelvedata_interval(interval))
    if not TWELVEDATA_API_KEY:
        diagnostic["last_stage"] = "missing_api_key"
        diagnostic["last_error"] = "TWELVEDATA_API_KEY nao configurada"
        return pd.DataFrame(), "TWELVEDATA_API_KEY nao configurada", diagnostic

    params = {
        "symbol": diagnostic["normalized_symbol"],
        "interval": normalized_interval,
        "outputsize": min(max(int(history_limit), 30), 5000),
        "timezone": "UTC",
        "order": "asc",
        "apikey": TWELVEDATA_API_KEY,
    }
    url = f"{TWELVEDATA_API_BASE.rstrip('/')}/time_series?{urllib_parse.urlencode(params)}"
    parsed_url = urllib_parse.urlparse(url)
    diagnostic["request_built"] = True
    diagnostic["last_stage"] = "request_built"
    diagnostic["request_host"] = parsed_url.netloc or ""
    diagnostic["request_path"] = parsed_url.path or ""
    request = urllib_request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "TradeOpsDesk/1.0",
        },
    )

    try:
        diagnostic["request_attempted"] = True
        diagnostic["last_stage"] = "request_sent"
        with urllib_request.urlopen(request, timeout=20) as response:
            diagnostic["response_received"] = True
            diagnostic["http_status"] = int(getattr(response, "status", 200) or 200)
            payload = json.loads(response.read().decode("utf-8"))
    except urllib_error.HTTPError as exc:
        diagnostic["request_attempted"] = True
        diagnostic["response_received"] = True
        diagnostic["http_status"] = int(exc.code)
        diagnostic["error_class"] = type(exc).__name__
        try:
            payload = json.loads(exc.read().decode("utf-8"))
        except Exception:
            payload = {}
        diagnostic["payload_status"] = str(payload.get("status") or "").strip()
        diagnostic["payload_code"] = str(payload.get("code") or "").strip()
        diagnostic["last_stage"] = "http_error"
        diagnostic["last_error"] = _safe_diagnostic_text(_twelvedata_error_message(payload, f"HTTP {exc.code}"))
        return pd.DataFrame(), _twelvedata_error_message(payload, f"HTTP {exc.code}"), diagnostic
    except Exception as exc:
        diagnostic["request_attempted"] = True
        diagnostic["last_stage"] = "request_exception"
        diagnostic["error_class"] = type(exc).__name__
        diagnostic["last_error"] = _safe_diagnostic_text(exc)
        return pd.DataFrame(), str(exc), diagnostic

    diagnostic["payload_status"] = str(payload.get("status") or "").strip()
    diagnostic["payload_code"] = str(payload.get("code") or "").strip()
    diagnostic["values_count"] = len(payload.get("values")) if isinstance(payload.get("values"), list) else 0
    diagnostic["last_stage"] = "response_received"

    frame = _normalize_twelvedata_prices(payload, history_limit)
    if frame.empty:
        diagnostic["last_stage"] = "empty_payload"
        diagnostic["last_error"] = _safe_diagnostic_text(
            _twelvedata_error_message(payload, "Resposta vazia do Twelve Data")
        )
        return pd.DataFrame(), _twelvedata_error_message(payload, "Resposta vazia do Twelve Data"), diagnostic
    diagnostic["used_live_data"] = True
    diagnostic["last_stage"] = "success"
    return frame, "", diagnostic


def _fetch_yahoo_frames(symbols: list[str], *, period: str, interval: str, history_limit: int) -> tuple[dict[str, pd.DataFrame], list[str]]:
    batch_df = pd.DataFrame()
    error_messages: list[str] = []

    try:
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        target: str | list[str] = symbols if len(symbols) > 1 else symbols[0]
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
    for symbol in symbols:
        normalized_frame = _normalize_downloaded_prices(_extract_symbol_download(batch_df, symbol), history_limit, "yahoo")
        if not normalized_frame.empty:
            frames[symbol] = normalized_frame

    return frames, error_messages


def _fetch_provider_frames(
    provider: str,
    symbols: list[str],
    *,
    period: str,
    interval: str,
    history_limit: int,
) -> tuple[dict[str, pd.DataFrame], list[str], dict[str, Any]]:
    normalized_provider = _normalize_provider_name(provider)
    if normalized_provider == "twelvedata":
        frames: dict[str, pd.DataFrame] = {}
        errors: list[str] = []
        diagnostics: list[dict[str, Any]] = []
        for symbol in symbols:
            frame, error_message, diagnostic = _fetch_twelvedata_frame(
                symbol,
                interval=interval,
                history_limit=history_limit,
            )
            diagnostics.append(diagnostic)
            if not frame.empty:
                frames[symbol] = frame
            elif error_message:
                errors.append(f"{symbol}: {error_message}")
        return frames, errors, {"twelvedata": _summarize_twelvedata_diagnostics(diagnostics, symbols=symbols)}

    if normalized_provider == "yahoo":
        frames, errors = _fetch_yahoo_frames(symbols, period=period, interval=interval, history_limit=history_limit)
        return frames, errors, _summarize_provider_diagnostics("yahoo", symbols, errors, interval=interval)

    errors = [f"provider nao suportado: {provider}"]
    return {}, errors, _summarize_provider_diagnostics(provider, symbols, errors, interval=interval)


def _resolve_effective_provider(provider_breakdown: dict[str, int]) -> str:
    active = [name for name, count in provider_breakdown.items() if int(count or 0) > 0 and name != "unknown"]
    if not active:
        return "unknown"
    if len(active) == 1:
        return active[0]
    return "mixed"


def _status_from_frames(
    frames: dict[str, pd.DataFrame],
    *,
    configured_provider: str,
    fallback_provider: str,
    provider_chain: list[str],
    symbols: list[str],
    requested_interval: str,
    requested_by: str,
    last_error: str = "",
    provider_diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source_breakdown = {"market": 0, "cached": 0, "fallback": 0, "unknown": 0}
    provider_breakdown = {"twelvedata": 0, "yahoo": 0, "synthetic": 0, "mixed": 0, "unknown": 0}
    live_symbols: list[str] = []
    cached_symbols: list[str] = []
    fallback_symbols: list[str] = []
    unknown_symbols: list[str] = []
    last_source = "unknown"

    for symbol in symbols:
        frame = frames.get(symbol, pd.DataFrame())
        source = frame_data_source(frame)
        provider_name = frame_provider_name(frame)
        last_source = source
        source_breakdown[source if source in source_breakdown else "unknown"] += 1
        provider_breakdown[provider_name if provider_name in provider_breakdown else "unknown"] += 1
        if source == "market":
            live_symbols.append(symbol)
        elif source == "cached":
            cached_symbols.append(symbol)
        elif source == "fallback":
            fallback_symbols.append(symbol)
        else:
            unknown_symbols.append(symbol)

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

    feed_status = classify_feed_status(
        status=status,
        last_source=last_source,
        source_breakdown=source_breakdown,
    )
    twelvedata_diag = dict((provider_diagnostics or {}).get("twelvedata") or {})
    response_status_code = None
    http_statuses = twelvedata_diag.get("http_statuses") or []
    if http_statuses:
        try:
            response_status_code = int(http_statuses[0])
        except Exception:
            response_status_code = None
    configured_service_name = str(SERVICE_NAME or RAILWAY_SERVICE_NAME or "").strip()
    provider_effective = _resolve_effective_provider(provider_breakdown)
    yahoo_diag = dict((provider_diagnostics or {}).get("yahoo") or {})
    effective_interval = str(requested_interval or "")
    if provider_effective in {"twelvedata", "mixed"}:
        effective_interval = str(twelvedata_diag.get("normalized_interval") or effective_interval)
    elif provider_effective == "yahoo":
        effective_interval = str(yahoo_diag.get("interval") or effective_interval)

    return {
        "provider": provider_effective,
        "provider_effective": provider_effective,
        "configured_provider": configured_provider,
        "fallback_provider": fallback_provider,
        "provider_chain": list(provider_chain),
        "provider_breakdown": provider_breakdown,
        "provider_diagnostics": dict(provider_diagnostics or {}),
        "status": status,
        "status_legacy": status,
        "feed_status": feed_status,
        "last_sync_at": _utc_now_iso(),
        "last_error": last_error,
        "last_source": last_source,
        "source_breakdown": source_breakdown,
        "symbols": symbols,
        "requested_symbols": list(symbols),
        "requested_interval": str(requested_interval or ""),
        "effective_interval": effective_interval,
        "live_symbols": live_symbols,
        "cached_symbols": cached_symbols,
        "fallback_symbols": fallback_symbols,
        "unknown_symbols": unknown_symbols,
        "requested_by": requested_by,
        "build_active": MARKET_DATA_BUILD_LABEL,
        "git_sha": str(RAILWAY_GIT_COMMIT_SHA or "").strip(),
        "source_commit_sha": str(APP_SOURCE_COMMIT_SHA or "").strip(),
        "build_timestamp": str(BUILD_TIMESTAMP or "").strip(),
        "service_name": configured_service_name or "unknown",
        "api_key_present": bool(twelvedata_diag.get("api_key_present")),
        "request_prepared": bool(twelvedata_diag.get("request_built")),
        "request_attempted": bool(twelvedata_diag.get("request_attempted")),
        "response_received": bool(twelvedata_diag.get("response_received")),
        "response_status_code": response_status_code,
        "last_stage": str(twelvedata_diag.get("last_stage") or ""),
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
    normalized_symbols = _normalize_symbols(symbols)
    configured_provider = _normalize_provider_name(provider or MARKET_DATA_PROVIDER, default="twelvedata")
    fallback_provider = _normalize_provider_name(MARKET_DATA_FALLBACK_PROVIDER, default="yahoo")
    active_provider_chain = _provider_chain(provider)

    frames: dict[str, pd.DataFrame] = {}
    error_messages: list[str] = []
    provider_diagnostics: dict[str, Any] = {}

    for provider_name in active_provider_chain:
        remaining = [symbol for symbol in normalized_symbols if symbol not in frames]
        if not remaining:
            break

        fresh_cache = _cached_frames(
            remaining,
            period=period,
            interval=interval,
            history_limit=history_limit,
            provider=provider_name,
            allow_stale=False,
        )
        for symbol, frame in fresh_cache.items():
            frames[symbol] = frame

        remaining = [symbol for symbol in normalized_symbols if symbol not in frames]
        if not remaining:
            continue

        live_frames, provider_errors, provider_debug = _fetch_provider_frames(
            provider_name,
            remaining,
            period=period,
            interval=interval,
            history_limit=history_limit,
        )
        if provider_debug:
            provider_diagnostics.update(dict(provider_debug or {}))
        if live_frames:
            _store_cached_frames(
                remaining,
                period=period,
                interval=interval,
                history_limit=history_limit,
                provider=provider_name,
                frames=live_frames,
            )
            frames.update(live_frames)

        error_messages.extend(f"{provider_name}: {message}" for message in provider_errors if message)

    if allow_stale:
        for provider_name in active_provider_chain:
            remaining = [symbol for symbol in normalized_symbols if symbol not in frames]
            if not remaining:
                break
            stale_frames = _cached_frames(
                remaining,
                period=period,
                interval=interval,
                history_limit=history_limit,
                provider=provider_name,
                allow_stale=True,
            )
            for symbol, frame in stale_frames.items():
                frames[symbol] = frame

    for symbol in normalized_symbols:
        if symbol not in frames:
            frames[symbol] = fallback_data(symbol, rows=max(180, int(history_limit)))

    last_error = " | ".join(message for message in error_messages if message).strip()
    return MarketDataBatchResult(
        frames=frames,
        status=_status_from_frames(
            frames,
            configured_provider=configured_provider,
            fallback_provider=fallback_provider,
            provider_chain=active_provider_chain,
            symbols=normalized_symbols,
            requested_interval=interval,
            requested_by=requested_by,
            last_error=last_error,
            provider_diagnostics=provider_diagnostics,
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
