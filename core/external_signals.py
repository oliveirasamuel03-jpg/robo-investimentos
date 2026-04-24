from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from typing import Any

from core.config import (
    EXTERNAL_SIGNAL_ALLOWED_SOURCES,
    EXTERNAL_SIGNAL_ALLOWED_TIMEFRAMES,
    EXTERNAL_SIGNAL_DEDUPE_SECONDS,
    EXTERNAL_SIGNAL_MAX_AGE_SECONDS,
    EXTERNAL_SIGNAL_SECRET,
    EXTERNAL_SIGNAL_WEBHOOK_ENABLED,
    SWING_VALIDATION_RECOMMENDED_WATCHLIST,
)

STATUS_DISABLED = "DISABLED"
STATUS_RECEIVED = "RECEIVED"
STATUS_ACCEPTED_FOR_AUDIT = "ACCEPTED_FOR_AUDIT"
STATUS_REJECTED = "REJECTED"
STATUS_EXPIRED = "EXPIRED"
STATUS_DUPLICATE = "DUPLICATE"
STATUS_IGNORED = "IGNORED"

REQUIRED_FIELDS = (
    "source",
    "strategy",
    "symbol",
    "timeframe",
    "side",
    "alert_price",
    "score",
    "ts",
)
VALID_SIDES = {"BUY", "SELL", "LONG", "SHORT"}
SENSITIVE_KEYS = {"token", "secret", "password", "api_key", "apikey", "authorization"}
RECENT_EVENTS_LIMIT = 20


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    return utc_now().isoformat()


def csv_values(raw: str | None, *, upper: bool = False) -> list[str]:
    values: list[str] = []
    for item in str(raw or "").split(","):
        normalized = item.strip()
        if not normalized:
            continue
        normalized = normalized.upper() if upper else normalized.lower()
        if normalized not in values:
            values.append(normalized)
    return values


def configured_external_signal_state() -> dict[str, Any]:
    allowed_sources = csv_values(EXTERNAL_SIGNAL_ALLOWED_SOURCES)
    allowed_timeframes = csv_values(EXTERNAL_SIGNAL_ALLOWED_TIMEFRAMES)
    return {
        "enabled": bool(EXTERNAL_SIGNAL_WEBHOOK_ENABLED),
        "webhook_configured": bool(EXTERNAL_SIGNAL_WEBHOOK_ENABLED and EXTERNAL_SIGNAL_SECRET and allowed_sources),
        "allowed_sources": ",".join(allowed_sources),
        "allowed_timeframes": ",".join(allowed_timeframes),
        "max_age_seconds": int(EXTERNAL_SIGNAL_MAX_AGE_SECONDS),
        "dedupe_seconds": int(EXTERNAL_SIGNAL_DEDUPE_SECONDS),
        "audit_only": True,
    }


def default_external_signal_state(reason: str = "External signal webhook disabled.") -> dict[str, Any]:
    return {
        **configured_external_signal_state(),
        "last_ts": None,
        "last_received_at": None,
        "last_source": "",
        "last_strategy": "",
        "last_symbol": "",
        "last_side": "",
        "last_timeframe": "",
        "last_alert_price": None,
        "last_score": 0.0,
        "last_status": STATUS_DISABLED,
        "last_reason": reason,
        "last_dedupe_key": "",
        "recent_events": [],
    }


def safe_payload(payload: dict[str, Any] | None) -> dict[str, Any]:
    def sanitize(value: Any, key: str = "") -> Any:
        if key.lower() in SENSITIVE_KEYS:
            return "[redacted]"
        if isinstance(value, dict):
            return {str(item_key): sanitize(item_value, str(item_key)) for item_key, item_value in value.items()}
        if isinstance(value, list):
            return [sanitize(item_value) for item_value in value[:20]]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    return sanitize(payload if isinstance(payload, dict) else {})


def parse_signal_timestamp(raw: Any) -> datetime | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def normalize_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    if score > 1.0 and score <= 100.0:
        score = score / 100.0
    return round(max(0.0, min(1.0, score)), 4)


def normalize_price(value: Any) -> float | None:
    try:
        price = float(value)
    except (TypeError, ValueError):
        return None
    if price <= 0:
        return None
    return price


def build_dedupe_key(*, source: str, strategy: str, symbol: str, timeframe: str, side: str, ts: str) -> str:
    raw_key = "|".join(
        [
            source.strip().lower(),
            strategy.strip().lower(),
            symbol.strip().upper(),
            timeframe.strip().lower(),
            side.strip().upper(),
            ts.strip(),
        ]
    )
    return sha256(raw_key.encode("utf-8")).hexdigest()[:32]


def _watchlist_from_state(state: dict[str, Any] | None) -> set[str]:
    trader_state = dict((state or {}).get("trader", {}) or {})
    watchlist = trader_state.get("watchlist") or SWING_VALIDATION_RECOMMENDED_WATCHLIST
    return {str(item or "").strip().upper() for item in watchlist if str(item or "").strip()}


def _recent_duplicate_found(recent_events: list[dict[str, Any]], dedupe_key: str, now: datetime) -> bool:
    for event in recent_events:
        if not isinstance(event, dict):
            continue
        if str(event.get("dedupe_key") or "") != dedupe_key:
            continue
        received_at = parse_signal_timestamp(event.get("received_at"))
        if received_at is None:
            continue
        age_seconds = (now - received_at).total_seconds()
        if age_seconds <= int(EXTERNAL_SIGNAL_DEDUPE_SECONDS):
            return True
    return False


def _event_payload(
    *,
    payload: dict[str, Any] | None,
    status: str,
    reason: str,
    received_at: datetime,
    source: str = "",
    strategy: str = "",
    symbol: str = "",
    timeframe: str = "",
    side: str = "",
    alert_price: float | None = None,
    score: float = 0.0,
    ts: str | None = None,
    dedupe_key: str = "",
) -> dict[str, Any]:
    return {
        "source": source,
        "strategy": strategy,
        "symbol": symbol,
        "timeframe": timeframe,
        "side": side,
        "alert_price": alert_price,
        "score": round(float(score or 0.0), 4),
        "ts": ts,
        "received_at": received_at.isoformat(),
        "status": status,
        "reason": reason,
        "raw_payload_safe": safe_payload(payload),
        "dedupe_key": dedupe_key,
        "audit_only": True,
        "trade_approved": False,
        "execution_authority": False,
    }


def validate_external_signal_payload(
    payload: dict[str, Any] | None,
    *,
    state: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    current_time = (now or utc_now()).astimezone(timezone.utc)
    raw_payload = dict(payload) if isinstance(payload, dict) else {}

    if not bool(EXTERNAL_SIGNAL_WEBHOOK_ENABLED):
        return _event_payload(
            payload=raw_payload,
            status=STATUS_DISABLED,
            reason="External signal webhook disabled.",
            received_at=current_time,
        )

    if not EXTERNAL_SIGNAL_SECRET:
        return _event_payload(
            payload=raw_payload,
            status=STATUS_REJECTED,
            reason="external_signal_secret_not_configured",
            received_at=current_time,
        )

    if str(raw_payload.get("token") or "") != str(EXTERNAL_SIGNAL_SECRET):
        return _event_payload(
            payload=raw_payload,
            status=STATUS_REJECTED,
            reason="invalid_token",
            received_at=current_time,
        )

    def missing_value(value: Any) -> bool:
        return value is None or (isinstance(value, str) and value.strip() == "")

    missing = [field for field in REQUIRED_FIELDS if missing_value(raw_payload.get(field))]
    if missing:
        return _event_payload(
            payload=raw_payload,
            status=STATUS_REJECTED,
            reason=f"missing_required_fields:{','.join(missing)}",
            received_at=current_time,
        )

    source = str(raw_payload.get("source") or "").strip().lower()
    strategy = str(raw_payload.get("strategy") or "").strip()
    symbol = str(raw_payload.get("symbol") or "").strip().upper()
    timeframe = str(raw_payload.get("timeframe") or "").strip().lower()
    side = str(raw_payload.get("side") or "").strip().upper()
    score = normalize_score(raw_payload.get("score"))
    alert_price = normalize_price(raw_payload.get("alert_price"))
    event_time = parse_signal_timestamp(raw_payload.get("ts"))

    allowed_sources = set(csv_values(EXTERNAL_SIGNAL_ALLOWED_SOURCES))
    if not allowed_sources:
        return _event_payload(
            payload=raw_payload,
            status=STATUS_REJECTED,
            reason="no_allowed_sources_configured",
            received_at=current_time,
            source=source,
            strategy=strategy,
            symbol=symbol,
            timeframe=timeframe,
            side=side,
            alert_price=alert_price,
            score=score,
        )
    if source not in allowed_sources:
        return _event_payload(
            payload=raw_payload,
            status=STATUS_REJECTED,
            reason="source_not_allowed",
            received_at=current_time,
            source=source,
            strategy=strategy,
            symbol=symbol,
            timeframe=timeframe,
            side=side,
            alert_price=alert_price,
            score=score,
        )

    if event_time is None:
        return _event_payload(
            payload=raw_payload,
            status=STATUS_REJECTED,
            reason="invalid_timestamp",
            received_at=current_time,
            source=source,
            strategy=strategy,
            symbol=symbol,
            timeframe=timeframe,
            side=side,
            alert_price=alert_price,
            score=score,
        )

    age_seconds = (current_time - event_time).total_seconds()
    if age_seconds > int(EXTERNAL_SIGNAL_MAX_AGE_SECONDS):
        return _event_payload(
            payload=raw_payload,
            status=STATUS_EXPIRED,
            reason="signal_expired",
            received_at=current_time,
            source=source,
            strategy=strategy,
            symbol=symbol,
            timeframe=timeframe,
            side=side,
            alert_price=alert_price,
            score=score,
            ts=event_time.isoformat(),
        )

    if symbol not in _watchlist_from_state(state):
        return _event_payload(
            payload=raw_payload,
            status=STATUS_REJECTED,
            reason="symbol_not_in_watchlist",
            received_at=current_time,
            source=source,
            strategy=strategy,
            symbol=symbol,
            timeframe=timeframe,
            side=side,
            alert_price=alert_price,
            score=score,
            ts=event_time.isoformat(),
        )

    allowed_timeframes = set(csv_values(EXTERNAL_SIGNAL_ALLOWED_TIMEFRAMES))
    if timeframe not in allowed_timeframes:
        return _event_payload(
            payload=raw_payload,
            status=STATUS_REJECTED,
            reason="timeframe_not_allowed",
            received_at=current_time,
            source=source,
            strategy=strategy,
            symbol=symbol,
            timeframe=timeframe,
            side=side,
            alert_price=alert_price,
            score=score,
            ts=event_time.isoformat(),
        )

    if side not in VALID_SIDES:
        return _event_payload(
            payload=raw_payload,
            status=STATUS_REJECTED,
            reason="side_not_allowed",
            received_at=current_time,
            source=source,
            strategy=strategy,
            symbol=symbol,
            timeframe=timeframe,
            side=side,
            alert_price=alert_price,
            score=score,
            ts=event_time.isoformat(),
        )

    if alert_price is None:
        return _event_payload(
            payload=raw_payload,
            status=STATUS_REJECTED,
            reason="invalid_alert_price",
            received_at=current_time,
            source=source,
            strategy=strategy,
            symbol=symbol,
            timeframe=timeframe,
            side=side,
            score=score,
            ts=event_time.isoformat(),
        )

    dedupe_key = build_dedupe_key(
        source=source,
        strategy=strategy,
        symbol=symbol,
        timeframe=timeframe,
        side=side,
        ts=event_time.isoformat(),
    )
    recent_events = list(dict((state or {}).get("external_signal", {}) or {}).get("recent_events", []) or [])
    if _recent_duplicate_found(recent_events, dedupe_key, current_time):
        return _event_payload(
            payload=raw_payload,
            status=STATUS_DUPLICATE,
            reason="duplicate_signal",
            received_at=current_time,
            source=source,
            strategy=strategy,
            symbol=symbol,
            timeframe=timeframe,
            side=side,
            alert_price=alert_price,
            score=score,
            ts=event_time.isoformat(),
            dedupe_key=dedupe_key,
        )

    return _event_payload(
        payload=raw_payload,
        status=STATUS_ACCEPTED_FOR_AUDIT,
        reason="accepted_for_audit_only_no_trade_authority",
        received_at=current_time,
        source=source,
        strategy=strategy,
        symbol=symbol,
        timeframe=timeframe,
        side=side,
        alert_price=alert_price,
        score=score,
        ts=event_time.isoformat(),
        dedupe_key=dedupe_key,
    )


def build_external_signal_state_update(
    event: dict[str, Any],
    *,
    previous_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    external_state = dict((previous_state or {}).get("external_signal", {}) or {})
    recent_events = [
        item
        for item in list(external_state.get("recent_events", []) or [])
        if isinstance(item, dict)
    ]
    recent_events.append(dict(event))
    recent_events = recent_events[-RECENT_EVENTS_LIMIT:]

    return {
        **configured_external_signal_state(),
        "last_ts": event.get("ts"),
        "last_received_at": event.get("received_at"),
        "last_source": event.get("source") or "",
        "last_strategy": event.get("strategy") or "",
        "last_symbol": event.get("symbol") or "",
        "last_side": event.get("side") or "",
        "last_timeframe": event.get("timeframe") or "",
        "last_alert_price": event.get("alert_price"),
        "last_score": float(event.get("score", 0.0) or 0.0),
        "last_status": event.get("status") or STATUS_IGNORED,
        "last_reason": event.get("reason") or "",
        "last_dedupe_key": event.get("dedupe_key") or "",
        "recent_events": recent_events,
    }


def process_external_signal_payload(
    payload: dict[str, Any] | None,
    *,
    state: dict[str, Any] | None = None,
    now: datetime | None = None,
    persist: bool = True,
) -> dict[str, Any]:
    from core.state_store import load_bot_state, log_event, update_external_signal_status

    current_state = state or load_bot_state()
    event = validate_external_signal_payload(payload, state=current_state, now=now)
    update_payload = build_external_signal_state_update(event, previous_state=current_state)

    if persist:
        update_external_signal_status(update_payload)

    status = str(event.get("status") or STATUS_IGNORED)
    symbol = str(event.get("symbol") or "none")
    source = str(event.get("source") or "none")
    reason = str(event.get("reason") or "")
    log_event(
        "INFO" if status == STATUS_ACCEPTED_FOR_AUDIT else "WARNING",
        (
            "[external_signal_summary] "
            f"status={status.lower()};source={source};symbol={symbol};"
            f"audit_only=1;trade_authority=0;reason={reason}"
        ),
    )
    if status in {STATUS_REJECTED, STATUS_EXPIRED, STATUS_DUPLICATE, STATUS_DISABLED}:
        log_event("WARNING", f"[external_signal_reject_reason] status={status.lower()};reason={reason}")

    return {"event": event, "state_update": update_payload}
