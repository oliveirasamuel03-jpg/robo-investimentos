from __future__ import annotations

from datetime import datetime, timezone

from core.config import (
    ALERT_FEED_FALLBACK_MAX_MINUTES,
    ALERT_HEARTBEAT_MAX_DELAY_SECONDS,
    ALERT_MAX_CONSECUTIVE_ERRORS,
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_iso_datetime(value: object) -> datetime | None:
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


def isoformat_or_empty(value: datetime | None) -> str:
    return value.astimezone(timezone.utc).isoformat() if value is not None else ""


def heartbeat_age_seconds(worker_heartbeat: object, now: datetime | None = None) -> int | None:
    heartbeat_at = parse_iso_datetime(worker_heartbeat)
    if heartbeat_at is None:
        return None

    current_time = now or utc_now()
    return max(0, int((current_time - heartbeat_at).total_seconds()))


def evaluate_production_health(state: dict | None, now: datetime | None = None) -> dict:
    current_time = now or utc_now()
    payload = state or {}
    production_state = payload.get("production", {}) or {}
    market_state = payload.get("market_data", {}) or {}
    broker_state = payload.get("broker", {}) or {}

    worker_status = str(payload.get("worker_status") or "offline").strip().lower()
    heartbeat_age = heartbeat_age_seconds(payload.get("worker_heartbeat"), now=current_time)
    last_execution_at = str(production_state.get("last_execution_at") or payload.get("last_run_at") or "")
    last_success_at = str(production_state.get("last_success_at") or "")
    feed_status = str(market_state.get("status") or "unknown").strip().lower()
    broker_status = str(broker_state.get("status") or "unknown").strip().lower()
    consecutive_errors = max(0, int(production_state.get("consecutive_errors", 0) or 0))

    fallback_since_at = parse_iso_datetime(
        production_state.get("fallback_since_at") or market_state.get("fallback_since_at")
    )
    fallback_age_minutes = (
        max(0, int((current_time - fallback_since_at).total_seconds() // 60))
        if fallback_since_at is not None
        else 0
    )

    level_rank = {"healthy": 0, "warning": 1, "critical": 2}
    health_level = "healthy"
    health_reason = "healthy"
    reasons: list[str] = []

    def raise_level(level: str, reason: str, message: str) -> None:
        nonlocal health_level, health_reason
        if level_rank[level] > level_rank[health_level]:
            health_level = level
            health_reason = reason
        reasons.append(message)

    if worker_status == "error":
        raise_level("critical", "worker_error", "Worker em estado de erro.")
    elif worker_status != "online":
        raise_level("warning", "worker_offline", "Worker fora do estado online.")

    if heartbeat_age is None:
        raise_level("warning", "missing_heartbeat", "Sem heartbeat recente do worker.")
    elif heartbeat_age >= ALERT_HEARTBEAT_MAX_DELAY_SECONDS * 2:
        raise_level("critical", "heartbeat_delayed", f"Heartbeat atrasado ha {heartbeat_age}s.")
    elif heartbeat_age >= ALERT_HEARTBEAT_MAX_DELAY_SECONDS:
        raise_level("warning", "heartbeat_delayed", f"Heartbeat atrasado ha {heartbeat_age}s.")

    if consecutive_errors >= ALERT_MAX_CONSECUTIVE_ERRORS:
        raise_level(
            "critical",
            "consecutive_errors",
            f"Falhas consecutivas acima do limite: {consecutive_errors}.",
        )
    elif consecutive_errors > 0:
        raise_level("warning", "consecutive_errors", f"Falhas consecutivas atuais: {consecutive_errors}.")

    is_feed_in_fallback = feed_status in {"error", "fallback"} or str(market_state.get("last_source") or "").lower() == "fallback"
    if is_feed_in_fallback and fallback_age_minutes >= ALERT_FEED_FALLBACK_MAX_MINUTES * 2:
        raise_level(
            "critical",
            "feed_fallback",
            f"Feed em fallback ha {fallback_age_minutes} min.",
        )
    elif is_feed_in_fallback and fallback_age_minutes >= ALERT_FEED_FALLBACK_MAX_MINUTES:
        raise_level(
            "warning",
            "feed_fallback",
            f"Feed em fallback ha {fallback_age_minutes} min.",
        )
    elif is_feed_in_fallback:
        raise_level("warning", "feed_fallback", "Feed em fallback recente.")

    broker_note = (
        "Broker em modo simulado (paper). Nenhuma ordem real sera enviada."
        if broker_status == "paper"
        else f"Broker em estado {broker_status or 'desconhecido'}."
    )

    if health_level == "healthy":
        reasons.append("Sistema saudavel.")

    if broker_note not in reasons:
        reasons.append(broker_note)

    health_message = " ".join(reasons).strip()

    return {
        "worker_status": worker_status,
        "heartbeat_age_seconds": heartbeat_age,
        "last_execution_at": last_execution_at,
        "last_success_at": last_success_at,
        "feed_status": feed_status,
        "broker_status": broker_status,
        "consecutive_errors": consecutive_errors,
        "fallback_since_at": isoformat_or_empty(fallback_since_at),
        "fallback_age_minutes": fallback_age_minutes,
        "health_level": health_level,
        "health_reason": health_reason,
        "health_message": health_message,
        "last_health_at": current_time.isoformat(),
    }
