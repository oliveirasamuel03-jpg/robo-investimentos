from __future__ import annotations

import smtplib
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage

from core.config import (
    ALERT_COOLDOWN_MINUTES,
    ALERT_EMAIL_ENABLED,
    ALERT_EMAIL_TO,
    ALERT_SEND_RECOVERY_EMAIL,
    APP_TITLE,
    PRODUCTION_MODE,
    SMTP_HOST,
    SMTP_PASSWORD,
    SMTP_PORT,
    SMTP_TIMEOUT_SECONDS,
    SMTP_USE_TLS,
    SMTP_USERNAME,
)
from core.production_monitor import parse_iso_datetime
from core.state_store import load_bot_state, log_event, save_bot_state


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _smtp_ready() -> bool:
    return bool(ALERT_EMAIL_TO and SMTP_HOST and SMTP_USERNAME and SMTP_PASSWORD)


def _next_alert_eligible_at(last_alert_sent_at: object, now: datetime | None = None) -> str:
    sent_at = parse_iso_datetime(last_alert_sent_at)
    if sent_at is None:
        return ""

    current_time = now or _utc_now()
    next_time = sent_at + timedelta(minutes=ALERT_COOLDOWN_MINUTES)
    if next_time <= current_time:
        return ""
    return next_time.isoformat()


def should_send_alert(
    state: dict | None = None,
    *,
    alert_type: str,
    now: datetime | None = None,
    force: bool = False,
) -> dict:
    current_time = now or _utc_now()
    payload = state or load_bot_state()
    production_state = payload.get("production", {}) or {}

    if force:
        return {"allowed": True, "reason": "forced", "next_alert_eligible_at": ""}
    if not PRODUCTION_MODE:
        return {"allowed": False, "reason": "production_mode_disabled", "next_alert_eligible_at": ""}
    if not ALERT_EMAIL_ENABLED:
        return {"allowed": False, "reason": "alert_email_disabled", "next_alert_eligible_at": ""}
    if not _smtp_ready():
        return {"allowed": False, "reason": "smtp_not_configured", "next_alert_eligible_at": ""}

    next_eligible_at = _next_alert_eligible_at(production_state.get("last_alert_sent_at"), now=current_time)
    if next_eligible_at:
        return {"allowed": False, "reason": "cooldown_active", "next_alert_eligible_at": next_eligible_at}

    return {"allowed": True, "reason": "ready", "next_alert_eligible_at": ""}


def _update_alert_state(updates: dict) -> dict:
    state = load_bot_state()
    production_state = state.get("production", {}) or {}
    for key, value in updates.items():
        production_state[key] = value
    state["production"] = production_state
    save_bot_state(state)
    return state


def send_email_alert(
    subject: str,
    body: str,
    *,
    alert_type: str = "generic",
    state: dict | None = None,
    now: datetime | None = None,
    force: bool = False,
) -> dict:
    current_time = now or _utc_now()
    payload = state or load_bot_state()
    eligibility = should_send_alert(payload, alert_type=alert_type, now=current_time, force=force)

    if not eligibility.get("allowed", False):
        updates = {
            "next_alert_eligible_at": str(eligibility.get("next_alert_eligible_at") or ""),
        }
        _update_alert_state(updates)
        return {"sent": False, "reason": str(eligibility.get("reason") or "blocked"), **eligibility}

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = SMTP_USERNAME or ALERT_EMAIL_TO
    message["To"] = ALERT_EMAIL_TO
    message.set_content(body)

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=SMTP_TIMEOUT_SECONDS) as server:
            if SMTP_USE_TLS:
                server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.send_message(message)
    except Exception as exc:
        log_event("ERROR", f"Falha ao enviar alerta por email: {exc}")
        _update_alert_state(
            {
                "last_alert_error": str(exc),
                "next_alert_eligible_at": "",
            }
        )
        return {"sent": False, "reason": str(exc), "next_alert_eligible_at": ""}

    next_eligible_at = (current_time + timedelta(minutes=ALERT_COOLDOWN_MINUTES)).isoformat()
    _update_alert_state(
        {
            "last_alert_sent_at": current_time.isoformat(),
            "last_alert_type": alert_type,
            "last_alert_subject": subject,
            "last_alert_error": "",
            "next_alert_eligible_at": next_eligible_at,
        }
    )
    log_event("WARNING", f"Alerta enviado por email: {subject}")
    return {"sent": True, "reason": "sent", "next_alert_eligible_at": next_eligible_at}


def send_recovery_email(
    state: dict | None = None,
    *,
    health_payload: dict | None = None,
    now: datetime | None = None,
) -> dict:
    current_time = now or _utc_now()
    payload = state or load_bot_state()
    production_state = payload.get("production", {}) or {}

    if not ALERT_SEND_RECOVERY_EMAIL:
        return {"sent": False, "reason": "recovery_disabled"}
    if not production_state.get("last_alert_sent_at"):
        return {"sent": False, "reason": "no_previous_alert"}

    last_recovery_at = parse_iso_datetime(production_state.get("last_recovery_email_at"))
    last_alert_at = parse_iso_datetime(production_state.get("last_alert_sent_at"))
    if last_recovery_at is not None and last_alert_at is not None and last_recovery_at >= last_alert_at:
        return {"sent": False, "reason": "recovery_already_sent"}

    current_health = health_payload or {}
    subject = f"[{APP_TITLE}] Recuperacao do ambiente de producao"
    body = (
        "O ambiente voltou ao estado saudavel.\n\n"
        f"Health level: {current_health.get('health_level', 'healthy')}\n"
        f"Mensagem: {current_health.get('health_message', 'Sistema saudavel.')}\n"
        f"Data UTC: {current_time.isoformat()}\n"
        f"Broker: {current_health.get('broker_status', 'paper')}\n"
        f"Feed: {current_health.get('feed_status', 'unknown')}\n"
    )
    result = send_email_alert(subject, body, alert_type="recovery", state=payload, now=current_time, force=True)
    if result.get("sent"):
        _update_alert_state({"last_recovery_email_at": current_time.isoformat()})
    return result
