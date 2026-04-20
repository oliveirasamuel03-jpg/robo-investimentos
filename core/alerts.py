from __future__ import annotations

import json
import smtplib
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from core.config import (
    ALERT_COOLDOWN_MINUTES,
    ALERT_EMAIL_ENABLED,
    ALERT_EMAIL_FROM,
    ALERT_EMAIL_PROVIDER,
    ALERT_EMAIL_TO,
    ALERT_SEND_RECOVERY_EMAIL,
    APP_TITLE,
    PRODUCTION_MODE,
    RESEND_API_BASE,
    RESEND_API_KEY,
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


def _email_provider() -> str:
    return str(ALERT_EMAIL_PROVIDER or "smtp").strip().lower() or "smtp"


def _smtp_ready() -> bool:
    return bool(ALERT_EMAIL_TO and SMTP_HOST and SMTP_USERNAME and SMTP_PASSWORD)


def _resend_ready() -> bool:
    return bool(ALERT_EMAIL_TO and ALERT_EMAIL_FROM and RESEND_API_KEY and RESEND_API_BASE)


def _provider_readiness_reason(provider: str) -> str | None:
    if provider == "smtp":
        return None if _smtp_ready() else "smtp_not_configured"
    if provider == "resend":
        return None if _resend_ready() else "resend_not_configured"
    return "unknown_alert_provider"


def _smtp_from_address() -> str:
    return ALERT_EMAIL_FROM or SMTP_USERNAME or ALERT_EMAIL_TO


def _resend_user_agent() -> str:
    app_name = str(APP_TITLE or "Trade Ops Desk").strip() or "Trade Ops Desk"
    safe_name = app_name.replace(" ", "-")
    return f"{safe_name}/1.0"


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
    provider = _email_provider()
    provider_reason = _provider_readiness_reason(provider)

    if force:
        if provider_reason is not None:
            return {"allowed": False, "reason": provider_reason, "next_alert_eligible_at": "", "provider": provider}
        return {"allowed": True, "reason": "forced", "next_alert_eligible_at": "", "provider": provider}
    if not PRODUCTION_MODE:
        return {
            "allowed": False,
            "reason": "production_mode_disabled",
            "next_alert_eligible_at": "",
            "provider": provider,
        }
    if not ALERT_EMAIL_ENABLED:
        return {"allowed": False, "reason": "alert_email_disabled", "next_alert_eligible_at": "", "provider": provider}

    if provider_reason is not None:
        return {"allowed": False, "reason": provider_reason, "next_alert_eligible_at": "", "provider": provider}

    next_eligible_at = _next_alert_eligible_at(production_state.get("last_alert_sent_at"), now=current_time)
    if next_eligible_at:
        return {
            "allowed": False,
            "reason": "cooldown_active",
            "next_alert_eligible_at": next_eligible_at,
            "provider": provider,
        }

    return {"allowed": True, "reason": "ready", "next_alert_eligible_at": "", "provider": provider}


def _update_alert_state(updates: dict) -> dict:
    state = load_bot_state()
    production_state = state.get("production", {}) or {}
    for key, value in updates.items():
        production_state[key] = value
    state["production"] = production_state
    save_bot_state(state)
    return state


def _send_via_smtp(subject: str, body: str) -> None:
    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = _smtp_from_address()
    message["To"] = ALERT_EMAIL_TO
    message.set_content(body)

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=SMTP_TIMEOUT_SECONDS) as server:
        if SMTP_USE_TLS:
            server.starttls()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        server.send_message(message)


def _send_via_resend(subject: str, body: str) -> str:
    payload = {
        "from": ALERT_EMAIL_FROM,
        "to": [ALERT_EMAIL_TO],
        "subject": subject,
        "text": body,
    }
    request = Request(
        RESEND_API_BASE,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {RESEND_API_KEY}",
            "Content-Type": "application/json",
            "User-Agent": _resend_user_agent(),
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=SMTP_TIMEOUT_SECONDS) as response:
            response_body = response.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Resend HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Resend connection error: {exc.reason}") from exc

    try:
        parsed = json.loads(response_body) if response_body else {}
    except json.JSONDecodeError:
        parsed = {}

    return str(parsed.get("id") or "")


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
    provider = str(eligibility.get("provider") or _email_provider())

    if not eligibility.get("allowed", False):
        updates = {
            "next_alert_eligible_at": str(eligibility.get("next_alert_eligible_at") or ""),
            "last_alert_provider": provider,
        }
        _update_alert_state(updates)
        return {"sent": False, "reason": str(eligibility.get("reason") or "blocked"), **eligibility}

    try:
        delivery_id = ""
        if provider == "smtp":
            _send_via_smtp(subject, body)
        elif provider == "resend":
            delivery_id = _send_via_resend(subject, body)
        else:
            raise RuntimeError(f"Provider de alerta desconhecido: {provider}")
    except Exception as exc:
        log_event("ERROR", f"Falha ao enviar alerta por email: {exc}")
        _update_alert_state(
            {
                "last_alert_error": str(exc),
                "last_alert_provider": provider,
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
            "last_alert_provider": provider,
            "last_alert_error": "",
            "next_alert_eligible_at": next_eligible_at,
        }
    )
    log_event(
        "WARNING",
        f"Alerta enviado por email via {provider}: {subject}" + (f" ({delivery_id})" if delivery_id else ""),
    )
    return {"sent": True, "reason": "sent", "next_alert_eligible_at": next_eligible_at, "provider": provider}


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
