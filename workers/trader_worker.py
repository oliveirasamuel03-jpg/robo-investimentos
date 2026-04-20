from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

from core.alerts import send_email_alert, send_final_validation_email, send_recovery_email
from core.config import ALERT_EMAIL_ENABLED, PRODUCTION_MODE
from core.production_monitor import evaluate_production_health
from core.retention import run_retention_job, should_run_retention_job
from core.state_store import (
    load_bot_state,
    log_event,
    save_bot_state,
    update_production_status,
    update_retention_status,
    update_validation_status,
    update_worker_heartbeat,
)
from core.swing_validation import refresh_swing_validation_cycle
from engines.trader_engine import run_trader_cycle

SLEEP_SECONDS = 60
PAUSED_SLEEP_SECONDS = 5


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def update_runtime_state(last_action: str, next_run_delta_seconds: int) -> None:
    state = load_bot_state()

    now = datetime.now(timezone.utc)

    state["last_action"] = last_action
    state["last_run_at"] = now.isoformat()
    state["next_run_at"] = (now + timedelta(seconds=next_run_delta_seconds)).isoformat()
    state["worker_status"] = "online"
    state["worker_heartbeat"] = now.isoformat()

    save_bot_state(state)


def mark_error(message: str) -> None:
    state = load_bot_state()

    now = datetime.now(timezone.utc)

    state["last_action"] = f"Erro: {message}"
    state["last_run_at"] = now.isoformat()
    state["next_run_at"] = (now + timedelta(seconds=SLEEP_SECONDS)).isoformat()
    state["worker_status"] = "error"
    state["worker_heartbeat"] = now.isoformat()

    save_bot_state(state)


def build_action_text(result: dict) -> str:
    cycle_result = result.get("cycle_result", {}) if isinstance(result, dict) else {}
    trades_executed = int(cycle_result.get("trades_executed", 0))

    if trades_executed > 0:
        return f"{trades_executed} trade(s) executado(s) nesta rodada"

    return "Rodada concluida sem novas operacoes"


def _send_health_alert(state: dict, health_payload: dict, *, alert_type: str, current_time: datetime) -> None:
    subject = f"[Trade Ops Desk] Alerta de producao: {alert_type}"
    body = (
        "O monitoramento do trader detectou uma condicao de atencao.\n\n"
        f"Health level: {health_payload.get('health_level')}\n"
        f"Motivo: {health_payload.get('health_reason')}\n"
        f"Mensagem: {health_payload.get('health_message')}\n"
        f"Worker: {health_payload.get('worker_status')}\n"
        f"Heartbeat age (s): {health_payload.get('heartbeat_age_seconds')}\n"
        f"Feed: {health_payload.get('feed_status')}\n"
        f"Broker: {health_payload.get('broker_status')}\n"
        f"Falhas consecutivas: {health_payload.get('consecutive_errors')}\n"
        f"Ultima execucao: {health_payload.get('last_execution_at')}\n"
        f"Ultimo sucesso: {health_payload.get('last_success_at')}\n"
        f"Data UTC: {current_time.isoformat()}\n"
    )
    send_email_alert(subject, body, alert_type=alert_type, state=state, now=current_time)


def _log_cycle_health(health_payload: dict) -> None:
    fallback_flag = 1 if str(health_payload.get("feed_status") or "").lower() == "error" else 0
    message = (
        "[cycle_health] "
        f"health_level={str(health_payload.get('health_level') or 'healthy').lower()};"
        f"feed_status={str(health_payload.get('feed_status') or 'unknown').lower()};"
        f"broker_status={str(health_payload.get('broker_status') or 'paper').lower()};"
        f"worker_status={str(health_payload.get('worker_status') or 'online').lower()};"
        f"consecutive_errors={int(health_payload.get('consecutive_errors') or 0)};"
        f"fallback={fallback_flag}"
    )
    log_event("INFO", message)


def _log_validation_cycle(validation_report: dict) -> None:
    metrics = dict(validation_report.get("metrics", {}) or {})
    rejections = dict(metrics.get("signal_rejections", {}) or {})
    current_context = dict(validation_report.get("current_market_context", {}) or {})
    message = (
        "[validation_signal] "
        f"day={int(validation_report.get('validation_day_number', 1) or 1)};"
        f"phase={str(validation_report.get('validation_phase') or '').lower().replace(' ', '_')};"
        f"approved={int(metrics.get('signals_approved', 0) or 0)};"
        f"rejected={int(metrics.get('signals_rejected', 0) or 0)};"
        f"against_trend={int(metrics.get('against_trend_entries', 0) or 0)};"
        f"weak_score={int(rejections.get('weak_score', 0) or 0)};"
        f"feed_unreliable={int(rejections.get('feed_unreliable', 0) or 0)};"
        f"context={str(current_context.get('market_context_status') or 'NEUTRO').upper()};"
        f"context_blocked={int(metrics.get('context_blocked_signals', 0) or 0)}"
    )
    log_event("INFO", message)


def _maybe_send_final_validation_email(validation_report: dict, *, current_time: datetime) -> None:
    if int(validation_report.get("validation_day_number", 0) or 0) < 10:
        return
    if not validation_report.get("final_validation_grade"):
        return

    state = load_bot_state()
    validation_state = state.get("validation", {}) or {}
    if bool(validation_state.get("final_email_sent", False)):
        return

    try:
        result = send_final_validation_email(validation_report)
    except Exception as exc:
        log_event("ERROR", f"Falha ao enviar email final da validacao swing: {exc}")
        return

    if result.get("sent"):
        update_validation_status(
            {
                "final_email_sent": True,
                "final_email_sent_at": current_time.isoformat(),
            }
        )
        log_event("INFO", "Email final da validacao swing enviado com sucesso.")
    else:
        log_event(
            "WARNING",
            f"Email final da validacao swing nao enviado: {result.get('reason') or 'motivo nao informado'}",
        )


def _run_daily_retention_maintenance() -> None:
    state = load_bot_state()
    if not should_run_retention_job(state):
        return

    retention_state = state.get("retention", {}) or {}
    current_time = datetime.now(timezone.utc)

    try:
        summary = run_retention_job(
            now=current_time,
            retention_days=int(retention_state.get("retention_days") or 60),
            archive_trader_orders=bool(retention_state.get("archive_trader_orders", False)),
        )
        update_retention_status(summary)
        last_summary = summary.get("last_summary", {}) or {}
        log_event(
            "INFO",
            "Retencao executada com sucesso: "
            f"relatorios={int(last_summary.get('trade_reports_archived_rows', 0) or 0)}, "
            f"logs={int(last_summary.get('bot_logs_archived_rows', 0) or 0)}, "
            f"weeklies={int(last_summary.get('weekly_reports_generated', 0) or 0)}",
        )
    except Exception as exc:
        update_retention_status(
            {
                "last_run_at": current_time.isoformat(),
                "last_error": str(exc),
                "last_error_at": current_time.isoformat(),
            }
        )
        log_event("ERROR", f"Falha na retencao automatica: {exc}")


def _refresh_production_monitor(*, cycle_success: bool, exception_message: str = "") -> dict:
    current_time = datetime.now(timezone.utc)
    state = load_bot_state()
    previous_production = state.get("production", {}) or {}
    previous_health_level = str(previous_production.get("health_level") or "healthy").lower()

    updates = {
        "enabled": PRODUCTION_MODE,
        "alert_email_enabled": ALERT_EMAIL_ENABLED,
        "last_execution_at": current_time.isoformat(),
    }

    if cycle_success:
        updates.update(
            {
                "last_success_at": current_time.isoformat(),
                "consecutive_errors": 0,
                "last_error": "",
                "last_error_at": "",
                "last_exception": "",
            }
        )
    else:
        updates.update(
            {
                "consecutive_errors": max(0, int(previous_production.get("consecutive_errors", 0) or 0)) + 1,
                "last_error": exception_message,
                "last_error_at": current_time.isoformat(),
                "last_exception": exception_message,
            }
        )

    update_production_status(updates)
    updated_state = load_bot_state()
    health_payload = evaluate_production_health(updated_state, now=current_time)
    update_production_status(health_payload)
    monitored_state = load_bot_state()

    if exception_message:
        _send_health_alert(
            monitored_state,
            health_payload,
            alert_type="critical_exception",
            current_time=current_time,
        )
        return health_payload

    if health_payload.get("health_level") in {"warning", "critical"} and health_payload.get("health_reason") not in {
        "",
        "healthy",
    }:
        _send_health_alert(
            monitored_state,
            health_payload,
            alert_type=str(health_payload.get("health_reason") or "health_warning"),
            current_time=current_time,
        )
    elif previous_health_level in {"warning", "critical"} and health_payload.get("health_level") == "healthy":
        send_recovery_email(monitored_state, health_payload=health_payload, now=current_time)

    return health_payload


def worker_loop() -> None:
    log_event("INFO", "Worker 24h iniciado")

    while True:
        try:
            current_time = datetime.now(timezone.utc)
            update_worker_heartbeat("online")
            state = load_bot_state()

            if state.get("bot_status") != "RUNNING":
                update_runtime_state(
                    last_action="Robo pausado. Aguardando ativacao.",
                    next_run_delta_seconds=PAUSED_SLEEP_SECONDS,
                )
                _refresh_production_monitor(cycle_success=True)
                _run_daily_retention_maintenance()
                validation_report = refresh_swing_validation_cycle(now=current_time)
                _maybe_send_final_validation_email(validation_report, current_time=current_time)
                time.sleep(PAUSED_SLEEP_SECONDS)
                continue

            result = run_trader_cycle()
            action_text = build_action_text(result)

            update_runtime_state(
                last_action=action_text,
                next_run_delta_seconds=SLEEP_SECONDS,
            )

            log_event("INFO", action_text)
            health_payload = _refresh_production_monitor(cycle_success=True)
            _log_cycle_health(health_payload)
            _run_daily_retention_maintenance()
            validation_report = refresh_swing_validation_cycle(
                cycle_result=result.get("cycle_result", {}),
                now=current_time,
            )
            _log_validation_cycle(validation_report)
            _maybe_send_final_validation_email(validation_report, current_time=current_time)

        except Exception as exc:
            error_msg = str(exc)
            log_event("ERROR", f"Erro no worker: {error_msg}")
            mark_error(error_msg)
            health_payload = _refresh_production_monitor(cycle_success=False, exception_message=error_msg)
            _log_cycle_health(health_payload)
            _run_daily_retention_maintenance()
            validation_report = refresh_swing_validation_cycle(now=datetime.now(timezone.utc))
            _maybe_send_final_validation_email(validation_report, current_time=datetime.now(timezone.utc))

        time.sleep(SLEEP_SECONDS)


if __name__ == "__main__":
    worker_loop()
