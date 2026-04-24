from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

from core.alerts import send_email_alert, send_final_validation_email, send_recovery_email
from core.config import (
    ALERT_EMAIL_ENABLED,
    BUILD_TIMESTAMP,
    MARKET_DATA_BUILD_LABEL,
    MARKET_DATA_FALLBACK_PROVIDER,
    MARKET_DATA_PROVIDER,
    PRODUCTION_MODE,
    RAILWAY_GIT_COMMIT_SHA,
    SERVICE_NAME,
)
from core.email_reports import final_report_path_reachable, process_report_email_delivery
from core.market_data import build_feed_quality_snapshot
from core.production_monitor import evaluate_production_health
from core.retention import run_retention_job, should_run_retention_job
from core.state_store import (
    load_bot_state,
    log_event,
    persist_worker_cycle_state,
    save_bot_state,
    update_production_status,
    update_retention_status,
    update_validation_status,
)
from core.swing_validation import refresh_swing_validation_cycle
from engines.trader_engine import refresh_daily_loss_guard, run_trader_cycle

SLEEP_SECONDS = 60
PAUSED_SLEEP_SECONDS = 5
WORKER_RUNTIME_STARTED_AT = datetime.now(timezone.utc).isoformat()
WORKER_PROCESS_ROLE = "worker"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def emit_startup_marker(message: str) -> None:
    print(message, flush=True)
    log_event("INFO", message)


def update_runtime_state(last_action: str, next_run_delta_seconds: int) -> None:
    persist_worker_cycle_state(
        last_action=last_action,
        next_run_delta_seconds=next_run_delta_seconds,
        worker_status="online",
        runtime_started_at=WORKER_RUNTIME_STARTED_AT,
        process_role=WORKER_PROCESS_ROLE,
    )


def mark_error(message: str) -> None:
    persist_worker_cycle_state(
        last_action=f"Erro: {message}",
        next_run_delta_seconds=SLEEP_SECONDS,
        worker_status="error",
        market_data_payload={
            "requested_by": "worker_cycle",
            "build_active": MARKET_DATA_BUILD_LABEL,
            "git_sha": str(RAILWAY_GIT_COMMIT_SHA or ""),
            "build_timestamp": str(BUILD_TIMESTAMP or ""),
            "service_name": str(SERVICE_NAME or ""),
            "process_role": WORKER_PROCESS_ROLE,
            "last_stage": "worker_exception",
            "last_error": str(message or ""),
        },
        runtime_started_at=WORKER_RUNTIME_STARTED_AT,
        process_role=WORKER_PROCESS_ROLE,
    )


def build_action_text(result: dict) -> str:
    cycle_result = result.get("cycle_result", {}) if isinstance(result, dict) else {}
    trades_executed = int(cycle_result.get("trades_executed", 0))
    if bool(cycle_result.get("entries_blocked_by_daily_loss", False)):
        return "Entradas bloqueadas por limite de perda diaria"

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
    fallback_flag = 1 if str(health_payload.get("feed_status") or "").strip().upper() == "FALLBACK" else 0
    message = (
        "[cycle_health] "
        f"health_level={str(health_payload.get('health_level') or 'healthy').lower()};"
        f"provider={str(health_payload.get('provider') or 'unknown').lower()};"
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
    weak_score_count = int(
        rejections.get("score_below_minimum", rejections.get("weak_score", 0)) or 0
    )
    feed_blocked_count = int(
        rejections.get("feed_quality_blocked", 0) or 0
    ) + int(rejections.get("fallback_blocked", 0) or 0) + int(rejections.get("provider_unknown", 0) or 0)
    message = (
        "[validation_signal] "
        f"day={int(validation_report.get('validation_day_number', 1) or 1)};"
        f"phase={str(validation_report.get('validation_phase') or '').lower().replace(' ', '_')};"
        f"approved={int(metrics.get('signals_approved', 0) or 0)};"
        f"rejected={int(metrics.get('signals_rejected', 0) or 0)};"
        f"against_trend={int(metrics.get('against_trend_entries', 0) or 0)};"
        f"weak_score={weak_score_count};"
        f"feed_unreliable={feed_blocked_count};"
        f"context={str(current_context.get('market_context_status') or 'NEUTRO').upper()};"
        f"context_blocked={int(metrics.get('context_blocked_signals', 0) or 0)}"
    )
    log_event("INFO", message)


def _log_feed_quality_summary(market_data_status: dict | None) -> None:
    quality = build_feed_quality_snapshot(market_data_status)
    success_rate = quality.get("twelvedata_success_rate")
    success_rate_label = "-" if success_rate is None else f"{float(success_rate or 0.0) * 100:.1f}%"
    message = (
        "[feed_quality_summary] "
        f"status={str(quality.get('feed_status') or 'UNKNOWN').lower()};"
        f"provider={str(quality.get('provider_effective') or 'unknown').lower()};"
        f"success_rate={success_rate_label};"
        f"live={int(quality.get('live_count') or 0)}/{int(quality.get('total_symbols') or 0)};"
        f"fallback={int(quality.get('fallback_count') or 0)};"
        f"last_success={str(quality.get('last_success_at') or 'na')};"
        f"reason={str(quality.get('fallback_reason') or 'none')}"
    )
    log_event("INFO", message)


def _log_signal_quality_summary(validation_report: dict) -> None:
    metrics = dict(validation_report.get("metrics", {}) or {})
    consistency = dict(validation_report.get("consistency", {}) or {})
    approval_rate = consistency.get("signal_approval_rate")
    approval_rate_label = "-" if approval_rate is None else f"{float(approval_rate or 0.0) * 100:.1f}%"
    message = (
        "[signal_quality_summary] "
        f"approved={int(metrics.get('signals_approved', 0) or 0)};"
        f"rejected={int(metrics.get('signals_rejected', 0) or 0)};"
        f"approval_rate={approval_rate_label};"
        f"sample={str(consistency.get('sample_quality_label') or 'Sem leitura').lower()};"
        f"posture={str(consistency.get('operational_posture_label') or 'Indefinida').lower()};"
        f"watchlist={'coerente' if bool(consistency.get('watchlist_phase_aligned')) else 'fora_da_fase'};"
        f"reading={str(consistency.get('signal_quality_label') or 'Baixa').lower()}"
    )
    log_event("INFO", message)


def _log_signal_rejection_summary(validation_report: dict, cycle_result: dict | None) -> None:
    cycle_validation = dict((cycle_result or {}).get("validation_cycle", {}) or {})
    rejection_summary = dict(cycle_validation.get("rejection_summary", {}) or validation_report.get("rejection_quality", {}) or {})
    top_reason = str(rejection_summary.get("top_rejection_reason") or "")
    top_layer = str(rejection_summary.get("top_rejection_layer") or "")
    top_strategy = str(rejection_summary.get("top_rejection_strategy") or "")
    reason_breakdown = dict(rejection_summary.get("rejected_by_reason", {}) or {})
    layer_breakdown = dict(rejection_summary.get("rejected_by_layer", {}) or {})
    log_event(
        "INFO",
        (
            "[signal_rejection_summary] "
            f"rejected={int(rejection_summary.get('total_rejected_signals', 0) or 0)};"
            f"events={int(rejection_summary.get('total_rejection_events', 0) or 0)};"
            f"top_reason={top_reason or 'none'};"
            f"top_layer={top_layer or 'none'};"
            f"top_strategy={top_strategy or 'none'}"
        ),
    )
    if top_reason:
        log_event(
            "INFO",
            (
                "[signal_rejection_top_reason] "
                f"reason={top_reason};"
                f"count={int(reason_breakdown.get(top_reason, 0) or 0)}"
            ),
        )
    if layer_breakdown:
        summary = ",".join(
            f"{str(layer)}:{int(count or 0)}"
            for layer, count in sorted(layer_breakdown.items(), key=lambda item: int(item[1] or 0), reverse=True)
        )
        log_event("INFO", f"[signal_rejection_layer_summary] {summary}")


def _log_macro_alert_summary(cycle_result: dict | None) -> None:
    payload = dict(cycle_result or {})
    alert = dict(payload.get("macro_alert", {}) or {})
    cycle_validation = dict(payload.get("validation_cycle", {}) or {})
    rejections = dict(cycle_validation.get("rejections", {}) or {})
    macro_blocks = int(rejections.get("macro_alert_guard", 0) or 0)
    log_event(
        "INFO",
        (
            "[macro_alert_summary] "
            f"active={1 if bool(alert.get('macro_alert_active', False)) else 0};"
            f"level={str(alert.get('macro_alert_level') or 'LOW').lower()};"
            f"window={str(alert.get('macro_alert_window_status') or 'INACTIVE').lower()};"
            f"currency={str(alert.get('macro_alert_currency') or 'none').lower()};"
            f"blocks_new_entries={1 if bool(alert.get('macro_alert_blocks_new_entries', False)) else 0};"
            f"penalty={float(alert.get('macro_alert_penalty', 0.0) or 0.0):.4f};"
            f"guard_blocks={macro_blocks}"
        ),
    )
    if macro_blocks > 0:
        log_event(
            "WARNING",
            (
                "[macro_alert_guard_reason] "
                f"reason={str(alert.get('macro_alert_reason') or 'macro_risk_active')};"
                f"blocked_signals={macro_blocks}"
            ),
        )


def _log_cycle_summary(*, action_text: str, market_data_status: dict | None, validation_report: dict) -> None:
    metrics = dict(validation_report.get("metrics", {}) or {})
    consistency = dict(validation_report.get("consistency", {}) or {})
    status = dict(market_data_status or {})
    message = (
        "[cycle_summary] "
        f"action={str(action_text or '').lower().replace(' ', '_')};"
        f"provider={str(status.get('provider_effective') or status.get('provider') or 'unknown').lower()};"
        f"feed={str(status.get('feed_status') or 'UNKNOWN').lower()};"
        f"positions={int(metrics.get('open_positions', 0) or 0)};"
        f"approved={int(metrics.get('signals_approved', 0) or 0)};"
        f"rejected={int(metrics.get('signals_rejected', 0) or 0)};"
        f"posture={str(consistency.get('operational_posture_label') or 'Indefinida').lower()}"
    )
    log_event("INFO", message)


def _maybe_send_final_validation_email(validation_report: dict, *, current_time: datetime) -> None:
    if final_report_path_reachable(validation_report):
        return
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


def _maybe_send_reporting_emails(validation_report: dict, *, current_time: datetime) -> None:
    try:
        process_report_email_delivery(validation_report=validation_report, now=current_time)
    except Exception as exc:
        log_event("ERROR", f"Falha no envio best-effort dos relatorios por email: {exc}")


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
    emit_startup_marker("[worker_startup_marker] WORKER_BUILD_MARKER_20260422_A")
    emit_startup_marker("[worker_startup] worker starting")
    emit_startup_marker(
        (
            "[worker_startup_marker] build info loaded "
            f"build={MARKET_DATA_BUILD_LABEL};git_sha={str(RAILWAY_GIT_COMMIT_SHA or '') or 'na'};"
            f"build_timestamp={str(BUILD_TIMESTAMP or '') or 'na'};service={str(SERVICE_NAME or '') or 'na'}"
        )
    )
    emit_startup_marker(
        (
            "[worker_startup] provider configured "
            f"primary={str(MARKET_DATA_PROVIDER or '').lower()};"
            f"fallback={str(MARKET_DATA_FALLBACK_PROVIDER or '').lower()}"
        )
    )
    load_bot_state()
    emit_startup_marker("[worker_startup] state store ready")
    emit_startup_marker("[worker_startup_marker] worker loop started")

    while True:
        try:
            current_time = datetime.now(timezone.utc)
            refresh_daily_loss_guard()
            state = load_bot_state()

            if state.get("bot_status") != "RUNNING":
                update_runtime_state(
                    last_action="Robo pausado. Aguardando ativacao.",
                    next_run_delta_seconds=PAUSED_SLEEP_SECONDS,
                )
                _refresh_production_monitor(cycle_success=True)
                _run_daily_retention_maintenance()
                validation_report = refresh_swing_validation_cycle(now=current_time)
                _maybe_send_reporting_emails(validation_report, current_time=current_time)
                _maybe_send_final_validation_email(validation_report, current_time=current_time)
                time.sleep(PAUSED_SLEEP_SECONDS)
                continue

            result = run_trader_cycle(persist_market_data=False)
            action_text = build_action_text(result)

            persist_worker_cycle_state(
                last_action=action_text,
                next_run_delta_seconds=SLEEP_SECONDS,
                worker_status="online",
                market_data_payload=result.get("cycle_result", {}).get("market_data_status"),
                runtime_started_at=WORKER_RUNTIME_STARTED_AT,
                process_role=WORKER_PROCESS_ROLE,
            )

            log_event("INFO", action_text)
            health_payload = _refresh_production_monitor(cycle_success=True)
            _log_cycle_health(health_payload)
            _run_daily_retention_maintenance()
            validation_report = refresh_swing_validation_cycle(
                cycle_result=result.get("cycle_result", {}),
                now=current_time,
            )
            _log_cycle_summary(
                action_text=action_text,
                market_data_status=result.get("cycle_result", {}).get("market_data_status"),
                validation_report=validation_report,
            )
            _log_feed_quality_summary(result.get("cycle_result", {}).get("market_data_status"))
            _log_validation_cycle(validation_report)
            _log_signal_quality_summary(validation_report)
            _log_signal_rejection_summary(validation_report, result.get("cycle_result", {}))
            _log_macro_alert_summary(result.get("cycle_result", {}))
            _maybe_send_reporting_emails(validation_report, current_time=current_time)
            _maybe_send_final_validation_email(validation_report, current_time=current_time)

        except Exception as exc:
            error_msg = str(exc)
            log_event("ERROR", f"Erro no worker: {error_msg}")
            mark_error(error_msg)
            health_payload = _refresh_production_monitor(cycle_success=False, exception_message=error_msg)
            _log_cycle_health(health_payload)
            _run_daily_retention_maintenance()
            validation_report = refresh_swing_validation_cycle(now=datetime.now(timezone.utc))
            _maybe_send_reporting_emails(validation_report, current_time=datetime.now(timezone.utc))
            _maybe_send_final_validation_email(validation_report, current_time=datetime.now(timezone.utc))

        time.sleep(SLEEP_SECONDS)


if __name__ == "__main__":
    worker_loop()
