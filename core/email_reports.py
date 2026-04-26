from __future__ import annotations

import json
import smtplib
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

from core.config import (
    APP_TITLE,
    REPORT_EMAIL_10DAY_ENABLED,
    REPORT_EMAIL_DAILY_ENABLED,
    REPORT_EMAIL_ENABLED,
    REPORT_EMAIL_FINAL_ENABLED,
    REPORT_EMAIL_FROM,
    REPORT_EMAIL_PROVIDER,
    REPORT_EMAIL_RESEND_API_BASE,
    REPORT_EMAIL_RESEND_API_KEY,
    REPORT_EMAIL_SMTP_HOST,
    REPORT_EMAIL_SMTP_PASSWORD,
    REPORT_EMAIL_SMTP_PORT,
    REPORT_EMAIL_SMTP_USERNAME,
    REPORT_EMAIL_TIMEOUT_SECONDS,
    REPORT_EMAIL_TO,
    REPORT_EMAIL_USE_TLS,
    REPORT_EMAIL_WEEKLY_ENABLED,
)
from core.market_data import build_feed_quality_snapshot
from core.retention import load_weekly_summary, read_weekly_report_rows
from core.signal_rejection_analysis import rejection_reason_label
from core.state_store import load_bot_state, log_event, update_email_reporting_status
from core.swing_validation import SWING_VALIDATION_DAYS
from core.trader_reports import calculate_trade_report_metrics, normalize_trade_reports_frame, read_trade_reports


REPORT_TYPE_DAILY = "daily"
REPORT_TYPE_WEEKLY = "weekly"
REPORT_TYPE_10DAY = "10day"
REPORT_TYPE_FINAL = "final_30day"

FINAL_CLASSIFICATION_LABELS = {
    "REPROVADO_ESTRATEGIA": "REPROVADO",
    "REPROVADO_INSTABILIDADE": "REPROVADO",
    "APROVADO_COM_AJUSTES": "APROVADO COM RESSALVAS",
}

BLOCK_OBJECTIVES = {
    1: "Coletar base operacional limpa e validar estabilidade com feed confiavel em PAPER.",
    2: "Validar qualidade dos sinais, disciplina dos filtros e consistencia da watchlist.",
    3: "Consolidar consistencia operacional e confirmar maturidade para a proxima decisao da fase.",
}


def _utc_now(now: datetime | None = None) -> datetime:
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        return current.replace(tzinfo=timezone.utc)
    return current.astimezone(timezone.utc)


def _safe_text(value: object, *, fallback: str = "Sem leitura consolidada") -> str:
    text = str(value or "").strip()
    return text or fallback


def _format_money(value: Any) -> str:
    return f"R$ {float(value or 0.0):,.2f}"


def _format_pct(value: Any, *, multiplier: float = 100.0, digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{float(value or 0.0) * multiplier:.{digits}f}%"


def _format_ratio(value: Any, *, digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{float(value or 0.0):.{digits}f}"


def _date_key(now: datetime) -> str:
    return _utc_now(now).date().isoformat()


def _week_key(now: datetime) -> str:
    iso_year, iso_week, _ = _utc_now(now).isocalendar()
    return f"{int(iso_year)}-W{int(iso_week):02d}"


def _week_key_internal(now: datetime) -> str:
    iso_year, iso_week, _ = _utc_now(now).isocalendar()
    return f"{int(iso_year)}_W{int(iso_week):02d}"


def _week_range_label(now: datetime) -> str:
    current = _utc_now(now)
    start = current - timedelta(days=current.weekday())
    end = start + timedelta(days=6)
    return f"{start.date().isoformat()} a {end.date().isoformat()} (UTC)"


def _day_bounds(now: datetime) -> tuple[datetime, datetime]:
    current = _utc_now(now)
    start = current.replace(hour=0, minute=0, second=0, microsecond=0)
    return start, start + timedelta(days=1)


def _filter_reports_between(reports_df: pd.DataFrame, start_at: datetime, end_at: datetime) -> pd.DataFrame:
    frame = normalize_trade_reports_frame(reports_df)
    if frame.empty:
        return frame
    closed_at = pd.to_datetime(frame["closed_at"], errors="coerce", utc=True)
    mask = closed_at.notna() & (closed_at >= start_at) & (closed_at < end_at)
    return frame.loc[mask].reset_index(drop=True)


def _profit_factor_from_reports(reports_df: pd.DataFrame) -> float | None:
    frame = normalize_trade_reports_frame(reports_df)
    if frame.empty:
        return None
    pnl = pd.to_numeric(frame.get("realized_pnl"), errors="coerce").dropna()
    if pnl.empty:
        return None
    gross_profit = float(pnl[pnl > 0].sum())
    gross_loss = abs(float(pnl[pnl < 0].sum()))
    if gross_loss <= 0:
        return None
    return round(gross_profit / gross_loss, 2)


def _avg_entry_score(reports_df: pd.DataFrame) -> float | None:
    frame = normalize_trade_reports_frame(reports_df)
    if frame.empty:
        return None
    entry_scores = pd.to_numeric(frame.get("entry_score"), errors="coerce").dropna()
    if entry_scores.empty:
        return None
    return round(float(entry_scores.mean()), 3)


def _email_provider() -> str:
    return str(REPORT_EMAIL_PROVIDER or "smtp").strip().lower() or "smtp"


def _smtp_ready() -> bool:
    return bool(REPORT_EMAIL_TO and REPORT_EMAIL_SMTP_HOST and REPORT_EMAIL_SMTP_USERNAME and REPORT_EMAIL_SMTP_PASSWORD)


def _resend_ready() -> bool:
    return bool(REPORT_EMAIL_TO and REPORT_EMAIL_FROM and REPORT_EMAIL_RESEND_API_KEY and REPORT_EMAIL_RESEND_API_BASE)


def _provider_readiness_reason(provider: str) -> str | None:
    if provider == "smtp":
        return None if _smtp_ready() else "smtp_not_configured"
    if provider == "resend":
        return None if _resend_ready() else "resend_not_configured"
    return "unknown_report_email_provider"


def _report_from_address() -> str:
    return REPORT_EMAIL_FROM or REPORT_EMAIL_SMTP_USERNAME or REPORT_EMAIL_TO


def _resend_user_agent() -> str:
    safe_name = str(APP_TITLE or "Trade Ops Desk").replace(" ", "-")
    return f"{safe_name}/report-email/1.0"


def _send_via_smtp(subject: str, body: str) -> None:
    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = _report_from_address()
    message["To"] = REPORT_EMAIL_TO
    message.set_content(body)

    with smtplib.SMTP(REPORT_EMAIL_SMTP_HOST, REPORT_EMAIL_SMTP_PORT, timeout=REPORT_EMAIL_TIMEOUT_SECONDS) as server:
        if REPORT_EMAIL_USE_TLS:
            server.starttls()
        server.login(REPORT_EMAIL_SMTP_USERNAME, REPORT_EMAIL_SMTP_PASSWORD)
        server.send_message(message)


def _send_via_resend(subject: str, body: str) -> str:
    payload = {
        "from": REPORT_EMAIL_FROM,
        "to": [REPORT_EMAIL_TO],
        "subject": subject,
        "text": body,
    }
    request = Request(
        REPORT_EMAIL_RESEND_API_BASE,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {REPORT_EMAIL_RESEND_API_KEY}",
            "Content-Type": "application/json",
            "User-Agent": _resend_user_agent(),
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=REPORT_EMAIL_TIMEOUT_SECONDS) as response:
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


def _send_report_email(subject: str, body: str) -> dict[str, Any]:
    provider = _email_provider()
    readiness_reason = _provider_readiness_reason(provider)
    if readiness_reason is not None:
        return {"sent": False, "reason": readiness_reason, "provider": provider}

    delivery_id = ""
    try:
        if provider == "smtp":
            _send_via_smtp(subject, body)
        elif provider == "resend":
            delivery_id = _send_via_resend(subject, body)
        else:
            raise RuntimeError(f"Provider de email de relatorio desconhecido: {provider}")
    except Exception as exc:
        return {"sent": False, "reason": str(exc), "provider": provider}

    return {"sent": True, "reason": "sent", "provider": provider, "delivery_id": delivery_id}


def build_email_reporting_status(state: dict | None = None) -> dict[str, Any]:
    payload = state or load_bot_state()
    email_state = dict(payload.get("email_reporting", {}) or {})
    provider = _email_provider()
    readiness_reason = _provider_readiness_reason(provider)
    enabled = bool(REPORT_EMAIL_ENABLED)
    configured = enabled and readiness_reason is None

    return {
        "enabled": enabled,
        "provider": provider,
        "destination": REPORT_EMAIL_TO or email_state.get("destination") or "",
        "from_address": _report_from_address(),
        "configured": configured,
        "warning": "" if configured else ("email_reporting_disabled" if not enabled else readiness_reason or "misconfigured"),
        "daily_enabled": bool(REPORT_EMAIL_DAILY_ENABLED),
        "weekly_enabled": bool(REPORT_EMAIL_WEEKLY_ENABLED),
        "ten_day_enabled": bool(REPORT_EMAIL_10DAY_ENABLED),
        "final_enabled": bool(REPORT_EMAIL_FINAL_ENABLED),
        "last_delivery_status": str(email_state.get("last_email_delivery_status") or ""),
        "last_delivery_reason": str(email_state.get("last_email_delivery_reason") or ""),
        "last_delivery_attempt_ts": str(email_state.get("last_email_delivery_attempt_ts") or ""),
        "last_delivery_success_ts": str(email_state.get("last_email_delivery_success_ts") or ""),
        "last_sent_report_type": str(email_state.get("last_email_delivery_report_type") or ""),
        "last_daily_report_email_date": str(email_state.get("last_daily_report_email_date") or ""),
        "last_weekly_report_email_week": str(email_state.get("last_weekly_report_email_week") or ""),
        "last_10day_report_email_block": str(email_state.get("last_10day_report_email_block") or ""),
        "last_final_report_email_ts": str(email_state.get("last_final_report_email_ts") or ""),
    }


def _signal_pipeline_lines(validation_report: dict[str, Any]) -> list[str]:
    metrics = dict(validation_report.get("metrics", {}) or {})
    consistency = dict(validation_report.get("consistency", {}) or {})
    received = int(metrics.get("signals_total", 0) or 0)
    validated = int(metrics.get("signals_approved", 0) or 0) + int(metrics.get("signals_rejected", 0) or 0)
    composite_accepted = int(metrics.get("signals_approved", 0) or 0)
    trade_eligible = int(metrics.get("trades_opened", 0) or 0)
    approval_rate = consistency.get("signal_approval_rate")
    return [
        f"- Signal received: {received}",
        f"- Signal validated: {validated}",
        f"- Composite accepted: {composite_accepted}",
        f"- Trade eligible: {trade_eligible}",
        (
            "- Validation approval rate: -"
            if approval_rate is None
            else f"- Validation approval rate: {float(approval_rate or 0.0) * 100:.1f}%"
        ),
    ]


def _composite_summary(validation_report: dict[str, Any], reports_df: pd.DataFrame) -> str:
    consistency = dict(validation_report.get("consistency", {}) or {})
    avg_entry_score = _avg_entry_score(reports_df)
    feed_rejection_diag = dict(validation_report.get("feed_rejection_consistency", {}) or {})
    rejection_quality = dict(validation_report.get("rejection_quality", {}) or {})
    top_reason = rejection_reason_label(
        feed_rejection_diag.get("dominant_rejection_reason") or rejection_quality.get("top_reason")
    )
    scope = str(feed_rejection_diag.get("dominant_rejection_scope") or "accumulated")
    scope_label = {
        "current_cycle": "do ciclo atual",
        "accumulated": "acumulado",
    }.get(scope, "de escopo ainda indefinido")
    if avg_entry_score is not None:
        return f"Media de entry_score das operacoes fechadas no periodo: {avg_entry_score:.3f}."
    approval_rate = consistency.get("signal_approval_rate")
    if approval_rate is not None:
        return (
            f"Sem score medio fechado suficiente; aprovacao atual em {float(approval_rate or 0.0) * 100:.1f}% "
            f"com gargalo dominante {scope_label} em '{top_reason}'."
        )
    return f"Sem leitura consolidada de composite score; principal bloqueio atual: {top_reason}."


def _multi_timeframe_summary(validation_report: dict[str, Any]) -> str:
    timeframe_label = _safe_text(validation_report.get("timeframe_label"), fallback="Diario (1D)")
    return (
        f"A fase atual opera em {timeframe_label}. Nao ha camada multi-timeframe ativa nesta etapa, "
        "entao o efeito consolidado segue neutro."
    )


def _calibration_preview_summary(validation_report: dict[str, Any]) -> str:
    preview = dict(validation_report.get("calibration_preview", {}) or {})
    if not preview:
        return "Calibration preview: no near-approved sample yet. Preview only; no threshold was changed."
    if not bool(preview.get("enabled", True)):
        return "Calibration preview: disabled. Preview only; no threshold was changed."
    near_count = int(preview.get("near_approved_count", 0) or 0)
    margin_text = "-"
    min_score = preview.get("min_score_current")
    floor = preview.get("preview_score_floor")
    if min_score is not None and floor is not None:
        margin_text = f"{float(min_score) - float(floor):.2f}"
    if near_count <= 0:
        return "Calibration preview: no near-approved sample yet. Preview only; no threshold was changed."
    return (
        f"Calibration preview: {near_count} near-approved signal(s) within {margin_text} of threshold "
        "under safe conditions. Preview only; no operational threshold was changed."
    )


def _short_audit_summary(state: dict[str, Any], validation_report: dict[str, Any]) -> str:
    market_state = dict(state.get("market_data", {}) or {})
    rejection_quality = dict(validation_report.get("rejection_quality", {}) or {})
    feed_rejection_diag = dict(validation_report.get("feed_rejection_consistency", {}) or {})
    provider_effective = _safe_text(market_state.get("provider_effective"), fallback="desconhecido")
    last_stage = _safe_text(market_state.get("last_stage"), fallback="sem registro")
    top_reason = rejection_reason_label(feed_rejection_diag.get("dominant_rejection_reason") or rejection_quality.get("top_reason"))
    scope = _safe_text(feed_rejection_diag.get("dominant_rejection_scope"), fallback="unknown")
    note = _safe_text(feed_rejection_diag.get("diagnostic_note"), fallback="Sem diagnostico feed x rejeicao consolidado.")
    return (
        f"Worker escreveu o estado via '{_safe_text(market_state.get('state_writer'), fallback='sem registro')}', "
        f"provider efetivo '{provider_effective}', estagio '{last_stage}' e rejeicao dominante "
        f"({scope}) '{top_reason}'. {note}"
    )


def _worker_reliability_summary(state: dict[str, Any]) -> str:
    production = dict(state.get("production", {}) or {})
    heartbeat_age = production.get("heartbeat_age_seconds")
    heartbeat_age_text = str(heartbeat_age) if heartbeat_age is not None else "-"
    consecutive_errors = int(production.get("consecutive_errors", 0) or 0)
    return (
        f"Worker={_safe_text(state.get('worker_status'), fallback='offline')}; "
        f"heartbeat_age={heartbeat_age_text}s; "
        f"falhas consecutivas={consecutive_errors}."
    )


def _ui_coherence_summary(state: dict[str, Any]) -> str:
    market_state = dict(state.get("market_data", {}) or {})
    if str(market_state.get("state_writer") or "").strip().lower() == "worker" and market_state.get("state_written_at"):
        return (
            f"Snapshot compartilhado confirmado: writer={market_state.get('state_writer')} "
            f"em {market_state.get('state_written_at')} (build {market_state.get('state_build_sha') or 'sem registro'})."
        )
    return "Snapshot compartilhado ainda sem prova completa no estado atual."


def _build_daily_email_body(state: dict[str, Any], validation_report: dict[str, Any], now: datetime) -> str:
    market_state = dict(state.get("market_data", {}) or {})
    production = dict(state.get("production", {}) or {})
    consistency = dict(validation_report.get("consistency", {}) or {})
    metrics = dict(validation_report.get("metrics", {}) or {})
    rejection_quality = dict(validation_report.get("rejection_quality", {}) or {})
    start_at, end_at = _day_bounds(now)
    daily_reports = _filter_reports_between(read_trade_reports(), start_at, end_at)
    daily_metrics = calculate_trade_report_metrics(daily_reports)
    daily_profit_factor = _profit_factor_from_reports(daily_reports)
    feed_quality = build_feed_quality_snapshot(market_state)
    dominant_strategy = _safe_text(rejection_quality.get("top_strategy"))
    daily_pnl_text = _format_money(daily_metrics.get("total_pnl", 0.0))
    daily_win_rate_text = _format_pct(daily_metrics.get("win_rate", 0.0))
    daily_payoff_text = _format_ratio(daily_metrics.get("payoff"))
    daily_profit_factor_text = _format_ratio(daily_profit_factor)
    blocked_trades = int(metrics.get("signals_rejected", 0) or 0)
    live_count = int(feed_quality.get("live_count") or 0)
    total_symbols = int(feed_quality.get("total_symbols") or 0)
    fallback_count = int(feed_quality.get("fallback_count") or 0)
    last_success_text = _safe_text(feed_quality.get("last_success_at"))
    health_level_text = _safe_text(production.get("health_level"), fallback="healthy")
    health_message_text = _safe_text(production.get("health_message"), fallback="Sem mensagem")
    validation_reading_text = _safe_text(consistency.get("validation_reading_message"))
    composite_summary = _composite_summary(validation_report, daily_reports)
    multi_timeframe_summary = _multi_timeframe_summary(validation_report)
    calibration_preview_summary = _calibration_preview_summary(validation_report)
    short_audit_summary = _short_audit_summary(state, validation_report)
    feed_rejection_diag = dict(validation_report.get("feed_rejection_consistency", {}) or {})
    feed_rejection_note = _safe_text(
        feed_rejection_diag.get("diagnostic_note"),
        fallback="Sem diagnostico feed x rejeicao consolidado.",
    )

    lines = [
        "[PAPER MODE] Nenhuma ordem real foi enviada.",
        "",
        f"Date: {_date_key(now)} (UTC)",
        f"Worker status: {_safe_text(state.get('worker_status'), fallback='offline')}",
        f"Heartbeat: {_safe_text(state.get('worker_heartbeat'))}",
        f"Feed status: {_safe_text(market_state.get('feed_status'), fallback='UNKNOWN')}",
        f"Effective provider: {_safe_text(market_state.get('provider_effective') or market_state.get('provider'), fallback='unknown')}",
        f"Context status: {_safe_text((state.get('market_context', {}) or {}).get('market_context_status'), fallback='NEUTRO')}",
        f"Dominant strategy: {dominant_strategy}",
        f"Composite score summary: {composite_summary}",
        calibration_preview_summary,
        f"Multi-timeframe summary: {multi_timeframe_summary}",
        "",
        "Signal pipeline:",
        *_signal_pipeline_lines(validation_report),
        "",
        "Daily result:",
        f"- Daily PnL: {daily_pnl_text}",
        f"- Win rate: {daily_win_rate_text}",
        f"- Payoff: {daily_payoff_text}",
        f"- Profit factor: {daily_profit_factor_text}",
        f"- Blocked trades: {blocked_trades}",
        "",
        "Short audit notes:",
        f"- {short_audit_summary}",
        f"- Feed quality: live={live_count}/{total_symbols} | fallback={fallback_count} | last_success={last_success_text}",
        f"- Feed/rejection consistency: {feed_rejection_note}",
        f"- Operational health: {health_level_text} | {health_message_text}",
        f"- Validation reading: {validation_reading_text}",
    ]
    return "\n".join(lines)


def _current_weekly_entry(state: dict[str, Any], now: datetime) -> dict[str, Any] | None:
    retention_state = dict(state.get("retention", {}) or {})
    weekly_index = list(retention_state.get("weekly_reports_index", []) or [])
    if not weekly_index:
        return None
    current_week = _week_key_internal(now)
    for item in weekly_index:
        if str(item.get("week_key") or "") == current_week:
            return dict(item)
    return dict(weekly_index[0]) if weekly_index else None


def _build_weekly_email_body(state: dict[str, Any], validation_report: dict[str, Any], now: datetime) -> str:
    entry = _current_weekly_entry(state, now)
    weekly_summary = load_weekly_summary(entry) if entry else {}
    weekly_rows = read_weekly_report_rows(entry) if entry else pd.DataFrame()
    weekly_profit_factor = _profit_factor_from_reports(weekly_rows)
    rejection_quality = dict(validation_report.get("rejection_quality", {}) or {})
    dominant_strategy = _safe_text(rejection_quality.get("top_strategy"))
    recommendation = _safe_text(weekly_summary.get("observation_final"))
    suggestions = list(weekly_summary.get("suggestions", []) or [])
    suggestion_text = _safe_text((suggestions[0] or {}).get("message")) if suggestions else "Sem recomendacao adicional no resumo semanal."
    weekly_trades = int(weekly_summary.get("trades_count", 0) or 0)
    weekly_pnl_text = _format_money(weekly_summary.get("pnl", 0.0))
    weekly_win_rate_text = _format_pct(weekly_summary.get("win_rate", 0.0))
    weekly_payoff_text = _format_ratio(weekly_summary.get("payoff"))
    weekly_profit_factor_text = _format_ratio(weekly_profit_factor)
    weekly_drawdown_text = _format_pct((validation_report.get("metrics", {}) or {}).get("max_drawdown_pct"))
    multi_timeframe_summary = _multi_timeframe_summary(validation_report)

    lines = [
        "[PAPER MODE] Nenhuma ordem real foi enviada.",
        "",
        f"Week range: {_week_range_label(now)}",
        f"Week key: {_week_key(now)}",
        f"Total trades: {weekly_trades}",
        f"PnL: {weekly_pnl_text}",
        f"Win rate: {weekly_win_rate_text}",
        f"Payoff: {weekly_payoff_text}",
        f"Profit factor: {weekly_profit_factor_text}",
        f"Drawdown: {weekly_drawdown_text}",
        f"Dominant strategy: {dominant_strategy}",
        "Webhook value summary: Sem leitura consolidada no runtime atual.",
        f"Multi-timeframe effect summary: {multi_timeframe_summary}",
        f"Weekly recommendation: {recommendation}",
        f"Suggestion highlight: {suggestion_text}",
    ]
    return "\n".join(lines)


def _current_block_number(day_number: int) -> int:
    return max(1, min(3, ((max(1, int(day_number)) - 1) // 10) + 1))


def _block_trigger_day(block_number: int) -> int:
    return {1: 10, 2: 20, 3: 30}.get(int(block_number), 30)


def _block_status_message(validation_report: dict[str, Any], block_number: int) -> str:
    metrics = dict(validation_report.get("metrics", {}) or {})
    consistency = dict(validation_report.get("consistency", {}) or {})
    if int(metrics.get("operational_errors", 0) or 0) >= 3:
        return "REPROVAR BLOCO"
    if float(metrics.get("fallback_cycle_pct", 0.0) or 0.0) > 30.0:
        return "REPROVAR BLOCO"
    if block_number == 1 and not bool(consistency.get("watchlist_phase_aligned")):
        return "AJUSTAR ANTES DE PROSSEGUIR"
    if bool(consistency.get("fine_tuning_ready")):
        return "APROVAR PARA PROXIMO BLOCO"
    return "MANTER COLETA CONTROLADA"


def _build_10day_email_body(state: dict[str, Any], validation_report: dict[str, Any], now: datetime) -> str:
    day_number = int(validation_report.get("validation_day_number", 1) or 1)
    block_number = _current_block_number(day_number)
    current_status = _safe_text(validation_report.get("phase_conclusion"))
    strengths = list(validation_report.get("successes", []) or [])
    risks = list(validation_report.get("errors", []) or [])
    performance = dict(validation_report.get("performance", {}) or {})
    metrics = dict(validation_report.get("metrics", {}) or {})
    approval_status = _block_status_message(validation_report, block_number)
    recommendation = _safe_text(validation_report.get("final_validation_reason") or validation_report.get("phase_conclusion"))
    pnl_text = _format_money(performance.get("pnl_total", 0.0))
    win_rate_text = _format_pct(performance.get("win_rate", 0.0))
    payoff_text = _format_ratio(performance.get("payoff"))
    feed_quality_text = _safe_text((state.get("market_data", {}) or {}).get("feed_status"), fallback="UNKNOWN")
    audit_summary = _short_audit_summary(state, validation_report)

    lines = [
        "[PAPER MODE] Nenhuma ordem real foi enviada.",
        "",
        f"Current block: {block_number}",
        f"Evaluation day: {day_number}",
        f"Block objective: {BLOCK_OBJECTIVES.get(block_number)}",
        "Key metrics:",
        f"- Signals approved: {int(metrics.get('signals_approved', 0) or 0)}",
        f"- Signals rejected: {int(metrics.get('signals_rejected', 0) or 0)}",
        f"- PnL: {pnl_text}",
        f"- Win rate: {win_rate_text}",
        f"- Payoff: {payoff_text}",
        f"- Feed quality: {feed_quality_text}",
        f"Current status: {current_status}",
        f"Approval for this block: {approval_status}",
        "Strengths:",
    ]
    if strengths:
        lines.extend(f"- {item}" for item in strengths[:4])
    else:
        lines.append("- Sem forcas consolidadas suficientes ate aqui.")
    lines.append("Risks:")
    if risks:
        lines.extend(f"- {item}" for item in risks[:4])
    else:
        lines.append("- Sem riscos dominantes fora do baseline atual.")
    lines.extend(
        [
            f"Recommendation for next block: {recommendation}",
            f"Audit summary: {audit_summary}",
        ]
    )
    return "\n".join(lines)


def _final_classification(validation_report: dict[str, Any], state: dict[str, Any]) -> str:
    final_grade = str(validation_report.get("final_validation_grade") or "").strip().upper()
    if final_grade in FINAL_CLASSIFICATION_LABELS:
        return FINAL_CLASSIFICATION_LABELS[final_grade]

    metrics = dict(validation_report.get("metrics", {}) or {})
    performance = dict(validation_report.get("performance", {}) or {})
    consistency = dict(validation_report.get("consistency", {}) or {})
    if final_grade == "APROVADO":
        if (
            float(performance.get("pnl_total", 0.0) or 0.0) > 0.0
            and (performance.get("payoff") is None or float(performance.get("payoff") or 0.0) >= 1.2)
            and float(metrics.get("fallback_cycle_pct", 0.0) or 0.0) <= 15.0
            and int(metrics.get("operational_errors", 0) or 0) <= 1
            and bool(consistency.get("fine_tuning_ready"))
        ):
            return "APROVADO PARA PLANEJAR PROXIMA FASE"
        return "APROVADO PARA CONTINUAR EM PAPER"
    if final_grade.startswith("REPROVADO"):
        return "REPROVADO"
    return "APROVADO COM RESSALVAS"


def _final_next_steps(validation_report: dict[str, Any], classification: str) -> str:
    if classification == "REPROVADO":
        return "Manter PAPER, revisar gargalos dominantes e repetir novo ciclo controlado."
    if classification == "APROVADO COM RESSALVAS":
        return "Continuar em PAPER com ajustes pequenos e nova rodada de validacao."
    if classification == "APROVADO PARA CONTINUAR EM PAPER":
        return "Seguir mais um ciclo em PAPER para ampliar amostra e consolidar consistencia."
    return "Documentar criterios de transicao e planejar a proxima fase ainda sem liberar ordem real."


def active_final_report_day() -> int:
    return int(SWING_VALIDATION_DAYS)


def active_final_model_label() -> str:
    return "10-day final" if active_final_report_day() <= 10 else "10+10+10"


def final_report_path_reachable(validation_report: dict[str, Any] | None) -> bool:
    report = dict(validation_report or {})
    day_number = int(report.get("validation_day_number", 0) or 0)
    final_grade = str(report.get("final_validation_grade") or "").strip()
    return day_number >= active_final_report_day() and bool(final_grade)


def _final_report_subject() -> str:
    if active_final_report_day() <= 10:
        return "[PAPER] Final 10-Day Phase Report"
    return "[PAPER] Final 30-Day Phase Report"


def _build_final_email_body(state: dict[str, Any], validation_report: dict[str, Any], now: datetime) -> str:
    metrics = dict(validation_report.get("metrics", {}) or {})
    consistency = dict(validation_report.get("consistency", {}) or {})
    market_state = dict(state.get("market_data", {}) or {})
    feed_quality = build_feed_quality_snapshot(market_state)
    classification = _final_classification(validation_report, state)
    main_risks = list(validation_report.get("errors", []) or [])
    generated_at_text = _utc_now(now).isoformat()
    final_reasoning = _safe_text(validation_report.get("final_validation_reason") or validation_report.get("verdict_message"))
    stability_summary = _worker_reliability_summary(state)
    decision_quality_summary = _safe_text(consistency.get("signal_quality_message"))
    consistency_summary = _safe_text(consistency.get("validation_reading_message"))
    drawdown_text = _format_pct(metrics.get("max_drawdown_pct"))
    full_reports = read_trade_reports()
    profit_factor_value = _profit_factor_from_reports(full_reports)
    profit_factor_text = _format_ratio(profit_factor_value)
    feed_status_text = _safe_text(market_state.get("feed_status"), fallback="UNKNOWN")
    provider_text = _safe_text(market_state.get("provider_effective") or market_state.get("provider"), fallback="unknown")
    live_count = int(feed_quality.get("live_count") or 0)
    total_symbols = int(feed_quality.get("total_symbols") or 0)
    worker_reliability_text = _worker_reliability_summary(state)
    coherence_summary = _ui_coherence_summary(state)
    next_steps = _final_next_steps(validation_report, classification)

    lines = [
        "[PAPER MODE] Nenhuma ordem real foi enviada.",
        "",
        f"Active evaluation model: {active_final_model_label()}",
        f"Final classification: {classification}",
        f"Generated at: {generated_at_text}",
        f"Final reasoning: {final_reasoning}",
        f"Stability summary: {stability_summary}",
        f"Decision quality summary: {decision_quality_summary}",
        f"Consistency summary: {consistency_summary}",
        f"Drawdown: {drawdown_text}",
        f"Profit factor: {profit_factor_text}",
        f"Feed quality summary: status={feed_status_text} | provider={provider_text} | live={live_count}/{total_symbols}",
        f"Worker reliability summary: {worker_reliability_text}",
        f"UI/log/state coherence summary: {coherence_summary}",
        "Main risks:",
    ]
    if main_risks:
        lines.extend(f"- {item}" for item in main_risks[:5])
    else:
        lines.append("- Sem riscos dominantes adicionais no fechamento.")
    lines.extend(
        [
            f"Recommended next steps: {next_steps}",
        ]
    )
    return "\n".join(lines)


def _daily_report_due(email_state: dict[str, Any], now: datetime) -> bool:
    return REPORT_EMAIL_ENABLED and REPORT_EMAIL_DAILY_ENABLED and str(email_state.get("last_daily_report_email_date") or "") != _date_key(now)


def _weekly_report_due(email_state: dict[str, Any], now: datetime) -> bool:
    return REPORT_EMAIL_ENABLED and REPORT_EMAIL_WEEKLY_ENABLED and str(email_state.get("last_weekly_report_email_week") or "") != _week_key(now)


def _ten_day_report_due(email_state: dict[str, Any], validation_report: dict[str, Any]) -> bool:
    if not (REPORT_EMAIL_ENABLED and REPORT_EMAIL_10DAY_ENABLED):
        return False
    day_number = int(validation_report.get("validation_day_number", 0) or 0)
    if day_number < 10:
        return False
    block_number = _current_block_number(day_number)
    if day_number < _block_trigger_day(block_number):
        return False
    return str(email_state.get("last_10day_report_email_block") or "") != str(block_number)


def _final_report_due(email_state: dict[str, Any], validation_report: dict[str, Any]) -> bool:
    if not (REPORT_EMAIL_ENABLED and REPORT_EMAIL_FINAL_ENABLED):
        return False
    day_number = int(validation_report.get("validation_day_number", 0) or 0)
    if day_number < active_final_report_day():
        return False
    if not str(validation_report.get("final_validation_grade") or "").strip():
        return False
    return not str(email_state.get("last_final_report_email_ts") or "").strip()


def _record_delivery_attempt(report_type: str, subject: str, reason: str, now: datetime) -> None:
    update_email_reporting_status(
        {
            "last_email_delivery_report_type": report_type,
            "last_email_delivery_subject": subject,
            "last_email_delivery_status": "attempted",
            "last_email_delivery_reason": reason,
            "last_email_delivery_attempt_ts": _utc_now(now).isoformat(),
        }
    )


def _record_delivery_result(report_type: str, subject: str, *, sent: bool, reason: str, provider: str, now: datetime, marker: str = "") -> None:
    payload = {
        "last_email_delivery_report_type": report_type,
        "last_email_delivery_subject": subject,
        "last_email_delivery_provider": provider,
        "last_email_delivery_status": "sent" if sent else "failed",
        "last_email_delivery_reason": reason,
        "last_email_delivery_attempt_ts": _utc_now(now).isoformat(),
    }
    if sent:
        payload["last_email_delivery_success_ts"] = _utc_now(now).isoformat()
        if report_type == REPORT_TYPE_DAILY:
            payload["last_daily_report_email_date"] = marker
        elif report_type == REPORT_TYPE_WEEKLY:
            payload["last_weekly_report_email_week"] = marker
        elif report_type == REPORT_TYPE_10DAY:
            payload["last_10day_report_email_block"] = marker
        elif report_type == REPORT_TYPE_FINAL:
            payload["last_final_report_email_ts"] = _utc_now(now).isoformat()
    update_email_reporting_status(payload)


def _send_due_report(*, report_type: str, subject: str, body: str, marker: str, now: datetime) -> dict[str, Any]:
    _record_delivery_attempt(report_type, subject, "pending_send", now)
    result = _send_report_email(subject, body)
    sent = bool(result.get("sent"))
    reason = str(result.get("reason") or ("sent" if sent else "unknown_failure"))
    provider = str(result.get("provider") or _email_provider())
    _record_delivery_result(report_type, subject, sent=sent, reason=reason, provider=provider, now=now, marker=marker)
    level = "INFO" if sent else "WARNING"
    log_event(level, f"[report_email_delivery] type={report_type};status={'sent' if sent else 'failed'};provider={provider};reason={reason}")
    return {"report_type": report_type, "subject": subject, "sent": sent, "reason": reason, "provider": provider}


def process_report_email_delivery(
    *,
    validation_report: dict[str, Any] | None = None,
    state: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    current_time = _utc_now(now)
    payload = state or load_bot_state()
    report = dict(validation_report or ((payload.get("validation", {}) or {}).get("last_report") or {}))
    email_state = dict(payload.get("email_reporting", {}) or {})
    status = build_email_reporting_status(payload)

    if not status.get("enabled"):
        update_email_reporting_status(
            {
                "last_email_delivery_status": "disabled",
                "last_email_delivery_reason": "email_reporting_disabled",
            }
        )
        return {"enabled": False, "sent": [], "failed": [], "skipped": ["email_reporting_disabled"]}

    if not status.get("configured"):
        reason = str(status.get("warning") or "misconfigured")
        update_email_reporting_status(
            {
                "last_email_delivery_status": "misconfigured",
                "last_email_delivery_reason": reason,
                "last_email_delivery_attempt_ts": current_time.isoformat(),
            }
        )
        log_event("WARNING", f"[report_email_delivery] type=runtime;status=misconfigured;reason={reason}")
        return {"enabled": True, "sent": [], "failed": [], "skipped": [reason]}

    sent: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []

    if _daily_report_due(email_state, current_time):
        subject = f"[PAPER] Daily Trading Report - {_date_key(current_time)}"
        result = _send_due_report(
            report_type=REPORT_TYPE_DAILY,
            subject=subject,
            body=_build_daily_email_body(payload, report, current_time),
            marker=_date_key(current_time),
            now=current_time,
        )
        (sent if result["sent"] else failed).append(result)

    if _weekly_report_due(email_state, current_time):
        subject = f"[PAPER] Weekly Trading Report - {_week_key(current_time)}"
        result = _send_due_report(
            report_type=REPORT_TYPE_WEEKLY,
            subject=subject,
            body=_build_weekly_email_body(payload, report, current_time),
            marker=_week_key(current_time),
            now=current_time,
        )
        (sent if result["sent"] else failed).append(result)

    if _ten_day_report_due(email_state, report):
        block_number = _current_block_number(int(report.get("validation_day_number", 1) or 1))
        trigger_day = _block_trigger_day(block_number)
        subject = f"[PAPER] Evaluation Block {block_number} Report - Day {trigger_day}"
        result = _send_due_report(
            report_type=REPORT_TYPE_10DAY,
            subject=subject,
            body=_build_10day_email_body(payload, report, current_time),
            marker=str(block_number),
            now=current_time,
        )
        (sent if result["sent"] else failed).append(result)

    if _final_report_due(email_state, report):
        subject = _final_report_subject()
        result = _send_due_report(
            report_type=REPORT_TYPE_FINAL,
            subject=subject,
            body=_build_final_email_body(payload, report, current_time),
            marker="final",
            now=current_time,
        )
        (sent if result["sent"] else failed).append(result)

    return {"enabled": True, "sent": sent, "failed": failed, "skipped": []}
