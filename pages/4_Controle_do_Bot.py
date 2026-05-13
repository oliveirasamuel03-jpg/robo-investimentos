from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from core.alerts import send_email_alert
from core.auth.guards import render_auth_toolbar, require_admin
from core.broker import broker_status_label, probe_broker_status
from core.config import ALERT_EMAIL_ENABLED, ALERT_EMAIL_FROM, ALERT_EMAIL_PROVIDER, PRODUCTION_MODE, SMTP_USERNAME
from core.email_reports import build_email_reporting_status
from core.external_signals import format_external_signal_events_for_display, process_external_signal_payload
from core.market_data import build_feed_quality_snapshot, classify_feed_status, format_market_timestamp, legacy_market_status
from core.macro_alerts import macro_alert_operational_effect
from core.production_monitor import evaluate_production_health
from core.signal_rejection_analysis import rejection_dominant_message, rejection_layer_label, rejection_reason_label
from core.state_store import (
    load_bot_state,
    log_event,
    resolve_market_data_views,
    reset_state,
    save_bot_state,
    update_broker_status,
    update_production_status,
)
from core.swing_validation import refresh_swing_validation_cycle, reset_swing_validation_cycle
from engines.trader_engine import run_trader_cycle


def market_data_status_label(raw_status: str | None) -> str:
    payload = raw_status if isinstance(raw_status, dict) else {"status": raw_status}
    return classify_feed_status(
        status=payload.get("feed_status") or payload.get("status"),
        last_source=payload.get("last_source"),
        source_breakdown=payload.get("source_breakdown"),
    )


def market_data_provider_label(raw_status: dict | None) -> str:
    payload = raw_status if isinstance(raw_status, dict) else {"provider": raw_status}
    provider = str(payload.get("provider") or "unknown").strip().lower()
    labels = {
        "twelvedata": "Twelve Data",
        "yahoo": "Yahoo",
        "synthetic": "Fallback sintetico",
        "mixed": "Twelve Data + Yahoo",
        "unknown": "Desconhecido",
    }
    return labels.get(provider, provider.upper())


def market_data_source_label(raw_source: str | None) -> str:
    payload = raw_source if isinstance(raw_source, dict) else {"last_source": raw_source}
    labels = {
        "market": "Mercado ao vivo",
        "cached": "Cache reaproveitado",
        "fallback": "Fallback sintetico",
        "mixed": "Misto",
        "unknown": "Desconhecido",
    }
    source = str(payload.get("last_source") or "unknown").strip().lower()
    base_label = labels.get(source, str(source or "Desconhecido"))
    provider_label = market_data_provider_label(payload)
    if source == "fallback":
        return base_label
    return f"{base_label} via {provider_label}"


def twelvedata_diagnostic_payload(raw_status: dict | None) -> dict:
    payload = raw_status if isinstance(raw_status, dict) else {}
    diagnostics = payload.get("provider_diagnostics", {}) or {}
    diagnostic = diagnostics.get("twelvedata", {}) if isinstance(diagnostics, dict) else {}
    return dict(diagnostic or {})


def audit_display_value(value: object) -> str:
    if value is None:
        return "NAO REGISTRADO NO ESTADO ATUAL"
    if isinstance(value, str) and not value.strip():
        return "NAO REGISTRADO NO ESTADO ATUAL"
    if isinstance(value, bool):
        return "Sim" if value else "Nao"
    return str(value)


def source_commit_display_value(value: object) -> str:
    if value is None:
        return "NAO INFORMADO PELO DEPLOY"
    if isinstance(value, str) and not value.strip():
        return "NAO INFORMADO PELO DEPLOY"
    return str(value)


def symbol_list_label(symbols: list[str] | None) -> str:
    values = [str(item).upper() for item in (symbols or []) if str(item)]
    return ", ".join(values) if values else "Nenhum"


def pct_label(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{float(value or 0.0) * 100:.1f}%"


def worker_instrumentation_confirmed(payload: dict | None) -> bool:
    data = payload or {}
    return (
        str(data.get("process_role") or "").strip().lower() == "worker"
        and str(data.get("state_writer") or "").strip().lower() == "worker"
        and bool(str(data.get("build_active") or "").strip())
        and bool(str(data.get("last_stage") or "").strip())
    )


def market_data_legacy_label(raw_status: dict | None) -> str:
    payload = raw_status or {}
    return legacy_market_status(
        status=payload.get("status_legacy") or payload.get("status"),
        last_source=payload.get("last_source"),
        source_breakdown=payload.get("source_breakdown"),
    )


def broker_mode_label(raw_mode: str | None) -> str:
    labels = {
        "paper": "Simulado",
        "live": "Real",
    }
    return labels.get(str(raw_mode or "paper").strip().lower(), str(raw_mode or "paper").title())


def bot_status_label(raw_status: str | None) -> str:
    labels = {
        "RUNNING": "Ligado",
        "PAUSED": "Pausado",
        "STOPPED": "Parado",
    }
    return labels.get(str(raw_status or "").upper(), str(raw_status or "Desconhecido"))


def health_level_label(raw_level: str | None) -> str:
    labels = {
        "healthy": "Saudavel",
        "warning": "Atencao",
        "critical": "Critico",
    }
    return labels.get(str(raw_level or "healthy").strip().lower(), str(raw_level or "healthy").title())


def market_context_label(raw_status: str | None) -> str:
    labels = {
        "FAVORAVEL": "Favoravel",
        "NEUTRO": "Neutro",
        "DESFAVORAVEL": "Desfavoravel",
        "CRITICO": "Critico",
    }
    return labels.get(str(raw_status or "NEUTRO").strip().upper(), str(raw_status or "NEUTRO").title())


current_user = require_admin()
render_auth_toolbar()

st.title("Controle Operacional")
st.caption(f"Painel administrativo do trader. Usuario: {current_user['username']}")
st.code("BUILD_UI_MARKER_20260422_A")

state = load_bot_state()
security_state = state.get("security", {}) or {}
operational_market_state, chart_market_state = resolve_market_data_views(state)
broker_state = update_broker_status(probe_broker_status(security_state, requested_by="admin_panel"))
state = load_bot_state()
production_state = update_production_status(evaluate_production_health(state))
state = load_bot_state()
production_state = state.get("production", {}) or {}
operational_market_state, chart_market_state = resolve_market_data_views(state)
broker_state = state.get("broker", {}) or broker_state
validation_report = refresh_swing_validation_cycle()
state = load_bot_state()
validation_state = state.get("validation", {}) or {}
market_context_state = state.get("market_context", {}) or {}
macro_alert_state = state.get("macro_alert", {}) or {}
external_signal_state = state.get("external_signal", {}) or {}
risk_state = state.get("risk", {}) or {}
daily_loss_limit_brl = float(risk_state.get("daily_loss_limit_brl", 0.0) or 0.0)
daily_loss_consumed_brl = float(risk_state.get("daily_loss_consumed_brl", 0.0) or 0.0)
daily_loss_remaining_brl = float(risk_state.get("daily_loss_remaining_brl", 0.0) or 0.0)
daily_realized_pnl_brl = float(risk_state.get("daily_realized_pnl_brl", 0.0) or 0.0)
daily_loss_day_key = str(risk_state.get("daily_loss_day_key") or "-")
daily_loss_block_active = bool(risk_state.get("daily_loss_block_active", False))
daily_loss_blocked_at = str(risk_state.get("daily_loss_blocked_at") or "")
daily_loss_block_reason = str(risk_state.get("daily_loss_block_reason") or "")

if bool(security_state.get("real_mode_enabled", False)):
    st.warning("Real trading enabled")

st.subheader("Modo producao")
st.caption("Monitoramento de saude, alertas e diagnostico operacional. PAPER TRADING permanece como padrao nesta etapa.")

production_mode_text = "Ativo" if PRODUCTION_MODE else "Inativo"
alert_mode_text = "Ativo" if ALERT_EMAIL_ENABLED else "Inativo"
alert_provider_text = str(production_state.get("alert_provider") or ALERT_EMAIL_PROVIDER or "smtp").upper()
configured_sender = ALERT_EMAIL_FROM or SMTP_USERNAME or "Sem registro"
health_level = str(production_state.get("health_level") or "healthy").lower()
health_message = str(production_state.get("health_message") or "Sem mensagem.")

if health_level == "healthy":
    st.success(f"{health_level_label(health_level)}: {health_message}")
elif health_level == "warning":
    st.warning(f"{health_level_label(health_level)}: {health_message}")
else:
    st.error(f"{health_level_label(health_level)}: {health_message}")

prod_c1, prod_c2, prod_c3, prod_c4 = st.columns(4)
prod_c1.metric("Status geral", health_level_label(health_level))
prod_c2.metric("Heartbeat age (s)", str(production_state.get("heartbeat_age_seconds") or 0))
prod_c3.metric("Falhas consecutivas", str(production_state.get("consecutive_errors") or 0))
prod_c4.metric("Alertas por email", alert_mode_text)

prod_c5, prod_c6, prod_c7, prod_c8 = st.columns(4)
prod_c5.metric("Modo producao", production_mode_text)
prod_c6.metric("Ultimo heartbeat", state.get("worker_heartbeat") or "Sem registro")
prod_c7.metric("Ultima execucao", production_state.get("last_execution_at") or "Sem registro")
prod_c8.metric("Ultimo sucesso", production_state.get("last_success_at") or "Sem registro")

prod_c9, prod_c10, prod_c11, prod_c12 = st.columns(4)
prod_c9.metric("Feed monitorado", market_data_status_label(production_state.get("feed_status")))
prod_c10.metric("Broker monitorado", broker_status_label(production_state.get("broker_status")))
prod_c11.metric("Ultimo alerta", production_state.get("last_alert_sent_at") or "Nenhum")
prod_c12.metric("Proximo alerta elegivel", production_state.get("next_alert_eligible_at") or "Agora")

st.caption(
    f"Provider de alerta: {alert_provider_text} | Remetente configurado: {configured_sender}"
)

st.subheader("Trava de perda diária (paper)")
st.caption(
    "Bloqueio operacional explícito: ao atingir o limite diário de perda, o robô não abre novas entradas. "
    "Posições já abertas continuam sob gestão normal."
)
if daily_loss_block_active:
    st.error("Trava diária ativa: novas entradas bloqueadas por perda diária.")
else:
    st.success("Trava diária pronta: novas entradas liberadas neste momento.")

risk_c1, risk_c2, risk_c3, risk_c4 = st.columns(4)
risk_c1.metric("Estado", "Bloqueado" if daily_loss_block_active else "Liberado")
risk_c2.metric("Limite diário", f"R$ {daily_loss_limit_brl:,.2f}")
risk_c3.metric("Perda consumida", f"R$ {daily_loss_consumed_brl:,.2f}")
risk_c4.metric("Limite restante", f"R$ {daily_loss_remaining_brl:,.2f}")
st.caption(
    f"Dia operacional UTC: {daily_loss_day_key} | "
    f"PnL realizado do dia (base da trava): R$ {daily_realized_pnl_brl:,.2f}"
)
if daily_loss_block_active:
    st.caption(
        f"Bloqueio ativado em: {format_market_timestamp(daily_loss_blocked_at)} | "
        f"Motivo: {daily_loss_block_reason or 'Limite diário atingido.'}"
    )
if risk_state.get("daily_loss_reset_at"):
    st.caption(
        f"Último reset automático na virada do dia UTC: "
        f"{format_market_timestamp(risk_state.get('daily_loss_reset_at'))}"
    )

st.subheader("Alerta macro de risco")
st.caption(
    "Camada operacional de risco em PAPER mode. Eventos macro nao disparam ordens; "
    "eles apenas reduzem confianca, restringem setups frageis ou bloqueiam entradas sob risco alto."
)
macro_active = bool(macro_alert_state.get("macro_alert_active", False))
macro_m1, macro_m2, macro_m3, macro_m4 = st.columns(4)
macro_m1.metric("Estado", "Ativo" if macro_active else "Inativo")
macro_m2.metric("Impacto", str(macro_alert_state.get("macro_alert_level") or "LOW"))
macro_m3.metric("Janela", str(macro_alert_state.get("macro_alert_window_status") or "INACTIVE"))
macro_m4.metric(
    "Bloqueia novas entradas",
    "Sim" if bool(macro_alert_state.get("macro_alert_blocks_new_entries", False)) else "Nao",
)
macro_m5, macro_m6, macro_m7, macro_m8 = st.columns(4)
macro_m5.metric("Moeda", str(macro_alert_state.get("macro_alert_currency") or "-"))
macro_m6.metric("Evento", str(macro_alert_state.get("macro_alert_title") or "Sem evento ativo"))
macro_m7.metric(
    "Minutos ate evento",
    str(macro_alert_state.get("macro_alert_minutes_to_event"))
    if macro_alert_state.get("macro_alert_minutes_to_event") is not None
    else "-",
)
macro_m8.metric("Penalidade", f"{float(macro_alert_state.get('macro_alert_penalty', 0.0) or 0.0):.2f}")
if macro_alert_state.get("macro_alert_time"):
    st.caption(f"Horario do evento: {format_market_timestamp(macro_alert_state.get('macro_alert_time'))}")
st.caption(f"Motivo: {macro_alert_state.get('macro_alert_reason') or 'Nenhum evento macro ativo.'}")
st.caption(f"Efeito operacional: {macro_alert_operational_effect(macro_alert_state)}")
st.caption("Seguranca: filtro de risco somente; PAPER TRADING obrigatorio; nenhuma ordem real habilitada.")

st.subheader("External signal audit")
st.caption(
    "FASE 3A audit-only: webhook externo e entrada complementar de auditoria. "
    "Nao executa trades, nao aprova entradas e nao altera score, estrategia ou broker."
)
external_enabled = bool(external_signal_state.get("enabled", False))
external_status = str(external_signal_state.get("last_status") or ("DISABLED" if not external_enabled else "IGNORED"))
external_score = float(external_signal_state.get("last_score", 0.0) or 0.0)
ext_c1, ext_c2, ext_c3, ext_c4 = st.columns(4)
ext_c1.metric("Webhook externo", "Ativo" if external_enabled else "Inativo")
ext_c2.metric("Status", external_status)
ext_c3.metric("Fonte", str(external_signal_state.get("last_source") or "Sem registro"))
ext_c4.metric("Score recebido", f"{external_score:.2f}")
ext_c5, ext_c6, ext_c7, ext_c8 = st.columns(4)
ext_c5.metric("Estrategia", str(external_signal_state.get("last_strategy") or "Sem registro"))
ext_c6.metric("Ativo", str(external_signal_state.get("last_symbol") or "Sem registro"))
ext_c7.metric("Lado", str(external_signal_state.get("last_side") or "Sem registro"))
ext_c8.metric("Timeframe", str(external_signal_state.get("last_timeframe") or "Sem registro"))
st.caption(
    f"Recebido em: {format_market_timestamp(external_signal_state.get('last_received_at')) if external_signal_state.get('last_received_at') else 'Sem registro'} | "
    f"Motivo: {external_signal_state.get('last_reason') or 'Sem sinal externo recebido.'}"
)
st.caption(
    "Seguranca: audit-only, PAPER TRADING obrigatorio, sem autoridade de execucao e sem bypass de guards."
)
if not external_enabled:
    st.info("Webhook externo desabilitado por padrao. O app continua operando como antes.")
elif not bool(external_signal_state.get("webhook_configured", False)):
    st.warning("Webhook externo habilitado, mas segredo ou fontes permitidas nao foram configurados.")

st.subheader("External signal audit test")
st.caption(
    "Harness interno admin-only para testar validacao e persistencia. "
    "Nao cria rota publica, nao executa trades e nao altera score, estrategia, guards ou broker."
)
if not bool(external_signal_state.get("test_panel_enabled", False)):
    st.info("Test panel disabled. Defina EXTERNAL_SIGNAL_TEST_PANEL_ENABLED=true apenas para validacao controlada.")
else:
    allowed_timeframes = [
        value.strip()
        for value in str(external_signal_state.get("allowed_timeframes") or "30s,1m,5m,15m,1h,1d").split(",")
        if value.strip()
    ] or ["15m"]
    watchlist_options = list((state.get("trader", {}) or {}).get("watchlist", []) or ["BTC-USD"])
    default_ts = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    if "external_signal_test_ts" not in st.session_state:
        st.session_state["external_signal_test_ts"] = default_ts
    freeze_test_ts = st.checkbox(
        "Freeze timestamp for repeat submissions",
        value=True,
        help="Use the same timestamp and same fields to test DUPLICATE. Use an old timestamp to test EXPIRED.",
    )
    if st.button("Refresh timestamp suggestion", disabled=freeze_test_ts, use_container_width=True):
        st.session_state["external_signal_test_ts"] = default_ts
    with st.form("external_signal_audit_test_form"):
        st.caption("Token de teste e usado somente na submissao; nao e exibido, logado ou persistido.")
        st.caption("Use the same timestamp and same fields to test DUPLICATE.")
        st.caption("Use an old timestamp to test EXPIRED.")
        test_token = st.text_input("Test token", value="", type="password")
        test_source = st.text_input("Source", value=str(external_signal_state.get("last_source") or "tradingview"))
        test_strategy = st.text_input("Strategy", value="audit_test")
        test_symbol = st.selectbox("Symbol", options=watchlist_options, index=0)
        test_timeframe = st.selectbox("Timeframe", options=allowed_timeframes, index=0)
        test_side = st.selectbox("Side", options=["BUY", "SELL", "LONG", "SHORT"], index=0)
        test_alert_price = st.number_input("Alert price", min_value=0.0, value=1.0, step=1.0)
        test_score = st.number_input("Score", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        test_ts = st.text_input("Signal timestamp UTC", key="external_signal_test_ts")
        submitted = st.form_submit_button("Submit audit-only test signal")

    if submitted:
        try:
            result = process_external_signal_payload(
                {
                    "token": test_token,
                    "source": test_source,
                    "strategy": test_strategy,
                    "symbol": test_symbol,
                    "timeframe": test_timeframe,
                    "side": test_side,
                    "alert_price": test_alert_price,
                    "score": test_score,
                    "ts": test_ts,
                    "extra": {"origin": "controle_do_bot_test_panel"},
                },
                persist=True,
            )
            event = dict(result.get("event", {}) or {})
            status = str(event.get("status") or "IGNORED")
            reason = str(event.get("reason") or "Sem motivo registrado.")
            if status == "ACCEPTED_FOR_AUDIT":
                st.success(f"Sinal aceito para auditoria somente: {status}. Nenhum trade foi aprovado.")
            else:
                st.warning(f"Sinal de teste nao aceito para auditoria: {status} | {reason}")
            state = load_bot_state()
            external_signal_state = state.get("external_signal", {}) or {}
        except Exception as exc:
            st.error(f"Falha segura no teste audit-only: {exc}")

recent_external_events = format_external_signal_events_for_display(external_signal_state, limit=10)
st.caption("Eventos recentes de sinal externo: audit-only, sem execucao e sem aprovacao de trade.")
if recent_external_events:
    st.dataframe(pd.DataFrame(recent_external_events), hide_index=True, use_container_width=True)
else:
    st.caption("Sem eventos recentes de sinal externo.")

act_c1, act_c2 = st.columns(2)
with act_c1:
    if st.button("Testar email", use_container_width=True):
        result = send_email_alert(
            "[Trade Ops Desk] Teste de alerta",
            "Email de teste do modo producao.\nNenhuma ordem real foi habilitada.\nBroker atual: PAPER.",
            alert_type="manual_test",
            force=True,
        )
        if result.get("sent"):
            st.success("Email de teste enviado.")
        else:
            st.warning(f"Email de teste nao enviado: {result.get('reason') or 'bloqueado'}")

with act_c2:
    if st.button("Recalcular saude", use_container_width=True):
        production_state = update_production_status(evaluate_production_health(load_bot_state()))
        st.info(f"Saude recalculada: {health_level_label(production_state.get('health_level'))}")

email_reporting_status = build_email_reporting_status(load_bot_state())
st.subheader("Email reporting")
st.caption("Entrega automatica de relatorios em PAPER mode. O envio e best-effort e nunca interfere no ciclo do worker.")
email_c1, email_c2, email_c3, email_c4 = st.columns(4)
email_c1.metric("Email reporting", "Ativo" if email_reporting_status.get("enabled") else "Inativo")
email_c2.metric("Destino", email_reporting_status.get("destination") or "Sem registro")
email_c3.metric(
    "Ultimo status",
    email_reporting_status.get("last_delivery_status") or "Sem registro",
)
email_c4.metric(
    "Ultimo tipo enviado",
    email_reporting_status.get("last_sent_report_type") or "Sem registro",
)

email_c5, email_c6, email_c7 = st.columns(3)
email_c5.metric(
    "Ultima tentativa",
    email_reporting_status.get("last_delivery_attempt_ts") or "Sem registro",
)
email_c6.metric(
    "Ultimo sucesso",
    email_reporting_status.get("last_delivery_success_ts") or "Sem registro",
)
email_c7.metric(
    "Provider",
    str(email_reporting_status.get("provider") or "sem registro").upper(),
)

if not email_reporting_status.get("enabled"):
    st.warning("Email reporting esta desabilitado. Os relatorios continuam sendo gerados localmente em PAPER mode.")
elif not email_reporting_status.get("configured"):
    st.warning(
        "Email reporting esta habilitado, mas a configuracao atual esta incompleta ou invalida. "
        "O worker continua operando normalmente e os relatorios seguem locais."
    )
else:
    st.success("Email reporting habilitado em PAPER mode, com envio best-effort.")

st.caption(
    f"Razao mais recente: {email_reporting_status.get('last_delivery_reason') or email_reporting_status.get('warning') or 'Sem registro'}"
)
st.caption(
    "Controles: "
    f"diario={'on' if email_reporting_status.get('daily_enabled') else 'off'} | "
    f"semanal={'on' if email_reporting_status.get('weekly_enabled') else 'off'} | "
    f"10dias={'on' if email_reporting_status.get('ten_day_enabled') else 'off'} | "
    f"final={'on' if email_reporting_status.get('final_enabled') else 'off'}"
)

st.subheader("Validacao swing 10 dias")
st.caption("Camada de validacao paper para swing profissional. Nenhuma ordem real e liberada nesta etapa.")

validation_status = str(validation_report.get("validation_status") or "running")
validation_phase = str(validation_report.get("validation_phase") or validation_state.get("validation_phase") or "Coleta e observacao")
final_grade = str(validation_report.get("final_validation_grade") or validation_state.get("final_validation_grade") or "").strip()
verdict_message = str(validation_report.get("verdict_message") or "").strip()
phase_conclusion = str(validation_report.get("phase_conclusion") or "Sem conclusao")

if final_grade == "APROVADO":
    st.success(verdict_message or "Ciclo aprovado.")
elif final_grade == "APROVADO_COM_AJUSTES":
    st.warning(verdict_message or "Ciclo aprovado com ajustes.")
elif final_grade in {"REPROVADO_ESTRATEGIA", "REPROVADO_INSTABILIDADE"}:
    st.error(verdict_message or "Ciclo reprovado.")
else:
    st.info(f"Fase atual: {validation_phase} | Conclusao parcial: {phase_conclusion}")

val_metrics = dict(validation_report.get("metrics", {}) or {})
val_perf = dict(validation_report.get("performance", {}) or {})
val_consistency = dict(validation_report.get("consistency", {}) or {})

val_c1, val_c2, val_c3, val_c4 = st.columns(4)
val_c1.metric("Dia atual", str(int(validation_report.get("validation_day_number", 1) or 1)))
val_c2.metric("Fase", validation_phase)
val_c3.metric("Status do ciclo", "Finalizado" if validation_status == "completed" else "Em validacao")
val_c4.metric("Timeframe", str(validation_report.get("timeframe_label") or validation_state.get("timeframe_label") or "Diario (1D)"))

val_c5, val_c6, val_c7, val_c8 = st.columns(4)
val_c5.metric("Trades fechados", str(int(val_metrics.get("trades_closed", 0) or 0)))
val_c6.metric("Win rate", f"{float(val_perf.get('win_rate', 0.0) or 0.0) * 100:.2f}%")
val_c7.metric("Payoff", "-" if val_perf.get("payoff") is None else f"{float(val_perf.get('payoff') or 0.0):.2f}")
val_c8.metric("PnL total", f"R$ {float(val_perf.get('pnl_total', 0.0) or 0.0):,.2f}")

val_c9, val_c10, val_c11, val_c12 = st.columns(4)
val_c9.metric("Sinais aprovados", str(int(val_metrics.get("signals_approved", 0) or 0)))
val_c10.metric("Sinais rejeitados", str(int(val_metrics.get("signals_rejected", 0) or 0)))
val_c11.metric(
    "Fallback do ciclo",
    "-" if val_metrics.get("fallback_cycle_pct") is None else f"{float(val_metrics.get('fallback_cycle_pct') or 0.0):.2f}%",
)
val_c12.metric("Erros operacionais", str(int(val_metrics.get("operational_errors", 0) or 0)))

cons_c1, cons_c2, cons_c3, cons_c4 = st.columns(4)
cons_c1.metric("Amostra atual", str(val_consistency.get("sample_quality_label") or "Sem leitura"))
cons_c2.metric("Postura", str(val_consistency.get("operational_posture_label") or "Indefinida"))
cons_c3.metric(
    "Drawdown max",
    "-"
    if val_metrics.get("max_drawdown_pct") is None
    else f"{float(val_metrics.get('max_drawdown_pct') or 0.0) * 100:.2f}%",
)
cons_c4.metric(
    "Watchlist da fase",
    "Coerente" if bool(val_consistency.get("watchlist_phase_aligned")) else "Fora da fase",
)
st.caption(
    "Aprovacao de sinais no ciclo: "
    + (
        "-"
        if val_consistency.get("signal_approval_rate") is None
        else f"{float(val_consistency.get('signal_approval_rate') or 0.0) * 100:.1f}%"
    )
)
if val_consistency.get("sample_quality_message"):
    st.caption(f"Amostra: {val_consistency.get('sample_quality_message')}")
if val_consistency.get("watchlist_message"):
    st.caption(f"Watchlist: {val_consistency.get('watchlist_message')}")
if val_consistency.get("capital_phase_aligned") is False:
    st.warning(
        "O capital atual do runtime nao coincide com o capital-base recomendado da fase. "
        "Para um novo ciclo limpo, use o reset operacional do trader."
    )

st.subheader("Qualidade de sinal da FASE 2")
signal_c1, signal_c2, signal_c3, signal_c4, signal_c5, signal_c6 = st.columns(6)
signal_c1.metric("Sinais aprovados", str(int(val_metrics.get("signals_approved", 0) or 0)))
signal_c2.metric("Sinais rejeitados", str(int(val_metrics.get("signals_rejected", 0) or 0)))
signal_c3.metric(
    "Taxa de aprovacao",
    "-"
    if val_consistency.get("signal_approval_rate") is None
    else f"{float(val_consistency.get('signal_approval_rate') or 0.0) * 100:.1f}%",
)
signal_c4.metric("Amostra do ciclo", str(val_consistency.get("sample_quality_label") or "Sem leitura"))
signal_c5.metric("Postura atual", str(val_consistency.get("operational_posture_label") or "Indefinida"))
signal_c6.metric("Leitura simples", str(val_consistency.get("signal_quality_label") or "Baixa"))

signal_d1, signal_d2 = st.columns(2)
with signal_d1:
    st.write(
        f"Consistencia da watchlist: "
        f"{'Coerente com a fase' if bool(val_consistency.get('watchlist_phase_aligned')) else 'Fora da fase'}"
    )
    if val_consistency.get("signal_quality_message"):
        st.caption(f"Sinal: {val_consistency.get('signal_quality_message')}")
with signal_d2:
    st.write(
        f"Ajuste fino: "
        f"{'Ja existe base minima' if bool(val_consistency.get('fine_tuning_ready')) else 'Ainda nao'}"
    )
    if val_consistency.get("validation_reading_message"):
        st.caption(f"Validacao: {val_consistency.get('validation_reading_message')}")

rejection_quality = dict(validation_report.get("rejection_quality", {}) or {})
feed_rejection_consistency = dict(
    validation_report.get("feed_rejection_consistency")
    or val_metrics.get("feed_rejection_consistency")
    or (state.get("validation", {}) or {}).get("feed_rejection_consistency", {})
    or {}
)
calibration_preview = dict(
    validation_report.get("calibration_preview")
    or state.get("calibration_preview", {})
    or {}
)
strategy_bottleneck = dict(
    validation_report.get("strategy_bottleneck")
    or state.get("strategy_bottleneck", {})
    or {}
)
strategy_structure_audit = dict(
    validation_report.get("strategy_structure_audit")
    or state.get("strategy_structure_audit", {})
    or {}
)
market_structure_audit = dict(
    validation_report.get("market_structure_audit")
    or state.get("market_structure_audit", {})
    or {}
)
fib_alignment_audit = dict(
    validation_report.get("fib_alignment_audit")
    or state.get("fib_alignment_audit", {})
    or {}
)
multi_timeframe_intraday_fetcher = dict(
    validation_report.get("multi_timeframe_intraday_fetcher")
    or state.get("multi_timeframe_intraday_fetcher", {})
    or {}
)
multi_timeframe_swing_audit = dict(
    validation_report.get("multi_timeframe_swing_audit")
    or state.get("multi_timeframe_swing_audit", {})
    or {}
)
bos_pivot_trace_audit = dict(
    validation_report.get("bos_pivot_trace_audit")
    or state.get("bos_pivot_trace_audit", {})
    or {}
)
strategy_decision_bridge_trace = dict(
    validation_report.get("strategy_decision_bridge_trace")
    or state.get("strategy_decision_bridge_trace", {})
    or {}
)
feed_scope_reconciliation = dict(
    validation_report.get("feed_scope_reconciliation")
    or state.get("feed_scope_reconciliation", {})
    or {}
)
shadow_decision_simulator = dict(
    validation_report.get("shadow_decision_simulator")
    or state.get("shadow_decision_simulator", {})
    or {}
)
phase2_fine_tune = dict(
    validation_report.get("phase2_fine_tune")
    or state.get("phase2_fine_tune", {})
    or {}
)
phase2_1_fine_tune = dict(
    validation_report.get("phase2_1_fine_tune")
    or state.get("phase2_1_fine_tune", {})
    or {}
)
rejection_top_reasons = rejection_quality.get("top_reasons", []) or []
st.subheader("Qualidade de rejeicao de sinal")
rej_c1, rej_c2, rej_c3, rej_c4 = st.columns(4)
rej_c1.metric(
    "Motivo dominante",
    rejection_reason_label(rejection_quality.get("top_reason")) if rejection_quality.get("top_reason") else "Sem leitura",
)
rej_c2.metric(
    "Camada dominante",
    rejection_layer_label(rejection_quality.get("top_layer")) if rejection_quality.get("top_layer") else "Sem leitura",
)
rej_c3.metric(
    "Setup mais bloqueado",
    str(rejection_quality.get("top_strategy") or "Sem leitura"),
)
rej_c4.metric(
    "Base minima para ajuste",
    "Sim" if bool(rejection_quality.get("has_minimum_sample")) else "Nao",
)

rej_d1, rej_d2 = st.columns(2)
with rej_d1:
    st.write("**Top 5 motivos de rejeicao**")
    if rejection_top_reasons:
        for item in rejection_top_reasons:
            st.caption(
                f"- {item.get('human_reason')}: {int(item.get('count', 0) or 0)} "
                f"({pct_label(item.get('pct'))})"
            )
    else:
        st.caption("Sem rejeicoes suficientes registradas ate o momento.")
with rej_d2:
    layer_breakdown = dict(rejection_quality.get("layer_breakdown", {}) or {})
    strategy_breakdown = dict(rejection_quality.get("strategy_breakdown", {}) or {})
    st.write(
        f"**Leitura curta:** "
        f"{rejection_dominant_message(rejection_quality.get('top_layer')) if rejection_quality.get('top_layer') else 'Sem leitura consolidada.'}"
    )
    if layer_breakdown:
        layer_lines = [
            f"{rejection_layer_label(layer)}={pct_label((int(count or 0) / max(int(rejection_quality.get('total_rejection_events', 0) or 0), 1)))}"
            for layer, count in sorted(layer_breakdown.items(), key=lambda item: int(item[1] or 0), reverse=True)
        ]
        st.caption("Camadas: " + " | ".join(layer_lines))
    if strategy_breakdown:
        top_strategy, top_strategy_count = sorted(
            strategy_breakdown.items(),
            key=lambda item: int(item[1] or 0),
            reverse=True,
        )[0]
        st.caption(f"Estrategia mais bloqueada: {top_strategy} ({int(top_strategy_count or 0)})")
if feed_rejection_consistency:
    st.caption(
        "Diagnostico feed x rejeicao: "
        f"{feed_rejection_consistency.get('diagnostic_note') or 'Sem leitura consolidada.'}"
    )
    st.caption(
        "Escopo: "
        f"dominante={feed_rejection_consistency.get('dominant_rejection_scope') or 'unknown'} | "
        f"atual={rejection_reason_label(feed_rejection_consistency.get('current_cycle_rejection_reason'))} | "
        f"acumulado={rejection_reason_label(feed_rejection_consistency.get('accumulated_rejection_reason'))} | "
        f"fallback atual/acumulado="
        f"{int(feed_rejection_consistency.get('fallback_rejection_current_cycle_count', 0) or 0)}/"
        f"{int(feed_rejection_consistency.get('fallback_rejection_accumulated_count', 0) or 0)} | "
        f"guards atual/acumulado="
        f"{int(feed_rejection_consistency.get('guard_rejection_current_cycle_count', 0) or 0)}/"
        f"{int(feed_rejection_consistency.get('guard_rejection_accumulated_count', 0) or 0)}"
    )
if calibration_preview:
    st.subheader("Calibration Preview - PREVIEW ONLY")
    st.caption(
        "Camada diagnostica conservadora: nao aprova trades, nao reduz thresholds, nao altera estrategia "
        "e mantem PAPER TRADING obrigatorio."
    )
    cal_c1, cal_c2, cal_c3, cal_c4, cal_c5 = st.columns(5)
    min_score = calibration_preview.get("min_score_current")
    preview_floor = calibration_preview.get("preview_score_floor")
    best_score = calibration_preview.get("best_score_seen")
    avg_gap = calibration_preview.get("avg_score_gap")
    avg_gap_label = "-" if avg_gap is None else f"{float(avg_gap):.3f}"
    cal_c1.metric("Min score atual", "-" if min_score is None else f"{float(min_score):.2f}")
    cal_c2.metric("Piso preview", "-" if preview_floor is None else f"{float(preview_floor):.2f}")
    cal_c3.metric("Quase aprovados", str(int(calibration_preview.get("near_approved_count", 0) or 0)))
    cal_c4.metric("Taxa preview", pct_label(calibration_preview.get("near_approved_rate")))
    cal_c5.metric("Melhor score visto", "-" if best_score is None else f"{float(best_score):.2f}")
    st.caption(
        f"Gap medio: {avg_gap_label} | "
        f"Top ativo: {calibration_preview.get('top_asset') or '-'} | "
        f"Top setup: {calibration_preview.get('top_setup') or '-'} | "
        f"Recomendacao: {calibration_preview.get('recommendation') or 'observe_more'}"
    )
    st.caption(calibration_preview.get("reason") or "Sem leitura de preview ainda.")
    preview_examples = list(calibration_preview.get("near_approved_examples", []) or [])[:5]
    if preview_examples:
        st.dataframe(pd.DataFrame(preview_examples), hide_index=True, use_container_width=True)
if strategy_bottleneck:
    st.subheader("Strategy Bottleneck - DIAGNOSTIC ONLY")
    st.caption(
        "Camada diagnostica: nao aprova trades, nao reduz thresholds, nao altera estrategia "
        "e mantem PAPER TRADING obrigatorio."
    )
    bot_b1, bot_b2, bot_b3, bot_b4 = st.columns(4)
    bot_b1.metric("Bottleneck dominante", strategy_bottleneck.get("dominant_bottleneck") or "-")
    bot_b2.metric("Setup dominante", strategy_bottleneck.get("dominant_setup") or "-")
    bot_b3.metric("Ativo dominante", strategy_bottleneck.get("dominant_asset") or "-")
    bot_b4.metric("Rejeicoes estrategia", str(int(strategy_bottleneck.get("total_strategy_rejections", 0) or 0)))
    bot_d1, bot_d2, bot_d3, bot_d4, bot_d5 = st.columns(5)
    bot_d1.metric("Score baixo", str(int(strategy_bottleneck.get("score_below_min_count", 0) or 0)))
    bot_d2.metric("Momentum fraco", str(int(strategy_bottleneck.get("momentum_weak_count", 0) or 0)))
    bot_d3.metric("Confirmacao fraca", str(int(strategy_bottleneck.get("secondary_confirmation_weak_count", 0) or 0)))
    bot_d4.metric("RSI fora", str(int(strategy_bottleneck.get("rsi_out_of_range_count", 0) or 0)))
    bot_d5.metric("Trend nao confirmada", str(int(strategy_bottleneck.get("trend_not_confirmed_count", 0) or 0)))
    st.caption(
        f"Volatilidade: {int(strategy_bottleneck.get('volatility_filter_count', 0) or 0)} | "
        f"Contexto: {int(strategy_bottleneck.get('context_filter_count', 0) or 0)} | "
        f"Recomendacao: {strategy_bottleneck.get('recommendation') or 'observe_more'}"
    )
    st.caption(strategy_bottleneck.get("reason") or "Sem diagnostico de bottleneck ainda.")
    for label, key in (
        ("Top ativos bloqueados", "top_assets_blocked"),
        ("Top setups bloqueados", "top_setups_blocked"),
        ("Top filtros", "top_filter_reasons"),
        ("Candidatos mais proximos", "closest_candidates"),
    ):
        items = list(strategy_bottleneck.get(key, []) or [])[:5]
        if items:
            st.caption(label)
            st.dataframe(pd.DataFrame(items), hide_index=True, use_container_width=True)
if strategy_structure_audit:
    st.subheader("AUDITORIA ESTRUTURAL DA ESTRATEGIA")
    st.caption(
        "SHADOW ONLY: compara setups internos sem aprovar trades, sem reduzir thresholds, "
        "sem alterar score real, broker ou execucao. PAPER TRADING permanece obrigatorio."
    )
    shadow_score = strategy_structure_audit.get("structural_audit_top_score")
    shadow_gap = strategy_structure_audit.get("structural_audit_top_gap")
    shadow_score_label = "-" if shadow_score is None else f"{float(shadow_score):.2f}"
    shadow_gap_label = "-" if shadow_gap is None else f"{float(shadow_gap):.4f}"
    shadow_c1, shadow_c2, shadow_c3, shadow_c4, shadow_c5 = st.columns(5)
    shadow_c1.metric("Setup shadow", strategy_structure_audit.get("structural_audit_top_setup") or "-")
    shadow_c2.metric("Ativo shadow", strategy_structure_audit.get("structural_audit_top_symbol") or "-")
    shadow_c3.metric("Melhor score", shadow_score_label)
    shadow_c4.metric("Gap ate aprovacao", shadow_gap_label)
    shadow_c5.metric("Candidatos shadow", str(int(strategy_structure_audit.get("structural_audit_candidates", 0) or 0)))
    st.caption(
        f"Bloqueador primario: {strategy_structure_audit.get('structural_audit_primary_blocker') or '-'} | "
        f"Bloqueador secundario: {strategy_structure_audit.get('structural_audit_secondary_blocker') or '-'} | "
        f"Recomendacao: {strategy_structure_audit.get('structural_audit_recommendation') or 'sem dados suficientes'}"
    )
    st.caption(f"Temporalidade: {strategy_structure_audit.get('structural_audit_timeframe_note') or 'Inconclusivo.'}")
    st.caption(f"RSI/momentum: {strategy_structure_audit.get('structural_audit_rsi_momentum_note') or 'Inconclusivo.'}")
    st.caption(f"Reversal: {strategy_structure_audit.get('structural_audit_reversal_note') or 'Inconclusivo.'}")
    st.caption(strategy_structure_audit.get("structural_audit_reason") or "Sem auditoria estrutural consolidada ainda.")
    setup_rows = list(strategy_structure_audit.get("structural_audit_setup_comparison", []) or [])[:5]
    if setup_rows:
        display_rows = [
            {
                "setup": row.get("setup"),
                "candidatos shadow": row.get("shadow_candidates"),
                "melhor ativo": row.get("best_symbol"),
                "melhor score": row.get("best_score"),
                "gap medio": row.get("average_gap"),
                "bloqueador dominante": row.get("dominant_blocker"),
                "recomendacao": row.get("recommendation"),
            }
            for row in setup_rows
            if isinstance(row, dict)
        ]
        if display_rows:
            st.dataframe(pd.DataFrame(display_rows), hide_index=True, use_container_width=True)
if market_structure_audit:
    st.subheader("AUDITORIA FIBONACCI + ESTRUTURA DE MERCADO")
    st.caption(
        "SHADOW ONLY: Fibonacci, price action, pivos e BOS sao apenas auditoria estrutural. "
        "Nao aprovam trade, nao alteram score real, nao mudam broker e mantem PAPER TRADING."
    )
    top_score = market_structure_audit.get("market_structure_top_score")
    score_label = "-" if top_score is None else f"{float(top_score):.2f}"
    best_candidates = list(market_structure_audit.get("market_structure_best_candidates", []) or [])
    top_candidate = dict(best_candidates[0] or {}) if best_candidates else {}
    ms_c1, ms_c2, ms_c3, ms_c4, ms_c5 = st.columns(5)
    ms_c1.metric("Melhor ativo", market_structure_audit.get("market_structure_top_symbol") or "-")
    ms_c2.metric("Direcao", top_candidate.get("structure_direction") or "-")
    ms_c3.metric("Zona Fibonacci", market_structure_audit.get("market_structure_top_zone") or "-")
    ms_c4.metric("Score estrutural", score_label)
    ms_c5.metric("Candidato shadow", "Sim" if bool(top_candidate.get("market_structure_shadow_candidate", False)) else "Nao")
    ms_d1, ms_d2, ms_d3, ms_d4 = st.columns(4)
    ms_d1.metric("Pivo", "Sim" if bool(top_candidate.get("pivot_detected", False)) else "Nao")
    ms_d2.metric("BOS", "Sim" if bool(top_candidate.get("bos_detected", False)) else "Nao")
    ms_d3.metric("Falso rompimento", "Sim" if bool(top_candidate.get("false_breakout_risk", False)) else "Nao")
    ms_d4.metric("Regime", top_candidate.get("market_regime") or "INCONCLUSIVE")
    st.caption(
        f"Confluencia trend_pullback_breakout: "
        f"{'Sim' if bool(top_candidate.get('structure_confirms_trend_pullback', False)) else 'Nao'} | "
        f"Recomendacao: {market_structure_audit.get('market_structure_top_recommendation') or 'sem dados suficientes'}"
    )
    confluence = dict(market_structure_audit.get("market_structure_setup_confluence", {}) or {})
    st.caption(
        "Confluencia por setup: "
        f"trend_pullback={int(confluence.get('trend_pullback', 0) or 0)} | "
        f"breakout={int(confluence.get('breakout', 0) or 0)} | "
        f"reversal={int(confluence.get('reversal', 0) or 0)} | "
        f"melhoraria qualidade={int(confluence.get('would_improve_quality', 0) or 0)}"
    )
    st.caption(
        f"Suficiencia: {market_structure_audit.get('market_structure_data_sufficiency') or 'NO_DATA'} | "
        f"Amostra minima: {'Sim' if bool(market_structure_audit.get('market_structure_minimum_sample_met', False)) else 'Nao'} | "
        f"Por que nao houve candidato: {market_structure_audit.get('market_structure_why_no_candidate') or '-'}"
    )
    if best_candidates:
        display_rows = [
            {
                "asset": row.get("symbol"),
                "structure_score": row.get("market_structure_score"),
                "fib_zone": row.get("current_fib_zone"),
                "bos": row.get("bos_detected"),
                "pivot": row.get("pivot_detected"),
                "false_breakout_risk": row.get("false_breakout_risk"),
                "confluence": ", ".join(list(row.get("confluence_notes", []) or [])[:2]),
                "recommendation": row.get("structure_recommendation"),
            }
            for row in best_candidates[:5]
            if isinstance(row, dict)
        ]
        if display_rows:
            st.dataframe(pd.DataFrame(display_rows), hide_index=True, use_container_width=True)
if fib_alignment_audit:
    st.subheader("ADERENCIA AO VIDEO/PDF FIBONACCI")
    st.caption(
        "SHADOW ONLY: esta camada mede aderencia objetiva a um checklist inspirado no video/PDF. "
        "Nao afirma equivalencia da estrategia, nao aprova trade e nao altera score real, broker ou thresholds."
    )
    alignment_score = fib_alignment_audit.get("fib_alignment_score")
    alignment_score_label = "-" if alignment_score is None else f"{float(alignment_score):.2f}"
    fa_c1, fa_c2, fa_c3, fa_c4 = st.columns(4)
    fa_c1.metric("Ativo analisado", fib_alignment_audit.get("fib_alignment_top_symbol") or "-")
    fa_c2.metric("Score aderencia", alignment_score_label)
    fa_c3.metric("Status", fib_alignment_audit.get("fib_alignment_status") or "insufficient_data")
    fa_c4.metric("Modo", fib_alignment_audit.get("fib_alignment_mode") or "SHADOW_ONLY")
    fa_d1, fa_d2, fa_d3, fa_d4, fa_d5 = st.columns(5)
    fa_d1.metric("Ancora fundo", fib_alignment_audit.get("fib_alignment_anchor_low_status") or "insufficient")
    fa_d2.metric("Ancora topo", fib_alignment_audit.get("fib_alignment_anchor_high_status") or "insufficient")
    fa_d3.metric("Zona Fibonacci", fib_alignment_audit.get("fib_alignment_zone_status") or "insufficient")
    fa_d4.metric("Pivo", fib_alignment_audit.get("fib_alignment_pivot_status") or "insufficient")
    fa_d5.metric("BOS", fib_alignment_audit.get("fib_alignment_bos_status") or "insufficient")
    st.caption(
        f"Confirmacao de entrada: {fib_alignment_audit.get('fib_alignment_entry_confirmation_status') or 'insufficient'} | "
        f"Confluencia setup: {fib_alignment_audit.get('fib_alignment_confluence_status') or 'insufficient'} | "
        f"Principal divergencia: {fib_alignment_audit.get('fib_alignment_why_differs') or '-'} | "
        f"Recomendacao: {fib_alignment_audit.get('fib_alignment_recommendation') or 'keep_shadow_only'}"
    )
    checklist_rows = [
        {
            "regra": row.get("item"),
            "esperado": row.get("esperado_pelo_video_pdf"),
            "detectado": row.get("detectado_pelo_app"),
            "status": row.get("status"),
            "motivo": row.get("motivo"),
        }
        for row in list(fib_alignment_audit.get("fib_alignment_checklist", []) or [])
        if isinstance(row, dict)
    ]
    if checklist_rows:
        st.dataframe(pd.DataFrame(checklist_rows[:8]), hide_index=True, use_container_width=True)
if multi_timeframe_swing_audit:
    st.subheader("FASE 2.5 - DIAGNOSTICO MULTI-TIMEFRAME SWING")
    st.caption(
        "SHADOW ONLY: 1D, 4H e 1H sao usados apenas para diagnosticar estrutura swing. "
        "Nao aprova trade, nao altera score real, nao muda broker e preserva PAPER TRADING."
    )
    top_alignment_score = multi_timeframe_swing_audit.get("top_alignment_score")
    top_alignment_score_label = "-" if top_alignment_score is None else f"{float(top_alignment_score):.2f}"
    mtf_candidates = [
        row for row in list(multi_timeframe_swing_audit.get("recent_candidates", []) or []) if isinstance(row, dict)
    ]
    top_mtf = dict(mtf_candidates[0] if mtf_candidates else {})
    pivot_timeframes = ", ".join(list(top_mtf.get("pivot_confirmed_timeframes", top_mtf.get("pivot_timeframes", [])) or [])) or "-"
    bos_timeframes = ", ".join(list(top_mtf.get("bos_confirmed_timeframes", top_mtf.get("bos_timeframes", [])) or [])) or "-"
    mtf_c1, mtf_c2, mtf_c3, mtf_c4 = st.columns(4)
    mtf_c1.metric("Status", multi_timeframe_swing_audit.get("mode") or "SHADOW_ONLY")
    mtf_c2.metric("Top ativo", multi_timeframe_swing_audit.get("top_symbol") or "-")
    mtf_c3.metric("Score alinhamento", top_alignment_score_label)
    mtf_c4.metric("Status alinhamento", multi_timeframe_swing_audit.get("top_alignment_status") or "INSUFFICIENT_DATA")
    mtf_d1, mtf_d2, mtf_d3, mtf_d4 = st.columns(4)
    mtf_d1.metric("Direcao 1D", top_mtf.get("daily_bias") or "INCONCLUSIVE")
    mtf_d2.metric("Estrutura 4H", top_mtf.get("h4_structure") or "INCONCLUSIVE")
    mtf_d3.metric("Confirmacao 1H", top_mtf.get("h1_confirmation") or "INCONCLUSIVE")
    mtf_d4.metric("Pivo detectado", pivot_timeframes)
    mtf_e1, mtf_e2, mtf_e3, mtf_e4 = st.columns(4)
    mtf_e1.metric("BOS detectado", bos_timeframes)
    mtf_e2.metric("Conflito dominante", multi_timeframe_swing_audit.get("dominant_conflict_reason") or "-")
    mtf_e3.metric("Recomendacao", multi_timeframe_swing_audit.get("top_recommendation") or "observe_more")
    mtf_e4.metric("Feed usado", multi_timeframe_swing_audit.get("feed_status") or "UNKNOWN")
    st.caption(
        f"Provider: {multi_timeframe_swing_audit.get('provider_effective') or '-'} | "
        f"Cache/TTL: {multi_timeframe_swing_audit.get('cache_status') or 'cycle_data_resample_only'} / "
        f"{int(multi_timeframe_swing_audit.get('cache_ttl_seconds', 0) or 0)}s | "
        f"Chamadas extras estimadas: {int(multi_timeframe_swing_audit.get('estimated_provider_calls', 0) or 0)} | "
        f"Guard provider: {multi_timeframe_swing_audit.get('provider_guard') or '-'}"
    )
    st.markdown("##### FASE 2.5A - DADOS INTRADAY 4H/1H")
    st.caption(
        "SHADOW ONLY: dados 4H/1H sao usados apenas para diagnosticar estrutura swing. "
        "Nao aprovam trade, nao alteram score real, nao mudam broker e preservam PAPER TRADING."
    )
    intraday_diags = [
        row for row in list(multi_timeframe_intraday_fetcher.get("diagnostics", []) or []) if isinstance(row, dict)
    ]
    fetch_c1, fetch_c2, fetch_c3, fetch_c4 = st.columns(4)
    fetch_c1.metric("Intraday fetch status", multi_timeframe_intraday_fetcher.get("intraday_data_quality") or "NO_DATA")
    fetch_c2.metric("Usa dados reais 4H/1H", "Sim" if bool(multi_timeframe_swing_audit.get("uses_real_intraday_data", False)) else "Nao")
    fetch_c3.metric("Timeframes disponiveis", ", ".join(list(multi_timeframe_intraday_fetcher.get("timeframes_available", []) or [])) or "-")
    fetch_c4.metric("Simbolos buscados", len(list(multi_timeframe_intraday_fetcher.get("symbols_requested", []) or [])))
    fetch_d1, fetch_d2, fetch_d3, fetch_d4 = st.columns(4)
    fetch_d1.metric("Cache hits", int(multi_timeframe_intraday_fetcher.get("cache_hits", 0) or 0))
    fetch_d2.metric("Cache misses", int(multi_timeframe_intraday_fetcher.get("cache_misses", 0) or 0))
    fetch_d3.metric("Chamadas provider", int(multi_timeframe_intraday_fetcher.get("provider_calls_attempted", 0) or 0))
    fetch_d4.metric("Puladas por guard", int(multi_timeframe_intraday_fetcher.get("provider_calls_skipped", 0) or 0))
    fetch_e1, fetch_e2, fetch_e3, fetch_e4 = st.columns(4)
    fetch_e1.metric("Qualidade 4H", multi_timeframe_swing_audit.get("h4_data_quality") or "missing")
    fetch_e2.metric("Qualidade 1H", multi_timeframe_swing_audit.get("h1_data_quality") or "missing")
    fetch_e3.metric("Motivo insuficiente", multi_timeframe_swing_audit.get("intraday_missing_reason") or multi_timeframe_intraday_fetcher.get("provider_guard_reason") or "-")
    fetch_e4.metric("Recomendacao", multi_timeframe_intraday_fetcher.get("intraday_fetch_recommendation") or "observe_more")
    mtf_diag_by_symbol = {
        row.get("symbol"): dict(row.get("timeframe_diagnostics", {}) or {})
        for row in mtf_candidates
        if isinstance(row, dict)
    }
    fetch_rows = [
        {
            "symbol": row.get("symbol"),
            "timeframe": row.get("timeframe"),
            "candles_available": row.get("candles_available"),
            "data_quality": row.get("data_quality"),
            "cache_status": row.get("cache_status"),
            "provider_call_attempted": row.get("provider_call_attempted"),
            "trend_direction": ((mtf_diag_by_symbol.get(row.get("symbol"), {}) or {}).get(row.get("timeframe"), {}) or {}).get("trend_direction"),
            "pivot_confirmed": ((mtf_diag_by_symbol.get(row.get("symbol"), {}) or {}).get(row.get("timeframe"), {}) or {}).get("pivot_confirmed"),
            "bos_confirmed": ((mtf_diag_by_symbol.get(row.get("symbol"), {}) or {}).get(row.get("timeframe"), {}) or {}).get("bos_confirmed"),
            "why_not_confirmed": row.get("quality_reason"),
        }
        for row in intraday_diags[:10]
    ]
    if fetch_rows:
        st.dataframe(pd.DataFrame(fetch_rows), hide_index=True, use_container_width=True)
    else:
        st.info("Sem diagnostico intraday 4H/1H registrado ainda.")
    if bos_pivot_trace_audit:
        st.markdown("##### FASE 2.5B - AUDITORIA FINA BOS/PIVO 4H/1H")
        st.caption(
            "SHADOW ONLY: BOS e pivos 4H/1H sao apenas diagnostico estrutural. "
            "Nao aprovam trade, nao alteram score real, nao mudam broker e preservam PAPER TRADING."
        )
        bos_c1, bos_c2, bos_c3, bos_c4 = st.columns(4)
        bos_c1.metric("Status", bos_pivot_trace_audit.get("mode") or "SHADOW_ONLY")
        bos_c2.metric("Top ativo", bos_pivot_trace_audit.get("top_symbol") or "-")
        bos_c3.metric("Top timeframe", bos_pivot_trace_audit.get("top_timeframe") or "-")
        bos_c4.metric("Relacao 1H/4H", bos_pivot_trace_audit.get("top_relationship") or "INSUFFICIENT_DATA")
        bos_d1, bos_d2, bos_d3, bos_d4 = st.columns(4)
        bos_d1.metric("Estado do pivo", bos_pivot_trace_audit.get("top_pivot_state") or "INSUFFICIENT_DATA")
        bos_d2.metric("Estado do BOS", bos_pivot_trace_audit.get("top_bos_state") or "INSUFFICIENT_DATA")
        bos_d3.metric("Faltando confirmar", bos_pivot_trace_audit.get("dominant_missing_piece") or bos_pivot_trace_audit.get("top_primary_missing_piece") or "-")
        bos_d4.metric("Should keep blocked", int(bos_pivot_trace_audit.get("should_keep_blocked_count", 0) or 0))
        bos_e1, bos_e2, bos_e3, bos_e4 = st.columns(4)
        bos_e1.metric("BOS por pavio", int(bos_pivot_trace_audit.get("wick_only_bos_count", 0) or 0))
        bos_e2.metric("Fechamento fraco", int(bos_pivot_trace_audit.get("weak_close_bos_count", 0) or 0))
        bos_e3.metric("BOS confirmado", int(bos_pivot_trace_audit.get("confirmed_bos_count", 0) or 0))
        bos_e4.metric("Reteste pendente", int(bos_pivot_trace_audit.get("retest_pending_count", 0) or 0))
        st.caption(
            f"Recomendacao: {bos_pivot_trace_audit.get('top_recommendation') or 'observe_more'} | "
            f"h4_bos_missing={int(bos_pivot_trace_audit.get('h4_bos_missing_count', 0) or 0)} | "
            f"h1_bos_only={int(bos_pivot_trace_audit.get('h1_bos_only_count', 0) or 0)}"
        )
        bos_rows = [
            {
                "symbol": row.get("symbol"),
                "timeframe": row.get("timeframe"),
                "trend_direction": row.get("trend_direction"),
                "pivot_state": row.get("pivot_state"),
                "bos_state": row.get("bos_state"),
                "relationship_to_higher_tf": row.get("relationship_to_higher_tf"),
                "bos_level": row.get("bos_level"),
                "last_close": row.get("last_close"),
                "close_distance_to_bos_pct": row.get("close_distance_to_bos_pct"),
                "wick_crossed_level": row.get("wick_crossed_level"),
                "close_confirmed_level": row.get("close_confirmed_level"),
                "retest_detected": row.get("retest_detected"),
                "false_breakout_risk": row.get("false_breakout_risk"),
                "why_pivot_not_confirmed": row.get("why_pivot_not_confirmed"),
                "why_bos_not_confirmed": row.get("why_bos_not_confirmed"),
                "supports_trend_pullback_breakout": row.get("supports_trend_pullback_breakout"),
                "should_keep_blocked": row.get("should_keep_blocked"),
                "recommendation": row.get("recommendation"),
            }
            for row in list(bos_pivot_trace_audit.get("recent_candidates", []) or [])[:10]
            if isinstance(row, dict)
        ]
        if bos_rows:
            st.dataframe(pd.DataFrame(bos_rows), hide_index=True, use_container_width=True)
        else:
            st.info("Sem candidatos BOS/Pivo 4H/1H registrados ainda.")
    if strategy_decision_bridge_trace:
        st.markdown("##### FASE 2.5B.1 - PONTE ENTRE ESTRUTURA E DECISAO REAL")
        st.caption(
            "SHADOW ONLY: esta camada explica por que uma estrutura confirmada no diagnostico ainda nao virou "
            "entrada real/paper oficial. Nao aprova trade, nao altera score, nao muda broker, nao muda "
            "thresholds e preserva PAPER TRADING."
        )
        bridge_c1, bridge_c2, bridge_c3, bridge_c4, bridge_c5 = st.columns(5)
        bridge_c1.metric("Status", strategy_decision_bridge_trace.get("mode") or "SHADOW_ONLY")
        bridge_c2.metric("Top ativo", strategy_decision_bridge_trace.get("top_symbol") or "-")
        bridge_c3.metric("Status da ponte", strategy_decision_bridge_trace.get("top_bridge_status") or "INSUFFICIENT_TRACE_DATA")
        bridge_c4.metric("Bloqueador real", strategy_decision_bridge_trace.get("top_real_blocker") or "-")
        bridge_c5.metric("Estrutura shadow", strategy_decision_bridge_trace.get("top_structure_status") or "-")
        bridge_d1, bridge_d2, bridge_d3, bridge_d4, bridge_d5 = st.columns(5)
        bridge_d1.metric("Reconciliacao", strategy_decision_bridge_trace.get("top_reconciliation_status") or "UNKNOWN_MISMATCH")
        bridge_d2.metric("Fallback mismatch", int(strategy_decision_bridge_trace.get("fallback_scope_mismatch_count", 0) or 0))
        bridge_d3.metric("Multi-TF vs BOS/Pivo", int(strategy_decision_bridge_trace.get("multi_tf_vs_bos_mismatch_count", 0) or 0))
        bridge_d4.metric("Deve manter bloqueado", int(strategy_decision_bridge_trace.get("should_keep_blocked_count", 0) or 0))
        bridge_d5.metric("Recomendacao", strategy_decision_bridge_trace.get("recommendation") or "observe_more")
        bridge_rows = [
            {
                "symbol": row.get("symbol"),
                "real_score": row.get("real_score"),
                "min_score": row.get("min_score"),
                "score_gap": row.get("score_gap"),
                "real_rejection_reason": row.get("real_rejection_reason"),
                "primary_real_blocker": row.get("primary_real_blocker"),
                "secondary_real_blocker": row.get("secondary_real_blocker"),
                "multi_tf_alignment_status": row.get("multi_tf_alignment_status"),
                "bos_state_4h": row.get("bos_state_4h"),
                "bos_state_1h": row.get("bos_state_1h"),
                "pivot_state_4h": row.get("pivot_state_4h"),
                "pivot_state_1h": row.get("pivot_state_1h"),
                "relationship_1h_4h": row.get("relationship_1h_4h"),
                "fallback_blocker_scope": row.get("fallback_blocker_scope"),
                "reconciliation_status": row.get("reconciliation_status"),
                "final_bridge_reason": row.get("final_bridge_reason"),
                "should_keep_blocked": row.get("should_keep_blocked"),
                "recommendation": row.get("recommendation"),
            }
            for row in list(strategy_decision_bridge_trace.get("recent_candidates", []) or [])[:10]
            if isinstance(row, dict)
        ]
        if bridge_rows:
            st.dataframe(pd.DataFrame(bridge_rows), hide_index=True, use_container_width=True)
        else:
            st.info("Sem candidatos de ponte entre estrutura e decisao real ainda.")
    if feed_scope_reconciliation:
        st.markdown("##### FASE 2.5B.1A - RECONCILIACAO DE ESCOPO DO FEED/FALLBACK")
        st.caption(
            "DIAGNOSTIC ONLY: esta camada separa fallback atual, acumulado e historico. "
            "Nao aprova trade, nao altera score, nao muda broker, nao muda thresholds e preserva PAPER TRADING."
        )
        fs_c1, fs_c2, fs_c3, fs_c4 = st.columns(4)
        fs_c1.metric("Feed atual", feed_scope_reconciliation.get("current_feed_status") or "UNKNOWN")
        fs_c2.metric("Fallback atual", int(feed_scope_reconciliation.get("current_fallback_count", 0) or 0))
        fs_c3.metric("Fallback acumulado", int(feed_scope_reconciliation.get("accumulated_fallback_count", 0) or 0))
        fs_c4.metric("Escopo do fallback", feed_scope_reconciliation.get("fallback_blocker_scope") or "UNKNOWN")
        fs_d1, fs_d2, fs_d3, fs_d4 = st.columns(4)
        fs_d1.metric("Feed atual limpo?", "Sim" if bool(feed_scope_reconciliation.get("current_feed_is_clean", False)) else "Nao")
        fs_d2.metric("Rejeicao atual dominante", feed_scope_reconciliation.get("dominant_rejection_current") or "-")
        fs_d3.metric("Rejeicao acumulada", feed_scope_reconciliation.get("dominant_rejection_accumulated") or "-")
        fs_d4.metric("Recomendacao", feed_scope_reconciliation.get("recommendation") or "observe_more")
        if bool(feed_scope_reconciliation.get("current_feed_is_clean", False)) and int(feed_scope_reconciliation.get("accumulated_fallback_count", 0) or 0) > 0:
            st.info("O fallback exibido e acumulado/historico, nao do ciclo atual.")
        st.caption(feed_scope_reconciliation.get("notes") or "Sem nota de reconciliacao de feed.")
    mtf_rows = [
        {
            "symbol": row.get("symbol"),
            "daily_bias": row.get("daily_bias"),
            "h4_structure": row.get("h4_structure"),
            "h1_confirmation": row.get("h1_confirmation"),
            "alignment_score": row.get("alignment_score"),
            "alignment_status": row.get("alignment_status"),
            "pivot_timeframes": ", ".join(list(row.get("pivot_confirmed_timeframes", row.get("pivot_timeframes", [])) or [])),
            "bos_timeframes": ", ".join(list(row.get("bos_confirmed_timeframes", row.get("bos_timeframes", [])) or [])),
            "supports_trend_pullback_breakout": row.get("supports_trend_pullback_breakout"),
            "missing_for_setup": ", ".join(list(row.get("missing_for_setup", []) or [])),
            "would_improve_signal_quality": row.get("would_improve_signal_quality"),
            "should_keep_blocked": row.get("should_keep_blocked"),
            "recommendation": row.get("recommendation"),
        }
        for row in mtf_candidates[:8]
    ]
    if mtf_rows:
        st.dataframe(pd.DataFrame(mtf_rows), hide_index=True, use_container_width=True)
    else:
        st.info("Sem candidatos multi-timeframe suficientes ainda.")
if shadow_decision_simulator:
    st.subheader("SIMULADOR SHADOW DE DECISAO - FASE 2.4")
    st.caption(
        "SHADOW ONLY: quase-aprovados sao simulados separadamente. "
        "Nao abre trade, nao cria posicao paper oficial, nao altera PnL, score, broker ou thresholds."
    )
    sd_c1, sd_c2, sd_c3, sd_c4, sd_c5 = st.columns(5)
    sd_c1.metric("Quase-aprovados preview", int(shadow_decision_simulator.get("preview_near_approved_count", shadow_decision_simulator.get("shadow_near_approved_count", 0)) or 0))
    sd_c2.metric("Safe", int(shadow_decision_simulator.get("shadow_safe_near_approved_count", 0) or 0))
    sd_c3.metric("Marginal", int(shadow_decision_simulator.get("shadow_marginal_near_approved_count", shadow_decision_simulator.get("shadow_marginal_count", 0)) or 0))
    sd_c4.metric("Teria entrado", int(shadow_decision_simulator.get("shadow_would_enter_count", 0) or 0))
    sd_c5.metric("Pendentes", int(shadow_decision_simulator.get("shadow_pending_count", 0) or 0))
    sd_d1, sd_d2, sd_d3, sd_d4 = st.columns(4)
    sd_d1.metric("Teria ganho", int(shadow_decision_simulator.get("shadow_would_win_count", 0) or 0))
    sd_d2.metric("Teria perdido", int(shadow_decision_simulator.get("shadow_would_lose_count", 0) or 0))
    sd_d3.metric("Melhor ativo", shadow_decision_simulator.get("shadow_best_symbol") or "-")
    sd_d4.metric("Melhor setup", shadow_decision_simulator.get("shadow_best_strategy") or "-")
    st.caption(
        f"Bloqueio dominante: {shadow_decision_simulator.get('shadow_dominant_block_reason') or '-'} | "
        f"Atual: {shadow_decision_simulator.get('dominant_exclusion_current_scope') or '-'} | "
        f"Acumulado: {shadow_decision_simulator.get('dominant_exclusion_accumulated_scope') or '-'} | "
        f"Escopo fallback: {shadow_decision_simulator.get('fallback_blocker_scope') or 'UNKNOWN'} | "
        f"Recomendacao: {shadow_decision_simulator.get('shadow_policy_recommendation') or 'observe_more'} | "
        f"Politica: {shadow_decision_simulator.get('shadow_entry_policy') or 'conservative_v1'}"
    )
    st.markdown("##### RASTREABILIDADE FASE 2.4D - ESCOPO NORMALIZADO")
    trace_preview_count = int(shadow_decision_simulator.get("preview_near_approved_count", 0) or 0)
    trace_received_current = int(
        shadow_decision_simulator.get(
            "shadow_current_cycle_received_count",
            shadow_decision_simulator.get("shadow_candidates_received_count", 0),
        )
        or 0
    )
    trace_new_unique_current = int(
        shadow_decision_simulator.get(
            "shadow_current_cycle_new_unique_count",
            shadow_decision_simulator.get("shadow_candidates_unique_count", 0),
        )
        or 0
    )
    trace_duplicate_current = int(
        shadow_decision_simulator.get(
            "shadow_current_cycle_duplicate_count",
            shadow_decision_simulator.get("shadow_current_cycle_ignored_count", 0),
        )
        or 0
    )
    trace_already_analyzed = int(
        shadow_decision_simulator.get("shadow_current_cycle_already_analyzed_count", trace_duplicate_current) or 0
    )
    trace_analyzed_current = int(
        shadow_decision_simulator.get(
            "shadow_current_cycle_analyzed_new_count",
            shadow_decision_simulator.get("shadow_current_cycle_analyzed_count", 0),
        )
        or 0
    )
    trace_classified_current = int(
        shadow_decision_simulator.get(
            "shadow_current_cycle_classified_new_count",
            shadow_decision_simulator.get("shadow_current_cycle_classified_count", 0),
        )
        or 0
    )
    trace_unsafe_current = int(
        shadow_decision_simulator.get(
            "shadow_current_cycle_unsafe_new_count",
            shadow_decision_simulator.get("shadow_current_cycle_unsafe_count", 0),
        )
        or 0
    )
    trace_primary_current = int(
        shadow_decision_simulator.get(
            "shadow_current_cycle_primary_blocked_new_count",
            shadow_decision_simulator.get("shadow_primary_blocked_count", 0),
        )
        or 0
    )
    trace_secondary_current = int(
        shadow_decision_simulator.get(
            "shadow_current_cycle_secondary_blocked_new_count",
            shadow_decision_simulator.get("shadow_secondary_blocked_count", 0),
        )
        or 0
    )
    trace_accumulated_unique = int(
        shadow_decision_simulator.get(
            "shadow_accumulated_unique_candidates_count",
            shadow_decision_simulator.get("shadow_accumulated_candidates_count", 0),
        )
        or 0
    )
    trace_accumulated_unsafe = int(
        shadow_decision_simulator.get(
            "shadow_accumulated_unsafe_unique_count",
            shadow_decision_simulator.get("shadow_accumulated_unsafe_count", 0),
        )
        or 0
    )
    trace_accumulated_raw = int(
        shadow_decision_simulator.get(
            "shadow_accumulated_raw_received_count",
            shadow_decision_simulator.get("shadow_accumulated_received_count", trace_received_current),
        )
        or 0
    )
    trace_duplicate_ratio = float(shadow_decision_simulator.get("shadow_duplicate_ratio", 0.0) or 0.0)
    trace_raw_to_unique_ratio = float(shadow_decision_simulator.get("shadow_raw_to_unique_ratio", 0.0) or 0.0)
    trace_table_scope = shadow_decision_simulator.get("shadow_counts_scope") or "current_cycle_and_accumulated_recent"
    tr_c1, tr_c2, tr_c3, tr_c4 = st.columns(4)
    tr_c1.metric("Preview near-approved", trace_preview_count)
    tr_c2.metric("Recebidos no ciclo", trace_received_current)
    tr_c3.metric("Novos unicos no ciclo", trace_new_unique_current)
    tr_c4.metric("Duplicados no ciclo", trace_duplicate_current)
    tr_d1, tr_d2, tr_d3, tr_d4 = st.columns(4)
    tr_d1.metric("Analisados novos no ciclo", trace_analyzed_current)
    tr_d2.metric("Unsafe novos no ciclo", trace_unsafe_current)
    tr_d3.metric("Ja analisados anteriormente", trace_already_analyzed)
    tr_d4.metric("Candidatos unicos acumulados", trace_accumulated_unique)
    tr_e1, tr_e2, tr_e3, tr_e4 = st.columns(4)
    tr_e1.metric("Unsafe acumulados unicos", trace_accumulated_unsafe)
    tr_e2.metric("Recebidos brutos acumulados", trace_accumulated_raw)
    tr_e3.metric("Taxa de duplicidade", f"{trace_duplicate_ratio:.0%}")
    tr_e4.metric("Escopo da tabela", trace_table_scope)
    st.caption(
        f"Motivo principal de exclusao: {shadow_decision_simulator.get('shadow_dominant_block_reason') or '-'} | "
        f"Atual: {shadow_decision_simulator.get('dominant_exclusion_current_scope') or '-'} | "
        f"Acumulado: {shadow_decision_simulator.get('dominant_exclusion_accumulated_scope') or '-'} | "
        f"Primario novo: {trace_primary_current} | Secundario novo: {trace_secondary_current} | "
        f"Ignorados: {shadow_decision_simulator.get('shadow_ignored_reason') or '-'}"
    )
    st.caption(
        "Candidatos duplicados nao sao reanalisados como novos, mas continuam aparecendo no acumulado recente para auditoria."
    )
    if trace_received_current > 0 and trace_new_unique_current == 0 and trace_duplicate_current == trace_received_current:
        st.info(
            f"Este ciclo nao gerou novos candidatos unicos; os {trace_received_current} recebidos ja haviam sido analisados antes. "
            "A tabela de acumulados abaixo mostra candidatos recentes ja classificados."
        )
    if trace_raw_to_unique_ratio >= 10.0:
        st.info(
            "Alto volume bruto recebido por repeticao de ciclos; leitura estrategica deve usar candidatos unicos, "
            "nao recebimentos brutos."
        )
    if bool(shadow_decision_simulator.get("shadow_scope_warning", shadow_decision_simulator.get("shadow_counter_warning", False))):
        st.warning(
            "Aviso de consistencia de escopo shadow: "
            f"{shadow_decision_simulator.get('shadow_scope_warning_reason') or shadow_decision_simulator.get('shadow_counter_warning_reason') or 'verificar contadores'}"
        )
    def _shadow_trace_row(row, default_scope):
        return {
            "symbol": row.get("symbol"),
            "setup": row.get("strategy"),
            "score": row.get("current_score"),
            "score_gap": row.get("score_gap"),
            "raw_near_approved": row.get("raw_near_approved"),
            "duplicate_candidate": row.get("duplicate_candidate"),
            "already_seen": row.get("already_seen"),
            "analyzed_this_cycle": row.get("analyzed_this_cycle"),
            "analyzed_previously": row.get("analyzed_previously"),
            "classified_by_shadow": row.get("classified_by_shadow"),
            "class": row.get("candidate_class"),
            "safe_candidate": row.get("safe_candidate"),
            "would_enter": row.get("shadow_would_enter"),
            "primary_blockers": ", ".join(list(row.get("primary_blocker_codes", row.get("primary_blockers", [])) or [])),
            "secondary_blockers": ", ".join(list(row.get("secondary_blocker_codes", row.get("secondary_blockers", [])) or [])),
            "why_not_safe": row.get("why_not_safe"),
            "why_would_not_enter": row.get("why_would_not_enter") or row.get("shadow_block_reason"),
            "count_scope": row.get("count_scope") or default_scope,
        }

    current_shadow_rows = [
        _shadow_trace_row(row, "current_cycle")
        for row in list(shadow_decision_simulator.get("shadow_current_cycle_candidates", []) or [])
        if isinstance(row, dict)
    ]
    accumulated_shadow_rows = [
        _shadow_trace_row(row, "accumulated_recent")
        for row in list(
            shadow_decision_simulator.get(
                "shadow_accumulated_recent_candidates",
                shadow_decision_simulator.get("shadow_recent_candidates", []),
            )
            or []
        )
        if isinstance(row, dict)
    ]
    st.markdown("###### CANDIDATOS NOVOS DO CICLO")
    if current_shadow_rows:
        st.dataframe(pd.DataFrame(current_shadow_rows[:8]), hide_index=True, use_container_width=True)
    else:
        st.info("Nenhum candidato novo unico neste ciclo.")
    st.markdown("###### CANDIDATOS ACUMULADOS RECENTES")
    if accumulated_shadow_rows:
        st.dataframe(pd.DataFrame(accumulated_shadow_rows[:8]), hide_index=True, use_container_width=True)
    else:
        st.info("Nenhum candidato acumulado recente para auditoria.")
if phase2_fine_tune:
    st.subheader("Ajuste Fino FASE 2")
    st.caption(
        "Relaxamento pequeno, auditavel e reversivel. PAPER only; nao altera threshold global, "
        "nao muda broker e nao remove guards."
    )
    fine_status = "Ativo" if bool(phase2_fine_tune.get("fine_tune_enabled", False)) else "Inativo"
    ft_c1, ft_c2, ft_c3, ft_c4 = st.columns(4)
    ft_c1.metric("Status", fine_status)
    ft_c2.metric("Alvo", str(phase2_fine_tune.get("fine_tune_target") or "-"))
    ft_c3.metric("Aplicado no ciclo", str(int(phase2_fine_tune.get("fine_tune_applied_count", 0) or 0)))
    ft_c4.metric("Bloqueado por guard", str(int(phase2_fine_tune.get("fine_tune_blocked_count", 0) or 0)))
    st.caption(f"Motivo: {phase2_fine_tune.get('fine_tune_reason') or 'Sem motivo registrado.'}")
    st.caption(
        f"Antes: {phase2_fine_tune.get('fine_tune_before') or '-'} | "
        f"Depois: {phase2_fine_tune.get('fine_tune_after') or '-'}"
    )
    if phase2_fine_tune.get("fine_tune_last_guard_reason"):
        st.caption(f"Ultimo guard acionado: {phase2_fine_tune.get('fine_tune_last_guard_reason')}")
    st.caption("Observacao: PAPER only, reversivel e sem autoridade para ordem real.")
if phase2_1_fine_tune:
    st.subheader("Ajuste Fino FASE 2.1")
    st.caption(
        "Ajuste conservador para multiplas falhas pequenas de momentum/confirmacao secundaria. "
        "PAPER only, reversivel, sem ordem real e sem reducao global de score."
    )
    ft21_status = "Ativo" if bool(phase2_1_fine_tune.get("phase2_1_fine_tune_enabled", False)) else "Inativo"
    ft21_gap = phase2_1_fine_tune.get("phase2_1_fine_tune_score_gap")
    ft21_gap_label = "-" if ft21_gap is None else f"{float(ft21_gap):.4f}"
    ft21_c1, ft21_c2, ft21_c3, ft21_c4 = st.columns(4)
    ft21_c1.metric("Status", ft21_status)
    ft21_c2.metric("Alvo", str(phase2_1_fine_tune.get("phase2_1_fine_tune_target") or "-"))
    ft21_c3.metric("Aplicado no ciclo", str(int(phase2_1_fine_tune.get("phase2_1_fine_tune_applied_count", 0) or 0)))
    ft21_c4.metric("Bloqueado no ciclo", str(int(phase2_1_fine_tune.get("phase2_1_fine_tune_blocked_count", 0) or 0)))
    st.caption(
        f"Ultima decisao: {phase2_1_fine_tune.get('phase2_1_fine_tune_last_decision') or '-'} | "
        f"Ultimo guard: {phase2_1_fine_tune.get('phase2_1_fine_tune_last_guard') or '-'} | "
        f"Score gap: {ft21_gap_label}"
    )
    st.caption(
        "Motivos permitidos: "
        f"{symbol_list_label(phase2_1_fine_tune.get('phase2_1_fine_tune_allowed_reasons'))} | "
        "Motivos bloqueadores: "
        f"{symbol_list_label(phase2_1_fine_tune.get('phase2_1_fine_tune_blocked_reasons'))}"
    )
    st.caption(f"Motivo: {phase2_1_fine_tune.get('phase2_1_fine_tune_reason') or 'Sem motivo registrado.'}")
    st.caption("Observacao: nao altera min_signal_score global, broker, webhook, FASE 3 ou execucao real.")

val_actions1, val_actions2 = st.columns(2)
with val_actions1:
    if st.button("Recalcular validacao swing", use_container_width=True):
        validation_report = refresh_swing_validation_cycle()
        st.info("Validacao swing recalculada.")
with val_actions2:
    if st.button("Reiniciar ciclo swing 10 dias", use_container_width=True):
        validation_report = reset_swing_validation_cycle()
        st.warning("Novo ciclo swing iniciado a partir de agora.")

validation_state = load_bot_state().get("validation", {}) or {}
st.caption(
    f"Inicio do ciclo: {validation_report.get('validation_started_at') or 'Sem registro'} | "
    f"Email final enviado: {'Sim' if validation_state.get('final_email_sent') else 'Nao'}"
)

if validation_report.get("final_validation_reason"):
    st.caption(f"Motivo final: {validation_report.get('final_validation_reason')}")

panel_c1, panel_c2 = st.columns(2)
with panel_c1:
    st.write("**Erros identificados**")
    errors = validation_report.get("errors", []) or []
    if errors:
        for item in errors:
            st.caption(f"- {item}")
    else:
        st.caption("Sem erros relevantes ate aqui.")

    st.write("**Ativos em destaque**")
    best_assets = pd.DataFrame(validation_report.get("best_assets", []) or [])
    if best_assets.empty:
        st.info("Sem ativos consistentes suficientes ate o momento.")
    else:
        st.dataframe(best_assets, use_container_width=True)

with panel_c2:
    st.write("**Acertos identificados**")
    successes = validation_report.get("successes", []) or []
    if successes:
        for item in successes:
            st.caption(f"- {item}")
    else:
        st.caption("Sem acertos destacados ate aqui.")

    st.write("**Ativos problematicos**")
    worst_assets = pd.DataFrame(validation_report.get("worst_assets", []) or [])
    if worst_assets.empty:
        st.info("Sem ativos problematicos suficientes ate o momento.")
    else:
        st.dataframe(worst_assets, use_container_width=True)

st.write("**Sugestoes analiticas**")
suggestions = validation_report.get("suggestions", []) or []
if suggestions:
    for suggestion in suggestions:
        st.caption(f"- {suggestion.get('message')}")
else:
    st.caption("Sem sugestoes novas neste momento.")

before_after = validation_report.get("before_after_comparison", {}) or {}
if before_after:
    before_after_c1, before_after_c2 = st.columns(2)
    before_payload = before_after.get("before", {}) or {}
    after_payload = before_after.get("after", {}) or {}
    with before_after_c1:
        st.write("**Antes da fase de ajuste**")
        st.caption(
            f"Trades: {int(before_payload.get('trades', 0) or 0)} | "
            f"Win rate: {float(before_payload.get('win_rate', 0.0) or 0.0) * 100:.2f}% | "
            f"PnL: R$ {float(before_payload.get('pnl', 0.0) or 0.0):,.2f}"
        )
    with before_after_c2:
        st.write("**Depois da fase de ajuste**")
        st.caption(
            f"Trades: {int(after_payload.get('trades', 0) or 0)} | "
            f"Win rate: {float(after_payload.get('win_rate', 0.0) or 0.0) * 100:.2f}% | "
            f"PnL: R$ {float(after_payload.get('pnl', 0.0) or 0.0):,.2f}"
        )
    notes = before_after.get("notes", []) or []
    if notes:
        for note in notes:
            st.caption(f"- {note}")

st.subheader("Contexto de mercado cripto")
st.caption("Filtro auxiliar de PAPER TRADING. O contexto nao dispara ordens; ele apenas endurece sinais fracos de cripto.")

context_c1, context_c2, context_c3, context_c4 = st.columns(4)
context_c1.metric("Status atual", market_context_label(market_context_state.get("market_context_status")))
context_c2.metric("Score", f"{float(market_context_state.get('market_context_score', 50.0) or 50.0):.1f}")
context_c3.metric(
    "Sinais bloqueados",
    str(int(val_metrics.get("context_blocked_signals", 0) or 0)),
)
context_c4.metric("PAPER", "Ativo")

st.caption(f"Motivo: {market_context_state.get('market_context_reason') or 'Sem motivo registrado.'}")
st.caption(f"Impacto no robo: {market_context_state.get('market_context_impact') or 'Sem impacto adicional.'}")
if validation_report.get("context_impact_estimate"):
    st.caption(f"Impacto estimado no periodo: {validation_report.get('context_impact_estimate')}")

context_status_counts = dict(val_metrics.get("context_status_counts", {}) or {})
if context_status_counts:
    st.caption(
        "Contexto por periodo: "
        f"FAVORAVEL={int(context_status_counts.get('FAVORAVEL', 0) or 0)} | "
        f"NEUTRO={int(context_status_counts.get('NEUTRO', 0) or 0)} | "
        f"DESFAVORAVEL={int(context_status_counts.get('DESFAVORAVEL', 0) or 0)} | "
        f"CRITICO={int(context_status_counts.get('CRITICO', 0) or 0)}"
    )
if market_context_state.get("market_context_regime"):
    watchlist_consistency = market_context_state.get("watchlist_consistency")
    watchlist_consistency_label = (
        "-"
        if watchlist_consistency is None
        else f"{float(watchlist_consistency or 0.0) * 100:.1f}%"
    )
    st.caption(
        "Regime observado: "
        f"{str(market_context_state.get('market_context_regime') or 'indefinido').capitalize()} | "
        f"Consistencia da watchlist: {watchlist_consistency_label}"
    )

feed_status_label = market_data_status_label(operational_market_state)
if feed_status_label == "LIVE":
    st.success(
        f"Feed classificado como LIVE via {market_data_provider_label(operational_market_state)}."
    )
elif feed_status_label == "DELAYED":
    st.warning(
        f"Feed classificado como DELAYED: usando {market_data_source_label(operational_market_state)}. "
        "A UI deixa isso explicito e o worker nao trata dado atrasado como se fosse ao vivo."
    )
elif feed_status_label == "FALLBACK":
    st.error(
        f"Feed classificado como FALLBACK via {market_data_provider_label(operational_market_state)}. "
        "O worker continua online, mas evita operar com dado nao confiavel."
    )
else:
    st.info("Status do feed ainda nao determinado.")

info_c1, info_c2, info_c3, info_c4, info_c5 = st.columns(5)
info_c1.metric("Status do bot", bot_status_label(state.get("bot_status")))
info_c2.metric("Provider de dados", market_data_provider_label(operational_market_state))
info_c3.metric("Status do feed", feed_status_label)
info_c4.metric("Fonte atual", market_data_source_label(operational_market_state))
info_c5.metric("Modo do broker", broker_mode_label(broker_state.get("mode")))

diag_c1, diag_c2 = st.columns(2)
with diag_c1:
    st.caption(
        f"Ultimo sync do feed: {format_market_timestamp(operational_market_state.get('last_sync_at'))}"
    )
    st.caption(
        f"Ultimo sucesso: {format_market_timestamp(operational_market_state.get('last_success_at'))}"
    )
with diag_c2:
    st.caption(f"Broker provider: {str(broker_state.get('provider', 'paper')).upper()}")
    st.caption(f"Status do broker: {broker_status_label(broker_state.get('status'))}")

if operational_market_state.get("last_error"):
    st.caption(f"Ultimo alerta de mercado: {operational_market_state.get('last_error')}")
if broker_state.get("warning"):
    st.caption(f"Observacao do broker: {broker_state.get('warning')}")
if production_state.get("last_alert_error"):
    st.caption(f"Ultima falha no envio de alerta: {production_state.get('last_alert_error')}")

worker_feed_quality = build_feed_quality_snapshot(operational_market_state)
chart_feed_quality = build_feed_quality_snapshot(chart_market_state)
st.subheader("Qualidade do feed")
st.caption("Separacao explicita entre o feed operacional do worker e o ultimo feed visual usado pelo grafico do Trader.")
feed_c1, feed_c2, feed_c3, feed_c4 = st.columns(4)
feed_c1.metric("Sucesso Twelve Data", pct_label(worker_feed_quality.get("twelvedata_success_rate")))
feed_c2.metric("Ultimo sucesso real", format_market_timestamp(worker_feed_quality.get("last_success_at")))
feed_c3.metric("Ativos live no ciclo", f"{int(worker_feed_quality.get('live_count') or 0)}/{int(worker_feed_quality.get('total_symbols') or 0)}")
feed_c4.metric("Ativos em fallback", str(int(worker_feed_quality.get("fallback_count") or 0)))

feed_detail_c1, feed_detail_c2 = st.columns(2)
with feed_detail_c1:
    st.write("**Feed operacional do worker**")
    st.write(
        f"Status: {market_data_status_label(operational_market_state)} | "
        f"Fonte: {market_data_source_label(operational_market_state)}"
    )
    st.write(f"Ativos live: {symbol_list_label(worker_feed_quality.get('live_symbols'))}")
    st.write(f"Ativos em fallback: {symbol_list_label(worker_feed_quality.get('fallback_symbols'))}")
    if worker_feed_quality.get("fallback_reason"):
        st.caption(f"Motivo do fallback operacional: {worker_feed_quality.get('fallback_reason')}")
    if worker_feed_quality.get("quality_message"):
        st.caption(worker_feed_quality.get("quality_message"))
with feed_detail_c2:
    st.write("**Feed do grafico do Trader**")
    st.write(
        f"Status: {market_data_status_label(chart_market_state)} | "
        f"Fonte: {market_data_source_label(chart_market_state)}"
    )
    st.write(f"Ativos live: {symbol_list_label(chart_feed_quality.get('live_symbols'))}")
    st.write(f"Ativos em fallback: {symbol_list_label(chart_feed_quality.get('fallback_symbols'))}")
    if chart_feed_quality.get("fallback_reason"):
        st.caption(f"Motivo do fallback visual: {chart_feed_quality.get('fallback_reason')}")
    st.caption("Fallback apenas visual do grafico nao altera o feed operacional do worker.")

with st.expander("Diagnostico do feed"):
    st.write("**Contexto operacional (worker):**")
    st.write(f"Classificacao atual: {market_data_status_label(operational_market_state)}")
    st.write(f"Taxonomia legada: {market_data_legacy_label(operational_market_state)}")
    st.write(f"Solicitado por: {operational_market_state.get('requested_by') or 'Sem registro'}")
    st.write(f"Ativos monitorados: {', '.join(operational_market_state.get('symbols', []) or []) or 'Sem registro'}")
    st.write(f"Ultimo sync: {format_market_timestamp(operational_market_state.get('last_sync_at'))}")
    st.code(
        json.dumps(operational_market_state.get("source_breakdown", {}) or {}, ensure_ascii=False, indent=2),
        language="json",
    )
    td_diag = twelvedata_diagnostic_payload(operational_market_state)
    if td_diag:
        st.write("**Diagnostico Twelve Data (worker):**")
        st.write(f"Build ativo: {td_diag.get('build_label') or 'Sem registro'}")
        st.write(f"Servico: {td_diag.get('service_name') or 'Sem registro'}")
        st.write(f"API key lida pelo processo: {'Sim' if td_diag.get('api_key_present') else 'Nao'}")
        st.write(f"Tamanho da chave: {int(td_diag.get('api_key_length') or 0)}")
        st.write(f"Base URL: {td_diag.get('api_base') or 'Sem registro'}")
        st.write(f"Host resolvido: {td_diag.get('api_base_host') or 'Sem registro'}")
        st.write(f"Base URL valida: {'Sim' if td_diag.get('api_base_valid') else 'Nao'}")
        st.write(f"Simbolo amostra: {td_diag.get('sample_symbol') or 'Sem registro'}")
        st.write(f"Simbolo normalizado: {td_diag.get('sample_normalized_symbol') or 'Sem registro'}")
        st.write(f"Request montado: {'Sim' if td_diag.get('request_built') else 'Nao'}")
        st.write(f"Request saiu do processo: {'Sim' if td_diag.get('request_attempted') else 'Nao'}")
        st.write(f"Resposta recebida: {'Sim' if td_diag.get('response_received') else 'Nao'}")
        st.write(f"Sucessos Twelve Data no ciclo: {int(td_diag.get('success_count') or 0)}")
        st.write(f"Ultimo estagio: {td_diag.get('last_stage') or 'Sem registro'}")
        if td_diag.get("http_statuses"):
            st.write(f"HTTP status observados: {', '.join(str(item) for item in (td_diag.get('http_statuses') or []))}")
        if td_diag.get("payload_codes"):
            st.write(f"Codigos retornados: {', '.join(str(item) for item in (td_diag.get('payload_codes') or []))}")
        if td_diag.get("last_error"):
            st.write(f"Ultimo erro Twelve Data: {td_diag.get('last_error')}")
        st.code(json.dumps(td_diag, ensure_ascii=False, indent=2), language="json")
    if chart_market_state:
        st.write("**Ultimo contexto visual (Trader):**")
        st.write(f"Classificacao atual: {market_data_status_label(chart_market_state)}")
        st.write(f"Taxonomia legada: {market_data_legacy_label(chart_market_state)}")
        st.write(f"Solicitado por: {chart_market_state.get('requested_by') or 'Sem registro'}")
        st.write(f"Ativos monitorados: {', '.join(chart_market_state.get('symbols', []) or []) or 'Sem registro'}")
        st.write(f"Ultimo sync: {format_market_timestamp(chart_market_state.get('last_sync_at'))}")
        st.code(
            json.dumps(chart_market_state.get("source_breakdown", {}) or {}, ensure_ascii=False, indent=2),
            language="json",
        )

st.markdown("### AUDITORIA DO WORKER")
st.write("controle_bot_ui_version: audit_v2")
st.caption(
    "Os campos abaixo mostram o commit real do deploy em producao, nao necessariamente o SHA da branch original."
)
worker_confirmed = worker_instrumentation_confirmed(operational_market_state)
if worker_confirmed:
    st.success("WORKER INSTRUMENTADO CONFIRMADO")
else:
    st.error("WORKER INSTRUMENTADO NAO COMPROVADO")

audit_c1, audit_c2 = st.columns(2)
with audit_c1:
    st.write(f"ui_audit_probe: {audit_display_value(operational_market_state.get('ui_audit_probe'))}")
    st.write(f"Build do deploy: {audit_display_value(operational_market_state.get('build_active'))}")
    st.write(f"SHA do deploy: {audit_display_value(operational_market_state.get('git_sha'))}")
    st.write(f"SHA de origem: {source_commit_display_value(operational_market_state.get('source_commit_sha'))}")
    st.write(f"Build timestamp: {audit_display_value(operational_market_state.get('build_timestamp'))}")
    st.write(f"Runtime iniciado em: {audit_display_value(operational_market_state.get('runtime_started_at'))}")
    st.write(f"Servico: {audit_display_value(operational_market_state.get('service_name'))}")
    st.write(f"Papel do processo: {audit_display_value(operational_market_state.get('process_role'))}")
    st.write(f"Ultima gravacao do estado: {audit_display_value(operational_market_state.get('state_written_at'))}")
    st.write(f"Writer do estado: {audit_display_value(operational_market_state.get('state_writer'))}")
    st.write(
        "SHA do deploy gravado no estado: "
        f"{audit_display_value(operational_market_state.get('state_build_sha'))}"
    )
    st.write(f"Schema do estado: {audit_display_value(operational_market_state.get('state_schema_version'))}")
with audit_c2:
    st.write(f"API key presente: {audit_display_value(operational_market_state.get('api_key_present'))}")
    st.write(f"Request preparado: {audit_display_value(operational_market_state.get('request_prepared'))}")
    st.write(f"Request tentado: {audit_display_value(operational_market_state.get('request_attempted'))}")
    st.write(f"Resposta recebida: {audit_display_value(operational_market_state.get('response_received'))}")
    st.write(f"Status code: {audit_display_value(operational_market_state.get('response_status_code'))}")
    st.write(f"Ultimo estagio: {audit_display_value(operational_market_state.get('last_stage'))}")
    st.write(f"Ultimo erro: {audit_display_value(operational_market_state.get('last_error'))}")
    requested_symbols = operational_market_state.get("requested_symbols") or operational_market_state.get("symbols") or []
    st.write(
        "Simbolos solicitados: "
        f"{audit_display_value(', '.join(str(item) for item in requested_symbols) if requested_symbols else '')}"
    )
    st.write(f"Provider efetivo: {audit_display_value(operational_market_state.get('provider_effective') or operational_market_state.get('provider'))}")

with st.expander("Diagnostico do broker"):
    st.write(f"**Provider:** {str(broker_state.get('provider', 'paper')).upper()}")
    st.write(f"**Status:** {broker_status_label(broker_state.get('status'))}")
    st.write(f"**Modo configurado:** {broker_mode_label(broker_state.get('configured_mode'))}")
    st.write(f"**Modo efetivo:** {broker_mode_label(broker_state.get('effective_mode'))}")
    st.write(f"**Conta:** {broker_state.get('account_id') or 'Sem registro'}")
    st.write(f"**Base URL:** {broker_state.get('base_url') or 'Sem registro'}")
    st.write(f"**API key configurada:** {'Sim' if broker_state.get('api_key_configured') else 'Nao'}")
    st.write(f"**API secret configurada:** {'Sim' if broker_state.get('api_secret_configured') else 'Nao'}")
    st.write(f"**Pode enviar ordens agora:** {'Sim' if broker_state.get('can_submit_orders') else 'Nao'}")
    st.write(f"**Execucao real habilitada nesta etapa:** {'Sim' if broker_state.get('execution_enabled') else 'Nao'}")

status_options = ["RUNNING", "PAUSED", "STOPPED"]
mode_options = ["Automatico", "Semi-automatico"]

status_index = status_options.index(state["bot_status"]) if state.get("bot_status") in status_options else 1
mode_index = mode_options.index(state["bot_mode"]) if state.get("bot_mode") in mode_options else 0

status = st.radio("Status do bot", status_options, index=status_index, format_func=bot_status_label, horizontal=True)
mode = st.selectbox("Modo do bot", mode_options, index=mode_index)

c1, c2, c3, c4 = st.columns(4)
with c1:
    if st.button("Salvar status", use_container_width=True):
        state["bot_status"] = status
        state["bot_mode"] = mode
        save_bot_state(state)
        log_event("INFO", f"Status atualizado para {status} / {mode}")
        st.success("Status salvo.")

with c2:
    if st.button("Executar ciclo agora", use_container_width=True):
        result = run_trader_cycle()
        cycle = result.get("cycle_result", {}) or {}
        risk = result.get("risk", {}) or {}
        blocked = bool(risk.get("daily_loss_block_active", False))
        st.success(
            f"Ciclo executado. Trades feitos: {int(cycle.get('trades_executed', 0) or 0)} | "
            f"Feed: {market_data_status_label(cycle.get('market_data_status'))} | "
            f"Trava diaria: {'BLOQUEADA' if blocked else 'LIBERADA'}"
        )

with c3:
    if st.button("Pausar entradas", use_container_width=True):
        state["bot_status"] = "PAUSED"
        save_bot_state(state)
        st.warning("Bot pausado.")

with c4:
    if st.button("Reset geral", use_container_width=True):
        reset_state()
        st.error("Estado resetado.")

st.info("Ligado abre e gerencia posicoes. Pausado nao abre novas posicoes. Parado paralisa tudo.")
