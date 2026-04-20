from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from core.alerts import send_email_alert
from core.auth.guards import render_auth_toolbar, require_admin
from core.broker import broker_status_label, probe_broker_status
from core.config import ALERT_EMAIL_ENABLED, ALERT_EMAIL_FROM, ALERT_EMAIL_PROVIDER, PRODUCTION_MODE, SMTP_USERNAME
from core.production_monitor import evaluate_production_health
from core.state_store import (
    load_bot_state,
    log_event,
    reset_state,
    save_bot_state,
    update_broker_status,
    update_production_status,
)
from core.swing_validation import refresh_swing_validation_cycle, reset_swing_validation_cycle
from engines.trader_engine import run_trader_cycle


def market_data_status_label(raw_status: str | None) -> str:
    labels = {
        "healthy": "Saudavel",
        "degraded": "Degradado",
        "cached": "Cache",
        "error": "Fallback",
        "unknown": "Desconhecido",
    }
    return labels.get(str(raw_status or "unknown").strip().lower(), str(raw_status or "Desconhecido"))


def market_data_source_label(raw_source: str | None) -> str:
    labels = {
        "market": "Mercado ao vivo",
        "cached": "Cache reaproveitado",
        "fallback": "Fallback sintetico",
        "mixed": "Misto",
        "unknown": "Desconhecido",
    }
    return labels.get(str(raw_source or "unknown").strip().lower(), str(raw_source or "Desconhecido"))


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

state = load_bot_state()
security_state = state.get("security", {}) or {}
market_data_state = state.get("market_data", {}) or {}
market_contexts = market_data_state.get("contexts", {}) or {}
operational_market_state = market_contexts.get("worker_cycle") or market_data_state
chart_market_state = market_contexts.get("trader_chart") or {}
broker_state = update_broker_status(probe_broker_status(security_state, requested_by="admin_panel"))
state = load_bot_state()
production_state = update_production_status(evaluate_production_health(state))
state = load_bot_state()
production_state = state.get("production", {}) or {}
operational_market_state = (state.get("market_data", {}) or {}).get("contexts", {}).get("worker_cycle") or state.get(
    "market_data",
    {},
)
chart_market_state = (state.get("market_data", {}) or {}).get("contexts", {}).get("trader_chart") or {}
broker_state = state.get("broker", {}) or broker_state
validation_report = refresh_swing_validation_cycle()
state = load_bot_state()
validation_state = state.get("validation", {}) or {}
market_context_state = state.get("market_context", {}) or {}

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

status_label = market_data_status_label(operational_market_state.get("status"))
if str(operational_market_state.get("status", "")).lower() == "healthy":
    st.success(f"Feed de mercado saudavel via {str(operational_market_state.get('provider', 'yahoo')).upper()}.")
elif str(operational_market_state.get("status", "")).lower() in {"cached", "degraded"}:
    st.warning(
        f"Feed degradado: usando {market_data_source_label(operational_market_state.get('last_source'))}. "
        "Operacoes novas podem ser reduzidas ou bloqueadas conforme a estrategia."
    )
elif str(operational_market_state.get("status", "")).lower() == "error":
    st.error(
        f"Feed em fallback via {str(operational_market_state.get('provider', 'yahoo')).upper()}. "
        "O worker continua online, mas evita operar com dado nao confiavel."
    )
else:
    st.info("Status do feed ainda nao determinado.")

info_c1, info_c2, info_c3, info_c4, info_c5 = st.columns(5)
info_c1.metric("Status do bot", bot_status_label(state.get("bot_status")))
info_c2.metric("Provider de dados", str(operational_market_state.get("provider", "yahoo")).upper())
info_c3.metric("Status do feed", status_label)
info_c4.metric("Fonte atual", market_data_source_label(operational_market_state.get("last_source")))
info_c5.metric("Modo do broker", broker_mode_label(broker_state.get("mode")))

diag_c1, diag_c2 = st.columns(2)
with diag_c1:
    st.caption(f"Ultimo sync do feed: {operational_market_state.get('last_sync_at') or 'Sem registro'}")
    st.caption(f"Ultimo sucesso: {operational_market_state.get('last_success_at') or 'Sem registro'}")
with diag_c2:
    st.caption(f"Broker provider: {str(broker_state.get('provider', 'paper')).upper()}")
    st.caption(f"Status do broker: {broker_status_label(broker_state.get('status'))}")

if operational_market_state.get("last_error"):
    st.caption(f"Ultimo alerta de mercado: {operational_market_state.get('last_error')}")
if broker_state.get("warning"):
    st.caption(f"Observacao do broker: {broker_state.get('warning')}")
if production_state.get("last_alert_error"):
    st.caption(f"Ultima falha no envio de alerta: {production_state.get('last_alert_error')}")

with st.expander("Diagnostico do feed"):
    st.write("**Contexto operacional (worker):**")
    st.write(f"Solicitado por: {operational_market_state.get('requested_by') or 'Sem registro'}")
    st.write(f"Ativos monitorados: {', '.join(operational_market_state.get('symbols', []) or []) or 'Sem registro'}")
    st.code(
        json.dumps(operational_market_state.get("source_breakdown", {}) or {}, ensure_ascii=False, indent=2),
        language="json",
    )
    if chart_market_state:
        st.write("**Ultimo contexto visual (Trader):**")
        st.write(f"Solicitado por: {chart_market_state.get('requested_by') or 'Sem registro'}")
        st.write(f"Ativos monitorados: {', '.join(chart_market_state.get('symbols', []) or []) or 'Sem registro'}")
        st.code(
            json.dumps(chart_market_state.get("source_breakdown", {}) or {}, ensure_ascii=False, indent=2),
            language="json",
        )

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
        st.success(
            f"Ciclo executado. Trades feitos: {int(cycle.get('trades_executed', 0) or 0)} | "
            f"Feed: {market_data_status_label((cycle.get('market_data_status') or {}).get('status'))}"
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
