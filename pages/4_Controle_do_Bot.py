from __future__ import annotations

import json

import streamlit as st

from core.alerts import send_email_alert
from core.auth.guards import render_auth_toolbar, require_admin
from core.broker import broker_status_label, probe_broker_status
from core.config import ALERT_EMAIL_ENABLED, PRODUCTION_MODE
from core.production_monitor import evaluate_production_health
from core.state_store import (
    load_bot_state,
    log_event,
    reset_state,
    save_bot_state,
    update_broker_status,
    update_production_status,
)
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

if bool(security_state.get("real_mode_enabled", False)):
    st.warning("Real trading enabled")

st.subheader("Modo producao")
st.caption("Monitoramento de saude, alertas e diagnostico operacional. PAPER TRADING permanece como padrao nesta etapa.")

production_mode_text = "Ativo" if PRODUCTION_MODE else "Inativo"
alert_mode_text = "Ativo" if ALERT_EMAIL_ENABLED else "Inativo"
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
