from __future__ import annotations

import json

import streamlit as st

from core.auth.guards import render_auth_toolbar, require_admin
from core.state_store import load_bot_state, log_event, reset_state, save_bot_state
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


current_user = require_admin()
render_auth_toolbar()

st.title("Controle Operacional")
st.caption(f"Painel administrativo do trader. Usuario: {current_user['username']}")

state = load_bot_state()
security_state = state.get("security", {}) or {}
market_data_state = state.get("market_data", {}) or {}
broker_state = state.get("broker", {}) or {}

if bool(security_state.get("real_mode_enabled", False)):
    st.warning("Real trading enabled")

status_label = market_data_status_label(market_data_state.get("status"))
if str(market_data_state.get("status", "")).lower() == "healthy":
    st.success(f"Feed de mercado saudavel via {str(market_data_state.get('provider', 'yahoo')).upper()}.")
elif str(market_data_state.get("status", "")).lower() in {"cached", "degraded"}:
    st.warning(
        f"Feed degradado: usando {market_data_source_label(market_data_state.get('last_source'))}. "
        "Operacoes novas podem ser reduzidas ou bloqueadas conforme a estrategia."
    )
elif str(market_data_state.get("status", "")).lower() == "error":
    st.error(
        f"Feed em fallback via {str(market_data_state.get('provider', 'yahoo')).upper()}. "
        "O worker continua online, mas evita operar com dado nao confiavel."
    )
else:
    st.info("Status do feed ainda nao determinado.")

info_c1, info_c2, info_c3, info_c4, info_c5 = st.columns(5)
info_c1.metric("Status do bot", bot_status_label(state.get("bot_status")))
info_c2.metric("Provider de dados", str(market_data_state.get("provider", "yahoo")).upper())
info_c3.metric("Status do feed", status_label)
info_c4.metric("Fonte atual", market_data_source_label(market_data_state.get("last_source")))
info_c5.metric("Modo do broker", broker_mode_label(broker_state.get("mode")))

diag_c1, diag_c2 = st.columns(2)
with diag_c1:
    st.caption(f"Ultimo sync do feed: {market_data_state.get('last_sync_at') or 'Sem registro'}")
    st.caption(f"Ultimo sucesso: {market_data_state.get('last_success_at') or 'Sem registro'}")
with diag_c2:
    st.caption(f"Broker provider: {str(broker_state.get('provider', 'paper')).upper()}")
    st.caption(f"Status do broker: {str(broker_state.get('status', 'paper')).title()}")

if market_data_state.get("last_error"):
    st.caption(f"Ultimo alerta de mercado: {market_data_state.get('last_error')}")

with st.expander("Diagnostico do feed"):
    st.write(f"**Solicitado por:** {market_data_state.get('requested_by') or 'Sem registro'}")
    st.write(f"**Ativos monitorados:** {', '.join(market_data_state.get('symbols', []) or []) or 'Sem registro'}")
    st.write("**Distribuicao por fonte:**")
    st.code(
        json.dumps(market_data_state.get("source_breakdown", {}) or {}, ensure_ascii=False, indent=2),
        language="json",
    )

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
