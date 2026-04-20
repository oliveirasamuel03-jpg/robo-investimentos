from __future__ import annotations

import streamlit as st

from core.auth.guards import render_auth_toolbar, require_admin
from core.state_store import load_bot_state, log_event, reset_state, save_bot_state
from engines.trader_engine import run_trader_cycle


current_user = require_admin()
render_auth_toolbar()

st.title("Controle Operacional")
st.caption(f"Painel administrativo do trader. Usuario: {current_user['username']}")

state = load_bot_state()
if bool((state.get("security", {}) or {}).get("real_mode_enabled", False)):
    st.warning("Real trading enabled")

market_data_state = state.get("market_data", {}) or {}
broker_state = state.get("broker", {}) or {}

info_c1, info_c2, info_c3, info_c4 = st.columns(4)
info_c1.metric("Provider de dados", str(market_data_state.get("provider", "yahoo")).upper())
info_c2.metric("Status do feed", str(market_data_state.get("status", "unknown")).title())
info_c3.metric("Fonte atual", str(market_data_state.get("last_source", "unknown")).title())
info_c4.metric("Broker", f"{str(broker_state.get('provider', 'paper')).upper()} / {str(broker_state.get('mode', 'paper')).upper()}")

if market_data_state.get("last_error"):
    st.caption(f"Ultimo alerta de mercado: {market_data_state.get('last_error')}")

status_options = ["RUNNING", "PAUSED", "STOPPED"]
mode_options = ["Automatico", "Semi-automatico"]

status_index = status_options.index(state["bot_status"]) if state.get("bot_status") in status_options else 1
mode_index = mode_options.index(state["bot_mode"]) if state.get("bot_mode") in mode_options else 0

status = st.radio("Status do bot", status_options, index=status_index)
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
    if st.button("Rodar trader", use_container_width=True):
        result = run_trader_cycle()
        st.write(result)
with c3:
    if st.button("Pausar entradas", use_container_width=True):
        state["bot_status"] = "PAUSED"
        save_bot_state(state)
        st.warning("Bot pausado.")
with c4:
    if st.button("Reset geral", use_container_width=True):
        reset_state()
        st.error("Estado resetado.")

st.info("RUNNING abre e gerencia posicoes. PAUSED nao abre novas posicoes. STOPPED paralisa tudo.")
