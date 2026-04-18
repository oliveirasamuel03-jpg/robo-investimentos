from __future__ import annotations

import streamlit as st

from core.state_store import load_bot_state, log_event, reset_state, save_bot_state
from engines.investment_research_engine import run_investment_pipeline
from engines.trader_engine import run_trader_cycle

st.title("Controle do Bot")
state = load_bot_state()

status = st.radio("Status do bot", ["RUNNING", "PAUSED", "STOPPED"], index=["RUNNING", "PAUSED", "STOPPED"].index(state["bot_status"]))
mode = st.selectbox("Modo do bot", ["Automático", "Semi-automático"], index=["Automático", "Semi-automático"].index(state["bot_mode"]))

c1, c2, c3, c4 = st.columns(4)
with c1:
    if st.button("Salvar status", use_container_width=True):
        state["bot_status"] = status
        state["bot_mode"] = mode
        save_bot_state(state)
        log_event("INFO", f"Status atualizado para {status} / {mode}")
        st.success("Status salvo.")
with c2:
    if st.button("Rodar Trader", use_container_width=True):
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

st.info("RUNNING abre e gerencia posições. PAUSED não abre novas posições. STOPPED paralisa tudo.")
