# -*- coding: utf-8 -*-
from __future__ import annotations

import streamlit as st

from core.auth.guards import is_admin, render_auth_toolbar, require_auth
from core.state_store import load_bot_state
from paper_trading_engine import run_paper_cycle


st.set_page_config(page_title="Robo Trader", layout="wide")

current_user = require_auth()
render_auth_toolbar()
state = load_bot_state()

st.title("Robo de Investimentos")
st.caption(f"Usuario atual: {current_user['username']} ({current_user['role']})")

if bool((state.get("security", {}) or {}).get("real_mode_enabled", False)):
    st.warning("Real trading enabled")

st.page_link("pages/1_Trader.py", label="Abrir Trader", icon="📈")
st.page_link("pages/2_Investimento.py", label="Abrir Investimento", icon="🏦")
st.page_link("pages/3_Carteira.py", label="Abrir Carteira", icon="💼")
st.page_link("pages/5_Historico.py", label="Abrir Historico", icon="🗂️")

if is_admin(current_user):
    st.page_link("pages/4_Controle_do_Bot.py", label="Abrir Controle do Bot", icon="🛠️")

st.markdown("---")
st.subheader("Execucao manual (legado)")
st.caption("Mantido por compatibilidade. Apenas administradores podem executar o ciclo manualmente por aqui.")

if st.button("Rodar ciclo", use_container_width=True, disabled=not is_admin(current_user)):
    result = run_paper_cycle()

    st.subheader("Resultado")
    st.write(result["report"])

    st.subheader("Equity")
    st.line_chart(result["equity"])

    st.subheader("Trades")
    st.write(result["trades"])
