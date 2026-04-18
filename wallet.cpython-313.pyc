from __future__ import annotations

import pandas as pd
import streamlit as st

from core.config import BOT_LOG_FILE, INVESTOR_ORDERS_FILE, TRADER_ORDERS_FILE

st.title("Histórico")

trader_orders = pd.read_csv(TRADER_ORDERS_FILE)
investor_orders = pd.read_csv(INVESTOR_ORDERS_FILE)
logs = pd.read_csv(BOT_LOG_FILE)

t1, t2, t3 = st.tabs(["Ordens Trader", "Pesquisa", "Logs"])
with t1:
    if not trader_orders.empty:
        st.dataframe(trader_orders.iloc[::-1], use_container_width=True)
    else:
        st.info("Sem ordens do trader.")
with t2:
    if not investor_orders.empty:
        st.dataframe(investor_orders.iloc[::-1], use_container_width=True)
    else:
        st.info("Sem eventos de pesquisa.")
with t3:
    if not logs.empty:
        st.dataframe(logs.iloc[::-1], use_container_width=True)
    else:
        st.info("Sem logs do bot.")
