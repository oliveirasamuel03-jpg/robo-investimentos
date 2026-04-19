from __future__ import annotations

import streamlit as st

from core.auth.guards import render_auth_toolbar, require_auth
from core.config import (
    BOT_LOG_COLUMNS,
    BOT_LOG_FILE,
    INVESTOR_ORDERS_COLUMNS,
    INVESTOR_ORDERS_FILE,
    TRADER_ORDERS_COLUMNS,
    TRADER_ORDERS_FILE,
)
from core.state_store import read_storage_table


require_auth()
render_auth_toolbar()

st.title("Histórico")

trader_orders = read_storage_table(TRADER_ORDERS_FILE, columns=TRADER_ORDERS_COLUMNS)
investor_orders = read_storage_table(INVESTOR_ORDERS_FILE, columns=INVESTOR_ORDERS_COLUMNS)
logs = read_storage_table(BOT_LOG_FILE, columns=BOT_LOG_COLUMNS)

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
