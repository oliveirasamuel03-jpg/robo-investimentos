from __future__ import annotations

import pandas as pd
import streamlit as st

from core.config import (
    MAX_HOLDING_MINUTES,
    MAX_TICKET,
    MIN_HOLDING_MINUTES,
    MIN_TICKET,
    TRADER_ORDERS_FILE,
)
from core.state_store import load_bot_state, save_bot_state
from engines.quant_bridge import build_paper_report, load_paper_state, read_paper_equity, read_paper_trades
from engines.trader_engine import reset_trader_module, run_trader_cycle, sync_platform_positions_from_paper

st.title("Trader Desk")
state = load_bot_state()
trader = state["trader"]

c1, c2, c3 = st.columns(3)
with c1:
    ticket = st.number_input(
        "Valor por operação (R$)",
        min_value=MIN_TICKET,
        max_value=MAX_TICKET,
        value=float(trader["ticket_value"]),
        step=10.0,
    )
with c2:
    holding = st.slider(
        "Holding máximo (min)",
        min_value=MIN_HOLDING_MINUTES,
        max_value=MAX_HOLDING_MINUTES,
        value=int(trader["holding_minutes"]),
        step=1,
    )
with c3:
    max_open = st.slider(
        "Máx. posições abertas",
        min_value=1,
        max_value=20,
        value=int(trader["max_open_positions"]),
        step=1,
    )

watchlist_text = st.text_input(
    "Watchlist do Trader",
    value=", ".join(trader.get("watchlist", [])),
    help="Use poucos ativos aqui para o trader rodar rápido. Ex.: BTC-USD, ETH-USD, PETR4.SA",
)

col_a, col_b, col_c = st.columns(3)

with col_a:
    if st.button("Salvar parâmetros Trader", use_container_width=True):
        trader["ticket_value"] = float(ticket)
        trader["holding_minutes"] = int(holding)
        trader["max_open_positions"] = int(max_open)
        trader["watchlist"] = [x.strip().upper() for x in watchlist_text.split(",") if x.strip()]
        state["trader"] = trader
        save_bot_state(state)
        st.success("Parâmetros do trader salvos.")

with col_b:
    if st.button("Rodar ciclo Trader", use_container_width=True):
        try:
            with st.spinner("Rodando ciclo do trader..."):
                result = run_trader_cycle()
            st.success(
                f"Ciclo do trader executado. Trades: {result.get('cycle_result', {}).get('trades_executed', 0)}"
            )
        except Exception as e:
            st.error(f"Erro ao rodar ciclo trader: {e}")

with col_c:
    if st.button("Resetar Trader", use_container_width=True):
        try:
            with st.spinner("Resetando trader..."):
                reset_trader_module()
            st.warning("Trader resetado.")
        except Exception as e:
            st.error(f"Erro ao resetar trader: {e}")

sync_platform_positions_from_paper()
paper_state = load_paper_state()
paper_report = build_paper_report(initial_capital=float(load_bot_state()["wallet_value"]))
paper_equity_df = read_paper_equity(limit=300)
paper_trades = read_paper_trades(limit=200)

st.subheader("Resumo do Paper Trading")
r1, r2, r3, r4 = st.columns(4)
r1.metric("Cash", f"R$ {paper_state.get('cash', 0.0):,.2f}")
r2.metric("Equity", f"R$ {paper_state.get('equity', 0.0):,.2f}")
r3.metric("Runs", f"{paper_state.get('run_count', 0)}")
r4.metric("Trades", f"{paper_report.get('trades_count', 0)}")

if not paper_equity_df.empty:
    st.subheader("Equity do Trader")
    st.line_chart(paper_equity_df.set_index("timestamp")["equity_after"])

st.subheader("Posições Trader")
positions = [p for p in load_bot_state().get("positions", []) if p.get("module") == "TRADER"]
if positions:
    st.dataframe(pd.DataFrame(positions), use_container_width=True)
else:
    st.info("Sem posições abertas no trader.")

st.subheader("Ordens Trader")
try:
    orders = pd.read_csv(TRADER_ORDERS_FILE)
    if not orders.empty:
        st.dataframe(orders.tail(200).iloc[::-1], use_container_width=True)
    else:
        st.info("Sem ordens trader no storage ainda.")
except Exception as e:
    st.error(f"Erro ao ler ordens trader: {e}")

st.subheader("Últimos trades do motor")
if paper_trades:
    st.dataframe(pd.DataFrame(paper_trades[::-1]), use_container_width=True)
else:
    st.info("Sem trades ainda no paper engine.")
