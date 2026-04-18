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
from core.ui import (
    inject_trading_desk_css,
    panel_close,
    panel_open,
    render_hero,
    render_kpi_grid,
    render_tags,
)
from engines.trader_engine import manage_open_trades, run_trader_cycle

st.set_page_config(page_title="Trader Desk", layout="wide")
inject_trading_desk_css()

render_hero(
    "Trader Desk",
    "Execução rápida · Ticket flexível · Holding de 1 minuto até 48 horas",
)

state = load_bot_state()
trader = state["trader"]

positions = [p for p in state.get("positions", []) if p.get("module") == "TRADER"]
open_positions = [p for p in positions if p.get("status") == "OPEN"]

render_kpi_grid(
    [
        ("Status", state.get("bot_status", "PAUSED"), "warn" if state.get("bot_status") == "PAUSED" else ""),
        ("Ticket", f"R$ {float(trader.get('ticket_value', 0.0)):,.2f}", ""),
        ("Holding", f"{int(trader.get('holding_minutes', 0))} min", ""),
        ("Abertas", f"{len(open_positions)}", ""),
        ("Máx abertas", f"{int(trader.get('max_open_positions', 0))}", ""),
        ("Trader ativo", "Sim" if trader.get("enabled", False) else "Não", "good" if trader.get("enabled", False) else "bad"),
    ]
)

left, center, right = st.columns([1.0, 1.4, 1.2], gap="large")

with left:
    panel_open("Configuração Trader")

    ticket = st.number_input(
        "Valor por operação (R$)",
        min_value=float(MIN_TICKET),
        max_value=float(MAX_TICKET),
        value=float(trader.get("ticket_value", 100.0)),
        step=10.0,
    )

    holding = st.slider(
        "Holding máximo (min)",
        min_value=int(MIN_HOLDING_MINUTES),
        max_value=int(MAX_HOLDING_MINUTES),
        value=int(trader.get("holding_minutes", 60)),
        step=1,
    )

    max_open = st.slider(
        "Máx. posições abertas",
        min_value=1,
        max_value=20,
        value=int(trader.get("max_open_positions", 3)),
        step=1,
    )

    watchlist_text = st.text_area(
        "Watchlist Trader",
        value=", ".join(trader.get("watchlist", [])),
        height=120,
    )

    enabled = st.toggle("Trader habilitado", value=bool(trader.get("enabled", True)))

    if st.button("Salvar parâmetros Trader", use_container_width=True):
        trader["ticket_value"] = float(ticket)
        trader["holding_minutes"] = int(holding)
        trader["max_open_positions"] = int(max_open)
        trader["watchlist"] = [x.strip().upper() for x in watchlist_text.split(",") if x.strip()]
        trader["enabled"] = bool(enabled)

        state["trader"] = trader
        save_bot_state(state)
        st.success("Parâmetros do trader salvos.")

    panel_close()

    panel_open("Ativos monitorados")
    watchlist = trader.get("watchlist", [])
    if watchlist:
        render_tags(watchlist)
    else:
        st.info("Sem ativos na watchlist.")
    panel_close()

with center:
    panel_open("Execução")

    a, b = st.columns(2)

    with a:
        if st.button("Rodar ciclo Trader", use_container_width=True):
            try:
                result = run_trader_cycle()
                st.success(f"Ciclo executado. Fechadas: {result.get('closed_positions', 0)}")
                if result.get("opened_trade"):
                    st.info(f"Nova operação aberta em {result['opened_trade']['asset']}")
            except Exception as e:
                st.error(f"Erro ao rodar ciclo trader: {e}")

    with b:
        if st.button("Gerenciar posições abertas", use_container_width=True):
            try:
                result = manage_open_trades()
                st.success(f"Gerenciamento concluído. Fechadas: {result.get('closed_positions', 0)}")
            except Exception as e:
                st.error(f"Erro ao gerenciar posições: {e}")

    st.subheader("Posições abertas")
    current_state = load_bot_state()
    current_positions = [
        p for p in current_state.get("positions", [])
        if p.get("module") == "TRADER" and p.get("status") == "OPEN"
    ]

    if current_positions:
        st.dataframe(pd.DataFrame(current_positions), use_container_width=True)
    else:
        st.info("Sem posições abertas no trader.")

    panel_close()

with right:
    panel_open("Ordens Trader")

    try:
        orders = pd.read_csv(TRADER_ORDERS_FILE)
        if not orders.empty:
            st.dataframe(orders.tail(80).iloc[::-1], use_container_width=True)
        else:
            st.info("Sem ordens registradas.")
    except Exception as e:
        st.error(f"Erro ao ler ordens: {e}")

    panel_close()