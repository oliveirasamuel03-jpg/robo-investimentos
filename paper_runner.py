from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.config import APP_TITLE
from core.state_store import ensure_storage, load_bot_state
from core.ui import inject_trading_desk_css, panel_close, panel_open, render_hero, render_kpi_grid, render_tags
from portfolio.wallet import compute_wallet_snapshot

st.set_page_config(page_title=APP_TITLE, layout="wide")
ensure_storage()


def _build_wallet_gauge(wallet_value: float, cash: float, capital_in_use: float):
    fig = go.Figure()

    used_pct = 0.0 if wallet_value <= 0 else min(capital_in_use / wallet_value * 100, 100)
    cash_pct = 0.0 if wallet_value <= 0 else min(cash / wallet_value * 100, 100)

    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=used_pct,
            number={"suffix": "%"},
            title={"text": "Capital em uso"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"thickness": 0.35},
                "steps": [
                    {"range": [0, 35], "color": "rgba(25,245,193,0.18)"},
                    {"range": [35, 70], "color": "rgba(255,204,102,0.18)"},
                    {"range": [70, 100], "color": "rgba(255,95,126,0.20)"},
                ],
            },
        )
    )

    fig.update_layout(
        template="plotly_dark",
        height=260,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        annotations=[
            dict(
                x=0.5,
                y=-0.12,
                xref="paper",
                yref="paper",
                text=f"Caixa livre: {cash_pct:.1f}%",
                showarrow=False,
                font=dict(size=13),
            )
        ],
    )
    return fig


def _build_bot_activity_chart(state: dict):
    positions = state.get("positions", [])
    open_trader = sum(1 for p in positions if p.get("module") == "TRADER" and p.get("status") == "OPEN")
    open_invest = sum(1 for p in positions if p.get("module") == "INVESTMENT" and p.get("status") == "OPEN")
    closed = sum(1 for p in positions if p.get("status") == "CLOSED")

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Trader Abertas", x=["Atividade"], y=[open_trader]))
    fig.add_trace(go.Bar(name="Investimento Abertas", x=["Atividade"], y=[open_invest]))
    fig.add_trace(go.Bar(name="Fechadas", x=["Atividade"], y=[closed]))

    fig.update_layout(
        template="plotly_dark",
        barmode="group",
        height=260,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title="Resumo de posições",
    )
    return fig


inject_trading_desk_css()

state = load_bot_state()
snapshot = compute_wallet_snapshot(state)

render_hero(
    APP_TITLE,
    "Trading Desk · Trader + Investimento · Controle operacional · Visual profissional de mesa",
)

render_kpi_grid(
    [
        ("Carteira", f"R$ {snapshot['wallet_value']:,.2f}", ""),
        ("Caixa", f"R$ {snapshot['cash']:,.2f}", ""),
        ("Capital em uso", f"R$ {snapshot['capital_in_use']:,.2f}", ""),
        ("PnL realizado", f"R$ {snapshot['realized_pnl']:,.2f}", "good" if snapshot["realized_pnl"] >= 0 else "bad"),
        ("Posições abertas", f"{snapshot['open_positions']}", ""),
        ("Status Bot", state["bot_status"], "warn" if state["bot_status"] == "PAUSED" else ""),
    ]
)

left, center, right = st.columns([0.95, 1.8, 1.0], gap="large")

with left:
    panel_open("Watchlist")
    trader_watchlist = state.get("trader", {}).get("watchlist", [])
    invest_watchlist = state.get("investment", {}).get("watchlist", [])
    st.caption("Trader")
    if trader_watchlist:
        render_tags(trader_watchlist[:12])
    else:
        st.info("Sem ativos no trader.")

    st.caption("Investimento")
    if invest_watchlist:
        render_tags(invest_watchlist[:12])
    else:
        st.info("Sem ativos no investimento.")
    panel_close()

    panel_open("Botões rápidos")
    c1, c2 = st.columns(2)
    with c1:
        st.page_link("pages/1_Trader.py", label="Trader")
        st.page_link("pages/3_Carteira.py", label="Carteira")
    with c2:
        st.page_link("pages/2_Investimento.py", label="Investimento")
        st.page_link("pages/4_Controle_do_Bot.py", label="Controle")
    st.page_link("pages/5_Historico.py", label="Histórico")
    panel_close()

    panel_open("Resumo operacional")
    st.write(f"Modo: **{state['bot_mode']}**")
    st.write(f"Ticket trader: **R$ {state['trader']['ticket_value']:,.2f}**")
    st.write(f"Holding trader: **{state['trader']['holding_minutes']} min**")
    st.write(f"Máx. posições trader: **{state['trader']['max_open_positions']}**")
    st.write(f"Budget investimento: **R$ {state['investment']['budget']:,.2f}**")
    panel_close()

with center:
    panel_open("Visão da mesa")
    chart_tab, positions_tab = st.tabs(["Painel central", "Posições"])
    with chart_tab:
        st.plotly_chart(
            _build_wallet_gauge(
                wallet_value=snapshot["wallet_value"],
                cash=snapshot["cash"],
                capital_in_use=snapshot["capital_in_use"],
            ),
            use_container_width=True,
        )
        st.plotly_chart(_build_bot_activity_chart(state), use_container_width=True)
    with positions_tab:
        positions = state.get("positions", [])
        if positions:
            df = pd.DataFrame(positions)
            if "status" in df.columns:
                open_df = df[df["status"] == "OPEN"].copy()
                closed_df = df[df["status"] == "CLOSED"].copy()

                st.subheader("Posições abertas")
                if not open_df.empty:
                    st.dataframe(open_df, use_container_width=True)
                else:
                    st.info("Sem posições abertas.")

                st.subheader("Últimas posições encerradas")
                if not closed_df.empty:
                    st.dataframe(closed_df.tail(20).iloc[::-1], use_container_width=True)
                else:
                    st.info("Sem posições fechadas.")
            else:
                st.dataframe(df, use_container_width=True)
        else:
            st.info("Ainda não há posições registradas.")
    panel_close()

with right:
    panel_open("Status do bot")
    st.metric("Bot", state["bot_status"])
    st.metric("Modo", state["bot_mode"])
    st.metric("Reserva de caixa", f"{float(state.get('reserve_cash_pct', 0.10))*100:.0f}%")
    panel_close()

    panel_open("Diagnóstico rápido")
    messages = []
    if state["bot_status"] == "STOPPED":
        messages.append("Bot completamente parado. Nenhuma nova ação será tomada.")
    elif state["bot_status"] == "PAUSED":
        messages.append("Bot pausado para novas entradas. Gerenciamento pode continuar.")
    else:
        messages.append("Bot ativo e apto a executar ciclos.")

    if snapshot["cash"] < state["trader"]["ticket_value"]:
        messages.append("Caixa atual abaixo do ticket configurado do trader.")
    if snapshot["open_positions"] == 0:
        messages.append("Nenhuma posição aberta no momento.")
    if snapshot["capital_in_use"] > snapshot["wallet_value"] * 0.7:
        messages.append("Capital em uso elevado para o tamanho da carteira.")

    for msg in messages:
        st.markdown(f"<div class='desk-log'>{msg}</div>", unsafe_allow_html=True)
    panel_close()

    panel_open("Próximos passos")
    st.write("• Use a página **Trader** para rodar ciclos e gerenciar posições.")
    st.write("• Use **Investimento** para pesquisa institucional e alocação.")
    st.write("• Use **Controle do Bot** para RUNNING / PAUSED / STOPPED.")
    st.write("• Use **Histórico** para auditoria das ordens e logs.")
    panel_close()
