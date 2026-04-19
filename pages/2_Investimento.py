from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core.auth.guards import render_auth_toolbar, require_auth
from core.config import INVESTOR_ORDERS_FILE
from core.state_store import append_csv_row, load_bot_state, save_bot_state
from engines.investment_research_engine import run_investment_pipeline
from institutional import build_institutional_dashboard_payload, render_institutional_dashboard


@st.cache_data(ttl=3600, show_spinner=False)
def cached_pipeline(**kwargs):
    return run_investment_pipeline(**kwargs)


def plot_equity(backtest_result, title: str):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=backtest_result["equity_curve"].index,
            y=backtest_result["equity_curve"]["equity"],
            mode="lines",
            name=title,
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=360,
        title=title,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def render_basic_summary(result: dict) -> None:
    rm = result["research_metrics"]
    hm = result["holdout_metrics"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Sharpe Pesquisa", f"{rm['sharpe']:.2f}")
    m2.metric("Max DD Pesquisa", f"{rm['max_drawdown']:.2%}")
    m3.metric("Sharpe Holdout", f"{hm['sharpe']:.2f}")
    m4.metric("Ativos", f"{len(result['prices_columns'])}")

    st.plotly_chart(plot_equity(result["research_bt"], "Equity Curve - Pesquisa"), use_container_width=True)
    st.plotly_chart(plot_equity(result["holdout_bt"], "Equity Curve - Holdout"), use_container_width=True)
    st.plotly_chart(
        px.histogram(result["mc"]["simulations"], x="sharpe", title="Distribuicao de Sharpe - Monte Carlo"),
        use_container_width=True,
    )
    with st.expander("Relatorio institucional"):
        st.json(result["report"])


require_auth()
render_auth_toolbar()

st.title("Investimento / Pesquisa Institucional")
state = load_bot_state()
investment_state = state.setdefault("investment", {})

c1, c2, c3 = st.columns(3)
with c1:
    start_date = st.text_input("Data inicial", value=state.get("start_date", "2019-01-01"))
with c2:
    threshold = st.slider("Limiar de probabilidade", 0.50, 0.85, float(state.get("probability_threshold", 0.60)), 0.01)
with c3:
    top_n = st.slider("Top N", 1, 10, int(state.get("top_n", 2)), 1)

c4, c5, c6 = st.columns(3)
with c4:
    fast_mode = st.toggle("Modo rapido", True)
with c5:
    delay = st.slider("Atraso de execucao", 1, 3, 1, 1)
with c6:
    holdout_ratio = st.slider("Holdout", 0.10, 0.60, 0.20, 0.05)

st.subheader("Classes de ativos")
a1, a2, a3 = st.columns(3)
with a1:
    include_brazil_stocks = st.checkbox("Acoes Brasil", value=bool(state.get("include_brazil_stocks", True)))
    include_us_stocks = st.checkbox("Acoes EUA", value=bool(state.get("include_us_stocks", True)))
with a2:
    include_etfs = st.checkbox("ETFs", value=bool(state.get("include_etfs", True)))
    include_fiis = st.checkbox("FIIs", value=bool(state.get("include_fiis", True)))
with a3:
    include_crypto = st.checkbox("Cripto", value=bool(state.get("include_crypto", True)))
    include_grains = st.checkbox("Graos", value=bool(state.get("include_grains", True)))

custom_tickers_text = st.text_input("Tickers extras", value=", ".join(state.get("custom_tickers", [])))
custom_tickers = [x.strip().upper() for x in custom_tickers_text.split(",") if x.strip()]

if st.button("Salvar parametros e rodar pesquisa", use_container_width=True):
    state["start_date"] = start_date
    state["probability_threshold"] = float(threshold)
    state["top_n"] = int(top_n)
    state["include_brazil_stocks"] = include_brazil_stocks
    state["include_us_stocks"] = include_us_stocks
    state["include_etfs"] = include_etfs
    state["include_fiis"] = include_fiis
    state["include_crypto"] = include_crypto
    state["include_grains"] = include_grains
    state["custom_tickers"] = custom_tickers
    save_bot_state(state)

    try:
        with st.spinner("Executando pesquisa institucional..."):
            result = cached_pipeline(
                start_date=start_date,
                initial_capital=float(state["wallet_value"]),
                probability_threshold=float(threshold),
                top_n=int(top_n),
                execution_delay=int(delay),
                fast_mode=bool(fast_mode),
                use_regime_filter=bool(state.get("use_regime_filter", True)),
                use_volatility_filter=bool(state.get("use_volatility_filter", False)),
                vol_threshold=float(state.get("vol_threshold", 0.03)),
                cash_threshold=float(state.get("cash_threshold", 0.50)),
                target_portfolio_vol=float(state.get("target_portfolio_vol", 0.08)),
                holdout_ratio=float(holdout_ratio),
                include_brazil_stocks=include_brazil_stocks,
                include_us_stocks=include_us_stocks,
                include_etfs=include_etfs,
                include_fiis=include_fiis,
                include_crypto=include_crypto,
                include_grains=include_grains,
                custom_tickers=custom_tickers,
            )
    except Exception as exc:
        st.error(f"Falha ao executar a pesquisa: {exc}")
    else:
        dashboard_payload = build_institutional_dashboard_payload(result)
        investment_state["last_report"] = result["report"]
        investment_state["last_dashboard"] = dashboard_payload
        save_bot_state(state)
        append_csv_row(
            INVESTOR_ORDERS_FILE,
            {
                "timestamp": pd.Timestamp.utcnow().isoformat(),
                "metric": "research_run",
                "value": len(result["prices_columns"]),
                "notes": "Pesquisa institucional executada",
            },
        )

        st.success("Pesquisa executada com sucesso.")
        render_basic_summary(result)
        render_institutional_dashboard(dashboard_payload)
else:
    last_dashboard = investment_state.get("last_dashboard", {})
    last_report = investment_state.get("last_report", {})
    if last_dashboard:
        st.info("Ultimo dashboard institucional salvo em estado local.")
        render_institutional_dashboard(last_dashboard)
    if last_report:
        st.info("Ultimo relatorio salvo em estado local.")
        st.json(last_report)
