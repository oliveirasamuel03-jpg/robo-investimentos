from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from core.config import (
    MAX_HOLDING_MINUTES,
    MAX_TICKET,
    MIN_HOLDING_MINUTES,
    MIN_TICKET,
    TRADER_ORDERS_FILE,
)
from core.state_store import load_bot_state, save_bot_state
from engines.quant_bridge import (
    build_paper_report,
    load_paper_state,
    read_paper_equity,
    read_paper_trades,
)
from engines.trader_engine import (
    reset_trader_module,
    run_trader_cycle,
    sync_platform_positions_from_paper,
)


@st.cache_data(ttl=60, show_spinner=False)
def load_chart_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(
        tickers=ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    ).copy()

    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)

    df = df.sort_index()

    for col in ["open", "high", "low", "close"]:
        if col not in df.columns:
            return pd.DataFrame()

    if "volume" not in df.columns:
        df["volume"] = 0.0

    df["ma9"] = df["close"].rolling(9).mean()
    df["ma21"] = df["close"].rolling(21).mean()
    return df


def load_trader_orders() -> pd.DataFrame:
    try:
        df = pd.read_csv(TRADER_ORDERS_FILE)
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return df

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df["timestamp"] = df["timestamp"].dt.tz_convert(None)

    if "asset" in df.columns:
        df["asset"] = df["asset"].astype(str).str.upper()

    if "side" in df.columns:
        df["side"] = df["side"].astype(str).str.upper()

    return df


def align_orders_to_chart(df_orders: pd.DataFrame, df_chart: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if df_orders.empty or df_chart.empty:
        return pd.DataFrame()

    orders = df_orders.copy()
    orders = orders[orders["asset"] == ticker.upper()].copy()
    if orders.empty:
        return pd.DataFrame()

    orders = orders.dropna(subset=["timestamp"])
    if orders.empty:
        return pd.DataFrame()

    chart_index = pd.DataFrame({"chart_time": df_chart.index}).sort_values("chart_time")
    orders = orders.sort_values("timestamp")

    aligned = pd.merge_asof(
        orders,
        chart_index,
        left_on="timestamp",
        right_on="chart_time",
        direction="nearest",
    )

    aligned["plot_time"] = aligned["chart_time"].fillna(aligned["timestamp"])

    if "price" in aligned.columns:
        aligned["plot_price"] = pd.to_numeric(aligned["price"], errors="coerce")
    else:
        aligned["plot_price"] = pd.NA

    missing_price = aligned["plot_price"].isna()
    if missing_price.any():
        close_map = df_chart["close"]
        aligned.loc[missing_price, "plot_price"] = aligned.loc[missing_price, "plot_time"].map(close_map)

    aligned["plot_price"] = pd.to_numeric(aligned["plot_price"], errors="coerce")
    aligned = aligned.dropna(subset=["plot_time", "plot_price"])
    return aligned


def add_trade_markers(fig: go.Figure, aligned_orders: pd.DataFrame) -> None:
    if aligned_orders.empty:
        return

    buys = aligned_orders[aligned_orders["side"] == "BUY"].copy()
    sells = aligned_orders[aligned_orders["side"] == "SELL"].copy()

    if not buys.empty:
        fig.add_trace(
            go.Scatter(
                x=buys["plot_time"],
                y=buys["plot_price"],
                mode="markers+text",
                name="BUY",
                text=["BUY"] * len(buys),
                textposition="top center",
                marker=dict(symbol="triangle-up", size=13, color="#19f5c1"),
                hovertemplate=(
                    "BUY<br>"
                    "Tempo: %{x}<br>"
                    "Preço: %{y:.2f}<br>"
                    "<extra></extra>"
                ),
            )
        )

    if not sells.empty:
        fig.add_trace(
            go.Scatter(
                x=sells["plot_time"],
                y=sells["plot_price"],
                mode="markers+text",
                name="SELL",
                text=["SELL"] * len(sells),
                textposition="bottom center",
                marker=dict(symbol="triangle-down", size=13, color="#ff5f7e"),
                hovertemplate=(
                    "SELL<br>"
                    "Tempo: %{x}<br>"
                    "Preço: %{y:.2f}<br>"
                    "<extra></extra>"
                ),
            )
        )


def build_candle_chart(
    df: pd.DataFrame,
    ticker: str,
    entry_price: float | None = None,
    aligned_orders: pd.DataFrame | None = None,
) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Preço",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["ma9"],
            mode="lines",
            name="MA 9",
            line=dict(width=1.5),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["ma21"],
            mode="lines",
            name="MA 21",
            line=dict(width=1.5),
        )
    )

    if entry_price is not None and entry_price > 0:
        fig.add_hline(
            y=entry_price,
            line_width=1.5,
            line_dash="dot",
            annotation_text=f"Entrada {entry_price:.2f}",
        )

    if aligned_orders is not None and not aligned_orders.empty:
        add_trade_markers(fig, aligned_orders)

    fig.update_layout(
        template="plotly_dark",
        height=520,
        title=f"Preço real - {ticker}",
        xaxis_title="Tempo",
        yaxis_title="Preço",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_rangeslider_visible=False,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def build_volume_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["volume"],
            name="Volume",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=220,
        title=f"Volume - {ticker}",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


st.set_page_config(page_title="Trader Desk", layout="wide")
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

positions = [p for p in load_bot_state().get("positions", []) if p.get("module") == "TRADER"]
open_positions = [p for p in positions if p.get("status") == "OPEN"]

st.subheader("Gráfico Trader com preço real")

watchlist = trader.get("watchlist", []) or ["AAPL"]
selected_ticker = st.selectbox("Ativo do gráfico", options=watchlist, index=0)

g1, g2 = st.columns([1.2, 1.0])
with g1:
    period = st.selectbox(
        "Período",
        options=["1d", "5d", "1mo", "3mo", "6mo"],
        index=2,
    )
with g2:
    interval = st.selectbox(
        "Intervalo",
        options=["1m", "5m", "15m", "30m", "60m", "1d"],
        index=2,
    )

selected_position = next(
    (p for p in open_positions if p.get("asset") == selected_ticker),
    None,
)
entry_price = float(selected_position.get("entry_price", 0.0)) if selected_position else None

chart_df = pd.DataFrame()
orders_df = load_trader_orders()

try:
    with st.spinner("Carregando preço real..."):
        chart_df = load_chart_data(selected_ticker, period, interval)
except Exception as e:
    st.error(f"Erro ao carregar gráfico: {e}")

if chart_df.empty:
    st.warning("Não foi possível carregar dados reais para esse ativo/período.")
else:
    aligned_orders = align_orders_to_chart(orders_df, chart_df, selected_ticker)

    left_chart, right_info = st.columns([2.2, 1.0])

    with left_chart:
        st.plotly_chart(
            build_candle_chart(
                chart_df,
                selected_ticker,
                entry_price=entry_price,
                aligned_orders=aligned_orders,
            ),
            use_container_width=True,
        )
        st.plotly_chart(
            build_volume_chart(chart_df, selected_ticker),
            use_container_width=True,
        )

    with right_info:
        last_close = float(chart_df["close"].iloc[-1])
        prev_close = float(chart_df["close"].iloc[-2]) if len(chart_df) > 1 else last_close
        delta_pct = ((last_close / prev_close) - 1.0) * 100 if prev_close else 0.0

        st.metric("Último preço", f"{last_close:,.2f}")
        st.metric("Variação", f"{delta_pct:.2f}%")
        st.metric("Máxima", f"{float(chart_df['high'].max()):,.2f}")
        st.metric("Mínima", f"{float(chart_df['low'].min()):,.2f}")

        st.subheader("Posição no ativo")
        if selected_position:
            qty = float(selected_position.get("qty", 0.0))
            avg_price = float(selected_position.get("entry_price", 0.0))
            market_value = qty * last_close
            unrealized = (last_close - avg_price) * qty

            st.write(f"**Status:** {selected_position.get('status', '-')}")
            st.write(f"**Quantidade:** {qty:,.6f}")
            st.write(f"**Preço médio:** {avg_price:,.2f}")
            st.write(f"**Valor de mercado:** R$ {market_value:,.2f}")
            st.write(f"**PnL não realizado:** R$ {unrealized:,.2f}")
        else:
            st.info("Nenhuma posição aberta nesse ativo no momento.")

        st.subheader("Trades no gráfico")
        if not aligned_orders.empty:
            st.dataframe(
                aligned_orders[["timestamp", "asset", "side", "plot_price"]].tail(20).iloc[::-1],
                use_container_width=True,
            )
        else:
            st.info("Sem trades do trader nesse ativo/período.")

if not paper_equity_df.empty:
    st.subheader("Equity do Trader")
    st.line_chart(paper_equity_df.set_index("timestamp")["equity_after"])

st.subheader("Posições Trader")
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
