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
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_convert(None)

    df = df.sort_index()

    needed = ["open", "high", "low", "close"]
    for col in needed:
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
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    if "asset" in df.columns:
        df["asset"] = df["asset"].astype(str).str.upper()

    if "side" in df.columns:
        df["side"] = df["side"].astype(str).str.upper()

    return df


def align_orders_to_chart(
    df_orders: pd.DataFrame,
    df_chart: pd.DataFrame,
    ticker: str,
) -> pd.DataFrame:
    if df_orders.empty or df_chart.empty:
        return pd.DataFrame()

    orders = df_orders.copy()
    orders = orders[orders["asset"] == ticker.upper()].copy()
    if orders.empty:
        return pd.DataFrame()

    orders["timestamp"] = pd.to_datetime(orders["timestamp"], errors="coerce")
    orders = orders.dropna(subset=["timestamp"]).copy()
    if orders.empty:
        return pd.DataFrame()

    if getattr(orders["timestamp"].dt, "tz", None) is not None:
        orders["timestamp"] = orders["timestamp"].dt.tz_convert(None)

    chart_index = pd.DataFrame({"chart_time": pd.to_datetime(df_chart.index, errors="coerce")})
    chart_index = chart_index.dropna(subset=["chart_time"]).copy()
    if chart_index.empty:
        return pd.DataFrame()

    if getattr(chart_index["chart_time"].dt, "tz", None) is not None:
        chart_index["chart_time"] = chart_index["chart_time"].dt.tz_convert(None)

    orders["ts_int"] = orders["timestamp"].view("int64")
    chart_index["ts_int"] = chart_index["chart_time"].view("int64")

    orders = orders.sort_values("ts_int")
    chart_index = chart_index.sort_values("ts_int")

    aligned = pd.merge_asof(
        orders,
        chart_index,
        left_on="ts_int",
        right_on="ts_int",
        direction="nearest",
    )

    aligned["plot_time"] = aligned["chart_time"]
    aligned["plot_price"] = pd.to_numeric(aligned.get("price"), errors="coerce")

    missing = aligned["plot_price"].isna()
    if missing.any():
        close_series = df_chart["close"]
        aligned.loc[missing, "plot_price"] = aligned.loc[missing, "plot_time"].map(close_series)

    aligned = aligned.dropna(subset=["plot_time", "plot_price"]).copy()
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
                name="Compra",
                text=["Compra"] * len(buys),
                textposition="top center",
                marker=dict(symbol="triangle-up", size=12, color="#19f5c1"),
                hovertemplate="Compra<br>Tempo: %{x}<br>Preço: %{y:.2f}<extra></extra>",
            )
        )

    if not sells.empty:
        fig.add_trace(
            go.Scatter(
                x=sells["plot_time"],
                y=sells["plot_price"],
                mode="markers+text",
                name="Venda",
                text=["Venda"] * len(sells),
                textposition="bottom center",
                marker=dict(symbol="triangle-down", size=12, color="#ff5f7e"),
                hovertemplate="Venda<br>Tempo: %{x}<br>Preço: %{y:.2f}<extra></extra>",
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
            name="Média 9",
            line=dict(width=1.5),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["ma21"],
            mode="lines",
            name="Média 21",
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
        title=f"Gráfico do ativo - {ticker}",
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


def br_money(value: float) -> str:
    return f"R$ {value:,.2f}"


def simple_signal_text(chart_df: pd.DataFrame) -> str:
    if chart_df.empty or len(chart_df) < 2:
        return "Aguardando dados"

    ma9 = chart_df["ma9"].iloc[-1]
    ma21 = chart_df["ma21"].iloc[-1]
    prev_ma9 = chart_df["ma9"].iloc[-2]
    prev_ma21 = chart_df["ma21"].iloc[-2]

    if pd.notna(ma9) and pd.notna(ma21) and pd.notna(prev_ma9) and pd.notna(prev_ma21):
        if ma9 > ma21 and prev_ma9 <= prev_ma21:
            return "Sinal de compra"
        if ma9 < ma21 and prev_ma9 >= prev_ma21:
            return "Sinal de venda"
        if ma9 > ma21:
            return "Tendência de alta"
        if ma9 < ma21:
            return "Tendência de baixa"

    return "Sem sinal claro"


st.set_page_config(page_title="Trader", layout="wide")

st.markdown(
    """
    <style>
    .status-card {
        padding: 14px 18px;
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 14px;
        background: rgba(255,255,255,0.02);
        margin-bottom: 8px;
    }
    .status-label {
        font-size: 0.95rem;
        opacity: 0.8;
        margin-bottom: 6px;
    }
    .status-value {
        font-size: 1.65rem;
        font-weight: 700;
    }
    .helper-box {
        padding: 12px 14px;
        border-radius: 12px;
        background: rgba(25, 245, 193, 0.08);
        border: 1px solid rgba(25, 245, 193, 0.18);
        margin-bottom: 12px;
    }
    .block-container {
        padding-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🤖 Trader Simples")
st.caption("Tela mais fácil de entender. O modo avançado fica logo abaixo.")

state = load_bot_state()
trader = state["trader"]

sync_platform_positions_from_paper()
state = load_bot_state()
paper_state = load_paper_state()
paper_report = build_paper_report(initial_capital=float(state["wallet_value"]))
paper_equity_df = read_paper_equity(limit=300)
paper_trades = read_paper_trades(limit=200)
positions = [p for p in state.get("positions", []) if p.get("module") == "TRADER"]
open_positions = [p for p in positions if p.get("status") == "OPEN"]

watchlist = trader.get("watchlist", []) or ["AAPL"]
selected_ticker = st.selectbox("Ativo", options=watchlist, index=0)

p1, p2 = st.columns(2)
with p1:
    period = st.selectbox("Período", options=["1d", "5d", "1mo", "3mo", "6mo"], index=2)
with p2:
    interval = st.selectbox("Intervalo", options=["1m", "5m", "15m", "30m", "60m", "1d"], index=2)

selected_position = next((p for p in open_positions if p.get("asset") == selected_ticker), None)
entry_price = float(selected_position.get("entry_price", 0.0)) if selected_position else None

chart_df = pd.DataFrame()
orders_df = load_trader_orders()

try:
    with st.spinner("Carregando gráfico..."):
        chart_df = load_chart_data(selected_ticker, period, interval)
except Exception as e:
    st.error(f"Erro ao carregar gráfico: {e}")

last_price = float(chart_df["close"].iloc[-1]) if not chart_df.empty else 0.0
signal_text = simple_signal_text(chart_df)
robot_status = state.get("bot_status", "PAUSED")
robot_label = "Ligado" if robot_status == "RUNNING" else "Pausado" if robot_status == "PAUSED" else "Desligado"

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(
        f"<div class='status-card'><div class='status-label'>Saldo</div><div class='status-value'>{br_money(float(paper_state.get('cash', 0.0)))}</div></div>",
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f"<div class='status-card'><div class='status-label'>Lucro / prejuízo</div><div class='status-value'>{br_money(float(paper_report.get('net_profit', 0.0)))}</div></div>",
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        f"<div class='status-card'><div class='status-label'>Robô</div><div class='status-value'>{robot_label}</div></div>",
        unsafe_allow_html=True,
    )

st.markdown(
    f"<div class='helper-box'><b>Sinal atual:</b> {signal_text}<br><b>Preço atual:</b> {last_price:,.2f} | <b>Operações abertas:</b> {len(open_positions)}</div>",
    unsafe_allow_html=True,
)

b1, b2, b3 = st.columns(3)
with b1:
    if st.button("▶ Iniciar robô", use_container_width=True):
        state = load_bot_state()
        state["bot_status"] = "RUNNING"
        save_bot_state(state)
        st.success("Robô ligado.")
        st.rerun()

with b2:
    if st.button("⏸ Pausar robô", use_container_width=True):
        state = load_bot_state()
        state["bot_status"] = "PAUSED"
        save_bot_state(state)
        st.warning("Robô pausado.")
        st.rerun()

with b3:
    if st.button("🔁 Rodar agora", use_container_width=True):
        try:
            with st.spinner("Executando ciclo do robô..."):
                result = run_trader_cycle()
            st.success(f"Ciclo executado. Trades feitos: {result.get('cycle_result', {}).get('trades_executed', 0)}")
            st.rerun()
        except Exception as e:
            st.error(f"Erro ao rodar ciclo trader: {e}")

if chart_df.empty:
    st.warning("Não foi possível carregar dados desse ativo agora.")
else:
    aligned_orders = align_orders_to_chart(orders_df, chart_df, selected_ticker)
    st.plotly_chart(
        build_candle_chart(
            chart_df,
            selected_ticker,
            entry_price=entry_price,
            aligned_orders=aligned_orders,
        ),
        use_container_width=True,
    )

    with st.expander("Mostrar volume"):
        st.plotly_chart(build_volume_chart(chart_df, selected_ticker), use_container_width=True)

    info1, info2, info3 = st.columns(3)
    with info1:
        st.metric("Preço atual", f"{last_price:,.2f}")
    with info2:
        max_price = float(chart_df["high"].max()) if not chart_df.empty else 0.0
        st.metric("Máxima do período", f"{max_price:,.2f}")
    with info3:
        min_price = float(chart_df["low"].min()) if not chart_df.empty else 0.0
        st.metric("Mínima do período", f"{min_price:,.2f}")

if selected_position:
    qty = float(selected_position.get("qty", 0.0))
    avg_price = float(selected_position.get("entry_price", 0.0))
    market_value = qty * last_price if last_price else 0.0
    unrealized = (last_price - avg_price) * qty if last_price else 0.0
    st.info(
        f"Operação aberta em {selected_ticker}: quantidade {qty:,.6f} | preço médio {avg_price:,.2f} | valor atual {br_money(market_value)} | resultado atual {br_money(unrealized)}"
    )
else:
    st.info("Nenhuma operação aberta nesse ativo no momento.")

with st.expander("Modo avançado"):
    st.markdown("### Configuração do robô")

    a1, a2, a3 = st.columns(3)
    with a1:
        ticket = st.number_input(
            "Valor por operação (R$)",
            min_value=MIN_TICKET,
            max_value=MAX_TICKET,
            value=float(trader["ticket_value"]),
            step=10.0,
        )
    with a2:
        holding = st.slider(
            "Tempo máximo da operação (min)",
            min_value=MIN_HOLDING_MINUTES,
            max_value=MAX_HOLDING_MINUTES,
            value=int(trader["holding_minutes"]),
            step=1,
        )
    with a3:
        max_open = st.slider(
            "Máx. operações abertas",
            min_value=1,
            max_value=20,
            value=int(trader["max_open_positions"]),
            step=1,
        )

    watchlist_text = st.text_input(
        "Lista de ativos",
        value=", ".join(trader.get("watchlist", [])),
        help="Exemplo: BTC-USD, ETH-USD, PETR4.SA",
    )

    ac1, ac2 = st.columns(2)
    with ac1:
        if st.button("Salvar configuração avançada", use_container_width=True):
            state = load_bot_state()
            state["trader"]["ticket_value"] = float(ticket)
            state["trader"]["holding_minutes"] = int(holding)
            state["trader"]["max_open_positions"] = int(max_open)
            state["trader"]["watchlist"] = [x.strip().upper() for x in watchlist_text.split(",") if x.strip()]
            save_bot_state(state)
            st.success("Configuração salva.")
            st.rerun()

    with ac2:
        if st.button("Resetar módulo trader", use_container_width=True):
            try:
                with st.spinner("Resetando trader..."):
                    reset_trader_module()
                st.warning("Trader resetado.")
                st.rerun()
            except Exception as e:
                st.error(f"Erro ao resetar trader: {e}")

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Operações abertas",
            "Ordens executadas",
            "Últimos trades",
            "Métricas",
        ]
    )

    with tab1:
        if positions:
            st.dataframe(pd.DataFrame(positions), use_container_width=True)
        else:
            st.info("Sem operações abertas no trader.")

    with tab2:
        try:
            orders = pd.read_csv(TRADER_ORDERS_FILE)
            if not orders.empty:
                st.dataframe(orders.tail(200).iloc[::-1], use_container_width=True)
            else:
                st.info("Sem ordens registradas ainda.")
        except Exception as e:
            st.error(f"Erro ao ler ordens trader: {e}")

    with tab3:
        if paper_trades:
            st.dataframe(pd.DataFrame(paper_trades[::-1]), use_container_width=True)
        else:
            st.info("Sem trades ainda no motor.")

    with tab4:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Cash", br_money(float(paper_state.get("cash", 0.0))))
        m2.metric("Equity", br_money(float(paper_state.get("equity", 0.0))))
        m3.metric("Execuções", f"{paper_state.get('run_count', 0)}")
        m4.metric("Trades", f"{paper_report.get('trades_count', 0)}")

        if not paper_equity_df.empty:
            st.line_chart(paper_equity_df.set_index("timestamp")["equity_after"])
        else:
            st.info("Ainda não há curva de patrimônio para mostrar.")
