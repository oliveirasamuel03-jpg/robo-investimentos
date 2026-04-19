from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from core.auth.guards import is_admin, render_auth_toolbar, require_auth
from core.auth.security import verify_password
from core.auth.users_store import (
    create_user,
    get_user,
    list_users,
    set_user_disabled,
    update_user_role,
)
from core.config import (
    MAX_HOLDING_MINUTES,
    MAX_TICKET,
    MIN_HOLDING_MINUTES,
    MIN_TICKET,
    TRADER_ORDERS_COLUMNS,
    TRADER_ORDERS_FILE,
)
from core.state_store import load_bot_state, read_storage_table, save_bot_state
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

    for col in ["open", "high", "low", "close"]:
        if col not in df.columns:
            return pd.DataFrame()

    if "volume" not in df.columns:
        df["volume"] = 0.0

    df["ma9"] = df["close"].rolling(9).mean()
    df["ma21"] = df["close"].rolling(21).mean()
    return df


def load_trader_orders() -> pd.DataFrame:
    df = read_storage_table(TRADER_ORDERS_FILE, columns=TRADER_ORDERS_COLUMNS)

    if df.empty:
        return df

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

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
        aligned.loc[missing, "plot_price"] = aligned.loc[missing, "plot_time"].map(df_chart["close"])

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
                mode="markers",
                name="Compra",
                marker=dict(symbol="triangle-up", size=13, color="#22c55e"),
                hovertemplate="Compra<br>Tempo: %{x}<br>PreÃ§o: %{y:.2f}<extra></extra>",
            )
        )

    if not sells.empty:
        fig.add_trace(
            go.Scatter(
                x=sells["plot_time"],
                y=sells["plot_price"],
                mode="markers",
                name="Venda",
                marker=dict(symbol="triangle-down", size=13, color="#ef4444"),
                hovertemplate="Venda<br>Tempo: %{x}<br>PreÃ§o: %{y:.2f}<extra></extra>",
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
            name="PreÃ§o",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["ma9"],
            mode="lines",
            name="MÃ©dia 9",
            line=dict(width=1.6, color="#8b5cf6"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["ma21"],
            mode="lines",
            name="MÃ©dia 21",
            line=dict(width=1.6, color="#facc15"),
        )
    )

    if entry_price is not None and entry_price > 0:
        fig.add_hline(
            y=entry_price,
            line_width=1.2,
            line_dash="dot",
            line_color="#38bdf8",
            annotation_text=f"Entrada {entry_price:.2f}",
        )

    if aligned_orders is not None and not aligned_orders.empty:
        add_trade_markers(fig, aligned_orders)

    fig.update_layout(
        template="plotly_dark",
        height=560,
        title=f"{ticker}",
        xaxis_title="Tempo",
        yaxis_title="PreÃ§o",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=45, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def build_volume_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["volume"],
            name="Volume",
            marker_color="#334155",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=180,
        title=f"Volume - {ticker}",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=35, b=10),
        showlegend=False,
    )
    return fig


def br_money(value: float) -> str:
    return f"R$ {value:,.2f}"


def br_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def simple_signal_text(chart_df: pd.DataFrame) -> str:
    if chart_df.empty or len(chart_df) < 2:
        return "Aguardando dados"

    ma9 = chart_df["ma9"].iloc[-1]
    ma21 = chart_df["ma21"].iloc[-1]
    prev_ma9 = chart_df["ma9"].iloc[-2]
    prev_ma21 = chart_df["ma21"].iloc[-2]

    if pd.notna(ma9) and pd.notna(ma21) and pd.notna(prev_ma9) and pd.notna(prev_ma21):
        if ma9 > ma21 and prev_ma9 <= prev_ma21:
            return "RobÃ´ encontrou sinal de compra"
        if ma9 < ma21 and prev_ma9 >= prev_ma21:
            return "RobÃ´ encontrou sinal de venda"
        if ma9 > ma21:
            return "Mercado em tendÃªncia de alta"
        if ma9 < ma21:
            return "Mercado em tendÃªncia de baixa"

    return "RobÃ´ aguardando oportunidade"


def signal_badge_color(signal_text: str) -> str:
    text = signal_text.lower()
    if "compra" in text or "alta" in text:
        return "#22c55e"
    if "venda" in text or "baixa" in text:
        return "#ef4444"
    return "#38bdf8"


def get_last_action_text(paper_trades: list[dict]) -> str:
    if not paper_trades:
        return "Nenhuma operaÃ§Ã£o recente"

    last = paper_trades[-1]
    side = str(last.get("side", "")).upper()
    asset = str(last.get("asset", "-"))
    price = float(last.get("price", 0.0) or 0.0)

    if side == "BUY":
        return f"Comprou {asset} a {price:,.2f}"
    if side == "SELL":
        return f"Vendeu {asset} a {price:,.2f}"
    return f"Ãšltima aÃ§Ã£o em {asset}"


def get_last_execution_text(paper_state: dict) -> str:
    updated_at = paper_state.get("updated_at") or paper_state.get("last_run_at")
    if not updated_at:
        return "Ainda nÃ£o executado"
    return str(updated_at)


def get_next_execution_text(robot_status: str) -> str:
    if robot_status != "RUNNING":
        return "Pausado"
    return "Em breve"


def parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None

    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def resolve_worker_status(state_runtime: dict) -> str:
    heartbeat = parse_iso_datetime(state_runtime.get("worker_heartbeat"))
    raw_status = str(state_runtime.get("worker_status", "offline") or "offline").lower()

    if heartbeat is None:
        return raw_status

    age_seconds = (datetime.now(timezone.utc) - heartbeat).total_seconds()
    if age_seconds <= 150:
        return "error" if raw_status == "error" else "online"

    return "offline" if raw_status == "online" else raw_status


def resolve_last_action(state_runtime: dict, paper_trades: list[dict]) -> str:
    last_action = str(state_runtime.get("last_action", "") or "").strip()
    if last_action and last_action != "Nenhuma aÃ§Ã£o recente":
        return last_action
    return get_last_action_text(paper_trades)


def resolve_last_execution(state_runtime: dict, paper_state: dict) -> str:
    return str(
        state_runtime.get("last_run_at")
        or paper_state.get("updated_at")
        or paper_state.get("last_trade_at")
        or ""
    )


def build_robot_log(signal_text: str, robot_label: str, paper_trades: list[dict], open_positions: list[dict]) -> list[str]:
    logs = []

    if robot_label == "Ligado":
        logs.append("RobÃ´ ativo e monitorando o mercado.")
    elif robot_label == "Pausado":
        logs.append("RobÃ´ pausado. NÃ£o farÃ¡ novas entradas.")
    else:
        logs.append("RobÃ´ desligado.")

    logs.append(signal_text)

    if paper_trades:
        logs.append(f"Ãšltima aÃ§Ã£o: {get_last_action_text(paper_trades)}")

    if open_positions:
        logs.append(f"OperaÃ§Ãµes abertas agora: {len(open_positions)}")
    else:
        logs.append("Nenhuma operaÃ§Ã£o aberta no momento.")

    return logs[:4]


st.set_page_config(page_title="Trader Premium Max", layout="wide")

current_user = require_auth()
render_auth_toolbar()

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1450px;
    }

    .main-title {
        font-size: 2.15rem;
        font-weight: 800;
        margin-bottom: 0.15rem;
        letter-spacing: -0.02em;
    }

    .subtitle {
        color: #94a3b8;
        margin-bottom: 1rem;
    }

    .glass-card {
        background: linear-gradient(180deg, rgba(15,23,42,0.92), rgba(2,6,23,0.96));
        border: 1px solid rgba(148,163,184,0.16);
        border-radius: 18px;
        padding: 18px 18px 14px 18px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.28);
        margin-bottom: 14px;
    }

    .hero-box {
        background: linear-gradient(135deg, rgba(30,41,59,0.95), rgba(2,6,23,0.98));
        border: 1px solid rgba(148,163,184,0.16);
        border-radius: 22px;
        padding: 18px;
        margin-bottom: 14px;
        box-shadow: 0 14px 35px rgba(0,0,0,0.30);
    }

    .metric-label {
        color: #94a3b8;
        font-size: 0.90rem;
        margin-bottom: 0.35rem;
    }

    .metric-value {
        font-size: 1.85rem;
        font-weight: 800;
        line-height: 1.1;
    }

    .metric-good {
        color: #22c55e;
    }

    .metric-bad {
        color: #ef4444;
    }

    .metric-neutral {
        color: #f8fafc;
    }

    .signal-pill {
        display: inline-block;
        padding: 8px 14px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.95rem;
        margin-top: 8px;
        color: white;
        animation: pulseGlow 2.2s infinite;
    }

    .status-line {
        color: #cbd5e1;
        font-size: 0.98rem;
        margin-top: 10px;
    }

    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        margin-bottom: 0.6rem;
    }

    .small-note {
        color: #94a3b8;
        font-size: 0.92rem;
    }

    .log-card {
        background: linear-gradient(180deg, rgba(15,23,42,0.82), rgba(2,6,23,0.92));
        border: 1px solid rgba(148,163,184,0.14);
        border-radius: 16px;
        padding: 14px;
        min-height: 280px;
    }

    .log-item {
        padding: 10px 12px;
        border-radius: 12px;
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(148,163,184,0.10);
        color: #e2e8f0;
        margin-bottom: 8px;
        font-size: 0.95rem;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(180deg, rgba(15,23,42,0.80), rgba(2,6,23,0.92));
        border: 1px solid rgba(148,163,184,0.14);
        border-radius: 16px;
        padding: 10px 12px;
    }

    div[data-testid="stExpander"] {
        border: 1px solid rgba(148,163,184,0.12);
        border-radius: 16px;
        overflow: hidden;
    }

    .stButton > button {
        height: 3.2rem;
        border-radius: 14px;
        font-weight: 700;
        font-size: 1rem;
        border: 1px solid rgba(255,255,255,0.10);
    }

    @keyframes pulseGlow {
        0% { box-shadow: 0 0 0 rgba(255,255,255,0.0); }
        50% { box-shadow: 0 0 18px rgba(255,255,255,0.14); }
        100% { box-shadow: 0 0 0 rgba(255,255,255,0.0); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='main-title'>ðŸ’Ž Trader Premium Max</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Visual premium, leitura simples e um robÃ´ com status mais inteligente e vivo.</div>",
    unsafe_allow_html=True,
)

state = load_bot_state()
trader = state["trader"]
security_state = state.get("security", {}) or {}
admin_mode = is_admin(current_user)

sync_platform_positions_from_paper()
state = load_bot_state()
security_state = state.get("security", {}) or {}
paper_state = load_paper_state()
paper_report = build_paper_report(initial_capital=float(state.get("wallet_value", 10000.0)))
paper_equity_df = read_paper_equity(limit=300)
paper_trades = read_paper_trades()[-200:]
positions = [p for p in state.get("positions", []) if p.get("module") == "TRADER"]
open_positions = [p for p in positions if p.get("status") == "OPEN"]

watchlist = trader.get("watchlist", []) or ["AAPL"]

top_left, top_right = st.columns([1.4, 1.0])

with top_left:
    selected_ticker = st.selectbox("Ativo", options=watchlist, index=0)

with top_right:
    tf1, tf2 = st.columns(2)
    with tf1:
        period = st.selectbox("PerÃ­odo", options=["1d", "5d", "1mo", "3mo", "6mo"], index=2)
    with tf2:
        interval = st.selectbox("Intervalo", options=["1m", "5m", "15m", "30m", "60m", "1d"], index=2)

selected_position = next((p for p in open_positions if p.get("asset") == selected_ticker), None)
entry_price = float(selected_position.get("entry_price", 0.0)) if selected_position else None

chart_df = pd.DataFrame()
orders_df = load_trader_orders()

try:
    with st.spinner("Carregando grÃ¡fico..."):
        chart_df = load_chart_data(selected_ticker, period, interval)
except Exception as e:
    st.error(f"Erro ao carregar grÃ¡fico: {e}")

last_price = float(chart_df["close"].iloc[-1]) if not chart_df.empty else 0.0
signal_text = simple_signal_text(chart_df)
signal_color = signal_badge_color(signal_text)

robot_status = state.get("bot_status", "PAUSED")
robot_label = "Ligado" if robot_status == "RUNNING" else "Pausado" if robot_status == "PAUSED" else "Desligado"
robot_class = "metric-good" if robot_label == "Ligado" else "metric-bad" if robot_label == "Desligado" else "metric-neutral"

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(
        f"""
        <div class="glass-card">
            <div class="metric-label">Saldo disponÃ­vel</div>
            <div class="metric-value metric-neutral">{br_money(float(paper_state.get('cash', 0.0)))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m2:
    pnl_value = float(paper_report.get("net_profit", 0.0))
    pnl_class = "metric-good" if pnl_value >= 0 else "metric-bad"
    st.markdown(
        f"""
        <div class="glass-card">
            <div class="metric-label">Lucro / prejuÃ­zo</div>
            <div class="metric-value {pnl_class}">{br_money(pnl_value)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m3:
    st.markdown(
        f"""
        <div class="glass-card">
            <div class="metric-label">Status do robÃ´</div>
            <div class="metric-value {robot_class}">{robot_label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m4:
    equity_value = float(paper_state.get("equity", 0.0))
    return_pct = 0.0
    initial_capital = float(state.get("wallet_value", 0.0))
    if initial_capital > 0:
        return_pct = (equity_value - initial_capital) / initial_capital
    st.markdown(
        f"""
        <div class="glass-card">
            <div class="metric-label">Resultado %</div>
            <div class="metric-value metric-neutral">{br_pct(return_pct)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    f"""
    <div class="hero-box">
        <div class="section-title">VisÃ£o rÃ¡pida do mercado</div>
        <div class="small-note">Ativo selecionado: <b>{selected_ticker}</b></div>
        <div class="signal-pill" style="background:{signal_color};">{signal_text}</div>
        <div class="status-line">
            PreÃ§o atual: <b>{last_price:,.2f}</b> &nbsp;&nbsp;|&nbsp;&nbsp;
            OperaÃ§Ãµes abertas: <b>{len(open_positions)}</b> &nbsp;&nbsp;|&nbsp;&nbsp;
            RobÃ´: <b>{robot_label}</b>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if bool(security_state.get("real_mode_enabled", False)):
    st.warning("Real trading enabled")

b1, b2, b3 = st.columns(3)

with b1:
    if st.button("â–¶ Iniciar robÃ´", use_container_width=True, disabled=not admin_mode):
        state = load_bot_state()
        state["bot_status"] = "RUNNING"
        save_bot_state(state)
        st.success("RobÃ´ ligado.")
        st.rerun()

with b2:
    if st.button("â¸ Pausar robÃ´", use_container_width=True, disabled=not admin_mode):
        state = load_bot_state()
        state["bot_status"] = "PAUSED"
        save_bot_state(state)
        st.warning("RobÃ´ pausado.")
        st.rerun()

with b3:
    if st.button("ðŸ” Rodar agora", use_container_width=True, disabled=not admin_mode):
        try:
            with st.spinner("Executando ciclo do robÃ´..."):
                result = run_trader_cycle()
            st.success(f"Ciclo executado. Trades feitos: {result.get('cycle_result', {}).get('trades_executed', 0)}")
            st.rerun()
        except Exception as e:
            st.error(f"Erro ao rodar ciclo trader: {e}")

if not admin_mode:
    st.info("Somente administradores podem iniciar, pausar ou executar ciclos do robÃ´.")

chart_col, side_col = st.columns([2.3, 0.95])

with chart_col:
    if chart_df.empty:
        st.warning("NÃ£o foi possÃ­vel carregar dados desse ativo agora.")
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

with side_col:
    if not chart_df.empty:
        current_max = float(chart_df["high"].max())
        current_min = float(chart_df["low"].min())
    else:
        current_max = 0.0
        current_min = 0.0

    st.metric("PreÃ§o atual", f"{last_price:,.2f}")
    st.metric("MÃ¡xima", f"{current_max:,.2f}")
    st.metric("MÃ­nima", f"{current_min:,.2f}")

    if selected_position:
        qty = float(selected_position.get("qty", 0.0))
        avg_price = float(selected_position.get("entry_price", 0.0))
        market_value = qty * last_price if last_price else 0.0
        unrealized = (last_price - avg_price) * qty if last_price else 0.0

        st.markdown("### OperaÃ§Ã£o aberta")
        st.write(f"**Quantidade:** {qty:,.6f}")
        st.write(f"**PreÃ§o mÃ©dio:** {avg_price:,.2f}")
        st.write(f"**Valor atual:** {br_money(market_value)}")
        st.write(f"**Resultado atual:** {br_money(unrealized)}")
    else:
        st.info("Nenhuma operaÃ§Ã£o aberta nesse ativo.")

    st.markdown("<div class='log-card'>", unsafe_allow_html=True)
    st.markdown("### RobÃ´ em tempo real")
    st.caption("Resumo vivo do comportamento do robÃ´")

    state_runtime = load_bot_state()

    last_action = resolve_last_action(state_runtime, paper_trades)
    last_execution = resolve_last_execution(state_runtime, paper_state)
    next_execution = state_runtime.get("next_run_at", "")
    worker_status = resolve_worker_status(state_runtime)
    worker_heartbeat = state_runtime.get("worker_heartbeat", "")

    st.write(f"**Status do worker:** {worker_status}")
    st.write(f"**Ãšltima aÃ§Ã£o:** {last_action}")
    st.write(f"**Ãšltima execuÃ§Ã£o:** {last_execution or 'Ainda nÃ£o executado'}")
    st.write(f"**PrÃ³xima anÃ¡lise:** {next_execution or 'Aguardando'}")
    st.write(f"**Heartbeat:** {worker_heartbeat or 'Sem sinal'}")

    for item in build_robot_log(signal_text, robot_label, paper_trades, open_positions):
        st.markdown(f"<div class='log-item'>{item}</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

with st.expander("Modo avanÃ§ado"):
    st.markdown("### ConfiguraÃ§Ã£o avanÃ§ada")

    a1, a2, a3 = st.columns(3)
    with a1:
        ticket = st.number_input(
            "Valor por operaÃ§Ã£o (R$)",
            min_value=MIN_TICKET,
            max_value=MAX_TICKET,
            value=float(trader["ticket_value"]),
            step=10.0,
        )
    with a2:
        holding = st.slider(
            "Tempo mÃ¡ximo da operaÃ§Ã£o (min)",
            min_value=MIN_HOLDING_MINUTES,
            max_value=MAX_HOLDING_MINUTES,
            value=int(trader["holding_minutes"]),
            step=1,
        )
    with a3:
        max_open = st.slider(
            "MÃ¡x. operaÃ§Ãµes abertas",
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
        if st.button("Salvar configuraÃ§Ã£o avanÃ§ada", use_container_width=True, disabled=not admin_mode):
            state = load_bot_state()
            state["trader"]["ticket_value"] = float(ticket)
            state["trader"]["holding_minutes"] = int(holding)
            state["trader"]["max_open_positions"] = int(max_open)
            state["trader"]["watchlist"] = [x.strip().upper() for x in watchlist_text.split(",") if x.strip()]
            save_bot_state(state)
            st.success("ConfiguraÃ§Ã£o salva.")
            st.rerun()

    with ac2:
        if st.button("Resetar mÃ³dulo trader", use_container_width=True, disabled=not admin_mode):
            try:
                with st.spinner("Resetando trader..."):
                    reset_trader_module()
                st.warning("Trader resetado.")
                st.rerun()
            except Exception as e:
                st.error(f"Erro ao resetar trader: {e}")

    if not admin_mode:
        st.info("A configuraÃ§Ã£o avanÃ§ada estÃ¡ em modo somente leitura para usuÃ¡rios sem permissÃ£o de admin.")

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "ðŸ“Š OperaÃ§Ãµes em andamento",
            "ðŸ“„ HistÃ³rico de ordens",
            "âš¡ Ãšltimos trades",
            "ðŸ“ˆ MÃ©tricas",
        ]
    )

    with tab1:
        if positions:
            st.dataframe(pd.DataFrame(positions), use_container_width=True)
        else:
            st.info("Sem operaÃ§Ãµes abertas no trader.")

    with tab2:
        try:
            orders = read_storage_table(TRADER_ORDERS_FILE, columns=TRADER_ORDERS_COLUMNS)
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
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Cash", br_money(float(paper_state.get("cash", 0.0))))
        k2.metric("Equity", br_money(float(paper_state.get("equity", 0.0))))
        k3.metric("ExecuÃ§Ãµes", f"{paper_state.get('run_count', 0)}")
        k4.metric("Trades", f"{paper_report.get('trades_count', 0)}")

        st.metric("P&L", br_money(float(paper_report.get("net_profit", 0.0))))

        if not paper_equity_df.empty:
            equity_plot = paper_equity_df.copy()
            if "timestamp" in equity_plot.columns:
                equity_plot["timestamp"] = pd.to_datetime(equity_plot["timestamp"], errors="coerce")
            else:
                equity_plot["timestamp"] = pd.RangeIndex(start=1, stop=len(equity_plot) + 1)

            fig_equity = go.Figure()
            fig_equity.add_trace(
                go.Scatter(
                    x=equity_plot["timestamp"],
                    y=equity_plot["equity"],
                    mode="lines",
                    name="Equity",
                    line=dict(width=2.2, color="#38bdf8"),
                )
            )
            fig_equity.update_layout(
                template="plotly_dark",
                height=280,
                margin=dict(l=10, r=10, t=25, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis_title="Tempo",
                yaxis_title="Equity",
                showlegend=False,
            )
            st.plotly_chart(fig_equity, use_container_width=True)
        else:
            st.info("Curva de equity ainda nao disponivel.")

if admin_mode:
    st.markdown("---")
    with st.expander("AdministraÃ§Ã£o", expanded=False):
        st.markdown("### Painel administrativo")
        admin_tab_users, admin_tab_real = st.tabs(["UsuÃ¡rios", "Real Mode"])

        with admin_tab_users:
            with st.form("create_user_form"):
                new_username = st.text_input("Novo usuÃ¡rio")
                new_password = st.text_input("Senha do novo usuÃ¡rio", type="password")
                new_role = st.selectbox("Perfil", ["user", "admin"])
                create_submit = st.form_submit_button("Criar usuÃ¡rio", use_container_width=True)

            if create_submit:
                try:
                    create_user(new_username, new_password, new_role)
                    st.success("UsuÃ¡rio criado com sucesso.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"NÃ£o foi possÃ­vel criar o usuÃ¡rio: {exc}")

            users = list_users()
            if users:
                users_df = pd.DataFrame(users)
                st.dataframe(users_df, use_container_width=True)

                selected_username = st.selectbox(
                    "Selecionar usuÃ¡rio",
                    options=[user["username"] for user in users],
                    key="admin_selected_username",
                )
                selected_user = next((user for user in users if user["username"] == selected_username), None)

                if selected_user is not None:
                    role_col, disable_col = st.columns(2)
                    with role_col:
                        new_user_role = st.selectbox(
                            "Alterar perfil",
                            ["user", "admin"],
                            index=["user", "admin"].index(selected_user["role"]),
                            key="admin_selected_role",
                        )
                        if st.button("Salvar perfil", use_container_width=True, key="admin_save_role"):
                            try:
                                update_user_role(selected_username, new_user_role)
                                st.success("Perfil atualizado.")
                                st.rerun()
                            except Exception as exc:
                                st.error(f"NÃ£o foi possÃ­vel atualizar o perfil: {exc}")

                    with disable_col:
                        target_disabled = bool(selected_user.get("disabled", False))
                        action_label = "Reabilitar usuÃ¡rio" if target_disabled else "Desabilitar usuÃ¡rio"
                        if st.button(action_label, use_container_width=True, key="admin_toggle_user"):
                            try:
                                set_user_disabled(selected_username, not target_disabled)
                                st.success("Status do usuÃ¡rio atualizado.")
                                st.rerun()
                            except Exception as exc:
                                st.error(f"NÃ£o foi possÃ­vel atualizar o usuÃ¡rio: {exc}")
            else:
                st.info("Nenhum usuÃ¡rio encontrado.")

        with admin_tab_real:
            latest_state = load_bot_state()
            latest_security = latest_state.get("security", {}) or {}
            real_mode_enabled = bool(latest_security.get("real_mode_enabled", False))

            if real_mode_enabled:
                st.error("Real mode estÃ¡ habilitado.")
            else:
                st.info("Real mode estÃ¡ desligado.")

            st.caption(f"Habilitado por: {latest_security.get('real_mode_enabled_by', '-') or '-'}")
            st.caption(f"Data: {latest_security.get('real_mode_enabled_at', '-') or '-'}")

            confirm_password = st.text_input("Confirmar senha admin", type="password", key="real_mode_password")

            if st.button(
                "Desabilitar real mode" if real_mode_enabled else "Habilitar real mode",
                use_container_width=True,
                key="toggle_real_mode",
            ):
                try:
                    stored_user = get_user(current_user["username"])
                    if not stored_user:
                        raise ValueError("FaÃ§a login com uma conta admin persistida para alterar o real mode.")
                    if stored_user.get("disabled", False):
                        raise ValueError("A conta admin estÃ¡ desabilitada.")
                    if not verify_password(confirm_password, str(stored_user.get("password_hash", ""))):
                        raise ValueError("ConfirmaÃ§Ã£o de senha invÃ¡lida.")

                    latest_state = load_bot_state()
                    latest_state.setdefault("security", {})
                    latest_state["security"]["real_mode_enabled"] = not real_mode_enabled
                    latest_state["security"]["real_mode_enabled_by"] = current_user["username"]
                    latest_state["security"]["real_mode_enabled_at"] = datetime.now(timezone.utc).isoformat()
                    save_bot_state(latest_state)
                    st.success("Real mode atualizado com sucesso.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"NÃ£o foi possÃ­vel alterar o real mode: {exc}")
