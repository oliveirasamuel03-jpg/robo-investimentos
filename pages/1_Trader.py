from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.auth.guards import is_admin, render_auth_toolbar, require_auth
from core.auth.security import verify_password
from core.auth.users_store import (
    create_user,
    get_user,
    list_users,
    set_user_disabled,
    update_user_role,
)
from core.market_data import (
    build_feed_quality_snapshot,
    classify_feed_status,
    fetch_market_data_frame,
    format_market_timestamp,
    legacy_market_status,
)
from core.external_signals import format_external_signal_events_for_display
from core.macro_alerts import macro_alert_operational_effect
from core.signal_rejection_analysis import rejection_layer_label, rejection_reason_label
from core.config import (
    MAX_HOLDING_MINUTES,
    MAX_TICKET,
    MIN_HOLDING_MINUTES,
    MIN_TICKET,
    SWING_VALIDATION_DISCOURAGED_ASSET_NOTES,
    SWING_VALIDATION_RECOMMENDED_WATCHLIST,
    SWING_VALIDATION_WATCHLIST_DETAILS,
    TRADER_ORDERS_COLUMNS,
    TRADER_ORDERS_FILE,
    VALIDATION_DEFAULT_ENTRY_AMOUNT_BRL,
    VALIDATION_DEFAULT_MAX_OPEN_POSITIONS,
    VALIDATION_INITIAL_CAPITAL_BRL,
    VALIDATION_MODE_DISPLAY,
)
from core.state_store import (
    load_bot_state,
    read_storage_table,
    resolve_market_data_views as resolve_persisted_market_data_views,
    save_bot_state,
    update_market_data_status,
)
from core.swing_validation import SWING_VALIDATION_MODE, apply_swing_validation_overrides
from core.trader_profiles import (
    get_trader_profile_config,
    list_trader_profiles,
    normalize_trader_profile,
)
from core.trader_reports import (
    calculate_trade_report_metrics,
    generate_trade_suggestions,
    read_trade_reports,
    summarize_reports_by_profile,
)
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
def load_chart_data(ticker: str, period: str, interval: str, refresh_key: int | None = None) -> dict:
    df, market_data_status = fetch_market_data_frame(
        ticker,
        period=period,
        interval=interval,
        history_limit=300,
        allow_stale=True,
        requested_by="trader_chart",
    )

    if df is None or df.empty:
        return {"frame": pd.DataFrame(), "market_data_status": market_data_status}

    df = df.copy()
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df = df.dropna(subset=["datetime"]).set_index("datetime")

    df.index = pd.to_datetime(df.index)
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_convert(None)

    df = df.sort_index()

    for col in ["open", "high", "low", "close"]:
        if col not in df.columns:
            return {"frame": pd.DataFrame(), "market_data_status": market_data_status}

    if "volume" not in df.columns:
        df["volume"] = 0.0

    df["ma9"] = df["close"].rolling(9).mean()
    df["ma21"] = df["close"].rolling(21).mean()
    return {"frame": df, "market_data_status": market_data_status}


def market_data_status_label(status: dict | None) -> str:
    payload = status or {}
    return classify_feed_status(
        status=payload.get("feed_status") or payload.get("status"),
        last_source=payload.get("last_source"),
        source_breakdown=payload.get("source_breakdown"),
    )


def market_data_provider_label(status: dict | None) -> str:
    raw = str((status or {}).get("provider", "unknown") or "unknown").strip().lower()
    labels = {
        "twelvedata": "Twelve Data",
        "yahoo": "Yahoo",
        "synthetic": "Fallback sintetico",
        "mixed": "Twelve Data + Yahoo",
        "unknown": "Desconhecido",
    }
    return labels.get(raw, raw.upper())


def market_data_source_label(status: dict | None) -> str:
    payload = status or {}
    raw = str(payload.get("last_source", "unknown") or "unknown").lower()
    labels = {
        "market": "Mercado ao vivo",
        "cached": "Cache reaproveitado",
        "fallback": "Fallback sintetico",
        "mixed": "Misto",
        "unknown": "Desconhecido",
    }
    base_label = labels.get(raw, raw.title())
    provider_label = market_data_provider_label(payload)
    if raw == "fallback":
        return base_label
    return f"{base_label} via {provider_label}"


def twelvedata_diagnostic_payload(status: dict | None) -> dict:
    payload = status or {}
    diagnostics = payload.get("provider_diagnostics", {}) or {}
    diagnostic = diagnostics.get("twelvedata", {}) if isinstance(diagnostics, dict) else {}
    return dict(diagnostic or {})


def chart_interval_summary(status: dict | None, requested_interval: str | None) -> str:
    payload = status or {}
    td_diag = twelvedata_diagnostic_payload(payload)
    requested = str(
        payload.get("requested_interval")
        or requested_interval
        or td_diag.get("interval_raw")
        or ""
    ).strip()
    effective = str(
        payload.get("effective_interval")
        or td_diag.get("normalized_interval")
        or requested
    ).strip()
    if not requested and not effective:
        return "Intervalo visual: Sem registro"
    if requested and effective and requested != effective:
        return f"Intervalo visual: solicitado {requested} | usado {effective}"
    return f"Intervalo visual: {effective or requested}"


def market_data_legacy_label(status: dict | None) -> str:
    payload = status or {}
    return legacy_market_status(
        status=payload.get("status_legacy") or payload.get("status"),
        last_source=payload.get("last_source"),
        source_breakdown=payload.get("source_breakdown"),
    )


def resolve_market_data_views(state: dict, chart_market_data_status: dict | None = None) -> tuple[dict, dict]:
    operational_market_data_status, chart_state = resolve_persisted_market_data_views(state)
    if chart_market_data_status:
        chart_state.update(dict(chart_market_data_status or {}))
    return operational_market_data_status, chart_state


def market_context_label(raw_status: str | None) -> str:
    labels = {
        "FAVORAVEL": "Favoravel",
        "NEUTRO": "Neutro",
        "DESFAVORAVEL": "Desfavoravel",
        "CRITICO": "Critico",
    }
    return labels.get(str(raw_status or "NEUTRO").strip().upper(), str(raw_status or "Neutro").title())


def daily_loss_guard_label(is_blocked: bool) -> str:
    return "Entradas bloqueadas" if is_blocked else "Entradas liberadas"


def symbol_list_label(symbols: list[str] | None) -> str:
    values = [str(item).upper() for item in (symbols or []) if str(item)]
    return ", ".join(values) if values else "Nenhum"


def pct_label(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{float(value or 0.0) * 100:.1f}%"


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

    orders["ts_int"] = orders["timestamp"].astype("int64")
    chart_index["ts_int"] = chart_index["chart_time"].astype("int64")

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
                hovertemplate="Compra<br>Tempo: %{x}<br>Preço: %{y:.2f}<extra></extra>",
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
            line=dict(width=1.6, color="#8b5cf6"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["ma21"],
            mode="lines",
            name="Média 21",
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
        yaxis_title="Preço",
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


def metric_class_for_value(value: float | None) -> str:
    if value is None:
        return "metric-neutral"
    if float(value) > 0:
        return "metric-good"
    if float(value) < 0:
        return "metric-bad"
    return "metric-neutral"


def resolve_last_trade_pnl(trade_reports_df: pd.DataFrame, paper_trades: list[dict]) -> float | None:
    if trade_reports_df is not None and not trade_reports_df.empty and "closed_at" in trade_reports_df.columns:
        working = trade_reports_df.copy()
        working["closed_at"] = pd.to_datetime(working["closed_at"], errors="coerce", utc=True)
        working["realized_pnl"] = pd.to_numeric(working.get("realized_pnl"), errors="coerce")
        working = working.dropna(subset=["closed_at"]).sort_values("closed_at")
        if not working.empty:
            last_trade_value = working.iloc[-1].get("realized_pnl")
            if pd.notna(last_trade_value):
                return round(float(last_trade_value), 2)

    for trade in reversed(paper_trades or []):
        if str(trade.get("side") or "").upper() != "SELL":
            continue
        realized = pd.to_numeric(pd.Series([trade.get("realized_pnl")]), errors="coerce").iloc[0]
        if pd.notna(realized):
            return round(float(realized), 2)

    return None


def build_pnl_summary(
    paper_state: dict,
    trade_reports_df: pd.DataFrame,
    open_positions: list[dict],
    paper_trades: list[dict],
) -> dict:
    initial_capital = float(paper_state.get("initial_capital", VALIDATION_INITIAL_CAPITAL_BRL) or VALIDATION_INITIAL_CAPITAL_BRL)
    cumulative_pnl = round(float(paper_state.get("realized_pnl", 0.0) or 0.0), 2)
    open_pnl = round(
        sum(float(position.get("unrealized_pnl", 0.0) or 0.0) for position in (open_positions or [])),
        2,
    )
    last_trade_pnl = resolve_last_trade_pnl(trade_reports_df, paper_trades)
    cycle_return_pct = ((cumulative_pnl + open_pnl) / initial_capital) if initial_capital > 0 else 0.0

    return {
        "initial_capital": initial_capital,
        "cumulative_pnl": cumulative_pnl,
        "last_trade_pnl": last_trade_pnl,
        "open_pnl": open_pnl,
        "cycle_return_pct": cycle_return_pct,
    }


def simple_signal_text(chart_df: pd.DataFrame) -> str:
    if chart_df.empty or len(chart_df) < 2:
        return "Aguardando dados"

    ma9 = chart_df["ma9"].iloc[-1]
    ma21 = chart_df["ma21"].iloc[-1]
    prev_ma9 = chart_df["ma9"].iloc[-2]
    prev_ma21 = chart_df["ma21"].iloc[-2]

    if pd.notna(ma9) and pd.notna(ma21) and pd.notna(prev_ma9) and pd.notna(prev_ma21):
        if ma9 > ma21 and prev_ma9 <= prev_ma21:
            return "Robô encontrou sinal de compra"
        if ma9 < ma21 and prev_ma9 >= prev_ma21:
            return "Robô encontrou sinal de venda"
        if ma9 > ma21:
            return "Mercado em tendência de alta"
        if ma9 < ma21:
            return "Mercado em tendência de baixa"

    return "Robô aguardando oportunidade"


def signal_badge_color(signal_text: str) -> str:
    text = signal_text.lower()
    if "compra" in text or "alta" in text:
        return "#22c55e"
    if "venda" in text or "baixa" in text:
        return "#ef4444"
    return "#38bdf8"


def get_last_action_text(paper_trades: list[dict]) -> str:
    if not paper_trades:
        return "Nenhuma operação recente"

    last = paper_trades[-1]
    side = str(last.get("side", "")).upper()
    asset = str(last.get("asset", "-"))
    price = float(last.get("price", 0.0) or 0.0)

    if side == "BUY":
        return f"Comprou {asset} a {price:,.2f}"
    if side == "SELL":
        return f"Vendeu {asset} a {price:,.2f}"
    return f"Última ação em {asset}"


def get_last_execution_text(paper_state: dict) -> str:
    updated_at = paper_state.get("updated_at") or paper_state.get("last_run_at")
    if not updated_at:
        return "Ainda não executado"
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


def resolve_last_action(state_runtime: dict, paper_trades: list[dict], open_positions: list[dict]) -> str:
    last_action = str(state_runtime.get("last_action", "") or "").strip()
    last_trade_action = get_last_action_text(paper_trades)

    if not last_action or last_action == "Nenhuma ação recente":
        return last_trade_action

    # Prefer the latest persisted SELL when the platform has no open position and the
    # saved action still points to an older BUY message.
    if not open_positions and paper_trades:
        last_side = str(paper_trades[-1].get("side", "")).upper()
        if last_side == "SELL" and last_action.startswith(("Comprou ", "Última ação em ")):
            return last_trade_action

    return last_action


def resolve_last_execution(state_runtime: dict, paper_state: dict) -> str:
    return str(
        state_runtime.get("last_run_at")
        or paper_state.get("updated_at")
        or paper_state.get("last_trade_at")
        or ""
    )


def build_robot_log(signal_text: str, robot_label: str, last_action_text: str, open_positions: list[dict]) -> list[str]:
    logs = []

    if robot_label == "Ligado":
        logs.append("Robô ativo e monitorando o mercado.")
    elif robot_label == "Pausado":
        logs.append("Robô pausado. Não fará novas entradas.")
    else:
        logs.append("Robô desligado.")

    logs.append(signal_text)

    if last_action_text:
        logs.append(f"Última ação: {last_action_text}")

    if open_positions:
        logs.append(f"Operações abertas agora: {len(open_positions)}")
    else:
        logs.append("Nenhuma operação aberta no momento.")

    return logs[:4]


def build_trade_pnl_chart(reports_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if reports_df is not None and not reports_df.empty:
        plot_df = reports_df.copy()
        plot_df["closed_at"] = pd.to_datetime(plot_df["closed_at"], errors="coerce")
        plot_df["realized_pnl"] = pd.to_numeric(plot_df["realized_pnl"], errors="coerce").fillna(0.0)
        plot_df = plot_df.dropna(subset=["closed_at"]).sort_values("closed_at")
        fig.add_trace(
            go.Bar(
                x=plot_df["closed_at"],
                y=plot_df["realized_pnl"],
                marker_color=[
                    "#22c55e" if float(v) >= 0 else "#ef4444" for v in plot_df["realized_pnl"].tolist()
                ],
                name="PnL por trade",
            )
        )
    fig.update_layout(
        template="plotly_dark",
        height=260,
        margin=dict(l=10, r=10, t=35, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title="PnL por trade fechado",
        showlegend=False,
    )
    return fig


def build_profile_distribution_chart(summary_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if summary_df is not None and not summary_df.empty:
        fig.add_trace(
            go.Pie(
                labels=summary_df["profile"],
                values=summary_df["trades"],
                hole=0.45,
                marker=dict(colors=["#38bdf8", "#22c55e", "#f97316", "#eab308"]),
                textinfo="label+percent",
            )
        )
    fig.update_layout(
        template="plotly_dark",
        height=260,
        margin=dict(l=10, r=10, t=35, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title="Distribuicao por perfil",
        showlegend=False,
    )
    return fig


def build_profile_win_rate_chart(summary_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if summary_df is not None and not summary_df.empty:
        fig.add_trace(
            go.Bar(
                x=summary_df["profile"],
                y=pd.to_numeric(summary_df["win_rate"], errors="coerce").fillna(0.0) * 100.0,
                marker_color="#22c55e",
                name="Win rate",
            )
        )
    fig.update_layout(
        template="plotly_dark",
        height=260,
        margin=dict(l=10, r=10, t=35, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title="Win rate por perfil",
        yaxis_title="%",
        showlegend=False,
    )
    return fig


def make_chart_refresh_key(auto_refresh: bool, refresh_seconds: int) -> int | None:
    if not auto_refresh:
        return None
    return int(time.time() // max(int(refresh_seconds), 30))


def build_trader_snapshot(
    selected_ticker: str,
    period: str,
    interval: str,
    refresh_key: int | None = None,
) -> dict:
    sync_platform_positions_from_paper()
    state = load_bot_state()
    security_state = state.get("security", {}) or {}
    risk_state = state.get("risk", {}) or {}
    trader_state = state.get("trader", {}) or {}
    active_profile = normalize_trader_profile(trader_state.get("profile"))
    active_profile_config = get_trader_profile_config(
        active_profile,
        base_ticket_value=float(trader_state.get("ticket_value", MIN_TICKET)),
        base_holding_minutes=int(trader_state.get("holding_minutes", MIN_HOLDING_MINUTES)),
        base_max_open_positions=int(trader_state.get("max_open_positions", 1)),
    )
    paper_state = load_paper_state()
    paper_report = build_paper_report(initial_capital=float(paper_state.get("initial_capital", VALIDATION_INITIAL_CAPITAL_BRL)))
    paper_equity_df = read_paper_equity(limit=300)
    paper_trades = read_paper_trades()[-200:]
    trade_reports_df = read_trade_reports(limit=300)
    trade_report_metrics = calculate_trade_report_metrics(trade_reports_df)
    profile_summary_df = summarize_reports_by_profile(trade_reports_df)
    trade_suggestions = generate_trade_suggestions(trade_reports_df)
    positions = [p for p in state.get("positions", []) if p.get("module") == "TRADER"]
    open_positions = [p for p in positions if p.get("status") == "OPEN"]
    pnl_summary = build_pnl_summary(paper_state, trade_reports_df, open_positions, paper_trades)
    selected_position = next((p for p in open_positions if p.get("asset") == selected_ticker), None)
    entry_price = float(selected_position.get("entry_price", 0.0)) if selected_position else None

    chart_df = pd.DataFrame()
    chart_error = ""
    operational_market_data_status, chart_market_data_status = resolve_market_data_views(state)
    orders_df = load_trader_orders()

    try:
        chart_payload = load_chart_data(selected_ticker, period, interval, refresh_key=refresh_key)
        chart_df = chart_payload.get("frame", pd.DataFrame())
        chart_market_data_status = update_market_data_status(chart_payload.get("market_data_status"))
        state = load_bot_state()
        risk_state = state.get("risk", {}) or {}
        operational_market_data_status, chart_market_data_status = resolve_market_data_views(state, chart_market_data_status)
    except Exception as exc:
        chart_error = str(exc)

    last_price = float(chart_df["close"].iloc[-1]) if not chart_df.empty else 0.0
    signal_text = simple_signal_text(chart_df)
    signal_color = signal_badge_color(signal_text)

    robot_status = state.get("bot_status", "PAUSED")
    robot_label = "Ligado" if robot_status == "RUNNING" else "Pausado" if robot_status == "PAUSED" else "Desligado"
    robot_class = "metric-good" if robot_label == "Ligado" else "metric-bad" if robot_label == "Desligado" else "metric-neutral"

    last_action = resolve_last_action(state, paper_trades, open_positions)
    last_execution = resolve_last_execution(state, paper_state)
    next_execution = state.get("next_run_at", "")
    worker_status = resolve_worker_status(state)
    worker_heartbeat = state.get("worker_heartbeat", "")

    current_max = float(chart_df["high"].max()) if not chart_df.empty else 0.0
    current_min = float(chart_df["low"].min()) if not chart_df.empty else 0.0

    return {
        "state": state,
        "security_state": security_state,
        "risk_state": risk_state,
        "paper_state": paper_state,
        "paper_report": paper_report,
        "paper_equity_df": paper_equity_df,
        "paper_trades": paper_trades,
        "positions": positions,
        "open_positions": open_positions,
        "active_profile": active_profile,
        "active_profile_config": active_profile_config,
        "selected_position": selected_position,
        "entry_price": entry_price,
        "chart_df": chart_df,
        "chart_error": chart_error,
        "orders_df": orders_df,
        "last_price": last_price,
        "signal_text": signal_text,
        "signal_color": signal_color,
        "robot_status": robot_status,
        "robot_label": robot_label,
        "robot_class": robot_class,
        "last_action": last_action,
        "last_execution": last_execution,
        "next_execution": next_execution,
        "worker_status": worker_status,
        "worker_heartbeat": worker_heartbeat,
        "current_max": current_max,
        "current_min": current_min,
        "market_data_status": operational_market_data_status,
        "chart_market_data_status": chart_market_data_status,
        "trade_reports_df": trade_reports_df,
        "trade_report_metrics": trade_report_metrics,
        "profile_summary_df": profile_summary_df,
        "trade_suggestions": trade_suggestions,
        "pnl_summary": pnl_summary,
    }


def render_trader_snapshot(snapshot: dict, selected_ticker: str) -> None:
    chart_df = snapshot["chart_df"]
    paper_state = snapshot["paper_state"]
    paper_report = snapshot["paper_report"]
    open_positions = snapshot["open_positions"]
    selected_position = snapshot["selected_position"]
    last_price = float(snapshot["last_price"])
    signal_text = snapshot["signal_text"]
    signal_color = snapshot["signal_color"]
    robot_label = snapshot["robot_label"]
    robot_class = snapshot["robot_class"]
    chart_error = snapshot["chart_error"]

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="metric-label">Saldo disponivel</div>
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
                <div class="metric-label">Lucro / prejuizo</div>
                <div class="metric-value {pnl_class}">{br_money(pnl_value)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m3:
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="metric-label">Status do robo</div>
                <div class="metric-value {robot_class}">{robot_label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m4:
        equity_value = float(paper_state.get("equity", 0.0))
        return_pct = 0.0
        initial_capital = float(snapshot["state"].get("wallet_value", 0.0))
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
            <div class="section-title">Visao rapida do mercado</div>
            <div class="small-note">Ativo selecionado: <b>{selected_ticker}</b></div>
            <div class="signal-pill" style="background:{signal_color};">{signal_text}</div>
            <div class="status-line">
                Preco atual: <b>{last_price:,.2f}</b> &nbsp;&nbsp;|&nbsp;&nbsp;
                Operacoes abertas: <b>{len(open_positions)}</b> &nbsp;&nbsp;|&nbsp;&nbsp;
                Robo: <b>{robot_label}</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if bool(snapshot["security_state"].get("real_mode_enabled", False)):
        st.warning("Real trading enabled")

    chart_col, side_col = st.columns([2.3, 0.95])

    with chart_col:
        if chart_error:
            st.error(f"Erro ao carregar grafico: {chart_error}")

        if chart_df.empty:
            st.warning("Nao foi possivel carregar dados desse ativo agora.")
        else:
            aligned_orders = align_orders_to_chart(snapshot["orders_df"], chart_df, selected_ticker)
            st.plotly_chart(
                build_candle_chart(
                    chart_df,
                    selected_ticker,
                    entry_price=snapshot["entry_price"],
                    aligned_orders=aligned_orders,
                ),
                use_container_width=True,
                key=f"snapshot_chart_{selected_ticker}",
            )

            with st.expander("Mostrar volume"):
                st.plotly_chart(
                    build_volume_chart(chart_df, selected_ticker),
                    use_container_width=True,
                    key=f"snapshot_volume_{selected_ticker}",
                )

    with side_col:
        st.metric("Preco atual", f"{last_price:,.2f}")
        st.metric("Maxima", f"{float(snapshot['current_max']):,.2f}")
        st.metric("Minima", f"{float(snapshot['current_min']):,.2f}")

        if selected_position:
            qty = float(selected_position.get("qty", 0.0))
            avg_price = float(selected_position.get("entry_price", 0.0))
            market_value = qty * last_price if last_price else 0.0
            unrealized = (last_price - avg_price) * qty if last_price else 0.0

            st.markdown("### Operacao aberta")
            st.write(f"**Quantidade:** {qty:,.6f}")
            st.write(f"**Preco medio:** {avg_price:,.2f}")
            st.write(f"**Valor atual:** {br_money(market_value)}")
            st.write(f"**Resultado atual:** {br_money(unrealized)}")
        else:
            st.info("Nenhuma operacao aberta nesse ativo.")

        st.markdown("<div class='log-card'>", unsafe_allow_html=True)
        st.markdown("### Robo em tempo real")
        st.caption("Resumo vivo do comportamento do robo")

        st.write(f"**Status do worker:** {snapshot['worker_status']}")
        st.write(f"**Ultima acao:** {snapshot['last_action']}")
        st.write(f"**Ultima execucao:** {snapshot['last_execution'] or 'Ainda nao executado'}")
        st.write(f"**Proxima analise:** {snapshot['next_execution'] or 'Aguardando'}")
        st.write(f"**Heartbeat:** {snapshot['worker_heartbeat'] or 'Sem sinal'}")
        st.write(f"**Provider de dados:** {market_data_provider_label(snapshot.get('market_data_status'))}")
        st.write(f"**Status do feed (worker):** {market_data_status_label(snapshot.get('market_data_status'))}")
        st.write(
            f"**Ultimo sync do feed:** "
            f"{format_market_timestamp((snapshot.get('market_data_status') or {}).get('last_sync_at'))}"
        )
        st.write(f"**Fonte operacional:** {market_data_source_label(snapshot.get('market_data_status'))}")

        for item in build_robot_log(signal_text, robot_label, snapshot["last_action"], open_positions):
            st.markdown(f"<div class='log-item'>{item}</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)


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

    .metric-help {
        color: #64748b;
        font-size: 0.78rem;
        margin-top: 0.35rem;
        line-height: 1.35;
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

st.markdown("<div class='main-title'>💎 Trader Premium Max</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Visual premium, leitura simples e um robô com status mais inteligente e vivo.</div>",
    unsafe_allow_html=True,
)

state = load_bot_state()
trader = state["trader"]
admin_mode = is_admin(current_user)
if admin_mode:
    st.code("TRADER_UI_MARKER_20260422_A")

sync_platform_positions_from_paper()
state = load_bot_state()
security_state = state.get("security", {}) or {}
paper_state = load_paper_state()
paper_report = build_paper_report(initial_capital=float(paper_state.get("initial_capital", VALIDATION_INITIAL_CAPITAL_BRL)))
paper_equity_df = read_paper_equity(limit=300)
paper_trades = read_paper_trades()[-200:]
positions = [p for p in state.get("positions", []) if p.get("module") == "TRADER"]
open_positions = [p for p in positions if p.get("status") == "OPEN"]

watchlist = trader.get("watchlist", []) or list(SWING_VALIDATION_RECOMMENDED_WATCHLIST)
recommended_watchlist = list(SWING_VALIDATION_RECOMMENDED_WATCHLIST)
non_recommended_watchlist = [asset for asset in watchlist if asset not in recommended_watchlist]
managed_assets_outside_watchlist = sorted(
    {
        str(position.get("asset") or "").upper()
        for position in open_positions
        if str(position.get("asset") or "").upper() and str(position.get("asset") or "").upper() not in watchlist
    }
)
ticker_options = list(watchlist) + [asset for asset in managed_assets_outside_watchlist if asset not in watchlist]

top_left, top_right = st.columns([1.4, 1.0])

with top_left:
    selected_ticker = st.selectbox("Ativo", options=ticker_options, index=0)

with top_right:
    tf1, tf2 = st.columns(2)
    with tf1:
        period = st.selectbox("Período", options=["1d", "5d", "1mo", "3mo", "6mo"], index=2)
    with tf2:
        interval = st.selectbox("Intervalo", options=["1m", "5m", "15m", "30m", "60m", "1d"], index=2)

live_c1, live_c2 = st.columns([1.1, 0.9])
with live_c1:
    auto_refresh_enabled = st.toggle(
        "Atualizacao automatica do painel",
        value=True,
        help="Atualiza o painel do Trader sem mudar a frequencia do worker.",
    )
with live_c2:
    live_refresh_seconds = st.selectbox(
        "Atualizar painel a cada",
        options=[15, 30, 60, 120],
        index=1,
        disabled=not auto_refresh_enabled,
    )

st.caption(
    "Periodo e intervalo mudam apenas o grafico. O worker continua no proprio ciclo em segundo plano, hoje cerca de 60 segundos quando ligado."
)

auto_refresh_fragment_supported = bool(auto_refresh_enabled and hasattr(st, "fragment"))

page_snapshot = build_trader_snapshot(
    selected_ticker,
    period,
    interval,
    refresh_key=make_chart_refresh_key(auto_refresh_fragment_supported, int(live_refresh_seconds)),
)

state = page_snapshot["state"]
trader = state["trader"]
security_state = page_snapshot["security_state"]
risk_state = page_snapshot["risk_state"]
paper_state = page_snapshot["paper_state"]
paper_report = page_snapshot["paper_report"]
paper_equity_df = page_snapshot["paper_equity_df"]
paper_trades = page_snapshot["paper_trades"]
trade_reports_df = page_snapshot["trade_reports_df"]
trade_report_metrics = page_snapshot["trade_report_metrics"]
profile_summary_df = page_snapshot["profile_summary_df"]
trade_suggestions = page_snapshot["trade_suggestions"]
positions = page_snapshot["positions"]
open_positions = page_snapshot["open_positions"]
active_profile = page_snapshot["active_profile"]
active_profile_config = page_snapshot["active_profile_config"]
selected_position = page_snapshot["selected_position"]
entry_price = page_snapshot["entry_price"]
chart_df = page_snapshot["chart_df"]
orders_df = page_snapshot["orders_df"]
chart_error = page_snapshot["chart_error"]
last_price = float(page_snapshot["last_price"])
signal_text = page_snapshot["signal_text"]
signal_color = page_snapshot["signal_color"]
robot_status = page_snapshot["robot_status"]
robot_label = page_snapshot["robot_label"]
robot_class = page_snapshot["robot_class"]
market_data_status = page_snapshot.get("market_data_status", {})
chart_market_data_status = page_snapshot.get("chart_market_data_status", {})

selected_position = next((p for p in open_positions if p.get("asset") == selected_ticker), None)
entry_price = float(selected_position.get("entry_price", 0.0)) if selected_position else None

chart_df = pd.DataFrame()
orders_df = load_trader_orders()

try:
    with st.spinner("Carregando gráfico..."):
        chart_payload = load_chart_data(
            selected_ticker,
            period,
            interval,
            refresh_key=make_chart_refresh_key(auto_refresh_fragment_supported, int(live_refresh_seconds)),
        )
        chart_df = chart_payload.get("frame", pd.DataFrame())
        market_data_status = update_market_data_status(chart_payload.get("market_data_status"))
except Exception as e:
    st.error(f"Erro ao carregar gráfico: {e}")

last_price = float(chart_df["close"].iloc[-1]) if not chart_df.empty else 0.0
signal_text = simple_signal_text(chart_df)
signal_color = signal_badge_color(signal_text)

robot_status = state.get("bot_status", "PAUSED")
robot_label = "Ligado" if robot_status == "RUNNING" else "Pausado" if robot_status == "PAUSED" else "Desligado"
robot_class = "metric-good" if robot_label == "Ligado" else "metric-bad" if robot_label == "Desligado" else "metric-neutral"

# Reaplica o snapshot unificado mais recente para manter os cards principais e o painel vivo
# lendo a mesma base de estado durante esta renderizacao.
state = page_snapshot["state"]
security_state = page_snapshot["security_state"]
risk_state = page_snapshot["risk_state"]
paper_state = page_snapshot["paper_state"]
paper_report = page_snapshot["paper_report"]
paper_equity_df = page_snapshot["paper_equity_df"]
paper_trades = page_snapshot["paper_trades"]
pnl_summary = page_snapshot["pnl_summary"]
positions = page_snapshot["positions"]
open_positions = page_snapshot["open_positions"]
selected_position = page_snapshot["selected_position"]
entry_price = page_snapshot["entry_price"]
chart_df = page_snapshot["chart_df"]
orders_df = page_snapshot["orders_df"]
chart_error = page_snapshot["chart_error"]
last_price = float(page_snapshot["last_price"])
signal_text = page_snapshot["signal_text"]
signal_color = page_snapshot["signal_color"]
robot_status = page_snapshot["robot_status"]
robot_label = page_snapshot["robot_label"]
robot_class = page_snapshot["robot_class"]
market_data_status = page_snapshot.get("market_data_status", market_data_status)
chart_market_data_status = page_snapshot.get("chart_market_data_status", chart_market_data_status)
market_context_state = state.get("market_context", {}) or {}
current_audit_state = load_bot_state()
macro_alert_state = current_audit_state.get("macro_alert", {}) or {}
external_signal_state = current_audit_state.get("external_signal", {}) or {}
validation_state = state.get("validation", {}) or {}
validation_last_report = (validation_state.get("last_report", {}) or {})
validation_consistency = dict(validation_last_report.get("consistency", {}) or {})
validation_metrics = dict(validation_last_report.get("metrics", {}) or {})
validation_rejection_quality = dict(validation_last_report.get("rejection_quality", {}) or {})
validation_feed_rejection_consistency = dict(
    validation_last_report.get("feed_rejection_consistency")
    or validation_metrics.get("feed_rejection_consistency")
    or validation_state.get("feed_rejection_consistency", {})
    or {}
)
calibration_preview = dict(
    validation_last_report.get("calibration_preview")
    or state.get("calibration_preview", {})
    or {}
)
strategy_bottleneck = dict(
    validation_last_report.get("strategy_bottleneck")
    or state.get("strategy_bottleneck", {})
    or {}
)
phase2_fine_tune = dict(
    validation_last_report.get("phase2_fine_tune")
    or current_audit_state.get("phase2_fine_tune", {})
    or state.get("phase2_fine_tune", {})
    or {}
)
phase2_1_fine_tune = dict(
    validation_last_report.get("phase2_1_fine_tune")
    or current_audit_state.get("phase2_1_fine_tune", {})
    or state.get("phase2_1_fine_tune", {})
    or {}
)
daily_loss_limit_brl = float(risk_state.get("daily_loss_limit_brl", 0.0) or 0.0)
daily_loss_consumed_brl = float(risk_state.get("daily_loss_consumed_brl", 0.0) or 0.0)
daily_loss_remaining_brl = float(risk_state.get("daily_loss_remaining_brl", 0.0) or 0.0)
daily_realized_pnl_brl = float(risk_state.get("daily_realized_pnl_brl", 0.0) or 0.0)
daily_loss_block_active = bool(risk_state.get("daily_loss_block_active", False))
daily_loss_block_reason = str(risk_state.get("daily_loss_block_reason") or "")
daily_loss_day_key = str(risk_state.get("daily_loss_day_key") or "-")
daily_loss_blocked_at = str(risk_state.get("daily_loss_blocked_at") or "")

m1, m2, m3, m4, m5 = st.columns(5)

with m1:
    st.markdown(
        f"""
        <div class="glass-card">
            <div class="metric-label">Saldo disponível</div>
            <div class="metric-value metric-neutral">{br_money(float(paper_state.get('cash', 0.0)))}</div>
            <div class="metric-help">Caixa simulado pronto para novas entradas.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m2:
    pnl_value = float(pnl_summary.get("cumulative_pnl", 0.0) or 0.0)
    pnl_class = metric_class_for_value(pnl_value)
    st.markdown(
        f"""
        <div class="glass-card">
            <div class="metric-label">PnL Acumulado</div>
            <div class="metric-value {pnl_class}">{br_money(pnl_value)}</div>
            <div class="metric-help">Resultado total realizado do ciclo atual.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m3:
    last_trade_pnl = pnl_summary.get("last_trade_pnl")
    last_trade_value = br_money(float(last_trade_pnl)) if last_trade_pnl is not None else "Sem trade fechado"
    last_trade_class = metric_class_for_value(last_trade_pnl)
    st.markdown(
        f"""
        <div class="glass-card">
            <div class="metric-label">Última Operação</div>
            <div class="metric-value {last_trade_class}">{last_trade_value}</div>
            <div class="metric-help">Resultado do último trade fechado.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m4:
    open_pnl = float(pnl_summary.get("open_pnl", 0.0) or 0.0)
    open_pnl_class = metric_class_for_value(open_pnl)
    st.markdown(
        f"""
        <div class="glass-card">
            <div class="metric-label">PnL em Aberto</div>
            <div class="metric-value {open_pnl_class}">{br_money(open_pnl)}</div>
            <div class="metric-help">Resultado das posições ainda abertas.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m5:
    cycle_return_pct = float(pnl_summary.get("cycle_return_pct", 0.0) or 0.0)
    st.markdown(
        f"""
        <div class="glass-card">
            <div class="metric-label">Retorno do ciclo</div>
            <div class="metric-value metric-neutral">{br_pct(cycle_return_pct)}</div>
            <div class="metric-help">Resultado realizado + aberto sobre o capital inicial.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.caption(
    "Configuração inicial desta fase: "
    f"capital base {br_money(VALIDATION_INITIAL_CAPITAL_BRL)} | "
    f"entrada padrão {br_money(VALIDATION_DEFAULT_ENTRY_AMOUNT_BRL)} | "
    f"até {int(VALIDATION_DEFAULT_MAX_OPEN_POSITIONS)} posições simultâneas | "
    f"modo {VALIDATION_MODE_DISPLAY} | PAPER TRADING obrigatório."
)

if daily_loss_block_active:
    st.error(
        "Trava diária de risco ativa: novas entradas estão bloqueadas por perda diária. "
        "As posições já abertas continuam sob gestão normal."
    )
else:
    st.success("Trava diária de risco ativa no sistema: novas entradas estão liberadas no momento.")

risk_c1, risk_c2, risk_c3, risk_c4 = st.columns(4)
risk_c1.metric("Trava diária", daily_loss_guard_label(daily_loss_block_active))
risk_c2.metric("Limite diário", br_money(daily_loss_limit_brl))
risk_c3.metric("Perda consumida", br_money(daily_loss_consumed_brl))
risk_c4.metric("Limite restante", br_money(daily_loss_remaining_brl))

st.caption(
    f"Dia operacional UTC: {daily_loss_day_key} | "
    f"PnL realizado do dia (base da trava): {br_money(daily_realized_pnl_brl)}"
)
if daily_loss_block_active:
    st.caption(
        f"Bloqueio ativado em: {format_market_timestamp(daily_loss_blocked_at)} | "
        f"Motivo: {daily_loss_block_reason or 'Limite diário atingido.'}"
    )

st.markdown("### Alerta macro de risco")
st.caption(
    "Filtro operacional de risco: nao e gatilho direto de compra/venda, nao autoriza trades sozinho "
    "e preserva PAPER TRADING."
)
macro_active = bool(macro_alert_state.get("macro_alert_active", False))
macro_c1, macro_c2, macro_c3, macro_c4 = st.columns(4)
macro_c1.metric("Alerta macro", "Ativo" if macro_active else "Inativo")
macro_c2.metric("Impacto", str(macro_alert_state.get("macro_alert_level") or "LOW"))
macro_c3.metric("Janela", str(macro_alert_state.get("macro_alert_window_status") or "INACTIVE"))
macro_c4.metric(
    "Penalidade",
    f"{float(macro_alert_state.get('macro_alert_penalty', 0.0) or 0.0):.2f}",
)
macro_event_time = macro_alert_state.get("macro_alert_time")
macro_minutes = macro_alert_state.get("macro_alert_minutes_to_event")
st.caption(
    f"Evento: {macro_alert_state.get('macro_alert_title') or 'Sem evento ativo'} | "
    f"Moeda: {macro_alert_state.get('macro_alert_currency') or '-'} | "
    f"Horario: {format_market_timestamp(macro_event_time) if macro_event_time else 'Sem registro'} | "
    f"Minutos ate o evento: {macro_minutes if macro_minutes is not None else '-'}"
)
st.caption(f"Efeito operacional: {macro_alert_operational_effect(macro_alert_state)}")

st.markdown("### External signal audit")
st.caption(
    "FASE 3A: sinais externos sao apenas entrada complementar de auditoria. "
    "Nao executam trades, nao aprovam entradas e nao alteram a estrategia interna. PAPER TRADING obrigatorio."
)
external_enabled = bool(external_signal_state.get("enabled", False))
external_status = str(external_signal_state.get("last_status") or ("DISABLED" if not external_enabled else "IGNORED"))
external_score = float(external_signal_state.get("last_score", 0.0) or 0.0)
external_c1, external_c2, external_c3, external_c4 = st.columns(4)
external_c1.metric("Webhook externo", "Ativo" if external_enabled else "Inativo")
external_c2.metric("Status", external_status)
external_c3.metric("Fonte", str(external_signal_state.get("last_source") or "Sem registro"))
external_c4.metric("Score recebido", f"{external_score:.2f}")
external_c5, external_c6, external_c7, external_c8 = st.columns(4)
external_c5.metric("Estrategia", str(external_signal_state.get("last_strategy") or "Sem registro"))
external_c6.metric("Ativo", str(external_signal_state.get("last_symbol") or "Sem registro"))
external_c7.metric("Lado", str(external_signal_state.get("last_side") or "Sem registro"))
external_c8.metric("Timeframe", str(external_signal_state.get("last_timeframe") or "Sem registro"))
st.caption(
    f"Recebido em: {format_market_timestamp(external_signal_state.get('last_received_at')) if external_signal_state.get('last_received_at') else 'Sem registro'} | "
    f"Motivo: {external_signal_state.get('last_reason') or 'Sem sinal externo recebido.'}"
)
st.caption("Autoridade: audit-only, sem poder de compra/venda, sem bypass de guards e sem impacto em score.")
if external_enabled and not bool(external_signal_state.get("webhook_configured", False)):
    st.warning("Webhook externo habilitado, mas configuracao incompleta. Sinais serao rejeitados com seguranca.")
recent_external_events = format_external_signal_events_for_display(external_signal_state, limit=10)
st.caption("Eventos recentes de sinal externo: audit-only, sem execucao e sem aprovacao de trade.")
if recent_external_events:
    st.dataframe(pd.DataFrame(recent_external_events), hide_index=True, use_container_width=True)
else:
    st.caption("Sem eventos recentes de sinal externo.")

st.markdown(
    f"""
    <div class="hero-box">
        <div class="section-title">Visão rápida do mercado</div>
        <div class="small-note">Ativo selecionado: <b>{selected_ticker}</b></div>
        <div class="signal-pill" style="background:{signal_color};">{signal_text}</div>
        <div style="margin-top:10px;">
            <span class="signal-pill" style="background:{active_profile_config['accent']};">
                Perfil ativo: {active_profile_config['name']} - {active_profile_config['description']}
            </span>
        </div>
        <div class="status-line">
            Preço atual: <b>{last_price:,.2f}</b> &nbsp;&nbsp;|&nbsp;&nbsp;
            Operações abertas: <b>{len(open_positions)}</b> &nbsp;&nbsp;|&nbsp;&nbsp;
            Robô: <b>{robot_label}</b>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

context_c1, context_c2, context_c3, context_c4 = st.columns(4)
context_c1.metric("Contexto cripto", market_context_label(market_context_state.get("market_context_status")))
context_c2.metric("Score de contexto", f"{float(market_context_state.get('market_context_score', 50.0) or 50.0):.1f}")
context_c3.metric(
    "Sinais barrados",
    str(int((validation_last_report.get("metrics", {}) or {}).get("context_blocked_signals", 0) or 0)),
)
context_c4.metric("PAPER", "Ativo")
st.caption(
    "Watchlist otimizada para validacao swing em paper. "
    f"Motivo atual: {market_context_state.get('market_context_reason') or 'Sem motivo registrado.'}"
)
st.caption(
    f"Impacto no robo: {market_context_state.get('market_context_impact') or 'Sem impacto adicional.'}"
)
st.caption(
    "Escopo atual: CRIPTO ONLY | PAPER TRADING | validacao swing de 10 dias. "
    "A selecao foi reduzida intencionalmente para priorizar qualidade de sinal e estabilidade."
)
st.caption(f"Watchlist atual: {', '.join(watchlist)}")
st.caption(f"Watchlist recomendada da fase: {', '.join(recommended_watchlist)}")
if non_recommended_watchlist:
    st.warning(
        "Ha ativos fora da watchlist recomendada desta fase: "
        f"{', '.join(non_recommended_watchlist)}. Eles continuam editaveis, mas nao sao os mais indicados agora."
    )
if managed_assets_outside_watchlist:
    st.warning(
        "Ha posicoes abertas fora da watchlist atual que seguem sob gestao normal: "
        f"{', '.join(managed_assets_outside_watchlist)}."
    )
if validation_consistency.get("capital_phase_aligned") is False:
    st.warning(
        "O capital atual do paper nao esta alinhado ao capital-base recomendado da fase. "
        "Para iniciar um ciclo limpo, salve a configuracao desejada e use 'Resetar modulo trader'."
    )
if validation_consistency.get("sample_quality_message"):
    st.caption(f"Amostra operacional: {validation_consistency.get('sample_quality_message')}")
if validation_consistency.get("watchlist_message"):
    st.caption(f"Leitura de consistencia: {validation_consistency.get('watchlist_message')}")

if bool(security_state.get("real_mode_enabled", False)):
    st.warning("Real trading enabled")

b1, b2, b3 = st.columns(3)

with b1:
    if st.button("▶ Iniciar robô", use_container_width=True, disabled=not admin_mode):
        state = load_bot_state()
        state["bot_status"] = "RUNNING"
        save_bot_state(state)
        st.success("Robô ligado.")
        st.rerun()

with b2:
    if st.button("⏸ Pausar robô", use_container_width=True, disabled=not admin_mode):
        state = load_bot_state()
        state["bot_status"] = "PAUSED"
        save_bot_state(state)
        st.warning("Robô pausado.")
        st.rerun()

with b3:
    if st.button("🔁 Rodar agora", use_container_width=True, disabled=not admin_mode):
        try:
            with st.spinner("Executando ciclo do robô..."):
                result = run_trader_cycle()
            st.success(f"Ciclo executado. Trades feitos: {result.get('cycle_result', {}).get('trades_executed', 0)}")
            st.rerun()
        except Exception as e:
            st.error(f"Erro ao rodar ciclo trader: {e}")

if not admin_mode:
    st.info("Somente administradores podem iniciar, pausar ou executar ciclos do robô.")

if auto_refresh_fragment_supported:
    st.markdown("### Painel ao vivo")
    st.caption(
        f"Atualizacao automatica ligada a cada {int(live_refresh_seconds)}s. O bloco abaixo acompanha o worker sem mudar a frequencia de execucao do robo."
    )

    @st.fragment(run_every=f"{int(live_refresh_seconds)}s")
    def render_live_monitor() -> None:
        live_snapshot = build_trader_snapshot(
            selected_ticker,
            period,
            interval,
            refresh_key=make_chart_refresh_key(True, int(live_refresh_seconds)),
        )

        live_risk_state = live_snapshot.get("risk_state", {}) or {}
        live_daily_loss_block_active = bool(live_risk_state.get("daily_loss_block_active", False))
        live_daily_loss_limit = float(live_risk_state.get("daily_loss_limit_brl", 0.0) or 0.0)
        live_daily_loss_consumed = float(live_risk_state.get("daily_loss_consumed_brl", 0.0) or 0.0)

        lm1, lm2, lm3, lm4, lm5, lm6 = st.columns(6)
        lm1.metric("Preco ao vivo", f"{float(live_snapshot['last_price']):,.2f}")
        lm2.metric("Worker", str(live_snapshot["worker_status"]))
        lm3.metric("Posicoes", f"{len(live_snapshot['open_positions'])}")
        lm4.metric("Bot", str(live_snapshot["robot_label"]))
        lm5.metric("Feed", market_data_status_label(live_snapshot.get("market_data_status")))
        lm6.metric("Risco diario", daily_loss_guard_label(live_daily_loss_block_active))

        st.caption(
            f"Ultima acao: {live_snapshot['last_action']} | "
            f"Ultima execucao: {live_snapshot['last_execution'] or 'Aguardando'} | "
            f"Heartbeat: {live_snapshot['worker_heartbeat'] or 'Sem sinal'} | "
            f"Feed operacional: {market_data_status_label(live_snapshot.get('market_data_status'))} | "
            f"Ultimo sync: {format_market_timestamp((live_snapshot.get('market_data_status') or {}).get('last_sync_at'))} | "
            f"Fonte operacional: {market_data_source_label(live_snapshot.get('market_data_status'))} | "
            f"Perda diaria: {br_money(live_daily_loss_consumed)} de {br_money(live_daily_loss_limit)}"
        )
        if live_daily_loss_block_active:
            st.caption(
                "Entradas bloqueadas no ciclo atual por limite de perda diaria. "
                f"Motivo: {live_risk_state.get('daily_loss_block_reason') or 'Limite atingido.'}"
            )
        live_chart_market_data_status = live_snapshot.get("chart_market_data_status") or {}
        if live_chart_market_data_status:
            st.caption(
                "Leitura do grafico nesta tela: "
                f"{market_data_status_label(live_chart_market_data_status)} | "
                f"Ultimo sync do grafico: {format_market_timestamp(live_chart_market_data_status.get('last_sync_at'))} | "
                f"Fonte do grafico: {market_data_source_label(live_chart_market_data_status)}"
            )
            st.caption(chart_interval_summary(live_chart_market_data_status, interval))

        worker_feed_quality = build_feed_quality_snapshot(live_snapshot.get("market_data_status"))
        chart_feed_quality = build_feed_quality_snapshot(live_chart_market_data_status)
        st.markdown("#### Qualidade do feed")
        feed_q1, feed_q2 = st.columns(2)
        with feed_q1:
            st.write("**Feed operacional do worker**")
            st.write(
                f"Status: {market_data_status_label(live_snapshot.get('market_data_status'))} | "
                f"Fonte: {market_data_source_label(live_snapshot.get('market_data_status'))}"
            )
            st.write(f"Sucesso Twelve Data no ciclo: {pct_label(worker_feed_quality.get('twelvedata_success_rate'))}")
            st.write(
                f"Ativos live no ciclo: "
                f"{symbol_list_label(worker_feed_quality.get('live_symbols'))}"
            )
            st.write(
                f"Ativos em fallback no ciclo: "
                f"{symbol_list_label(worker_feed_quality.get('fallback_symbols'))}"
            )
            if worker_feed_quality.get("fallback_reason"):
                st.caption(f"Motivo do fallback operacional: {worker_feed_quality.get('fallback_reason')}")
            if worker_feed_quality.get("quality_message"):
                st.caption(worker_feed_quality.get("quality_message"))
        with feed_q2:
            st.write("**Feed do grafico desta tela**")
            st.write(
                f"Status: {market_data_status_label(live_chart_market_data_status)} | "
                f"Fonte: {market_data_source_label(live_chart_market_data_status)}"
            )
            st.write(
                f"Ativos live no grafico: "
                f"{symbol_list_label(chart_feed_quality.get('live_symbols'))}"
            )
            st.write(
                f"Ativos em fallback no grafico: "
                f"{symbol_list_label(chart_feed_quality.get('fallback_symbols'))}"
            )
            st.caption(chart_interval_summary(live_chart_market_data_status, interval))
            if chart_feed_quality.get("fallback_reason"):
                st.caption(f"Motivo do fallback visual: {chart_feed_quality.get('fallback_reason')}")
            td_chart_diag = twelvedata_diagnostic_payload(live_chart_market_data_status)
            if td_chart_diag.get("request_attempted"):
                st.caption(
                    "Diagnostico visual do Twelve Data: "
                    f"intervalo solicitado {td_chart_diag.get('interval_raw') or interval} | "
                    f"intervalo usado {td_chart_diag.get('normalized_interval') or td_chart_diag.get('interval_raw') or interval} | "
                    f"estagio {td_chart_diag.get('last_stage') or 'sem-registro'}"
                )
            st.caption("O worker usa o feed operacional acima. O grafico pode cair em fallback sem mudar a fonte operacional.")

        live_chart_df = live_snapshot["chart_df"]
        if not live_chart_df.empty:
            aligned_orders = align_orders_to_chart(live_snapshot["orders_df"], live_chart_df, selected_ticker)
            st.plotly_chart(
                build_candle_chart(
                    live_chart_df,
                    selected_ticker,
                    entry_price=live_snapshot["entry_price"],
                    aligned_orders=aligned_orders,
                ),
                use_container_width=True,
                key=f"live_chart_{selected_ticker}",
            )
        else:
            st.info("Sem dados recentes para o painel ao vivo neste momento.")

    render_live_monitor()

chart_col, side_col = st.columns([2.3, 0.95])

with chart_col:
    if chart_error:
        st.caption(f"Fallback do grafico ativado: {chart_error}")
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
            key=f"main_chart_{selected_ticker}",
        )

        with st.expander("Mostrar volume"):
            st.plotly_chart(
                build_volume_chart(chart_df, selected_ticker),
                use_container_width=True,
                key=f"main_volume_{selected_ticker}",
            )

with side_col:
    if not chart_df.empty:
        current_max = float(chart_df["high"].max())
        current_min = float(chart_df["low"].min())
    else:
        current_max = 0.0
        current_min = 0.0

    st.metric("Preço atual", f"{last_price:,.2f}")
    st.metric("Máxima", f"{current_max:,.2f}")
    st.metric("Mínima", f"{current_min:,.2f}")

    if selected_position:
        qty = float(selected_position.get("qty", 0.0))
        avg_price = float(selected_position.get("entry_price", 0.0))
        market_value = qty * last_price if last_price else 0.0
        unrealized = (last_price - avg_price) * qty if last_price else 0.0

        st.markdown("### Operação aberta")
        st.write(f"**Quantidade:** {qty:,.6f}")
        st.write(f"**Preço médio:** {avg_price:,.2f}")
        st.write(f"**Valor atual:** {br_money(market_value)}")
        st.write(f"**Resultado atual:** {br_money(unrealized)}")
    else:
        st.info("Nenhuma operação aberta nesse ativo.")

    st.markdown("<div class='log-card'>", unsafe_allow_html=True)
    st.markdown("### Robô em tempo real")
    st.caption("Resumo vivo do comportamento do robô")

    state_runtime = load_bot_state()

    last_action = resolve_last_action(state_runtime, paper_trades, open_positions)
    last_execution = resolve_last_execution(state_runtime, paper_state)
    next_execution = state_runtime.get("next_run_at", "")
    worker_status = resolve_worker_status(state_runtime)
    worker_heartbeat = state_runtime.get("worker_heartbeat", "")

    st.write(f"**Status do worker:** {worker_status}")
    st.write(f"**Última ação:** {last_action}")
    st.write(f"**Última execução:** {last_execution or 'Ainda não executado'}")
    st.write(f"**Próxima análise:** {next_execution or 'Aguardando'}")
    st.write(f"**Heartbeat:** {worker_heartbeat or 'Sem sinal'}")
    st.write(f"**Provider de dados:** {market_data_provider_label(market_data_status)}")
    st.write(f"**Status do feed (worker):** {market_data_status_label(market_data_status)}")
    st.write(f"**Ultimo sync do feed:** {format_market_timestamp((market_data_status or {}).get('last_sync_at'))}")
    st.write(f"**Fonte operacional:** {market_data_source_label(market_data_status)}")
    st.write(f"**Trava diaria de risco:** {daily_loss_guard_label(daily_loss_block_active)}")
    st.write(f"**Perda consumida hoje:** {br_money(daily_loss_consumed_brl)} / {br_money(daily_loss_limit_brl)}")
    td_diag = twelvedata_diagnostic_payload(market_data_status)
    if td_diag:
        st.caption(
            "Diagnostico Twelve Data: "
            f"build {td_diag.get('build_label') or 'sem-registro'} | "
            f"chave lida: {'Sim' if td_diag.get('api_key_present') else 'Nao'} | "
            f"request saiu: {'Sim' if td_diag.get('request_attempted') else 'Nao'} | "
            f"estagio: {td_diag.get('last_stage') or 'sem-registro'}"
        )
        if td_diag.get("last_error"):
            st.caption(f"Ultimo erro Twelve Data: {td_diag.get('last_error')}")
    if market_data_status.get("state_writer") or market_data_status.get("state_written_at"):
        st.caption(
            "Estado operacional compartilhado: "
            f"writer {market_data_status.get('state_writer') or 'nao-registrado'} | "
            f"gravado em {market_data_status.get('state_written_at') or 'nao-registrado'} | "
            f"build do deploy {market_data_status.get('build_active') or 'nao-registrado'}"
        )
    if admin_mode:
        st.caption(
            "Admin audit snapshot: "
            f"ui_audit_probe={market_data_status.get('ui_audit_probe') or 'NAO REGISTRADO NO ESTADO ATUAL'} | "
            f"state_writer={market_data_status.get('state_writer') or 'NAO REGISTRADO NO ESTADO ATUAL'} | "
            f"state_written_at={market_data_status.get('state_written_at') or 'NAO REGISTRADO NO ESTADO ATUAL'} | "
            f"sha_deploy={market_data_status.get('git_sha') or 'NAO REGISTRADO NO ESTADO ATUAL'} | "
            f"sha_origem={market_data_status.get('source_commit_sha') or 'NAO INFORMADO PELO DEPLOY'}"
        )
    if daily_loss_block_active:
        st.write(f"**Bloqueio ativado em:** {format_market_timestamp(daily_loss_blocked_at)}")
        st.caption(f"Motivo da trava: {daily_loss_block_reason or 'Limite diário atingido.'}")
    st.caption(f"Taxonomia legada preservada: {market_data_legacy_label(market_data_status)}")
    if chart_market_data_status:
        st.caption(
            "Grafico desta tela: "
            f"{market_data_status_label(chart_market_data_status)} | "
            f"Ultimo sync do grafico: {format_market_timestamp(chart_market_data_status.get('last_sync_at'))} | "
            f"Fonte do grafico: {market_data_source_label(chart_market_data_status)}"
        )
        st.caption(chart_interval_summary(chart_market_data_status, interval))

    for item in build_robot_log(signal_text, robot_label, last_action, open_positions):
        st.markdown(f"<div class='log-item'>{item}</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("### Performance e relatorios do Trader")
st.caption("Trades fechados, distribuicao por perfil e sugestoes analiticas. Nenhum ajuste e aplicado automaticamente nesta etapa.")

perf_m1, perf_m2, perf_m3, perf_m4 = st.columns(4)
perf_m1.metric("Trades fechados", f"{int(trade_report_metrics['total_trades'])}")
perf_m2.metric("Win rate", f"{float(trade_report_metrics['win_rate'] or 0.0) * 100:.2f}%")
perf_m3.metric("Payoff", "-" if trade_report_metrics["payoff"] is None else f"{float(trade_report_metrics['payoff']):.2f}")
perf_m4.metric("PnL fechado", br_money(float(trade_report_metrics["total_pnl"] or 0.0)))

cons_m1, cons_m2, cons_m3, cons_m4 = st.columns(4)
cons_m1.metric("Amostra do ciclo", str(validation_consistency.get("sample_quality_label") or "Sem leitura"))
cons_m2.metric("Postura operacional", str(validation_consistency.get("operational_posture_label") or "Indefinida"))
cons_m3.metric(
    "Aprovacao de sinais",
    "-"
    if validation_consistency.get("signal_approval_rate") is None
    else f"{float(validation_consistency.get('signal_approval_rate') or 0.0) * 100:.1f}%",
)
cons_m4.metric(
    "Drawdown max",
    "-"
    if validation_metrics.get("max_drawdown_pct") is None
    else f"{float(validation_metrics.get('max_drawdown_pct') or 0.0) * 100:.2f}%",
)
signal_m1, signal_m2, signal_m3, signal_m4, signal_m5 = st.columns(5)
signal_m1.metric("Sinais aprovados", str(int(validation_metrics.get("signals_approved", 0) or 0)))
signal_m2.metric("Sinais rejeitados", str(int(validation_metrics.get("signals_rejected", 0) or 0)))
signal_m3.metric("Qualidade do sinal", str(validation_consistency.get("signal_quality_label") or "Baixa"))
signal_m4.metric(
    "Watchlist da fase",
    "Coerente" if bool(validation_consistency.get("watchlist_phase_aligned")) else "Fora da fase",
)
signal_m5.metric(
    "Ajuste fino",
    "Base minima" if bool(validation_consistency.get("fine_tuning_ready")) else "Aguardar",
)
if validation_consistency.get("operational_posture_message"):
    st.caption(f"Postura atual: {validation_consistency.get('operational_posture_message')}")
if validation_consistency.get("signal_quality_message"):
    st.caption(f"Leitura de sinal: {validation_consistency.get('signal_quality_message')}")
if validation_consistency.get("validation_reading_message"):
    st.caption(f"Leitura da validacao: {validation_consistency.get('validation_reading_message')}")

rej_m1, rej_m2, rej_m3, rej_m4 = st.columns(4)
rej_m1.metric(
    "Motivo principal de rejeicao",
    rejection_reason_label(validation_rejection_quality.get("top_reason"))
    if validation_rejection_quality.get("top_reason")
    else "Sem leitura",
)
rej_m2.metric(
    "Camada dominante",
    rejection_layer_label(validation_rejection_quality.get("top_layer"))
    if validation_rejection_quality.get("top_layer")
    else "Sem leitura",
)
rej_m3.metric(
    "Setup mais bloqueado",
    str(validation_rejection_quality.get("top_strategy") or "Sem leitura"),
)
rej_m4.metric(
    "Base minima para ajuste fino",
    "Sim" if bool(validation_rejection_quality.get("has_minimum_sample")) else "Nao",
)
top_reasons = validation_rejection_quality.get("top_reasons", []) or []
if top_reasons:
    lead_reason = dict(top_reasons[0] or {})
    st.caption(
        "Rejeicao dominante agora: "
        f"{lead_reason.get('human_reason') or 'Sem leitura'} "
        f"({pct_label(lead_reason.get('pct'))})."
    )
elif validation_metrics.get("signals_rejected", 0):
    st.caption("Ja existem rejeicoes no ciclo, mas ainda sem detalhe consolidado suficiente.")
if validation_feed_rejection_consistency:
    st.caption(
        "Diagnostico feed x rejeicao: "
        f"{validation_feed_rejection_consistency.get('diagnostic_note') or 'Sem leitura consolidada.'}"
    )
    st.caption(
        "Escopo da rejeicao: "
        f"atual={rejection_reason_label(validation_feed_rejection_consistency.get('current_cycle_rejection_reason'))} | "
        f"acumulado={rejection_reason_label(validation_feed_rejection_consistency.get('accumulated_rejection_reason'))} | "
        f"fallback atual/acumulado="
        f"{int(validation_feed_rejection_consistency.get('fallback_rejection_current_cycle_count', 0) or 0)}/"
        f"{int(validation_feed_rejection_consistency.get('fallback_rejection_accumulated_count', 0) or 0)} | "
        f"estrategia atual/acumulado="
        f"{int(validation_feed_rejection_consistency.get('strategy_rejection_current_cycle_count', 0) or 0)}/"
        f"{int(validation_feed_rejection_consistency.get('strategy_rejection_accumulated_count', 0) or 0)}"
    )
if calibration_preview:
    st.markdown("#### Calibration Preview - PREVIEW ONLY")
    st.caption(
        "Diagnostico conservador: nao aprova trades, nao reduz thresholds, nao altera estrategia "
        "e preserva PAPER TRADING."
    )
    cal_m1, cal_m2, cal_m3, cal_m4, cal_m5 = st.columns(5)
    min_score = calibration_preview.get("min_score_current")
    preview_floor = calibration_preview.get("preview_score_floor")
    best_score = calibration_preview.get("best_score_seen")
    avg_gap = calibration_preview.get("avg_score_gap")
    avg_gap_label = "-" if avg_gap is None else f"{float(avg_gap):.3f}"
    cal_m1.metric("Min score atual", "-" if min_score is None else f"{float(min_score):.2f}")
    cal_m2.metric("Piso preview", "-" if preview_floor is None else f"{float(preview_floor):.2f}")
    cal_m3.metric("Quase aprovados", str(int(calibration_preview.get("near_approved_count", 0) or 0)))
    cal_m4.metric("Taxa preview", pct_label(calibration_preview.get("near_approved_rate")))
    cal_m5.metric("Melhor score visto", "-" if best_score is None else f"{float(best_score):.2f}")
    st.caption(
        f"Gap medio: {avg_gap_label} | "
        f"Top ativo: {calibration_preview.get('top_asset') or '-'} | "
        f"Top setup: {calibration_preview.get('top_setup') or '-'} | "
        f"Recomendacao: {calibration_preview.get('recommendation') or 'observe_more'}"
    )
    st.caption(calibration_preview.get("reason") or "Sem leitura de preview ainda.")
    preview_examples = list(calibration_preview.get("near_approved_examples", []) or [])[:5]
    if preview_examples:
        st.dataframe(pd.DataFrame(preview_examples), hide_index=True, use_container_width=True)
if strategy_bottleneck:
    st.markdown("#### Strategy Bottleneck - DIAGNOSTIC ONLY")
    st.caption(
        "Diagnostico interno: nao aprova trades, nao reduz thresholds, nao altera estrategia "
        "e preserva PAPER TRADING."
    )
    bot_m1, bot_m2, bot_m3, bot_m4 = st.columns(4)
    bot_m1.metric("Bottleneck dominante", strategy_bottleneck.get("dominant_bottleneck") or "-")
    bot_m2.metric("Setup dominante", strategy_bottleneck.get("dominant_setup") or "-")
    bot_m3.metric("Ativo dominante", strategy_bottleneck.get("dominant_asset") or "-")
    bot_m4.metric("Rejeicoes estrategia", str(int(strategy_bottleneck.get("total_strategy_rejections", 0) or 0)))
    bot_c1, bot_c2, bot_c3, bot_c4, bot_c5 = st.columns(5)
    bot_c1.metric("Score baixo", str(int(strategy_bottleneck.get("score_below_min_count", 0) or 0)))
    bot_c2.metric("Momentum fraco", str(int(strategy_bottleneck.get("momentum_weak_count", 0) or 0)))
    bot_c3.metric("Confirmacao fraca", str(int(strategy_bottleneck.get("secondary_confirmation_weak_count", 0) or 0)))
    bot_c4.metric("RSI fora", str(int(strategy_bottleneck.get("rsi_out_of_range_count", 0) or 0)))
    bot_c5.metric("Trend nao confirmada", str(int(strategy_bottleneck.get("trend_not_confirmed_count", 0) or 0)))
    st.caption(
        f"Volatilidade: {int(strategy_bottleneck.get('volatility_filter_count', 0) or 0)} | "
        f"Contexto: {int(strategy_bottleneck.get('context_filter_count', 0) or 0)} | "
        f"Recomendacao: {strategy_bottleneck.get('recommendation') or 'observe_more'}"
    )
    st.caption(strategy_bottleneck.get("reason") or "Sem diagnostico de bottleneck ainda.")
    for label, key in (
        ("Top ativos bloqueados", "top_assets_blocked"),
        ("Top setups bloqueados", "top_setups_blocked"),
        ("Top filtros", "top_filter_reasons"),
        ("Candidatos mais proximos", "closest_candidates"),
    ):
        items = list(strategy_bottleneck.get(key, []) or [])[:5]
        if items:
            st.caption(label)
            st.dataframe(pd.DataFrame(items), hide_index=True, use_container_width=True)
if phase2_fine_tune:
    st.markdown("#### Ajuste Fino FASE 2")
    st.caption(
        "Relaxamento conservador e reversivel: PAPER only, nao altera score minimo global, "
        "nao muda broker e permanece protegido pelos guards existentes."
    )
    fine_status = "Ativo" if bool(phase2_fine_tune.get("fine_tune_enabled", False)) else "Inativo"
    ft_c1, ft_c2, ft_c3, ft_c4 = st.columns(4)
    ft_c1.metric("Status", fine_status)
    ft_c2.metric("Alvo", str(phase2_fine_tune.get("fine_tune_target") or "-"))
    ft_c3.metric("Aplicado no ciclo", str(int(phase2_fine_tune.get("fine_tune_applied_count", 0) or 0)))
    ft_c4.metric("Bloqueado por guard", str(int(phase2_fine_tune.get("fine_tune_blocked_count", 0) or 0)))
    st.caption(f"Motivo: {phase2_fine_tune.get('fine_tune_reason') or 'Sem motivo registrado.'}")
    st.caption(
        f"Antes: {phase2_fine_tune.get('fine_tune_before') or '-'} | "
        f"Depois: {phase2_fine_tune.get('fine_tune_after') or '-'}"
    )
    if phase2_fine_tune.get("fine_tune_last_guard_reason"):
        st.caption(f"Ultimo guard acionado: {phase2_fine_tune.get('fine_tune_last_guard_reason')}")
    st.caption("Observacao: ajuste PAPER only, reversivel e sem autoridade para ordem real.")
if phase2_1_fine_tune:
    st.markdown("#### Ajuste Fino FASE 2.1")
    st.caption(
        "Ajuste conservador para multiplas falhas pequenas de momentum/confirmacao secundaria. "
        "PAPER only, reversivel, sem ordem real e sem reducao global de score."
    )
    ft21_status = "Ativo" if bool(phase2_1_fine_tune.get("phase2_1_fine_tune_enabled", False)) else "Inativo"
    ft21_gap = phase2_1_fine_tune.get("phase2_1_fine_tune_score_gap")
    ft21_gap_label = "-" if ft21_gap is None else f"{float(ft21_gap):.4f}"
    ft21_c1, ft21_c2, ft21_c3, ft21_c4 = st.columns(4)
    ft21_c1.metric("Status", ft21_status)
    ft21_c2.metric("Alvo", str(phase2_1_fine_tune.get("phase2_1_fine_tune_target") or "-"))
    ft21_c3.metric("Aplicado no ciclo", str(int(phase2_1_fine_tune.get("phase2_1_fine_tune_applied_count", 0) or 0)))
    ft21_c4.metric("Bloqueado no ciclo", str(int(phase2_1_fine_tune.get("phase2_1_fine_tune_blocked_count", 0) or 0)))
    st.caption(
        f"Ultima decisao: {phase2_1_fine_tune.get('phase2_1_fine_tune_last_decision') or '-'} | "
        f"Ultimo guard: {phase2_1_fine_tune.get('phase2_1_fine_tune_last_guard') or '-'} | "
        f"Score gap: {ft21_gap_label}"
    )
    st.caption(
        "Motivos permitidos: "
        f"{symbol_list_label(phase2_1_fine_tune.get('phase2_1_fine_tune_allowed_reasons'))} | "
        "Motivos bloqueadores: "
        f"{symbol_list_label(phase2_1_fine_tune.get('phase2_1_fine_tune_blocked_reasons'))}"
    )
    st.caption(f"Motivo: {phase2_1_fine_tune.get('phase2_1_fine_tune_reason') or 'Sem motivo registrado.'}")
    st.caption("Observacao: nao altera min_signal_score global, broker, webhook, FASE 3 ou execucao real.")

perf_chart_left, perf_chart_right = st.columns(2)
with perf_chart_left:
    if not paper_equity_df.empty:
        equity_plot = paper_equity_df.copy()
        if "timestamp" in equity_plot.columns:
            equity_plot["timestamp"] = pd.to_datetime(equity_plot["timestamp"], errors="coerce")
        else:
            equity_plot["timestamp"] = pd.RangeIndex(start=1, stop=len(equity_plot) + 1)

        fig_runtime_equity = go.Figure()
        fig_runtime_equity.add_trace(
            go.Scatter(
                x=equity_plot["timestamp"],
                y=equity_plot["equity"],
                mode="lines",
                name="Equity",
                line=dict(width=2.2, color="#38bdf8"),
            )
        )
        fig_runtime_equity.update_layout(
            template="plotly_dark",
            height=280,
            margin=dict(l=10, r=10, t=25, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Tempo",
            yaxis_title="Equity",
            showlegend=False,
        )
        st.plotly_chart(fig_runtime_equity, use_container_width=True, key="perf_equity_curve")
    else:
        st.info("Curva de capital ainda nao disponivel.")

with perf_chart_right:
    if trade_reports_df.empty:
        st.info("Os relatorios vao aparecer aqui quando a primeira operacao for fechada.")
    else:
        st.plotly_chart(build_trade_pnl_chart(trade_reports_df), use_container_width=True, key="trade_pnl_chart")

perf_chart_bottom_left, perf_chart_bottom_right = st.columns(2)
with perf_chart_bottom_left:
    if profile_summary_df.empty:
        st.info("Ainda nao ha distribuicao por perfil para exibir.")
    else:
        st.plotly_chart(
            build_profile_distribution_chart(profile_summary_df),
            use_container_width=True,
            key="trade_profile_distribution",
        )

with perf_chart_bottom_right:
    if profile_summary_df.empty:
        st.info("Win rate por perfil sera exibido quando houver trades fechados.")
    else:
        st.plotly_chart(
            build_profile_win_rate_chart(profile_summary_df),
            use_container_width=True,
            key="trade_profile_win_rate",
        )

latest_reports_col, suggestions_col = st.columns([1.6, 1.0])
with latest_reports_col:
    st.markdown("#### Ultimos trades fechados")
    if trade_reports_df.empty:
        st.info("Sem trades fechados ainda.")
    else:
        latest_reports = trade_reports_df.sort_values("closed_at", ascending=False).head(10)
        st.dataframe(latest_reports, use_container_width=True)

with suggestions_col:
    st.markdown("#### Sugestoes analiticas")
    if not trade_suggestions:
        st.info("Ainda nao ha sugestoes suficientes. Continue operando para gerar amostra.")
    else:
        for suggestion in trade_suggestions:
            severity = str(suggestion.get("severity", "info"))
            message = str(suggestion.get("message", "") or "")
            if severity == "warning":
                st.warning(message)
            elif severity == "success":
                st.success(message)
            else:
                st.info(message)

with st.expander("Modo avançado"):
    st.markdown("### Configuração avançada")

    profile_options = list_trader_profiles()
    profile_names = [item["name"] for item in profile_options]
    current_profile = normalize_trader_profile(trader.get("profile"))

    a1, a2, a3, a4 = st.columns(4)
    with a1:
        selected_profile_name = st.selectbox(
            "Perfil operacional",
            options=profile_names,
            index=profile_names.index(current_profile) if current_profile in profile_names else 1,
        )
    with a2:
        ticket = st.number_input(
            "Valor por operação (R$)",
            min_value=MIN_TICKET,
            max_value=MAX_TICKET,
            value=float(trader["ticket_value"]),
            step=10.0,
        )
    with a3:
        holding = st.slider(
            "Tempo máximo da operação (min)",
            min_value=MIN_HOLDING_MINUTES,
            max_value=MAX_HOLDING_MINUTES,
            value=int(trader["holding_minutes"]),
            step=1,
        )
    with a4:
        max_open = st.slider(
            "Máx. operações abertas",
            min_value=1,
            max_value=20,
            value=int(trader["max_open_positions"]),
            step=1,
        )

    selected_profile_preview = get_trader_profile_config(
        selected_profile_name,
        base_ticket_value=float(ticket),
        base_holding_minutes=int(holding),
        base_max_open_positions=int(max_open),
    )
    swing_validation_active = (
        str(load_bot_state().get("validation", {}).get("validation_mode") or "").strip().lower() == SWING_VALIDATION_MODE
    )
    effective_profile_preview = (
        apply_swing_validation_overrides(selected_profile_name, selected_profile_preview)
        if swing_validation_active
        else selected_profile_preview
    )
    st.caption(selected_profile_preview["description"])
    if swing_validation_active:
        st.info(
            "Modo swing 10 dias ativo: a engine opera com timeframe diario e holding de dias, "
            "mesmo que o preview base abaixo continue mostrando a configuracao manual."
        )
        st.caption(
            "Configuracao inicial recomendada para este ciclo: "
            f"capital base {br_money(VALIDATION_INITIAL_CAPITAL_BRL)} | "
            f"entrada padrao {br_money(VALIDATION_DEFAULT_ENTRY_AMOUNT_BRL)} | "
            f"ate {int(VALIDATION_DEFAULT_MAX_OPEN_POSITIONS)} posicoes abertas | "
            f"modo {VALIDATION_MODE_DISPLAY}."
        )

    preview_c1, preview_c2, preview_c3, preview_c4 = st.columns(4)
    preview_c1.metric("Ticket efetivo", br_money(float(effective_profile_preview["ticket_value"])))
    preview_c2.metric("Holding efetivo", f"{int(effective_profile_preview['holding_minutes'])} min")
    preview_c3.metric("Posicoes efetivas", f"{int(effective_profile_preview['max_open_positions'])}")
    preview_c4.metric("Score minimo", f"{float(effective_profile_preview['min_signal_score']):.2f}")
    st.caption(
        "Cooldown de reentrada: "
        f"{int(effective_profile_preview['reentry_cooldown_minutes'])} min"
        " | Saidas suaves depois de: "
        f"{int(effective_profile_preview['min_position_age_minutes'])} min"
    )

    watchlist_text = st.text_input(
        "Lista de ativos",
        value=", ".join(trader.get("watchlist", [])),
        help="Exemplo: BTC-USD, ETH-USD, BNB-USD, SOL-USD, LINK-USD",
    )

    st.caption(
        "Watchlist padrao recomendada para esta fase: "
        f"{', '.join(recommended_watchlist)}"
    )
    st.caption(
        "Para iniciar um ciclo limpo com essa configuracao, salve os parametros e use "
        "'Resetar modulo trader'. O historico consolidado continua preservado."
    )
    st.markdown("**Logica da watchlist recomendada**")
    st.caption(
        "Selecao otimizada para swing de 10 dias, CRIPTO ONLY e PAPER TRADING. "
        "O objetivo aqui e validar qualidade de sinal com poucos ativos liquidos e legiveis."
    )
    st.dataframe(pd.DataFrame(SWING_VALIDATION_WATCHLIST_DETAILS), use_container_width=True)
    for note in SWING_VALIDATION_DISCOURAGED_ASSET_NOTES:
        st.caption(f"- {note}")

    ac1, ac2, ac3 = st.columns(3)

    with ac1:
        if st.button("Salvar configuração avançada", use_container_width=True, disabled=not admin_mode):
            state = load_bot_state()
            state["trader"]["profile"] = str(selected_profile_name)
            state["trader"]["ticket_value"] = float(ticket)
            state["trader"]["holding_minutes"] = int(holding)
            state["trader"]["max_open_positions"] = int(max_open)
            state["trader"]["watchlist"] = [x.strip().upper() for x in watchlist_text.split(",") if x.strip()]
            save_bot_state(state)
            st.success("Configuração salva.")
            st.rerun()

    with ac2:
        if st.button("Aplicar watchlist recomendada", use_container_width=True, disabled=not admin_mode):
            state = load_bot_state()
            state["trader"]["watchlist"] = list(recommended_watchlist)
            save_bot_state(state)
            st.success("Watchlist recomendada aplicada.")
            st.rerun()

    with ac3:
        if st.button("Resetar módulo trader", use_container_width=True, disabled=not admin_mode):
            try:
                with st.spinner("Resetando trader..."):
                    reset_trader_module()
                st.warning("Trader resetado.")
                st.rerun()
            except Exception as e:
                st.error(f"Erro ao resetar trader: {e}")

    if not admin_mode:
        st.info("A configuração avançada está em modo somente leitura para usuários sem permissão de admin.")

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📊 Operações em andamento",
            "📄 Histórico de ordens",
            "⚡ Últimos trades",
            "📈 Métricas",
        ]
    )

    with tab1:
        if positions:
            st.dataframe(pd.DataFrame(positions), use_container_width=True)
        else:
            st.info("Sem operações abertas no trader.")

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
        k3.metric("Execuções", f"{paper_state.get('run_count', 0)}")
        k4.metric("Trades", f"{paper_report.get('trades_count', 0)}")

        pnl_m1, pnl_m2, pnl_m3 = st.columns(3)
        pnl_m1.metric("PnL Acumulado", br_money(float(pnl_summary.get("cumulative_pnl", 0.0) or 0.0)))
        pnl_m2.metric(
            "Última Operação",
            br_money(float(pnl_summary["last_trade_pnl"])) if pnl_summary.get("last_trade_pnl") is not None else "Sem trade fechado",
        )
        pnl_m3.metric("PnL em Aberto", br_money(float(pnl_summary.get("open_pnl", 0.0) or 0.0)))

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
            st.plotly_chart(fig_equity, use_container_width=True, key="main_equity_curve")
        else:
            st.info("Curva de equity ainda nao disponivel.")

if admin_mode:
    st.markdown("---")
    with st.expander("Administração", expanded=False):
        st.markdown("### Painel administrativo")
        admin_tab_users, admin_tab_real = st.tabs(["Usuários", "Real Mode"])

        with admin_tab_users:
            with st.form("create_user_form"):
                new_username = st.text_input("Novo usuário")
                new_password = st.text_input("Senha do novo usuário", type="password")
                new_role = st.selectbox("Perfil", ["user", "admin"])
                create_submit = st.form_submit_button("Criar usuário", use_container_width=True)

            if create_submit:
                try:
                    create_user(new_username, new_password, new_role)
                    st.success("Usuário criado com sucesso.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Não foi possível criar o usuário: {exc}")

            users = list_users()
            if users:
                users_df = pd.DataFrame(users)
                st.dataframe(users_df, use_container_width=True)

                selected_username = st.selectbox(
                    "Selecionar usuário",
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
                                st.error(f"Não foi possível atualizar o perfil: {exc}")

                    with disable_col:
                        target_disabled = bool(selected_user.get("disabled", False))
                        action_label = "Reabilitar usuário" if target_disabled else "Desabilitar usuário"
                        if st.button(action_label, use_container_width=True, key="admin_toggle_user"):
                            try:
                                set_user_disabled(selected_username, not target_disabled)
                                st.success("Status do usuário atualizado.")
                                st.rerun()
                            except Exception as exc:
                                st.error(f"Não foi possível atualizar o usuário: {exc}")
            else:
                st.info("Nenhum usuário encontrado.")

        with admin_tab_real:
            latest_state = load_bot_state()
            latest_security = latest_state.get("security", {}) or {}
            real_mode_enabled = bool(latest_security.get("real_mode_enabled", False))

            if real_mode_enabled:
                st.error("Real mode está habilitado.")
            else:
                st.info("Real mode está desligado.")

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
                        raise ValueError("Faça login com uma conta admin persistida para alterar o real mode.")
                    if stored_user.get("disabled", False):
                        raise ValueError("A conta admin está desabilitada.")
                    if not verify_password(confirm_password, str(stored_user.get("password_hash", ""))):
                        raise ValueError("Confirmação de senha inválida.")

                    latest_state = load_bot_state()
                    latest_state.setdefault("security", {})
                    latest_state["security"]["real_mode_enabled"] = not real_mode_enabled
                    latest_state["security"]["real_mode_enabled_by"] = current_user["username"]
                    latest_state["security"]["real_mode_enabled_at"] = datetime.now(timezone.utc).isoformat()
                    save_bot_state(latest_state)
                    st.success("Real mode atualizado com sucesso.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Não foi possível alterar o real mode: {exc}")
