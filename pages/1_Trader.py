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
strategy_structure_audit = dict(
    validation_last_report.get("strategy_structure_audit")
    or current_audit_state.get("strategy_structure_audit", {})
    or state.get("strategy_structure_audit", {})
    or {}
)
market_structure_audit = dict(
    validation_last_report.get("market_structure_audit")
    or current_audit_state.get("market_structure_audit", {})
    or state.get("market_structure_audit", {})
    or {}
)
fib_alignment_audit = dict(
    validation_last_report.get("fib_alignment_audit")
    or current_audit_state.get("fib_alignment_audit", {})
    or state.get("fib_alignment_audit", {})
    or {}
)
multi_timeframe_intraday_fetcher = dict(
    validation_last_report.get("multi_timeframe_intraday_fetcher")
    or current_audit_state.get("multi_timeframe_intraday_fetcher", {})
    or state.get("multi_timeframe_intraday_fetcher", {})
    or {}
)
multi_timeframe_swing_audit = dict(
    validation_last_report.get("multi_timeframe_swing_audit")
    or current_audit_state.get("multi_timeframe_swing_audit", {})
    or state.get("multi_timeframe_swing_audit", {})
    or {}
)
bos_pivot_trace_audit = dict(
    validation_last_report.get("bos_pivot_trace_audit")
    or current_audit_state.get("bos_pivot_trace_audit", {})
    or state.get("bos_pivot_trace_audit", {})
    or {}
)
strategy_decision_bridge_trace = dict(
    validation_last_report.get("strategy_decision_bridge_trace")
    or current_audit_state.get("strategy_decision_bridge_trace", {})
    or state.get("strategy_decision_bridge_trace", {})
    or {}
)
feed_scope_reconciliation = dict(
    validation_last_report.get("feed_scope_reconciliation")
    or current_audit_state.get("feed_scope_reconciliation", {})
    or state.get("feed_scope_reconciliation", {})
    or {}
)
no_setup_eligible_decomposition = dict(
    validation_last_report.get("no_setup_eligible_decomposition")
    or current_audit_state.get("no_setup_eligible_decomposition", {})
    or state.get("no_setup_eligible_decomposition", {})
    or {}
)
reversal_blocker_routing_audit = dict(
    validation_last_report.get("reversal_blocker_routing_audit")
    or current_audit_state.get("reversal_blocker_routing_audit", {})
    or state.get("reversal_blocker_routing_audit", {})
    or {}
)
setup_blocker_taxonomy_audit = dict(
    validation_last_report.get("setup_blocker_taxonomy_audit")
    or current_audit_state.get("setup_blocker_taxonomy_audit", {})
    or state.get("setup_blocker_taxonomy_audit", {})
    or {}
)
bos_confirmation_quality_audit = dict(
    validation_last_report.get("bos_confirmation_quality_audit")
    or current_audit_state.get("bos_confirmation_quality_audit", {})
    or state.get("bos_confirmation_quality_audit", {})
    or {}
)
h1_confirmation_after_h4_bos_audit = dict(
    validation_last_report.get("h1_confirmation_after_h4_bos_audit")
    or current_audit_state.get("h1_confirmation_after_h4_bos_audit", {})
    or state.get("h1_confirmation_after_h4_bos_audit", {})
    or {}
)
shadow_decision_simulator = dict(
    validation_last_report.get("shadow_decision_simulator")
    or current_audit_state.get("shadow_decision_simulator", {})
    or state.get("shadow_decision_simulator", {})
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
if strategy_structure_audit:
    st.markdown("#### AUDITORIA ESTRUTURAL DA ESTRATEGIA")
    st.caption(
        "SHADOW ONLY: compara setups internos sem aprovar trades, sem reduzir thresholds, "
        "sem alterar score real, broker ou execucao. PAPER TRADING permanece obrigatorio."
    )
    shadow_score = strategy_structure_audit.get("structural_audit_top_score")
    shadow_gap = strategy_structure_audit.get("structural_audit_top_gap")
    shadow_score_label = "-" if shadow_score is None else f"{float(shadow_score):.2f}"
    shadow_gap_label = "-" if shadow_gap is None else f"{float(shadow_gap):.4f}"
    shadow_c1, shadow_c2, shadow_c3, shadow_c4, shadow_c5 = st.columns(5)
    shadow_c1.metric("Setup shadow", strategy_structure_audit.get("structural_audit_top_setup") or "-")
    shadow_c2.metric("Ativo shadow", strategy_structure_audit.get("structural_audit_top_symbol") or "-")
    shadow_c3.metric("Melhor score", shadow_score_label)
    shadow_c4.metric("Gap ate aprovacao", shadow_gap_label)
    shadow_c5.metric("Candidatos shadow", str(int(strategy_structure_audit.get("structural_audit_candidates", 0) or 0)))
    st.caption(
        f"Bloqueador primario: {strategy_structure_audit.get('structural_audit_primary_blocker') or '-'} | "
        f"Bloqueador secundario: {strategy_structure_audit.get('structural_audit_secondary_blocker') or '-'} | "
        f"Recomendacao: {strategy_structure_audit.get('structural_audit_recommendation') or 'sem dados suficientes'}"
    )
    st.caption(f"Temporalidade: {strategy_structure_audit.get('structural_audit_timeframe_note') or 'Inconclusivo.'}")
    st.caption(f"RSI/momentum: {strategy_structure_audit.get('structural_audit_rsi_momentum_note') or 'Inconclusivo.'}")
    st.caption(f"Reversal: {strategy_structure_audit.get('structural_audit_reversal_note') or 'Inconclusivo.'}")
    st.caption(strategy_structure_audit.get("structural_audit_reason") or "Sem auditoria estrutural consolidada ainda.")
    setup_rows = list(strategy_structure_audit.get("structural_audit_setup_comparison", []) or [])[:5]
    if setup_rows:
        display_rows = [
            {
                "setup": row.get("setup"),
                "candidatos shadow": row.get("shadow_candidates"),
                "melhor ativo": row.get("best_symbol"),
                "melhor score": row.get("best_score"),
                "gap medio": row.get("average_gap"),
                "bloqueador dominante": row.get("dominant_blocker"),
                "recomendacao": row.get("recommendation"),
            }
            for row in setup_rows
            if isinstance(row, dict)
        ]
        if display_rows:
            st.dataframe(pd.DataFrame(display_rows), hide_index=True, use_container_width=True)
if market_structure_audit:
    st.markdown("#### AUDITORIA FIBONACCI + ESTRUTURA DE MERCADO")
    st.caption(
        "SHADOW ONLY: Fibonacci, price action, pivos e BOS sao apenas auditoria estrutural. "
        "Nao aprovam trade, nao alteram score real, nao mudam broker e preservam PAPER TRADING."
    )
    top_score = market_structure_audit.get("market_structure_top_score")
    score_label = "-" if top_score is None else f"{float(top_score):.2f}"
    best_candidates = list(market_structure_audit.get("market_structure_best_candidates", []) or [])
    top_candidate = dict(best_candidates[0] or {}) if best_candidates else {}
    ms_c1, ms_c2, ms_c3, ms_c4, ms_c5 = st.columns(5)
    ms_c1.metric("Melhor ativo", market_structure_audit.get("market_structure_top_symbol") or "-")
    ms_c2.metric("Direcao", top_candidate.get("structure_direction") or "-")
    ms_c3.metric("Zona Fibonacci", market_structure_audit.get("market_structure_top_zone") or "-")
    ms_c4.metric("Score estrutural", score_label)
    ms_c5.metric("Candidato shadow", "Sim" if bool(top_candidate.get("market_structure_shadow_candidate", False)) else "Nao")
    ms_d1, ms_d2, ms_d3, ms_d4 = st.columns(4)
    ms_d1.metric("Pivo", "Sim" if bool(top_candidate.get("pivot_detected", False)) else "Nao")
    ms_d2.metric("BOS", "Sim" if bool(top_candidate.get("bos_detected", False)) else "Nao")
    ms_d3.metric("Falso rompimento", "Sim" if bool(top_candidate.get("false_breakout_risk", False)) else "Nao")
    ms_d4.metric("Regime", top_candidate.get("market_regime") or "INCONCLUSIVE")
    st.caption(
        f"Confluencia trend_pullback_breakout: "
        f"{'Sim' if bool(top_candidate.get('structure_confirms_trend_pullback', False)) else 'Nao'} | "
        f"Recomendacao: {market_structure_audit.get('market_structure_top_recommendation') or 'sem dados suficientes'}"
    )
    confluence = dict(market_structure_audit.get("market_structure_setup_confluence", {}) or {})
    st.caption(
        "Confluencia por setup: "
        f"trend_pullback={int(confluence.get('trend_pullback', 0) or 0)} | "
        f"breakout={int(confluence.get('breakout', 0) or 0)} | "
        f"reversal={int(confluence.get('reversal', 0) or 0)} | "
        f"melhoraria qualidade={int(confluence.get('would_improve_quality', 0) or 0)}"
    )
    st.caption(
        f"Suficiencia: {market_structure_audit.get('market_structure_data_sufficiency') or 'NO_DATA'} | "
        f"Amostra minima: {'Sim' if bool(market_structure_audit.get('market_structure_minimum_sample_met', False)) else 'Nao'} | "
        f"Por que nao houve candidato: {market_structure_audit.get('market_structure_why_no_candidate') or '-'}"
    )
    if best_candidates:
        display_rows = [
            {
                "asset": row.get("symbol"),
                "structure_score": row.get("market_structure_score"),
                "fib_zone": row.get("current_fib_zone"),
                "bos": row.get("bos_detected"),
                "pivot": row.get("pivot_detected"),
                "false_breakout_risk": row.get("false_breakout_risk"),
                "confluence": ", ".join(list(row.get("confluence_notes", []) or [])[:2]),
                "recommendation": row.get("structure_recommendation"),
            }
            for row in best_candidates[:5]
            if isinstance(row, dict)
        ]
        if display_rows:
            st.dataframe(pd.DataFrame(display_rows), hide_index=True, use_container_width=True)
if fib_alignment_audit:
    st.markdown("#### ADERENCIA AO VIDEO/PDF FIBONACCI")
    st.caption(
        "SHADOW ONLY: esta camada mede aderencia objetiva a um checklist inspirado no video/PDF. "
        "Nao afirma equivalencia da estrategia, nao aprova trade e nao altera score real ou broker."
    )
    alignment_score = fib_alignment_audit.get("fib_alignment_score")
    alignment_score_label = "-" if alignment_score is None else f"{float(alignment_score):.2f}"
    fa_c1, fa_c2, fa_c3, fa_c4 = st.columns(4)
    fa_c1.metric("Ativo analisado", fib_alignment_audit.get("fib_alignment_top_symbol") or "-")
    fa_c2.metric("Score aderencia", alignment_score_label)
    fa_c3.metric("Status", fib_alignment_audit.get("fib_alignment_status") or "insufficient_data")
    fa_c4.metric("Modo", fib_alignment_audit.get("fib_alignment_mode") or "SHADOW_ONLY")
    fa_d1, fa_d2, fa_d3, fa_d4, fa_d5 = st.columns(5)
    fa_d1.metric("Ancora fundo", fib_alignment_audit.get("fib_alignment_anchor_low_status") or "insufficient")
    fa_d2.metric("Ancora topo", fib_alignment_audit.get("fib_alignment_anchor_high_status") or "insufficient")
    fa_d3.metric("Zona Fibonacci", fib_alignment_audit.get("fib_alignment_zone_status") or "insufficient")
    fa_d4.metric("Pivo", fib_alignment_audit.get("fib_alignment_pivot_status") or "insufficient")
    fa_d5.metric("BOS", fib_alignment_audit.get("fib_alignment_bos_status") or "insufficient")
    st.caption(
        f"Confirmacao de entrada: {fib_alignment_audit.get('fib_alignment_entry_confirmation_status') or 'insufficient'} | "
        f"Confluencia setup: {fib_alignment_audit.get('fib_alignment_confluence_status') or 'insufficient'} | "
        f"Principal divergencia: {fib_alignment_audit.get('fib_alignment_why_differs') or '-'} | "
        f"Recomendacao: {fib_alignment_audit.get('fib_alignment_recommendation') or 'keep_shadow_only'}"
    )
    checklist_rows = [
        {
            "regra": row.get("item"),
            "esperado": row.get("esperado_pelo_video_pdf"),
            "detectado": row.get("detectado_pelo_app"),
            "status": row.get("status"),
            "motivo": row.get("motivo"),
        }
        for row in list(fib_alignment_audit.get("fib_alignment_checklist", []) or [])
        if isinstance(row, dict)
    ]
    if checklist_rows:
        st.dataframe(pd.DataFrame(checklist_rows[:8]), hide_index=True, use_container_width=True)
if multi_timeframe_swing_audit:
    st.markdown("#### FASE 2.5 - DIAGNOSTICO MULTI-TIMEFRAME SWING")
    st.caption(
        "SHADOW ONLY: 1D, 4H e 1H sao usados apenas para diagnosticar estrutura swing. "
        "Nao aprova trade, nao altera score real, nao muda broker e preserva PAPER TRADING."
    )
    top_alignment_score = multi_timeframe_swing_audit.get("top_alignment_score")
    top_alignment_score_label = "-" if top_alignment_score is None else f"{float(top_alignment_score):.2f}"
    mtf_candidates = [
        row for row in list(multi_timeframe_swing_audit.get("recent_candidates", []) or []) if isinstance(row, dict)
    ]
    top_mtf = dict(mtf_candidates[0] if mtf_candidates else {})
    pivot_timeframes = ", ".join(list(top_mtf.get("pivot_confirmed_timeframes", top_mtf.get("pivot_timeframes", [])) or [])) or "-"
    bos_timeframes = ", ".join(list(top_mtf.get("bos_confirmed_timeframes", top_mtf.get("bos_timeframes", [])) or [])) or "-"
    mtf_c1, mtf_c2, mtf_c3, mtf_c4 = st.columns(4)
    mtf_c1.metric("Status", multi_timeframe_swing_audit.get("mode") or "SHADOW_ONLY")
    mtf_c2.metric("Top ativo", multi_timeframe_swing_audit.get("top_symbol") or "-")
    mtf_c3.metric("Score alinhamento", top_alignment_score_label)
    mtf_c4.metric("Status alinhamento", multi_timeframe_swing_audit.get("top_alignment_status") or "INSUFFICIENT_DATA")
    mtf_d1, mtf_d2, mtf_d3, mtf_d4 = st.columns(4)
    mtf_d1.metric("Direcao 1D", top_mtf.get("daily_bias") or "INCONCLUSIVE")
    mtf_d2.metric("Estrutura 4H", top_mtf.get("h4_structure") or "INCONCLUSIVE")
    mtf_d3.metric("Confirmacao 1H", top_mtf.get("h1_confirmation") or "INCONCLUSIVE")
    mtf_d4.metric("Pivo detectado", pivot_timeframes)
    mtf_e1, mtf_e2, mtf_e3, mtf_e4 = st.columns(4)
    mtf_e1.metric("BOS detectado", bos_timeframes)
    mtf_e2.metric("Conflito dominante", multi_timeframe_swing_audit.get("dominant_conflict_reason") or "-")
    mtf_e3.metric("Recomendacao", multi_timeframe_swing_audit.get("top_recommendation") or "observe_more")
    mtf_e4.metric("Feed usado", multi_timeframe_swing_audit.get("feed_status") or "UNKNOWN")
    st.caption(
        f"Provider: {multi_timeframe_swing_audit.get('provider_effective') or '-'} | "
        f"Cache/TTL: {multi_timeframe_swing_audit.get('cache_status') or 'cycle_data_resample_only'} / "
        f"{int(multi_timeframe_swing_audit.get('cache_ttl_seconds', 0) or 0)}s | "
        f"Chamadas extras estimadas: {int(multi_timeframe_swing_audit.get('estimated_provider_calls', 0) or 0)} | "
        f"Guard provider: {multi_timeframe_swing_audit.get('provider_guard') or '-'}"
    )
    st.markdown("##### FASE 2.5A - DADOS INTRADAY 4H/1H")
    st.caption(
        "SHADOW ONLY: dados 4H/1H sao usados apenas para diagnosticar estrutura swing. "
        "Nao aprovam trade, nao alteram score real, nao mudam broker e preservam PAPER TRADING."
    )
    intraday_diags = [
        row for row in list(multi_timeframe_intraday_fetcher.get("diagnostics", []) or []) if isinstance(row, dict)
    ]
    fetch_c1, fetch_c2, fetch_c3, fetch_c4 = st.columns(4)
    fetch_c1.metric("Intraday fetch status", multi_timeframe_intraday_fetcher.get("intraday_data_quality") or "NO_DATA")
    fetch_c2.metric("Usa dados reais 4H/1H", "Sim" if bool(multi_timeframe_swing_audit.get("uses_real_intraday_data", False)) else "Nao")
    fetch_c3.metric("Timeframes disponiveis", ", ".join(list(multi_timeframe_intraday_fetcher.get("timeframes_available", []) or [])) or "-")
    fetch_c4.metric("Simbolos buscados", len(list(multi_timeframe_intraday_fetcher.get("symbols_requested", []) or [])))
    fetch_d1, fetch_d2, fetch_d3, fetch_d4 = st.columns(4)
    fetch_d1.metric("Cache hits", int(multi_timeframe_intraday_fetcher.get("cache_hits", 0) or 0))
    fetch_d2.metric("Cache misses", int(multi_timeframe_intraday_fetcher.get("cache_misses", 0) or 0))
    fetch_d3.metric("Chamadas provider", int(multi_timeframe_intraday_fetcher.get("provider_calls_attempted", 0) or 0))
    fetch_d4.metric("Puladas por guard", int(multi_timeframe_intraday_fetcher.get("provider_calls_skipped", 0) or 0))
    fetch_e1, fetch_e2, fetch_e3, fetch_e4 = st.columns(4)
    fetch_e1.metric("Qualidade 4H", multi_timeframe_swing_audit.get("h4_data_quality") or "missing")
    fetch_e2.metric("Qualidade 1H", multi_timeframe_swing_audit.get("h1_data_quality") or "missing")
    fetch_e3.metric("Motivo insuficiente", multi_timeframe_swing_audit.get("intraday_missing_reason") or multi_timeframe_intraday_fetcher.get("provider_guard_reason") or "-")
    fetch_e4.metric("Recomendacao", multi_timeframe_intraday_fetcher.get("intraday_fetch_recommendation") or "observe_more")
    mtf_diag_by_symbol = {
        row.get("symbol"): dict(row.get("timeframe_diagnostics", {}) or {})
        for row in mtf_candidates
        if isinstance(row, dict)
    }
    fetch_rows = [
        {
            "symbol": row.get("symbol"),
            "timeframe": row.get("timeframe"),
            "candles_available": row.get("candles_available"),
            "data_quality": row.get("data_quality"),
            "cache_status": row.get("cache_status"),
            "provider_call_attempted": row.get("provider_call_attempted"),
            "trend_direction": ((mtf_diag_by_symbol.get(row.get("symbol"), {}) or {}).get(row.get("timeframe"), {}) or {}).get("trend_direction"),
            "pivot_confirmed": ((mtf_diag_by_symbol.get(row.get("symbol"), {}) or {}).get(row.get("timeframe"), {}) or {}).get("pivot_confirmed"),
            "bos_confirmed": ((mtf_diag_by_symbol.get(row.get("symbol"), {}) or {}).get(row.get("timeframe"), {}) or {}).get("bos_confirmed"),
            "why_not_confirmed": row.get("quality_reason"),
        }
        for row in intraday_diags[:10]
    ]
    if fetch_rows:
        st.dataframe(pd.DataFrame(fetch_rows), hide_index=True, use_container_width=True)
    else:
        st.info("Sem diagnostico intraday 4H/1H registrado ainda.")
    if bos_pivot_trace_audit:
        st.markdown("##### FASE 2.5B - AUDITORIA FINA BOS/PIVO 4H/1H")
        st.caption(
            "SHADOW ONLY: BOS e pivos 4H/1H sao apenas diagnostico estrutural. "
            "Nao aprovam trade, nao alteram score real, nao mudam broker e preservam PAPER TRADING."
        )
        bos_c1, bos_c2, bos_c3, bos_c4 = st.columns(4)
        bos_c1.metric("Status", bos_pivot_trace_audit.get("mode") or "SHADOW_ONLY")
        bos_c2.metric("Top ativo", bos_pivot_trace_audit.get("top_symbol") or "-")
        bos_c3.metric("Top timeframe", bos_pivot_trace_audit.get("top_timeframe") or "-")
        bos_c4.metric("Relacao 1H/4H", bos_pivot_trace_audit.get("top_relationship") or "INSUFFICIENT_DATA")
        bos_d1, bos_d2, bos_d3, bos_d4 = st.columns(4)
        bos_d1.metric("Estado do pivo", bos_pivot_trace_audit.get("top_pivot_state") or "INSUFFICIENT_DATA")
        bos_d2.metric("Estado do BOS", bos_pivot_trace_audit.get("top_bos_state") or "INSUFFICIENT_DATA")
        bos_d3.metric("Faltando confirmar", bos_pivot_trace_audit.get("dominant_missing_piece") or bos_pivot_trace_audit.get("top_primary_missing_piece") or "-")
        bos_d4.metric("Should keep blocked", int(bos_pivot_trace_audit.get("should_keep_blocked_count", 0) or 0))
        bos_e1, bos_e2, bos_e3, bos_e4 = st.columns(4)
        bos_e1.metric("BOS por pavio", int(bos_pivot_trace_audit.get("wick_only_bos_count", 0) or 0))
        bos_e2.metric("Fechamento fraco", int(bos_pivot_trace_audit.get("weak_close_bos_count", 0) or 0))
        bos_e3.metric("BOS confirmado", int(bos_pivot_trace_audit.get("confirmed_bos_count", 0) or 0))
        bos_e4.metric("Reteste pendente", int(bos_pivot_trace_audit.get("retest_pending_count", 0) or 0))
        st.caption(
            f"Risco de falso rompimento e recomendacao sao diagnosticos: "
            f"{bos_pivot_trace_audit.get('top_recommendation') or 'observe_more'} | "
            f"h4_bos_missing={int(bos_pivot_trace_audit.get('h4_bos_missing_count', 0) or 0)} | "
            f"h1_bos_only={int(bos_pivot_trace_audit.get('h1_bos_only_count', 0) or 0)}"
        )
        bos_rows = [
            {
                "symbol": row.get("symbol"),
                "timeframe": row.get("timeframe"),
                "trend_direction": row.get("trend_direction"),
                "pivot_state": row.get("pivot_state"),
                "bos_state": row.get("bos_state"),
                "relationship_to_higher_tf": row.get("relationship_to_higher_tf"),
                "bos_level": row.get("bos_level"),
                "last_close": row.get("last_close"),
                "close_distance_to_bos_pct": row.get("close_distance_to_bos_pct"),
                "wick_crossed_level": row.get("wick_crossed_level"),
                "close_confirmed_level": row.get("close_confirmed_level"),
                "retest_detected": row.get("retest_detected"),
                "false_breakout_risk": row.get("false_breakout_risk"),
                "why_pivot_not_confirmed": row.get("why_pivot_not_confirmed"),
                "why_bos_not_confirmed": row.get("why_bos_not_confirmed"),
                "supports_trend_pullback_breakout": row.get("supports_trend_pullback_breakout"),
                "should_keep_blocked": row.get("should_keep_blocked"),
                "recommendation": row.get("recommendation"),
            }
            for row in list(bos_pivot_trace_audit.get("recent_candidates", []) or [])[:10]
            if isinstance(row, dict)
        ]
        if bos_rows:
            st.dataframe(pd.DataFrame(bos_rows), hide_index=True, use_container_width=True)
        else:
            st.info("Sem candidatos BOS/Pivo 4H/1H registrados ainda.")
    if strategy_decision_bridge_trace:
        st.markdown("##### FASE 2.5B.1 - PONTE ENTRE ESTRUTURA E DECISAO REAL")
        st.caption(
            "SHADOW ONLY: esta camada explica por que uma estrutura confirmada no diagnostico ainda nao virou "
            "entrada real/paper oficial. Nao aprova trade, nao altera score, nao muda broker, nao muda "
            "thresholds e preserva PAPER TRADING."
        )
        bridge_c1, bridge_c2, bridge_c3, bridge_c4, bridge_c5 = st.columns(5)
        bridge_c1.metric("Status", strategy_decision_bridge_trace.get("mode") or "SHADOW_ONLY")
        bridge_c2.metric("Top ativo", strategy_decision_bridge_trace.get("top_symbol") or "-")
        bridge_c3.metric("Status da ponte", strategy_decision_bridge_trace.get("top_bridge_status") or "INSUFFICIENT_TRACE_DATA")
        bridge_c4.metric("Bloqueador real", strategy_decision_bridge_trace.get("top_real_blocker") or "-")
        bridge_c5.metric("Estrutura shadow", strategy_decision_bridge_trace.get("top_structure_status") or "-")
        bridge_d1, bridge_d2, bridge_d3, bridge_d4, bridge_d5 = st.columns(5)
        bridge_d1.metric("Reconciliacao", strategy_decision_bridge_trace.get("top_reconciliation_status") or "UNKNOWN_MISMATCH")
        bridge_d2.metric("Fallback mismatch", int(strategy_decision_bridge_trace.get("fallback_scope_mismatch_count", 0) or 0))
        bridge_d3.metric("Multi-TF vs BOS/Pivo", int(strategy_decision_bridge_trace.get("multi_tf_vs_bos_mismatch_count", 0) or 0))
        bridge_d4.metric("Deve manter bloqueado", int(strategy_decision_bridge_trace.get("should_keep_blocked_count", 0) or 0))
        bridge_d5.metric("Recomendacao", strategy_decision_bridge_trace.get("recommendation") or "observe_more")
        bridge_rows = [
            {
                "symbol": row.get("symbol"),
                "real_score": row.get("real_score"),
                "min_score": row.get("min_score"),
                "score_gap": row.get("score_gap"),
                "real_rejection_reason": row.get("real_rejection_reason"),
                "primary_real_blocker": row.get("primary_real_blocker"),
                "secondary_real_blocker": row.get("secondary_real_blocker"),
                "multi_tf_alignment_status": row.get("multi_tf_alignment_status"),
                "bos_state_4h": row.get("bos_state_4h"),
                "bos_state_1h": row.get("bos_state_1h"),
                "pivot_state_4h": row.get("pivot_state_4h"),
                "pivot_state_1h": row.get("pivot_state_1h"),
                "relationship_1h_4h": row.get("relationship_1h_4h"),
                "fallback_blocker_scope": row.get("fallback_blocker_scope"),
                "reconciliation_status": row.get("reconciliation_status"),
                "final_bridge_reason": row.get("final_bridge_reason"),
                "should_keep_blocked": row.get("should_keep_blocked"),
                "recommendation": row.get("recommendation"),
            }
            for row in list(strategy_decision_bridge_trace.get("recent_candidates", []) or [])[:10]
            if isinstance(row, dict)
        ]
        if bridge_rows:
            st.dataframe(pd.DataFrame(bridge_rows), hide_index=True, use_container_width=True)
        else:
            st.info("Sem candidatos de ponte entre estrutura e decisao real ainda.")
    if feed_scope_reconciliation:
        st.markdown("##### FASE 2.5B.1A - RECONCILIACAO DE ESCOPO DO FEED/FALLBACK")
        st.caption(
            "DIAGNOSTIC ONLY: esta camada separa fallback atual, acumulado e historico. "
            "Nao aprova trade, nao altera score, nao muda broker, nao muda thresholds e preserva PAPER TRADING."
        )
        fs_c1, fs_c2, fs_c3, fs_c4 = st.columns(4)
        fs_c1.metric("Feed atual", feed_scope_reconciliation.get("current_feed_status") or "UNKNOWN")
        fs_c2.metric("Fallback atual", int(feed_scope_reconciliation.get("current_fallback_count", 0) or 0))
        fs_c3.metric("Fallback acumulado", int(feed_scope_reconciliation.get("accumulated_fallback_count", 0) or 0))
        fs_c4.metric("Escopo do fallback", feed_scope_reconciliation.get("fallback_blocker_scope") or "UNKNOWN")
        fs_d1, fs_d2, fs_d3, fs_d4 = st.columns(4)
        fs_d1.metric("Feed atual limpo?", "Sim" if bool(feed_scope_reconciliation.get("current_feed_is_clean", False)) else "Nao")
        fs_d2.metric("Rejeicao atual dominante", feed_scope_reconciliation.get("dominant_rejection_current") or "-")
        fs_d3.metric("Rejeicao acumulada", feed_scope_reconciliation.get("dominant_rejection_accumulated") or "-")
        fs_d4.metric("Recomendacao", feed_scope_reconciliation.get("recommendation") or "observe_more")
        if bool(feed_scope_reconciliation.get("current_feed_is_clean", False)) and int(feed_scope_reconciliation.get("accumulated_fallback_count", 0) or 0) > 0:
            st.info("O fallback exibido e acumulado/historico, nao do ciclo atual.")
        st.caption(feed_scope_reconciliation.get("notes") or "Sem nota de reconciliacao de feed.")
    if no_setup_eligible_decomposition:
        st.markdown("##### FASE 2.5B.2 - DECOMPOSICAO NO_SETUP_ELIGIBLE")
        st.caption(
            "DIAGNOSTIC ONLY: esta camada explica por que NO_SETUP_ELIGIBLE bloqueou candidatos. "
            "Nao aprova trade, nao altera score, nao muda broker, nao muda thresholds e preserva PAPER TRADING."
        )
        ns_c1, ns_c2, ns_c3, ns_c4, ns_c5 = st.columns(5)
        ns_c1.metric("Top ativo", no_setup_eligible_decomposition.get("top_symbol") or "-")
        ns_c2.metric("Setup analisado", no_setup_eligible_decomposition.get("top_setup") or "trend_pullback_breakout")
        ns_c3.metric("Bloqueador real", no_setup_eligible_decomposition.get("top_real_blocker") or "-")
        ns_c4.metric("Bucket principal", no_setup_eligible_decomposition.get("top_reason_bucket") or "INSUFFICIENT_DATA")
        ns_c5.metric("Deve manter bloqueado", "Sim" if bool(no_setup_eligible_decomposition.get("should_keep_blocked", True)) else "Nao")
        ns_d1, ns_d2, ns_d3, ns_d4, ns_d5 = st.columns(5)
        ns_d1.metric("Score", no_setup_eligible_decomposition.get("top_score") if no_setup_eligible_decomposition.get("top_score") is not None else "-")
        ns_d2.metric("Min score", no_setup_eligible_decomposition.get("top_min_score") if no_setup_eligible_decomposition.get("top_min_score") is not None else "-")
        ns_d3.metric("Gap", no_setup_eligible_decomposition.get("top_score_gap") if no_setup_eligible_decomposition.get("top_score_gap") is not None else "-")
        ns_d4.metric("Estrutura confirmada", int(no_setup_eligible_decomposition.get("structure_confirmed_count", 0) or 0))
        ns_d5.metric("Feed atual limpo?", "Sim" if bool(no_setup_eligible_decomposition.get("current_feed_is_clean", False)) else "Nao")
        ns_e1, ns_e2, ns_e3 = st.columns(3)
        ns_e1.metric("No setup count", int(no_setup_eligible_decomposition.get("no_setup_eligible_count", 0) or 0))
        ns_e2.metric("Near no setup", int(no_setup_eligible_decomposition.get("near_approved_no_setup_count", 0) or 0))
        ns_e3.metric("Recomendacao", no_setup_eligible_decomposition.get("recommendation") or "insufficient_data")
        no_setup_rows = [
            {
                "symbol": row.get("symbol"),
                "setup": row.get("setup"),
                "score": row.get("score"),
                "gap": row.get("score_gap"),
                "primary_real_blocker": row.get("primary_real_blocker"),
                "secondary_real_blocker": row.get("secondary_real_blocker"),
                "reason_bucket": row.get("reason_bucket"),
                "multi_tf_alignment_status": row.get("multi_tf_alignment_status"),
                "bos_state": row.get("bos_state"),
                "pivot_state": row.get("pivot_state"),
                "suggested_future_study": row.get("suggested_future_study"),
                "should_keep_blocked": row.get("should_keep_blocked"),
            }
            for row in list(no_setup_eligible_decomposition.get("candidates", []) or [])[:10]
            if isinstance(row, dict)
        ]
        if no_setup_rows:
            st.dataframe(pd.DataFrame(no_setup_rows), hide_index=True, use_container_width=True)
        else:
            st.info("Sem amostra NO_SETUP_ELIGIBLE suficiente para decomposicao.")
        st.caption(no_setup_eligible_decomposition.get("notes") or "Diagnostico sem autoridade operacional.")
    if reversal_blocker_routing_audit:
        st.markdown("##### FASE 2.5B.2A - AUDITORIA DE ROTEAMENTO DO BLOQUEADOR DE REVERSAO")
        st.caption(
            "DIAGNOSTIC ONLY: esta camada explica por que REVERSAL_NOT_ELIGIBLE apareceu no roteamento do setup. "
            "Nao aprova trade, nao altera score, nao muda broker, nao muda thresholds e preserva PAPER TRADING."
        )
        rr_c1, rr_c2, rr_c3, rr_c4, rr_c5 = st.columns(5)
        rr_c1.metric("Top ativo", reversal_blocker_routing_audit.get("top_symbol") or "-")
        rr_c2.metric("Setup analisado", reversal_blocker_routing_audit.get("top_setup") or "trend_pullback_breakout")
        rr_c3.metric("Blocker observado", reversal_blocker_routing_audit.get("observed_blocker") or "REVERSAL_NOT_ELIGIBLE")
        rr_c4.metric("Status de roteamento", reversal_blocker_routing_audit.get("top_route_status") or "INSUFFICIENT_DATA")
        rr_c5.metric("Bucket alternativo", reversal_blocker_routing_audit.get("top_alternative_bucket") or "-")
        rr_d1, rr_d2, rr_d3, rr_d4, rr_d5 = st.columns(5)
        rr_d1.metric("Score", reversal_blocker_routing_audit.get("top_score") if reversal_blocker_routing_audit.get("top_score") is not None else "-")
        rr_d2.metric("Min score", reversal_blocker_routing_audit.get("top_min_score") if reversal_blocker_routing_audit.get("top_min_score") is not None else "-")
        rr_d3.metric("Gap", reversal_blocker_routing_audit.get("top_score_gap") if reversal_blocker_routing_audit.get("top_score_gap") is not None else "-")
        rr_d4.metric("Feed atual limpo?", "Sim" if bool(reversal_blocker_routing_audit.get("current_feed_is_clean", False)) else "Nao")
        rr_d5.metric("Recomendacao", reversal_blocker_routing_audit.get("recommendation") or "insufficient_data")
        rr_e1, rr_e2, rr_e3, rr_e4 = st.columns(4)
        rr_e1.metric("Reversal blockers", int(reversal_blocker_routing_audit.get("reversal_blocker_count", 0) or 0))
        rr_e2.metric("Trend + reversal", int(reversal_blocker_routing_audit.get("trend_candidates_with_reversal_blocker", 0) or 0))
        rr_e3.metric("Deve manter bloqueado", "Sim" if bool(reversal_blocker_routing_audit.get("should_keep_blocked", True)) else "Nao")
        rr_e4.metric("Seguro alterar estrategia agora?", "Sim" if bool(reversal_blocker_routing_audit.get("safe_to_change_strategy_now", False)) else "Nao")
        reversal_rows = [
            {
                "symbol": row.get("symbol"),
                "setup": row.get("setup"),
                "score": row.get("score"),
                "score_gap": row.get("score_gap"),
                "primary_real_blocker": row.get("primary_real_blocker"),
                "secondary_real_blocker": row.get("secondary_real_blocker"),
                "route_status": row.get("route_status"),
                "alternative_bucket": row.get("alternative_bucket"),
                "multi_tf_alignment_status": row.get("multi_tf_alignment_status"),
                "bos_state": row.get("bos_state"),
                "pivot_state": row.get("pivot_state"),
                "suggested_future_study": row.get("suggested_future_study"),
                "should_keep_blocked": row.get("should_keep_blocked"),
                "safe_to_change_strategy_now": row.get("safe_to_change_strategy_now"),
            }
            for row in list(reversal_blocker_routing_audit.get("candidates", []) or [])[:10]
            if isinstance(row, dict)
        ]
        if reversal_rows:
            st.dataframe(pd.DataFrame(reversal_rows), hide_index=True, use_container_width=True)
        else:
            st.info("Sem amostra REVERSAL_NOT_ELIGIBLE suficiente para auditoria de roteamento.")
        st.caption(reversal_blocker_routing_audit.get("notes") or "Diagnostico sem autoridade operacional.")
    if setup_blocker_taxonomy_audit:
        st.markdown("##### FASE 2.5B.2B - CLAREZA DE TAXONOMIA SETUP/BLOQUEADOR")
        st.caption(
            "DIAGNOSTIC ONLY: esta camada clarifica a taxonomia dos bloqueios entre setup de tendencia, "
            "reversao, score, BOS, pivo, pullback e Multi-TF. Nao aprova trade, nao altera score, "
            "nao muda broker, nao muda thresholds e preserva PAPER TRADING."
        )
        tx_c1, tx_c2, tx_c3, tx_c4, tx_c5 = st.columns(5)
        tx_c1.metric("Top ativo", setup_blocker_taxonomy_audit.get("top_symbol") or "-")
        tx_c2.metric("Setup analisado", setup_blocker_taxonomy_audit.get("top_setup") or "trend_pullback_breakout")
        tx_c3.metric("Blocker oficial primario", setup_blocker_taxonomy_audit.get("official_primary_blocker") or "-")
        tx_c4.metric("Blocker oficial secundario", setup_blocker_taxonomy_audit.get("official_secondary_blocker") or "-")
        tx_c5.metric("Status taxonomia", setup_blocker_taxonomy_audit.get("taxonomy_status") or "INSUFFICIENT_DATA")
        tx_d1, tx_d2, tx_d3, tx_d4, tx_d5 = st.columns(5)
        tx_d1.metric("Razao normalizada prim.", setup_blocker_taxonomy_audit.get("normalized_primary_reason") or "UNKNOWN")
        tx_d2.metric("Razao normalizada sec.", setup_blocker_taxonomy_audit.get("normalized_secondary_reason") or "UNKNOWN")
        tx_d3.metric("Confianca", setup_blocker_taxonomy_audit.get("taxonomy_confidence") if setup_blocker_taxonomy_audit.get("taxonomy_confidence") is not None else "-")
        tx_d4.metric("Feed atual limpo?", "Sim" if bool(setup_blocker_taxonomy_audit.get("current_feed_is_clean", False)) else "Nao")
        tx_d5.metric("Recomendacao", setup_blocker_taxonomy_audit.get("recommendation") or "insufficient_data")
        tx_e1, tx_e2, tx_e3, tx_e4 = st.columns(4)
        tx_e1.metric("Deve manter bloqueado", "Sim" if bool(setup_blocker_taxonomy_audit.get("should_keep_blocked", True)) else "Nao")
        tx_e2.metric("Seguro alterar estrategia agora?", "Sim" if bool(setup_blocker_taxonomy_audit.get("safe_to_change_strategy_now", False)) else "Nao")
        tx_e3.metric("Seguro alterar threshold agora?", "Sim" if bool(setup_blocker_taxonomy_audit.get("safe_to_change_threshold_now", False)) else "Nao")
        tx_e4.metric("Taxonomias mistas", int(setup_blocker_taxonomy_audit.get("mixed_taxonomy_count", 0) or 0))
        taxonomy_rows = [
            {
                "symbol": row.get("symbol"),
                "setup": row.get("setup"),
                "score": row.get("score"),
                "score_gap": row.get("score_gap"),
                "official_primary_blocker": row.get("official_primary_blocker"),
                "official_secondary_blocker": row.get("official_secondary_blocker"),
                "normalized_primary_reason": row.get("normalized_primary_reason"),
                "normalized_secondary_reason": row.get("normalized_secondary_reason"),
                "taxonomy_status": row.get("taxonomy_status"),
                "route_status": row.get("route_status"),
                "no_setup_bucket": row.get("no_setup_bucket"),
                "bos_state": row.get("bos_state"),
                "pivot_state": row.get("pivot_state"),
                "suggested_ui_message": row.get("suggested_ui_message"),
                "suggested_future_study": row.get("suggested_future_study"),
                "should_keep_blocked": row.get("should_keep_blocked"),
            }
            for row in list(setup_blocker_taxonomy_audit.get("candidates", []) or [])[:10]
            if isinstance(row, dict)
        ]
        if taxonomy_rows:
            st.dataframe(pd.DataFrame(taxonomy_rows), hide_index=True, use_container_width=True)
        else:
            st.info("Sem amostra suficiente para clarificar taxonomia setup/bloqueador.")
        st.caption(setup_blocker_taxonomy_audit.get("notes") or "Diagnostico sem autoridade operacional.")
    if bos_confirmation_quality_audit:
        st.markdown("##### FASE 2.5B.2C - QUALIDADE DA CONFIRMACAO DE BOS")
        st.caption(
            "DIAGNOSTIC ONLY: esta camada explica por que o BOS nao confirmou ou por que a confirmacao "
            "estrutural ainda e fraca. Nao aprova trade, nao altera score, nao muda broker, nao muda "
            "thresholds e preserva PAPER TRADING."
        )
        bq_c1, bq_c2, bq_c3, bq_c4, bq_c5 = st.columns(5)
        bq_c1.metric("Top ativo", bos_confirmation_quality_audit.get("top_symbol") or "-")
        bq_c2.metric("Setup analisado", bos_confirmation_quality_audit.get("top_setup") or "trend_pullback_breakout")
        bq_c3.metric("Timeframe principal", bos_confirmation_quality_audit.get("top_timeframe") or "-")
        bq_c4.metric("Status BOS", bos_confirmation_quality_audit.get("bos_quality_status") or "INSUFFICIENT_DATA")
        bq_c5.metric("Motivo falha BOS", bos_confirmation_quality_audit.get("bos_failure_reason") or "insufficient_data")
        bq_d1, bq_d2, bq_d3, bq_d4, bq_d5 = st.columns(5)
        bq_d1.metric("Pivo", bos_confirmation_quality_audit.get("pivot_state") or "INSUFFICIENT_DATA")
        bq_d2.metric("BOS 1H", bos_confirmation_quality_audit.get("h1_bos_state") or "INSUFFICIENT_DATA")
        bq_d3.metric("BOS 4H", bos_confirmation_quality_audit.get("h4_bos_state") or "INSUFFICIENT_DATA")
        bq_d4.metric("Relacao 1H/4H", bos_confirmation_quality_audit.get("h1_h4_relationship") or "INSUFFICIENT_DATA")
        bq_d5.metric("Multi-TF", bos_confirmation_quality_audit.get("multi_tf_alignment_status") or "INSUFFICIENT_DATA")
        bq_e1, bq_e2, bq_e3, bq_e4, bq_e5 = st.columns(5)
        bq_e1.metric("Dist. fechamento", bos_confirmation_quality_audit.get("close_distance_to_bos_pct") if bos_confirmation_quality_audit.get("close_distance_to_bos_pct") is not None else "-")
        bq_e2.metric("Cruzou por pavio?", "Sim" if bool(bos_confirmation_quality_audit.get("wick_crossed_level", False)) else "Nao")
        bq_e3.metric("Fechou alem?", "Sim" if bool(bos_confirmation_quality_audit.get("close_confirmed_beyond_level", False)) else "Nao")
        bq_e4.metric("Reteste pendente?", "Sim" if bool(bos_confirmation_quality_audit.get("retest_pending", False)) else "Nao")
        bq_e5.metric("Reteste confirmado?", "Sim" if bool(bos_confirmation_quality_audit.get("retest_confirmed", False)) else "Nao")
        bq_f1, bq_f2, bq_f3, bq_f4 = st.columns(4)
        bq_f1.metric("Feed atual limpo?", "Sim" if bool(bos_confirmation_quality_audit.get("current_feed_is_clean", False)) else "Nao")
        bq_f2.metric("Recomendacao", bos_confirmation_quality_audit.get("recommendation") or "insufficient_data")
        bq_f3.metric("Deve manter bloqueado", "Sim" if bool(bos_confirmation_quality_audit.get("should_keep_blocked", True)) else "Nao")
        bq_f4.metric("Seguro alterar threshold?", "Sim" if bool(bos_confirmation_quality_audit.get("safe_to_change_threshold_now", False)) else "Nao")
        bos_quality_rows = [
            {
                "symbol": row.get("symbol"),
                "setup": row.get("setup"),
                "score": row.get("score"),
                "score_gap": row.get("score_gap"),
                "bos_quality_status": row.get("bos_quality_status"),
                "bos_failure_reason": row.get("bos_failure_reason"),
                "h1_bos_state": row.get("h1_bos_state"),
                "h4_bos_state": row.get("h4_bos_state"),
                "pivot_state": row.get("pivot_state"),
                "close_distance_to_bos_pct": row.get("close_distance_to_bos_pct"),
                "wick_crossed_level": row.get("wick_crossed_level"),
                "close_confirmed_beyond_level": row.get("close_confirmed_beyond_level"),
                "retest_pending": row.get("retest_pending"),
                "retest_confirmed": row.get("retest_confirmed"),
                "suggested_ui_message": row.get("suggested_ui_message"),
                "suggested_future_study": row.get("suggested_future_study"),
                "should_keep_blocked": row.get("should_keep_blocked"),
            }
            for row in list(bos_confirmation_quality_audit.get("candidates", []) or [])[:10]
            if isinstance(row, dict)
        ]
        if bos_quality_rows:
            st.dataframe(pd.DataFrame(bos_quality_rows), hide_index=True, use_container_width=True)
        else:
            st.info("Sem amostra suficiente para auditar qualidade de BOS.")
        st.caption(bos_confirmation_quality_audit.get("notes") or "Diagnostico sem autoridade operacional.")
    if h1_confirmation_after_h4_bos_audit:
        st.markdown("##### FASE 2.5B.2D - CONFIRMACAO 1H APOS BOS 4H")
        st.caption(
            "DIAGNOSTIC ONLY: esta camada explica por que uma estrutura confirmada no 4H ainda nao "
            "recebeu confirmacao suficiente no 1H. Nao aprova trade, nao altera score, nao muda "
            "broker, nao muda thresholds e preserva PAPER TRADING."
        )
        h1_c1, h1_c2, h1_c3, h1_c4, h1_c5 = st.columns(5)
        h1_c1.metric("Top ativo", h1_confirmation_after_h4_bos_audit.get("top_symbol") or "-")
        h1_c2.metric(
            "Setup analisado",
            h1_confirmation_after_h4_bos_audit.get("top_setup") or "trend_pullback_breakout",
        )
        h1_c3.metric("BOS 4H", h1_confirmation_after_h4_bos_audit.get("h4_bos_state") or "INSUFFICIENT_DATA")
        h1_c4.metric("Reteste 4H", h1_confirmation_after_h4_bos_audit.get("h4_retest_state") or "UNKNOWN")
        h1_c5.metric("BOS 1H", h1_confirmation_after_h4_bos_audit.get("h1_bos_state") or "INSUFFICIENT_DATA")
        h1_d1, h1_d2, h1_d3, h1_d4, h1_d5 = st.columns(5)
        h1_d1.metric("Estado confirmacao 1H", h1_confirmation_after_h4_bos_audit.get("h1_confirmation_state") or "UNKNOWN")
        h1_d2.metric("Status confirmacao 1H", h1_confirmation_after_h4_bos_audit.get("h1_confirmation_status") or "INSUFFICIENT_DATA")
        h1_d3.metric("Motivo falha 1H", h1_confirmation_after_h4_bos_audit.get("h1_failure_reason") or "insufficient_data")
        h1_d4.metric("Qualidade dados 1H", h1_confirmation_after_h4_bos_audit.get("h1_data_quality") or "missing")
        h1_d5.metric("Alinhamento 1H/4H", h1_confirmation_after_h4_bos_audit.get("h1_h4_alignment") or "UNKNOWN")
        h1_e1, h1_e2, h1_e3, h1_e4, h1_e5 = st.columns(5)
        h1_e1.metric("Direcao 4H", h1_confirmation_after_h4_bos_audit.get("h4_trend_direction") or "INCONCLUSIVE")
        h1_e2.metric("Direcao 1H", h1_confirmation_after_h4_bos_audit.get("h1_trend_direction") or "INCONCLUSIVE")
        h1_e3.metric("Reteste 1H", h1_confirmation_after_h4_bos_audit.get("h1_retest_state") or "UNKNOWN")
        h1_e4.metric("Pivo 1H", h1_confirmation_after_h4_bos_audit.get("h1_pivot_state") or "INSUFFICIENT_DATA")
        h1_e5.metric("Risco de timing", h1_confirmation_after_h4_bos_audit.get("h1_entry_timing_risk") or "UNKNOWN")
        h1_f1, h1_f2, h1_f3, h1_f4 = st.columns(4)
        h1_f1.metric("Feed atual limpo?", "Sim" if bool(h1_confirmation_after_h4_bos_audit.get("current_feed_is_clean", False)) else "Nao")
        h1_f2.metric("Recomendacao", h1_confirmation_after_h4_bos_audit.get("recommendation") or "insufficient_data")
        h1_f3.metric("Deve manter bloqueado", "Sim" if bool(h1_confirmation_after_h4_bos_audit.get("should_keep_blocked", True)) else "Nao")
        h1_f4.metric("Seguro alterar threshold?", "Sim" if bool(h1_confirmation_after_h4_bos_audit.get("safe_to_change_threshold_now", False)) else "Nao")
        h1_g1, h1_g2 = st.columns(2)
        h1_g1.metric("Seguro alterar estrategia?", "Sim" if bool(h1_confirmation_after_h4_bos_audit.get("safe_to_change_strategy_now", False)) else "Nao")
        h1_g2.metric("Fallback scope", h1_confirmation_after_h4_bos_audit.get("fallback_blocker_scope") or "UNKNOWN")
        h1_after_h4_rows = [
            {
                "symbol": row.get("symbol"),
                "setup": row.get("setup"),
                "score": row.get("score"),
                "score_gap": row.get("score_gap"),
                "h4_bos_state": row.get("h4_bos_state"),
                "h4_retest_state": row.get("h4_retest_state"),
                "h1_bos_state": row.get("h1_bos_state"),
                "h1_confirmation_status": row.get("h1_confirmation_status"),
                "h1_failure_reason": row.get("h1_failure_reason"),
                "h1_data_quality": row.get("h1_data_quality"),
                "h1_h4_alignment": row.get("h1_h4_alignment"),
                "h1_entry_timing_risk": row.get("h1_entry_timing_risk"),
                "suggested_ui_message": row.get("suggested_ui_message"),
                "suggested_future_study": row.get("suggested_future_study"),
                "should_keep_blocked": row.get("should_keep_blocked"),
            }
            for row in list(h1_confirmation_after_h4_bos_audit.get("candidates", []) or [])[:10]
            if isinstance(row, dict)
        ]
        if h1_after_h4_rows:
            st.dataframe(pd.DataFrame(h1_after_h4_rows), hide_index=True, use_container_width=True)
        else:
            st.info("Sem amostra suficiente para auditar confirmacao 1H apos BOS 4H.")
        st.caption(h1_confirmation_after_h4_bos_audit.get("notes") or "Diagnostico sem autoridade operacional.")
    mtf_rows = [
        {
            "symbol": row.get("symbol"),
            "daily_bias": row.get("daily_bias"),
            "h4_structure": row.get("h4_structure"),
            "h1_confirmation": row.get("h1_confirmation"),
            "alignment_score": row.get("alignment_score"),
            "alignment_status": row.get("alignment_status"),
            "pivot_timeframes": ", ".join(list(row.get("pivot_confirmed_timeframes", row.get("pivot_timeframes", [])) or [])),
            "bos_timeframes": ", ".join(list(row.get("bos_confirmed_timeframes", row.get("bos_timeframes", [])) or [])),
            "supports_trend_pullback_breakout": row.get("supports_trend_pullback_breakout"),
            "missing_for_setup": ", ".join(list(row.get("missing_for_setup", []) or [])),
            "would_improve_signal_quality": row.get("would_improve_signal_quality"),
            "should_keep_blocked": row.get("should_keep_blocked"),
            "recommendation": row.get("recommendation"),
        }
        for row in mtf_candidates[:8]
    ]
    if mtf_rows:
        st.dataframe(pd.DataFrame(mtf_rows), hide_index=True, use_container_width=True)
    else:
        st.info("Sem candidatos multi-timeframe suficientes ainda.")
if shadow_decision_simulator:
    st.markdown("#### SIMULADOR SHADOW DE DECISAO - FASE 2.4")
    st.caption(
        "SHADOW ONLY: quase-aprovados sao simulados separadamente. "
        "Nao abre trade, nao cria posicao paper oficial, nao altera PnL, score, broker ou thresholds."
    )
    sd_c1, sd_c2, sd_c3, sd_c4, sd_c5 = st.columns(5)
    sd_c1.metric("Quase-aprovados preview", int(shadow_decision_simulator.get("preview_near_approved_count", shadow_decision_simulator.get("shadow_near_approved_count", 0)) or 0))
    sd_c2.metric("Safe", int(shadow_decision_simulator.get("shadow_safe_near_approved_count", 0) or 0))
    sd_c3.metric("Marginal", int(shadow_decision_simulator.get("shadow_marginal_near_approved_count", shadow_decision_simulator.get("shadow_marginal_count", 0)) or 0))
    sd_c4.metric("Teria entrado", int(shadow_decision_simulator.get("shadow_would_enter_count", 0) or 0))
    sd_c5.metric("Pendentes", int(shadow_decision_simulator.get("shadow_pending_count", 0) or 0))
    sd_d1, sd_d2, sd_d3, sd_d4 = st.columns(4)
    sd_d1.metric("Teria ganho", int(shadow_decision_simulator.get("shadow_would_win_count", 0) or 0))
    sd_d2.metric("Teria perdido", int(shadow_decision_simulator.get("shadow_would_lose_count", 0) or 0))
    sd_d3.metric("Melhor ativo", shadow_decision_simulator.get("shadow_best_symbol") or "-")
    sd_d4.metric("Melhor setup", shadow_decision_simulator.get("shadow_best_strategy") or "-")
    st.caption(
        f"Bloqueio dominante: {shadow_decision_simulator.get('shadow_dominant_block_reason') or '-'} | "
        f"Atual: {shadow_decision_simulator.get('dominant_exclusion_current_scope') or '-'} | "
        f"Acumulado: {shadow_decision_simulator.get('dominant_exclusion_accumulated_scope') or '-'} | "
        f"Escopo fallback: {shadow_decision_simulator.get('fallback_blocker_scope') or 'UNKNOWN'} | "
        f"Recomendacao: {shadow_decision_simulator.get('shadow_policy_recommendation') or 'observe_more'} | "
        f"Politica: {shadow_decision_simulator.get('shadow_entry_policy') or 'conservative_v1'}"
    )
    st.markdown("##### RASTREABILIDADE FASE 2.4D - ESCOPO NORMALIZADO")
    trace_preview_count = int(shadow_decision_simulator.get("preview_near_approved_count", 0) or 0)
    trace_received_current = int(
        shadow_decision_simulator.get(
            "shadow_current_cycle_received_count",
            shadow_decision_simulator.get("shadow_candidates_received_count", 0),
        )
        or 0
    )
    trace_new_unique_current = int(
        shadow_decision_simulator.get(
            "shadow_current_cycle_new_unique_count",
            shadow_decision_simulator.get("shadow_candidates_unique_count", 0),
        )
        or 0
    )
    trace_duplicate_current = int(
        shadow_decision_simulator.get(
            "shadow_current_cycle_duplicate_count",
            shadow_decision_simulator.get("shadow_current_cycle_ignored_count", 0),
        )
        or 0
    )
    trace_already_analyzed = int(
        shadow_decision_simulator.get("shadow_current_cycle_already_analyzed_count", trace_duplicate_current) or 0
    )
    trace_analyzed_current = int(
        shadow_decision_simulator.get(
            "shadow_current_cycle_analyzed_new_count",
            shadow_decision_simulator.get("shadow_current_cycle_analyzed_count", 0),
        )
        or 0
    )
    trace_classified_current = int(
        shadow_decision_simulator.get(
            "shadow_current_cycle_classified_new_count",
            shadow_decision_simulator.get("shadow_current_cycle_classified_count", 0),
        )
        or 0
    )
    trace_unsafe_current = int(
        shadow_decision_simulator.get(
            "shadow_current_cycle_unsafe_new_count",
            shadow_decision_simulator.get("shadow_current_cycle_unsafe_count", 0),
        )
        or 0
    )
    trace_primary_current = int(
        shadow_decision_simulator.get(
            "shadow_current_cycle_primary_blocked_new_count",
            shadow_decision_simulator.get("shadow_primary_blocked_count", 0),
        )
        or 0
    )
    trace_secondary_current = int(
        shadow_decision_simulator.get(
            "shadow_current_cycle_secondary_blocked_new_count",
            shadow_decision_simulator.get("shadow_secondary_blocked_count", 0),
        )
        or 0
    )
    trace_accumulated_unique = int(
        shadow_decision_simulator.get(
            "shadow_accumulated_unique_candidates_count",
            shadow_decision_simulator.get("shadow_accumulated_candidates_count", 0),
        )
        or 0
    )
    trace_accumulated_unsafe = int(
        shadow_decision_simulator.get(
            "shadow_accumulated_unsafe_unique_count",
            shadow_decision_simulator.get("shadow_accumulated_unsafe_count", 0),
        )
        or 0
    )
    trace_accumulated_raw = int(
        shadow_decision_simulator.get(
            "shadow_accumulated_raw_received_count",
            shadow_decision_simulator.get("shadow_accumulated_received_count", trace_received_current),
        )
        or 0
    )
    trace_duplicate_ratio = float(shadow_decision_simulator.get("shadow_duplicate_ratio", 0.0) or 0.0)
    trace_raw_to_unique_ratio = float(shadow_decision_simulator.get("shadow_raw_to_unique_ratio", 0.0) or 0.0)
    trace_table_scope = shadow_decision_simulator.get("shadow_counts_scope") or "current_cycle_and_accumulated_recent"
    tr_c1, tr_c2, tr_c3, tr_c4 = st.columns(4)
    tr_c1.metric("Preview near-approved", trace_preview_count)
    tr_c2.metric("Recebidos no ciclo", trace_received_current)
    tr_c3.metric("Novos unicos no ciclo", trace_new_unique_current)
    tr_c4.metric("Duplicados no ciclo", trace_duplicate_current)
    tr_d1, tr_d2, tr_d3, tr_d4 = st.columns(4)
    tr_d1.metric("Analisados novos no ciclo", trace_analyzed_current)
    tr_d2.metric("Unsafe novos no ciclo", trace_unsafe_current)
    tr_d3.metric("Ja analisados anteriormente", trace_already_analyzed)
    tr_d4.metric("Candidatos unicos acumulados", trace_accumulated_unique)
    tr_e1, tr_e2, tr_e3, tr_e4 = st.columns(4)
    tr_e1.metric("Unsafe acumulados unicos", trace_accumulated_unsafe)
    tr_e2.metric("Recebidos brutos acumulados", trace_accumulated_raw)
    tr_e3.metric("Taxa de duplicidade", f"{trace_duplicate_ratio:.0%}")
    tr_e4.metric("Escopo da tabela", trace_table_scope)
    st.caption(
        f"Motivo principal de exclusao: {shadow_decision_simulator.get('shadow_dominant_block_reason') or '-'} | "
        f"Atual: {shadow_decision_simulator.get('dominant_exclusion_current_scope') or '-'} | "
        f"Acumulado: {shadow_decision_simulator.get('dominant_exclusion_accumulated_scope') or '-'} | "
        f"Primario novo: {trace_primary_current} | Secundario novo: {trace_secondary_current} | "
        f"Ignorados: {shadow_decision_simulator.get('shadow_ignored_reason') or '-'}"
    )
    st.caption(
        "Candidatos duplicados nao sao reanalisados como novos, mas continuam aparecendo no acumulado recente para auditoria."
    )
    if trace_received_current > 0 and trace_new_unique_current == 0 and trace_duplicate_current == trace_received_current:
        st.info(
            f"Este ciclo nao gerou novos candidatos unicos; os {trace_received_current} recebidos ja haviam sido analisados antes. "
            "A tabela de acumulados abaixo mostra candidatos recentes ja classificados."
        )
    if trace_raw_to_unique_ratio >= 10.0:
        st.info(
            "Alto volume bruto recebido por repeticao de ciclos; leitura estrategica deve usar candidatos unicos, "
            "nao recebimentos brutos."
        )
    if bool(shadow_decision_simulator.get("shadow_scope_warning", shadow_decision_simulator.get("shadow_counter_warning", False))):
        st.warning(
            "Aviso de consistencia de escopo shadow: "
            f"{shadow_decision_simulator.get('shadow_scope_warning_reason') or shadow_decision_simulator.get('shadow_counter_warning_reason') or 'verificar contadores'}"
        )
    def _shadow_trace_row(row, default_scope):
        return {
            "symbol": row.get("symbol"),
            "setup": row.get("strategy"),
            "score": row.get("current_score"),
            "score_gap": row.get("score_gap"),
            "raw_near_approved": row.get("raw_near_approved"),
            "duplicate_candidate": row.get("duplicate_candidate"),
            "already_seen": row.get("already_seen"),
            "analyzed_this_cycle": row.get("analyzed_this_cycle"),
            "analyzed_previously": row.get("analyzed_previously"),
            "classified_by_shadow": row.get("classified_by_shadow"),
            "class": row.get("candidate_class"),
            "safe_candidate": row.get("safe_candidate"),
            "would_enter": row.get("shadow_would_enter"),
            "primary_blockers": ", ".join(list(row.get("primary_blocker_codes", row.get("primary_blockers", [])) or [])),
            "secondary_blockers": ", ".join(list(row.get("secondary_blocker_codes", row.get("secondary_blockers", [])) or [])),
            "why_not_safe": row.get("why_not_safe"),
            "why_would_not_enter": row.get("why_would_not_enter") or row.get("shadow_block_reason"),
            "count_scope": row.get("count_scope") or default_scope,
        }

    current_shadow_rows = [
        _shadow_trace_row(row, "current_cycle")
        for row in list(shadow_decision_simulator.get("shadow_current_cycle_candidates", []) or [])
        if isinstance(row, dict)
    ]
    accumulated_shadow_rows = [
        _shadow_trace_row(row, "accumulated_recent")
        for row in list(
            shadow_decision_simulator.get(
                "shadow_accumulated_recent_candidates",
                shadow_decision_simulator.get("shadow_recent_candidates", []),
            )
            or []
        )
        if isinstance(row, dict)
    ]
    st.markdown("###### CANDIDATOS NOVOS DO CICLO")
    if current_shadow_rows:
        st.dataframe(pd.DataFrame(current_shadow_rows[:8]), hide_index=True, use_container_width=True)
    else:
        st.info("Nenhum candidato novo unico neste ciclo.")
    st.markdown("###### CANDIDATOS ACUMULADOS RECENTES")
    if accumulated_shadow_rows:
        st.dataframe(pd.DataFrame(accumulated_shadow_rows[:8]), hide_index=True, use_container_width=True)
    else:
        st.info("Nenhum candidato acumulado recente para auditoria.")
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
