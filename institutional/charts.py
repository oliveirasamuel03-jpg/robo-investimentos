from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


DARK_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=12, r=12, t=42, b=12),
)


def build_equity_chart(curve_df: pd.DataFrame, title: str = "Curva de Equity") -> go.Figure:
    fig = go.Figure()
    if curve_df is not None and not curve_df.empty:
        fig.add_trace(
            go.Scatter(
                x=curve_df["timestamp"],
                y=curve_df["equity"],
                mode="lines",
                name="Equity",
                line=dict(width=2.6, color="#38bdf8"),
            )
        )
    fig.update_layout(height=300, title=title, **DARK_LAYOUT)
    return fig


def build_drawdown_chart(drawdown_df: pd.DataFrame, title: str = "Drawdown") -> go.Figure:
    fig = go.Figure()
    if drawdown_df is not None and not drawdown_df.empty:
        fig.add_trace(
            go.Scatter(
                x=drawdown_df["timestamp"],
                y=drawdown_df["drawdown"],
                fill="tozeroy",
                mode="lines",
                name="Drawdown",
                line=dict(width=1.8, color="#ef4444"),
            )
        )
    fig.update_layout(height=240, title=title, **DARK_LAYOUT)
    return fig


def build_asset_performance_chart(asset_rows: list[dict], title: str = "Performance por Ativo") -> go.Figure:
    fig = go.Figure()
    if asset_rows:
        df = pd.DataFrame(asset_rows)
        fig.add_trace(
            go.Bar(
                x=df["asset"],
                y=df["pnl"],
                marker_color=["#22c55e" if value >= 0 else "#ef4444" for value in df["pnl"]],
                name="PnL",
            )
        )
    fig.update_layout(height=290, title=title, **DARK_LAYOUT)
    return fig
