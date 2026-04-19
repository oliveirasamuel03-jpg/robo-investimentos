from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from .analytics import summarize_backtest_by_asset
from .charts import build_asset_performance_chart, build_drawdown_chart, build_equity_chart
from .performance import build_drawdown_series, build_equity_curve_df, summarize_performance


def _fmt_pct(value: float) -> str:
    return f"{float(value) * 100:.2f}%"


def _safe_records(df: pd.DataFrame, limit: int | None = None) -> list[dict]:
    if df is None or df.empty:
        return []
    out = df.copy()
    if limit is not None:
        out = out.tail(limit)
    return out.to_dict(orient="records")


def _drawdown_records(curve_df: pd.DataFrame) -> list[dict]:
    if curve_df is None or curve_df.empty:
        return []
    drawdown = build_drawdown_series(curve_df)
    return pd.DataFrame({"timestamp": curve_df["timestamp"], "drawdown": drawdown}).to_dict(orient="records")


def classify_strategy_health(research_metrics: dict, holdout_metrics: dict) -> str:
    research_sharpe = float(research_metrics.get("sharpe", 0.0) or 0.0)
    holdout_sharpe = float(holdout_metrics.get("sharpe", 0.0) or 0.0)
    holdout_drawdown = float(holdout_metrics.get("max_drawdown", 0.0) or 0.0)

    if holdout_sharpe >= 0.8 and holdout_drawdown > -0.10:
        return "GREEN"
    if holdout_sharpe < 0.1 or holdout_drawdown <= -0.18 or research_sharpe <= 0.0:
        return "RED"
    return "YELLOW"


def build_institutional_dashboard_payload(result: dict) -> dict:
    research_bt = result.get("research_bt", {}) or {}
    holdout_bt = result.get("holdout_bt", {}) or {}
    wf = result.get("wf", {}) or {}

    research_curve = build_equity_curve_df(research_bt.get("equity_curve"))
    holdout_curve = build_equity_curve_df(holdout_bt.get("equity_curve"))

    research_metrics = summarize_performance(
        research_bt.get("equity_curve"),
        returns_input=research_bt.get("returns"),
    )
    holdout_metrics = summarize_performance(
        holdout_bt.get("equity_curve"),
        returns_input=holdout_bt.get("returns"),
    )

    by_asset_research = summarize_backtest_by_asset(
        research_bt.get("asset_contribution"),
        research_bt.get("effective_weights"),
    )
    by_asset_holdout = summarize_backtest_by_asset(
        holdout_bt.get("asset_contribution"),
        holdout_bt.get("effective_weights"),
    )

    return {
        "generated_at": datetime.utcnow().isoformat(),
        "health": classify_strategy_health(research_metrics, holdout_metrics),
        "report": result.get("report", {}) or {},
        "research_metrics": research_metrics,
        "holdout_metrics": holdout_metrics,
        "research_curve": _safe_records(research_curve),
        "holdout_curve": _safe_records(holdout_curve),
        "research_drawdown": _drawdown_records(research_curve),
        "holdout_drawdown": _drawdown_records(holdout_curve),
        "research_by_asset": by_asset_research,
        "holdout_by_asset": by_asset_holdout,
        "fold_metrics": _safe_records(wf.get("fold_metrics"), limit=50),
        "aggregate_metrics": wf.get("aggregate_metrics", {}) or {},
        "degradation_analysis": wf.get("degradation_analysis", {}) or {},
        "monte_carlo": result.get("mc", {}) or {},
        "prices_columns": list(result.get("prices_columns", []) or []),
    }


def _load_frame(records: list[dict], numeric_cols: list[str]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def render_institutional_dashboard(payload: dict) -> None:
    if not payload:
        st.info("Sem dashboard institucional salvo ainda.")
        return

    research_metrics = payload.get("research_metrics", {}) or {}
    holdout_metrics = payload.get("holdout_metrics", {}) or {}
    health = payload.get("health", "YELLOW")

    st.markdown("### Dashboard Institucional")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Strategy Health", health)
    c2.metric("Sharpe Pesquisa", f"{float(research_metrics.get('sharpe', 0.0)):.2f}")
    c3.metric("Sharpe Holdout", f"{float(holdout_metrics.get('sharpe', 0.0)):.2f}")
    c4.metric("Max DD Holdout", _fmt_pct(float(holdout_metrics.get("max_drawdown", 0.0) or 0.0)))
    c5.metric("Ativos", str(len(payload.get("prices_columns", []) or [])))

    research_curve = _load_frame(payload.get("research_curve", []), ["equity", "returns"])
    holdout_curve = _load_frame(payload.get("holdout_curve", []), ["equity", "returns"])
    research_drawdown = _load_frame(payload.get("research_drawdown", []), ["drawdown"])
    holdout_drawdown = _load_frame(payload.get("holdout_drawdown", []), ["drawdown"])
    research_by_asset = payload.get("research_by_asset", []) or []
    holdout_by_asset = payload.get("holdout_by_asset", []) or []

    tabs = st.tabs(["Performance", "Equity & Drawdown", "By-Asset", "Report"])

    with tabs[0]:
        left, right = st.columns(2)
        with left:
            st.caption("Pesquisa")
            st.json(research_metrics)
        with right:
            st.caption("Holdout")
            st.json(holdout_metrics)

        if payload.get("aggregate_metrics"):
            st.caption("Walk-forward agregado")
            st.json(payload["aggregate_metrics"])

        if payload.get("degradation_analysis"):
            st.caption("Degradacao / estabilidade")
            st.json(payload["degradation_analysis"])

    with tabs[1]:
        col_left, col_right = st.columns(2)
        with col_left:
            st.plotly_chart(build_equity_chart(research_curve, title="Equity - Pesquisa"), use_container_width=True)
            st.plotly_chart(build_drawdown_chart(research_drawdown, title="Drawdown - Pesquisa"), use_container_width=True)
        with col_right:
            st.plotly_chart(build_equity_chart(holdout_curve, title="Equity - Holdout"), use_container_width=True)
            st.plotly_chart(build_drawdown_chart(holdout_drawdown, title="Drawdown - Holdout"), use_container_width=True)

    with tabs[2]:
        if research_by_asset:
            st.caption("Pesquisa")
            st.plotly_chart(build_asset_performance_chart(research_by_asset, title="Pesquisa por ativo"), use_container_width=True)
            st.dataframe(pd.DataFrame(research_by_asset), use_container_width=True)
        else:
            st.info("Sem analytics por ativo na janela de pesquisa.")

        if holdout_by_asset:
            st.caption("Holdout")
            st.plotly_chart(build_asset_performance_chart(holdout_by_asset, title="Holdout por ativo"), use_container_width=True)
            st.dataframe(pd.DataFrame(holdout_by_asset), use_container_width=True)

    with tabs[3]:
        if payload.get("report"):
            st.json(payload["report"])
        else:
            st.info("Sem relatorio institucional disponivel.")

        if payload.get("fold_metrics"):
            st.caption("Ultimos folds")
            st.dataframe(pd.DataFrame(payload["fold_metrics"]), use_container_width=True)
