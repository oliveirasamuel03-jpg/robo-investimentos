from __future__ import annotations

from datetime import timedelta

import pandas as pd
import streamlit as st

from core.auth.guards import render_auth_toolbar, require_auth
from core.config import (
    BOT_LOG_COLUMNS,
    BOT_LOG_FILE,
    TRADER_ORDERS_COLUMNS,
    TRADER_ORDERS_FILE,
)
from core.state_store import read_storage_table
from core.trader_reports import (
    calculate_trade_report_metrics,
    filter_trade_reports,
    read_trade_reports,
)


require_auth()
render_auth_toolbar()

st.title("Historico Operacional")

trader_orders = read_storage_table(TRADER_ORDERS_FILE, columns=TRADER_ORDERS_COLUMNS)
trade_reports = read_trade_reports()
logs = read_storage_table(BOT_LOG_FILE, columns=BOT_LOG_COLUMNS)

t1, t2, t3 = st.tabs(["Ordens", "Relatorios Trader", "Logs"])

with t1:
    if not trader_orders.empty:
        st.dataframe(trader_orders.iloc[::-1], use_container_width=True)
    else:
        st.info("Sem ordens do trader.")

with t2:
    if trade_reports.empty:
        st.info("Sem relatorios de trades fechados ainda.")
    else:
        reports_view = trade_reports.copy()
        reports_view["closed_at"] = pd.to_datetime(reports_view["closed_at"], errors="coerce")
        reports_view["opened_at"] = pd.to_datetime(reports_view["opened_at"], errors="coerce")

        filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
        profile_options = ["Todos"] + sorted(
            [x for x in reports_view["profile"].dropna().astype(str).unique().tolist() if x]
        )
        asset_options = ["Todos"] + sorted(
            [x for x in reports_view["asset"].dropna().astype(str).unique().tolist() if x]
        )
        status_options = ["Todos"] + sorted(
            [x for x in reports_view["status_final"].dropna().astype(str).unique().tolist() if x]
        )

        with filter_col1:
            selected_profile = st.selectbox("Perfil", options=profile_options, index=0)
        with filter_col2:
            selected_asset = st.selectbox("Ativo", options=asset_options, index=0)
        with filter_col3:
            selected_status = st.selectbox("Status", options=status_options, index=0)
        with filter_col4:
            period_choice = st.selectbox("Periodo", options=["Tudo", "7 dias", "30 dias", "90 dias"], index=0)

        end_at = reports_view["closed_at"].max()
        start_at = None
        if pd.notna(end_at):
            if period_choice == "7 dias":
                start_at = end_at - timedelta(days=7)
            elif period_choice == "30 dias":
                start_at = end_at - timedelta(days=30)
            elif period_choice == "90 dias":
                start_at = end_at - timedelta(days=90)

        filtered_reports = filter_trade_reports(
            reports_view,
            profile=selected_profile,
            asset=selected_asset,
            status_final=selected_status,
            start_at=start_at.to_pydatetime() if isinstance(start_at, pd.Timestamp) else start_at,
            end_at=end_at.to_pydatetime() if isinstance(end_at, pd.Timestamp) else end_at,
        )
        metrics = calculate_trade_report_metrics(filtered_reports)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Trades fechados", f"{int(metrics['total_trades'])}")
        m2.metric("PnL acumulado", f"R$ {float(metrics['total_pnl'] or 0.0):,.2f}")
        m3.metric("Win rate", f"{float(metrics['win_rate'] or 0.0) * 100:.2f}%")
        m4.metric("Payoff", "-" if metrics["payoff"] is None else f"{float(metrics['payoff']):.2f}")

        m5, m6, m7, m8 = st.columns(4)
        m5.metric("Media ganhos", "-" if metrics["avg_win"] is None else f"R$ {float(metrics['avg_win']):,.2f}")
        m6.metric("Media perdas", "-" if metrics["avg_loss"] is None else f"R$ {float(metrics['avg_loss']):,.2f}")
        m7.metric("Duracao media", "-" if metrics["avg_duration"] is None else f"{float(metrics['avg_duration']):.1f} min")
        if metrics["best_trade"] is None:
            m8.metric("Melhor / pior", "-")
        else:
            m8.metric(
                "Melhor / pior",
                f"R$ {float(metrics['best_trade']):,.2f} / R$ {float(metrics['worst_trade'] or 0.0):,.2f}",
            )

        if filtered_reports.empty:
            st.info("Nenhum relatorio encontrado com os filtros atuais.")
        else:
            display_df = filtered_reports.sort_values("closed_at", ascending=False).copy()
            export_df = display_df.copy()
            export_df["opened_at"] = export_df["opened_at"].astype("string")
            export_df["closed_at"] = export_df["closed_at"].astype("string")
            st.download_button(
                "Exportar relatorios filtrados (CSV)",
                data=export_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="trader_trade_reports_export.csv",
                mime="text/csv",
                use_container_width=False,
                key="download_trade_reports_csv",
            )
            st.dataframe(display_df, use_container_width=True)

with t3:
    if not logs.empty:
        st.dataframe(logs.iloc[::-1], use_container_width=True)
    else:
        st.info("Sem logs do bot.")
