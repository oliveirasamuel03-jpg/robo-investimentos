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
from core.retention import get_retention_summary, load_weekly_summary, read_weekly_report_rows
from core.state_store import load_bot_state, read_storage_table
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
state = load_bot_state()
retention_summary = get_retention_summary(state)
validation_report = dict((state.get("validation", {}) or {}).get("last_report", {}) or {})
validation_consistency = dict(validation_report.get("consistency", {}) or {})

t1, t2, t3, t4 = st.tabs(["Ordens", "Relatorios Trader", "Validacao Semanal", "Logs"])

with t1:
    if not trader_orders.empty:
        st.dataframe(trader_orders.iloc[::-1], use_container_width=True)
    else:
        st.info("Sem ordens do trader.")

with t2:
    if retention_summary["enabled"]:
        st.caption(
            f"Retencao automatica ativa: runtime preserva os ultimos {retention_summary['retention_days']} dias. "
            "Dados mais antigos sao arquivados sem apagar historico."
        )
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
    status_c1, status_c2, status_c3, status_c4 = st.columns(4)
    status_c1.metric("Retencao", "Ativa" if retention_summary["enabled"] else "Inativa")
    status_c2.metric("Ultimo arquivamento", retention_summary["last_success_at"] or "Sem registro")
    status_c3.metric("Dias no runtime", str(retention_summary["retention_days"]))
    status_c4.metric("Relatorios semanais", str(retention_summary["weekly_reports_generated"]))

    status_c5, status_c6, status_c7, status_c8 = st.columns(4)
    status_c5.metric("Relatorios arquivados", str(retention_summary["trade_reports_archived_rows"]))
    status_c6.metric("Logs arquivados", str(retention_summary["bot_logs_archived_rows"]))
    status_c7.metric("Ordens arquivadas", str(retention_summary["trader_orders_archived_rows"]))
    status_c8.metric("Intervalo da rotina", f"{retention_summary['run_interval_hours']} h")

    if validation_consistency:
        cons_c1, cons_c2, cons_c3, cons_c4 = st.columns(4)
        cons_c1.metric("Amostra atual", str(validation_consistency.get("sample_quality_label") or "Sem leitura"))
        cons_c2.metric("Postura", str(validation_consistency.get("operational_posture_label") or "Indefinida"))
        cons_c3.metric(
            "Drawdown max",
            "-"
            if validation_report.get("metrics", {}).get("max_drawdown_pct") is None
            else f"{float(validation_report.get('metrics', {}).get('max_drawdown_pct') or 0.0) * 100:.2f}%",
        )
        cons_c4.metric(
            "Watchlist da fase",
            "Coerente" if bool(validation_consistency.get("watchlist_phase_aligned")) else "Fora da fase",
        )
        st.caption(validation_consistency.get("sample_quality_message") or "")
        st.caption(validation_consistency.get("watchlist_message") or "")

    if retention_summary["archive_trader_orders"]:
        st.caption("Ordens antigas tambem entram na politica de retencao nesta configuracao.")
    else:
        st.caption("Ordens antigas permanecem no runtime nesta etapa para preservar a aba Ordens sem mudar o fluxo atual.")

    if retention_summary["last_error"]:
        st.warning(f"Ultima falha da retencao: {retention_summary['last_error']}")

    weekly_index = retention_summary["weekly_reports_index"]
    if not weekly_index:
        st.info("Sem relatorios semanais disponiveis ainda.")
    else:
        options = {
            f"{item.get('week_label')} | {str(item.get('location') or 'runtime').title()} | "
            f"{int(item.get('trades_count') or 0)} trade(s)": item
            for item in weekly_index
        }
        selected_label = st.selectbox("Semana de validacao", options=list(options.keys()), index=0)
        selected_report = options[selected_label]
        selected_summary = load_weekly_summary(selected_report)
        selected_rows = read_weekly_report_rows(selected_report)

        week_c1, week_c2, week_c3, week_c4 = st.columns(4)
        week_c1.metric("Trades", str(int(selected_summary.get("trades_count", 0) or 0)))
        week_c2.metric("PnL semanal", f"R$ {float(selected_summary.get('pnl', 0.0) or 0.0):,.2f}")
        week_c3.metric("Win rate", f"{float(selected_summary.get('win_rate', 0.0) or 0.0) * 100:.2f}%")
        week_c4.metric("Payoff", "-" if selected_summary.get("payoff") is None else f"{float(selected_summary.get('payoff')):.2f}")

        week_c5, week_c6, week_c7, week_c8 = st.columns(4)
        week_c5.metric("Erros operacionais", str(int(selected_summary.get("operational_errors", 0) or 0)))
        fallback_pct = selected_summary.get("fallback_cycle_pct")
        week_c6.metric("Ciclos em fallback", "-" if fallback_pct is None else f"{float(fallback_pct):.2f}%")
        week_c7.metric("Trades cedo demais", str(int(selected_summary.get("early_closed_trades", 0) or 0)))
        week_c8.metric("Trades tempo demais", str(int(selected_summary.get("late_held_trades", 0) or 0)))

        observation = str(selected_summary.get("observation_final") or "Sem observacao").strip()
        if observation == "apto para continuar em simulacao":
            st.success(f"Observacao final: {observation}")
        elif observation == "precisa ajuste":
            st.warning(f"Observacao final: {observation}")
        else:
            st.error(f"Observacao final: {observation}")

        asset_c1, asset_c2 = st.columns(2)
        with asset_c1:
            st.write("**Melhores ativos da semana**")
            best_assets = pd.DataFrame(selected_summary.get("best_assets", []) or [])
            if best_assets.empty:
                st.info("Sem trades fechados com ganho relevante nesta semana.")
            else:
                st.dataframe(best_assets, use_container_width=True)
        with asset_c2:
            st.write("**Piores ativos da semana**")
            worst_assets = pd.DataFrame(selected_summary.get("worst_assets", []) or [])
            if worst_assets.empty:
                st.info("Sem trades fechados negativos relevantes nesta semana.")
            else:
                st.dataframe(worst_assets, use_container_width=True)

        suggestions = selected_summary.get("suggestions", []) or []
        st.write("**Sugestoes analiticas semanais**")
        if suggestions:
            for suggestion in suggestions:
                st.caption(f"- {suggestion}")
        else:
            st.caption("Sem sugestoes adicionais para a semana selecionada.")

        if not selected_rows.empty:
            weekly_export_df = selected_rows.copy()
            weekly_export_df["opened_at"] = weekly_export_df["opened_at"].astype("string")
            weekly_export_df["closed_at"] = weekly_export_df["closed_at"].astype("string")
            st.download_button(
                "Exportar consolidado semanal (CSV)",
                data=weekly_export_df.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"weekly_validation_{selected_report.get('week_key')}.csv",
                mime="text/csv",
                use_container_width=False,
                key=f"download_weekly_validation_{selected_report.get('week_key')}",
            )
            with st.expander("Trades da semana"):
                st.dataframe(selected_rows.sort_values("closed_at", ascending=False), use_container_width=True)
        else:
            st.info("A semana selecionada nao possui trades fechados; o resumo usa logs e saude operacional.")

with t4:
    if not logs.empty:
        st.dataframe(logs.iloc[::-1], use_container_width=True)
    else:
        st.info("Sem logs do bot.")
