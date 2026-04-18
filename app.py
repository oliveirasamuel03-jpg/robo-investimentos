from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_engine import load_data
from ml_engine import MLEngineConfig, EnsembleProbabilityModel, build_feature_panel
from strategy_engine import StrategyConfig, combined_market_filter, generate_target_weights
from walk_forward import WalkForwardConfig, run_walk_forward_validation
from metrics import compute_all_metrics
from monte_carlo import MonteCarloConfig, run_monte_carlo
from report_generator import build_institutional_report, save_report
from risk_engine import RiskConfig, apply_portfolio_risk_overlay
from paper_trading_engine import (
    PaperTradingConfig,
    build_paper_report,
    ensure_paper_files,
    load_paper_state,
    read_paper_equity,
    read_paper_trades,
    reset_paper_state,
    run_paper_cycle,
)


st.set_page_config(page_title="Invest Pro Bot", layout="wide")


def inject_css() -> None:
    st.markdown(
        """
        <style>
            .stApp {
                background: linear-gradient(180deg, #02070d 0%, #06111a 45%, #041019 100%);
            }
            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, rgba(8, 14, 24, 0.98), rgba(10, 20, 32, 0.98));
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=3600, show_spinner=False)
def cached_load_data(start_date: str, fast_mode: bool):
    history_limit = 500 if fast_mode else 1500
    return load_data(start=start_date, history_limit=history_limit)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_features(prices: pd.DataFrame):
    return build_feature_panel(prices, MLEngineConfig())


def _safe_reset_index_with_date(df: pd.DataFrame) -> pd.DataFrame:
    out = df.reset_index().copy()
    if "date" not in out.columns:
        out = out.rename(columns={out.columns[0]: "date"})
    out["date"] = pd.to_datetime(out["date"])
    return out


def _build_dynamic_wf_config(
    feature_panel: pd.DataFrame,
    initial_capital: float,
    execution_delay: int,
    holdout_ratio: float,
) -> WalkForwardConfig:
    unique_dates = pd.Index(sorted(pd.to_datetime(feature_panel["date"]).unique()))
    n_dates = len(unique_dates)

    if n_dates < 120:
        raise ValueError(
            f"Poucas datas úteis após limpeza ({n_dates}). "
            f"Desligue o modo rápido ou use uma data inicial mais antiga."
        )

    holdout_size = max(20, int(n_dates * holdout_ratio))
    research_size = n_dates - holdout_size

    if research_size < 60:
        raise ValueError(
            f"Com holdout de {holdout_ratio:.0%}, sobraram poucas datas para pesquisa. "
            f"Reduza o holdout ou desligue o modo rápido."
        )

    train_window = max(40, min(160, int(research_size * 0.55)))
    test_window = max(10, min(20, int(research_size * 0.12)))
    step_size = max(5, min(15, test_window))
    embargo = 5

    while train_window + embargo + test_window + 5 >= research_size and train_window > 30:
        train_window -= 10
    while train_window + embargo + test_window + 5 >= research_size and test_window > 5:
        test_window -= 5

    if train_window + embargo + test_window + 5 >= research_size:
        raise ValueError(
            f"Não há dados suficientes para walk-forward com holdout de {holdout_ratio:.0%}. "
            f"Reduza o holdout ou desligue o modo rápido."
        )

    return WalkForwardConfig(
        train_window=train_window,
        test_window=test_window,
        step_size=step_size,
        embargo=embargo,
        min_assets=3,
        initial_capital=initial_capital,
        fee_rate=0.0005,
        slippage_rate=0.0005,
        execution_delay=execution_delay,
        holdout_ratio=holdout_ratio,
    )


def run_pipeline(
    start_date: str,
    initial_capital: float,
    probability_threshold: float,
    top_n: int,
    execution_delay: int,
    fast_mode: bool,
    use_regime_filter: bool,
    use_volatility_filter: bool,
    vol_threshold: float,
    cash_threshold: float,
    target_portfolio_vol: float,
    holdout_ratio: float,
):
    prices = cached_load_data(start_date, fast_mode)
    features = cached_features(prices)

    strategy_config = StrategyConfig(
        probability_threshold=probability_threshold,
        top_n=top_n,
        max_weight_per_asset=0.35,
        cash_threshold=cash_threshold,
        target_portfolio_vol=target_portfolio_vol,
    )

    wf_config = _build_dynamic_wf_config(
        feature_panel=features,
        initial_capital=initial_capital,
        execution_delay=execution_delay,
        holdout_ratio=holdout_ratio,
    )

    filters = combined_market_filter(
        prices,
        use_regime_filter=use_regime_filter,
        use_volatility_filter=use_volatility_filter,
        vol_threshold=vol_threshold,
    )
    filters = _safe_reset_index_with_date(filters)

    risk_config = RiskConfig(
        max_weight_per_asset=0.35,
        max_gross_exposure=1.0,
        max_net_exposure=1.0,
        max_drawdown_circuit_breaker=0.12,
        recovery_exposure=0.35,
        min_nav_fraction=0.70,
    )

    def model_factory():
        return EnsembleProbabilityModel(MLEngineConfig())

    def signal_builder(pred_df: pd.DataFrame) -> pd.DataFrame:
        weighted = generate_target_weights(
            pred_df,
            prices=prices,
            config=strategy_config,
        ).copy()

        weighted["date"] = pd.to_datetime(weighted["date"])

        weighted = weighted.merge(
            filters[["date", "regime", "vol_filter", "trade_allowed"]],
            on="date",
            how="left",
        )

        weighted["trade_allowed"] = weighted["trade_allowed"].fillna(False)
        weighted.loc[~weighted["trade_allowed"], "weight"] = 0.0

        out_rows = []

        for _, group in weighted.groupby("date", sort=True):
            base_weights = group.set_index("asset")["weight"]

            safe_weights = apply_portfolio_risk_overlay(
                base_weights,
                equity_curve=None,
                config=risk_config,
            )

            tmp = group.copy()
            tmp["weight"] = tmp["asset"].map(safe_weights).fillna(0.0)

            out_rows.append(tmp[["date", "asset", "weight"]])

        return pd.concat(out_rows, ignore_index=True)

    wf = run_walk_forward_validation(
        prices,
        features,
        model_factory,
        signal_builder,
        wf_config,
    )

    research_bt = wf["backtest_result"]
    holdout_bt = wf["holdout_backtest"]

    research_metrics = compute_all_metrics(
        research_bt["returns"]["return"],
        research_bt["equity_curve"]["equity"],
    )

    holdout_metrics = wf["holdout_metrics"]

    mc = run_monte_carlo(
        holdout_bt["returns"]["return"],
        MonteCarloConfig(),
    )

    report = build_institutional_report(research_metrics, mc, wf)
    save_report(report)

    return {
        "wf": wf,
        "research_bt": research_bt,
        "holdout_bt": holdout_bt,
        "research_metrics": research_metrics,
        "holdout_metrics": holdout_metrics,
        "mc": mc,
        "report": report,
        "wf_config_used": {
            "train_window": wf["fold_metrics"]["n_train_rows"].iloc[0] if not wf["fold_metrics"].empty else None,
            "test_window": None,
            "step_size": None,
            "embargo": None,
            "holdout_ratio": holdout_ratio,
        },
        "filters": filters,
    }


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
    fig.update_layout(template="plotly_dark", height=360, title=title)
    return fig


def plot_drawdown(backtest_result, title: str):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=backtest_result["drawdown"].index,
            y=backtest_result["drawdown"].values,
            mode="lines",
            fill="tozeroy",
            name=title,
        )
    )
    fig.update_layout(template="plotly_dark", height=320, title=title)
    return fig


def plot_holdout_comparison(strategy_bt, spy_bt, random_bt):
    df = pd.DataFrame(index=strategy_bt["equity_curve"].index)
    df["Strategy"] = strategy_bt["equity_curve"]["equity"]
    df["SPY"] = spy_bt["equity_curve"]["equity"].reindex(df.index).ffill()
    df["Random"] = random_bt["equity_curve"]["equity"].reindex(df.index).ffill()

    fig = go.Figure()
    for col in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines", name=col))

    fig.update_layout(template="plotly_dark", height=420, title="Holdout Final: Strategy vs SPY vs Random")
    return fig


def plot_filter_states(filters_df: pd.DataFrame):
    if filters_df.empty:
        return go.Figure()

    plot_df = filters_df.copy()
    for col in ["regime", "vol_filter", "trade_allowed"]:
        plot_df[col] = plot_df[col].astype(int)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_df["date"], y=plot_df["regime"], mode="lines", name="Bull Regime"))
    fig.add_trace(go.Scatter(x=plot_df["date"], y=plot_df["vol_filter"], mode="lines", name="Volatilidade OK"))
    fig.add_trace(go.Scatter(x=plot_df["date"], y=plot_df["trade_allowed"], mode="lines", name="Operar Permitido"))
    fig.update_layout(template="plotly_dark", height=320, title="Estados dos Filtros de Mercado", yaxis=dict(tickmode="array", tickvals=[0, 1]))
    return fig


def plot_paper_equity(paper_equity_df: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=paper_equity_df["timestamp"],
            y=paper_equity_df["equity_after"],
            mode="lines+markers",
            name="Paper Equity",
        )
    )
    fig.update_layout(template="plotly_dark", height=320, title="Paper Trading Equity")
    return fig


def main():
    inject_css()
    ensure_paper_files()

    st.title("Invest Pro Bot - Sistema de Pesquisa e Negociação Quantitativa Institucional")
    st.caption("Robustez > rentabilidade | estatística > intuição | validação > pressupostos")

    st.sidebar.header("Configuração de pesquisa")

    start_date = st.sidebar.text_input("Data de início", "2019-01-01")
    capital = st.sidebar.number_input("Capital inicial", min_value=1000.0, value=10000.0, step=1000.0)
    threshold = st.sidebar.slider("Limiar de probabilidade", 0.50, 0.85, 0.60, 0.01)
    top_n = st.sidebar.slider("Principais ativos N", 1, 5, 2, 1)
    delay = st.sidebar.slider("Atraso de execução (barras)", 1, 3, 1, 1)

    fast_mode = st.sidebar.toggle("Modo rápido para Streamlit", True)

    st.sidebar.subheader("Filtros de mercado")
    bull_filter = st.sidebar.toggle("Filtro bull/bear", True)
    vol_filter = st.sidebar.toggle("Filtro de volatilidade", False)
    vol_limit = st.sidebar.slider("Limite de volatilidade", 0.01, 0.05, 0.03, 0.001)

    st.sidebar.subheader("Controles de convicção")
    cash_threshold = st.sidebar.slider("Threshold para ficar em caixa", 0.50, 0.85, 0.50, 0.01)
    target_vol = st.sidebar.slider("Vol alvo do portfólio", 0.05, 0.20, 0.08, 0.01)

    st.sidebar.subheader("Validação final")
    holdout_ratio = st.sidebar.slider("Parcela final fora da amostra", 0.10, 0.60, 0.20, 0.05)

    run = st.sidebar.button("Executar pesquisa institucional", use_container_width=True)

    if not run:
        st.info("Configure e execute.")
    else:
        try:
            result = run_pipeline(
                start_date,
                capital,
                threshold,
                top_n,
                delay,
                fast_mode,
                bull_filter,
                vol_filter,
                vol_limit,
                cash_threshold,
                target_vol,
                holdout_ratio,
            )
        except Exception as e:
            st.error(f"Erro: {e}")
            result = None

        if result is not None:
            wf = result["wf"]
            research_metrics = result["research_metrics"]
            holdout_metrics = result["holdout_metrics"]

            st.subheader("Métricas da pesquisa (walk-forward)")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Sharpe Pesquisa", f"{research_metrics['sharpe']:.2f}")
            c2.metric("Sortino Pesquisa", f"{research_metrics['sortino']:.2f}")
            c3.metric("Calmar Pesquisa", f"{research_metrics['calmar']:.2f}")
            c4.metric("Max DD Pesquisa", f"{research_metrics['max_drawdown']:.2%}")

            st.subheader("Métricas do holdout final")
            h1, h2, h3 = st.columns(3)
            h1.metric("Strategy Sharpe", f"{holdout_metrics['sharpe']:.2f}")
            h2.metric("SPY Sharpe", f"{wf['holdout_spy_metrics']['sharpe']:.2f}")
            h3.metric("Random Sharpe", f"{wf['holdout_random_metrics']['sharpe']:.2f}")

            st.plotly_chart(
                plot_holdout_comparison(
                    result["holdout_bt"],
                    wf["holdout_spy_backtest"],
                    wf["holdout_random_backtest"],
                ),
                use_container_width=True,
            )

            left, right = st.columns(2)
            with left:
                st.plotly_chart(plot_equity(result["research_bt"], "Equity Curve - Pesquisa"), use_container_width=True)
            with right:
                st.plotly_chart(plot_equity(result["holdout_bt"], "Equity Curve - Holdout"), use_container_width=True)

            left2, right2 = st.columns(2)
            with left2:
                st.plotly_chart(plot_drawdown(result["research_bt"], "Drawdown - Pesquisa"), use_container_width=True)
            with right2:
                st.plotly_chart(plot_drawdown(result["holdout_bt"], "Drawdown - Holdout"), use_container_width=True)

            st.plotly_chart(
                px.histogram(result["mc"]["simulations"], x="sharpe", title="Distribuição de Sharpe - Monte Carlo (Holdout)"),
                use_container_width=True,
            )

            st.plotly_chart(plot_filter_states(result["filters"]), use_container_width=True)
            st.dataframe(wf["fold_metrics"], use_container_width=True)

            with st.expander("Relatório institucional"):
                st.json(result["report"])

    st.divider()
    st.subheader("Paper Trading")

    paper_cfg = PaperTradingConfig(
        start_date=start_date,
        initial_capital=capital,
        probability_threshold=threshold,
        top_n=top_n,
        use_regime_filter=bull_filter,
        use_volatility_filter=vol_filter,
        vol_threshold=vol_limit,
        cash_threshold=cash_threshold,
        target_portfolio_vol=target_vol,
    )

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Rodar ciclo de paper trading", use_container_width=True):
            try:
                cycle_result = run_paper_cycle(paper_cfg)
                st.success(f"Ciclo executado com sucesso. Trades: {cycle_result['trades_executed']}")
            except Exception as e:
                st.error(f"Erro no paper trading: {e}")

    with col_b:
        if st.button("Resetar paper trading", use_container_width=True):
            reset_paper_state(paper_cfg)
            st.warning("Paper trading resetado.")

    state = load_paper_state()
    paper_equity_df = read_paper_equity(limit=300)
    paper_trades = read_paper_trades(limit=200)
    paper_report = build_paper_report(initial_capital=capital)

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Cash", f"{state.get('cash', 0.0):,.2f}")
    p2.metric("Equity", f"{state.get('equity', 0.0):,.2f}")
    p3.metric("Posições", f"{len(state.get('positions', {}))}")
    p4.metric("Runs", f"{state.get('run_count', 0)}")

    if not paper_equity_df.empty:
        st.plotly_chart(plot_paper_equity(paper_equity_df), use_container_width=True)

    st.markdown("### Relatório automático do paper trading")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Retorno %", f"{paper_report['return_pct']:.2f}%")
    r2.metric("Max Drawdown %", f"{paper_report['max_drawdown_pct']:.2f}%")
    r3.metric("Trades", f"{paper_report['trades_count']}")
    r4.metric("Trades fechados", f"{paper_report['closed_trades_count']}")

    r5, r6, r7, r8 = st.columns(4)
    r5.metric("Win Rate %", f"{paper_report['win_rate_pct']:.2f}%")
    r6.metric("Payoff", f"{paper_report['payoff_ratio']:.2f}")
    r7.metric("PnL realizado", f"{paper_report['realized_pnl']:.2f}")
    r8.metric("Ativo mais operado", f"{paper_report['most_traded_asset'] or '-'}")

    left3, right3 = st.columns(2)

    with left3:
        st.markdown("### Posições atuais")
        positions = state.get("positions", {}) or {}
        if positions:
            rows = []
            last_prices = state.get("last_prices", {}) or {}
            for asset, pos in positions.items():
                qty = float(pos.get("quantity", 0.0))
                px = float(last_prices.get(asset, pos.get("last_price", 0.0)))
                avg = float(pos.get("avg_price", px))
                rows.append(
                    {
                        "asset": asset,
                        "quantity": qty,
                        "avg_price": avg,
                        "last_price": px,
                        "market_value": qty * px,
                        "unrealized_pnl": (px - avg) * qty,
                    }
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.info("Sem posições abertas no paper trading.")

    with right3:
        st.markdown("### Últimos trades")
        if paper_trades:
            st.dataframe(pd.DataFrame(paper_trades[::-1]), use_container_width=True)
        else:
            st.info("Sem trades ainda.")

    st.markdown("### Diagnóstico automático")
    if paper_report["diagnosis"]:
        for item in paper_report["diagnosis"]:
            st.write(f"- {item}")
    else:
        st.info("Ainda não há dados suficientes para diagnóstico.")

    with st.expander("Estado bruto do paper trading"):
        st.json(state)


if __name__ == "__main__":
    main()
