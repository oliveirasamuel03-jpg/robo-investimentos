from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_engine import load_data
from ml_engine import MLEngineConfig, EnsembleProbabilityModel, build_feature_panel
from strategy_engine import (
    StrategyConfig,
    combined_market_filter,
    generate_target_weights,
)
from portfolio_optimizer import OptimizerConfig, build_rolling_optimized_weights
from walk_forward import WalkForwardConfig, run_walk_forward_validation
from metrics import compute_all_metrics
from monte_carlo import MonteCarloConfig, run_monte_carlo
from report_generator import build_institutional_report, save_report


st.set_page_config(
    page_title="Invest Pro Bot - Institutional Quant Research",
    layout="wide",
)


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
def cached_load_data(start_date: str, fast_mode: bool) -> pd.DataFrame:
    history_limit = 450 if fast_mode else 900
    return load_data(start=start_date, history_limit=history_limit)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_build_feature_panel(prices: pd.DataFrame) -> pd.DataFrame:
    ml_config = MLEngineConfig()
    return build_feature_panel(prices, config=ml_config)


def run_pipeline(
    start_date: str,
    initial_capital: float,
    probability_threshold: float,
    top_n: int,
    optimizer_method: str,
    execution_delay: int,
    fast_mode: bool,
    use_regime_filter: bool,
    use_volatility_filter: bool,
    vol_threshold: float,
):
    progress = st.progress(0, text="Iniciando pipeline institucional...")

    progress.progress(10, text="Carregando dados de mercado...")
    prices = cached_load_data(start_date=start_date, fast_mode=fast_mode)

    progress.progress(25, text="Construindo features quantitativas...")
    feature_panel = cached_build_feature_panel(prices)

    ml_config = MLEngineConfig()

    strategy_config = StrategyConfig(
        probability_threshold=probability_threshold,
        top_n=top_n,
        max_weight_per_asset=0.40,
        weight_method="score",
        long_only=True,
        min_assets=1,
    )

    optimizer_config = OptimizerConfig(
        method=optimizer_method,
        lookback_window=40 if fast_mode else 60,
        max_weight=0.40,
        min_weight=0.0,
        target_gross_exposure=1.0,
    )

    if fast_mode:
        wf_config = WalkForwardConfig(
            train_window=252,
            test_window=15,
            step_size=15,
            min_assets=3,
            initial_capital=initial_capital,
            fee_rate=0.0005,
            slippage_rate=0.0005,
            execution_delay=execution_delay,
        )
        mc_config = MonteCarloConfig(
            n_simulations=200,
            block_size=5,
            random_state=42,
            periods_per_year=252,
        )
    else:
        wf_config = WalkForwardConfig(
            train_window=252 * 2,
            test_window=21,
            step_size=21,
            min_assets=3,
            initial_capital=initial_capital,
            fee_rate=0.0005,
            slippage_rate=0.0005,
            execution_delay=execution_delay,
        )
        mc_config = MonteCarloConfig(
            n_simulations=500,
            block_size=5,
            random_state=42,
            periods_per_year=252,
        )

    filters_df = combined_market_filter(
        prices=prices,
        benchmark="SPY",
        fast_ma_window=50,
        slow_ma_window=200,
        vol_window=20,
        vol_threshold=vol_threshold,
        use_regime_filter=use_regime_filter,
        use_volatility_filter=use_volatility_filter,
    )

    def model_factory():
        return EnsembleProbabilityModel(config=ml_config)

    def signal_builder(pred_df: pd.DataFrame) -> pd.DataFrame:
        weighted = generate_target_weights(pred_df, config=strategy_config)
        weighted["selected"] = weighted["weight"] > 0
        weighted["date"] = pd.to_datetime(weighted["date"])

        weighted = weighted.merge(
            filters_df.reset_index().rename(columns={"index": "date"}),
            on="date",
            how="left",
        )

        weighted["trade_allowed"] = weighted["trade_allowed"].fillna(False)

        weighted.loc[weighted["trade_allowed"] == False, "weight"] = 0.0
        weighted.loc[weighted["trade_allowed"] == False, "selected"] = False

        optimized_matrix = build_rolling_optimized_weights(
            prices=prices,
            selected_signals=weighted[["date", "asset", "selected", "probability", "weight"]],
            config=optimizer_config,
        )

        optimized_long = (
            optimized_matrix.reset_index()
            .melt(id_vars="date", var_name="asset", value_name="weight")
            .sort_values(["date", "asset"])
        )
        return optimized_long

    progress.progress(50, text="Executando walk-forward validation...")
    wf_result = run_walk_forward_validation(
        prices=prices,
        feature_data=feature_panel,
        model_factory=model_factory,
        signal_builder=signal_builder,
        config=wf_config,
    )

    progress.progress(72, text="Calculando métricas institucionais...")
    backtest_result = wf_result["backtest_result"]

    weight_cols = [c for c in backtest_result["portfolio_history"].columns if str(c).startswith("w_")]
    weights_df = backtest_result["portfolio_history"][weight_cols].copy()
    weights_df.columns = [c.replace("w_", "") for c in weights_df.columns]

    metrics = compute_all_metrics(
        returns=backtest_result["returns"]["return"],
        equity=backtest_result["equity_curve"]["equity"],
        weights=weights_df,
        initial_capital=initial_capital,
    )

    progress.progress(85, text="Rodando Monte Carlo...")
    mc_result = run_monte_carlo(
        returns=backtest_result["returns"]["return"],
        config=mc_config,
    )

    progress.progress(95, text="Gerando relatório institucional...")
    report = build_institutional_report(
        metrics=metrics,
        monte_carlo=mc_result,
        walk_forward=wf_result,
    )
    save_report(report)

    progress.progress(100, text="Pesquisa concluída.")
    progress.empty()

    return {
        "prices": prices,
        "feature_panel": feature_panel,
        "wf_result": wf_result,
        "backtest_result": backtest_result,
        "metrics": metrics,
        "mc_result": mc_result,
        "report": report,
        "weights_df": weights_df,
        "filters_df": filters_df,
    }


def plot_equity_curve(equity_curve: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=equity_curve.index,
            y=equity_curve["equity"],
            mode="lines",
            name="Equity",
        )
    )
    fig.update_layout(
        title="Equity Curve",
        template="plotly_dark",
        height=420,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def plot_drawdown(drawdown: pd.Series):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            mode="lines",
            fill="tozeroy",
            name="Drawdown",
        )
    )
    fig.update_layout(
        title="Drawdown",
        template="plotly_dark",
        height=320,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def plot_monte_carlo_histogram(mc_df: pd.DataFrame):
    fig = px.histogram(
        mc_df,
        x="sharpe",
        nbins=30,
        title="Distribuição de Sharpe - Monte Carlo",
    )
    fig.update_layout(
        template="plotly_dark",
        height=360,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def plot_allocation(weights_df: pd.DataFrame):
    if weights_df.empty:
        return go.Figure()

    latest = weights_df.iloc[-1]
    latest = latest[latest > 0].sort_values(ascending=False)

    if latest.empty:
        return go.Figure()

    fig = px.pie(
        names=latest.index,
        values=latest.values,
        title="Alocação Final do Portfólio",
    )
    fig.update_layout(
        template="plotly_dark",
        height=360,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def plot_strategy_comparison(metrics: dict, mc_result: dict, wf_result: dict):
    comparison = pd.DataFrame(
        {
            "Métrica": ["Sharpe", "Sortino", "Calmar", "Robustez MC", "WFE"],
            "Valor": [
                metrics.get("sharpe", 0.0),
                metrics.get("sortino", 0.0),
                metrics.get("calmar", 0.0),
                mc_result.get("robustness_score", 0.0),
                wf_result.get("degradation_analysis", {}).get("wfe_proxy", 0.0),
            ],
        }
    )

    fig = px.bar(
        comparison,
        x="Métrica",
        y="Valor",
        title="Comparação de Qualidade da Estratégia",
    )
    fig.update_layout(
        template="plotly_dark",
        height=360,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def plot_filter_states(filters_df: pd.DataFrame):
    if filters_df.empty:
        return go.Figure()

    plot_df = filters_df.copy().astype(int)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["regime"], mode="lines", name="Bull Regime"))
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["vol_filter"], mode="lines", name="Volatilidade OK"))
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["trade_allowed"], mode="lines", name="Operar Permitido"))

    fig.update_layout(
        title="Estados dos Filtros de Mercado",
        template="plotly_dark",
        height=320,
        margin=dict(l=20, r=20, t=50, b=20),
        yaxis=dict(tickmode="array", tickvals=[0, 1]),
    )
    return fig


def main():
    inject_css()

    st.title("Invest Pro Bot - Sistema de Pesquisa e Negociação Quantitativa Institucional")
    st.caption("Robustez > rentabilidade | estatística > intuição | validação > pressupostos")

    st.sidebar.header("Configuração de pesquisa")

    start_date = st.sidebar.text_input("Data de início", value="2019-01-01")
    initial_capital = st.sidebar.number_input("Capital inicial", min_value=1000.0, value=10000.0, step=1000.0)
    probability_threshold = st.sidebar.slider("Limiar de probabilidade", 0.50, 0.85, 0.65, 0.01)
    top_n = st.sidebar.slider("Principais ativos N", 1, 6, 5, 1)
    optimizer_method = st.sidebar.selectbox(
        "Método de otimização",
        ["equal_weight", "risk_parity", "markowitz"],
        index=1,
        format_func=lambda x: {
            "equal_weight": "peso igual",
            "risk_parity": "paridade de risco",
            "markowitz": "markowitz",
        }[x],
    )
    execution_delay = st.sidebar.slider("Atraso de execução (barras)", 1, 3, 1, 1)
    fast_mode = st.sidebar.toggle("Modo rápido para Streamlit", value=True)

    st.sidebar.subheader("Filtros de mercado")
    use_regime_filter = st.sidebar.toggle("Filtro bull/bear", value=True)
    use_volatility_filter = st.sidebar.toggle("Filtro de volatilidade", value=True)
    vol_threshold = st.sidebar.slider("Limite de volatilidade", 0.010, 0.050, 0.025, 0.001)

    run_button = st.sidebar.button("Executar pesquisa institucional", use_container_width=True)

    if not run_button:
        st.info("Configure os parâmetros e clique em 'Executar pesquisa institucional'.")
        return

    try:
        result = run_pipeline(
            start_date=start_date,
            initial_capital=initial_capital,
            probability_threshold=probability_threshold,
            top_n=top_n,
            optimizer_method=optimizer_method,
            execution_delay=execution_delay,
            fast_mode=fast_mode,
            use_regime_filter=use_regime_filter,
            use_volatility_filter=use_volatility_filter,
            vol_threshold=vol_threshold,
        )
    except Exception as e:
        st.error(f"Erro ao executar pipeline: {e}")
        return

    backtest_result = result["backtest_result"]
    metrics = result["metrics"]
    mc_result = result["mc_result"]
    wf_result = result["wf_result"]
    weights_df = result["weights_df"]
    filters_df = result["filters_df"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sharpe", f"{metrics.get('sharpe', 0.0):.2f}")
    col2.metric("Sortino", f"{metrics.get('sortino', 0.0):.2f}")
    col3.metric("Calmar", f"{metrics.get('calmar', 0.0):.2f}")
    col4.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0.0):.2%}")

    st.plotly_chart(
        plot_equity_curve(backtest_result["equity_curve"]),
        use_container_width=True,
    )

    st.plotly_chart(
        plot_drawdown(backtest_result["drawdown"]),
        use_container_width=True,
    )

    left, right = st.columns(2)

    with left:
        st.plotly_chart(
            plot_monte_carlo_histogram(mc_result["simulations"]),
            use_container_width=True,
        )

    with right:
        st.plotly_chart(
            plot_allocation(weights_df),
            use_container_width=True,
        )

    st.subheader("Filtros de mercado")
    st.plotly_chart(
        plot_filter_states(filters_df),
        use_container_width=True,
    )

    st.subheader("Resultados do Walk-Forward")
    st.dataframe(wf_result["fold_metrics"], use_container_width=True)

    st.subheader("Painel de Comparação da Estratégia")
    st.plotly_chart(
        plot_strategy_comparison(metrics, mc_result, wf_result),
        use_container_width=True,
    )

    with st.expander("Relatório institucional"):
        st.json(result["report"])


if __name__ == "__main__":
    main()
