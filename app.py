from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_engine import load_data
from ml_engine import MLEngineConfig, EnsembleProbabilityModel, build_feature_panel
from strategy_engine import StrategyConfig, generate_target_weights
from portfolio_optimizer import OptimizerConfig, build_rolling_optimized_weights
from walk_forward import WalkForwardConfig, run_walk_forward_validation
from metrics import compute_all_metrics
from monte_carlo import MonteCarloConfig, run_monte_carlo
from report_generator import build_institutional_report, save_report


st.set_page_config(
    page_title="Invest Pro Bot - Institutional Quant Research",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def run_pipeline(
    start_date: str,
    initial_capital: float,
    probability_threshold: float,
    top_n: int,
    optimizer_method: str,
    execution_delay: int,
):
    prices = load_data(start=start_date)

    ml_config = MLEngineConfig()
    feature_panel = build_feature_panel(prices, config=ml_config)

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
        lookback_window=60,
        max_weight=0.40,
        min_weight=0.0,
        target_gross_exposure=1.0,
    )

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

    def model_factory():
        return EnsembleProbabilityModel(config=ml_config)

    def signal_builder(pred_df: pd.DataFrame) -> pd.DataFrame:
        weighted = generate_target_weights(pred_df, config=strategy_config)
        weighted["selected"] = weighted["weight"] > 0

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

    wf_result = run_walk_forward_validation(
        prices=prices,
        feature_data=feature_panel,
        model_factory=model_factory,
        signal_builder=signal_builder,
        config=wf_config,
    )

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

    mc_result = run_monte_carlo(
        returns=backtest_result["returns"]["return"],
        config=MonteCarloConfig(
            n_simulations=1000,
            block_size=5,
            random_state=42,
            periods_per_year=252,
        ),
    )

    report = build_institutional_report(
        metrics=metrics,
        monte_carlo=mc_result,
        walk_forward=wf_result,
    )
    save_report(report)

    return {
        "prices": prices,
        "feature_panel": feature_panel,
        "wf_result": wf_result,
        "backtest_result": backtest_result,
        "metrics": metrics,
        "mc_result": mc_result,
        "report": report,
        "weights_df": weights_df,
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
        nbins=40,
        title="Monte Carlo Sharpe Distribution",
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
        title="Latest Portfolio Allocation",
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
            "Metric": ["Sharpe", "Sortino", "Calmar", "MC Robustness", "WFE"],
            "Value": [
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
        x="Metric",
        y="Value",
        title="Strategy Quality Comparison",
    )
    fig.update_layout(
        template="plotly_dark",
        height=360,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def main():
    st.title("Invest Pro Bot - Institutional Quant Research & Trading System")
    st.caption("Robustness > profitability | statistics > intuition | validation > assumptions")

    st.sidebar.header("Research Configuration")

    start_date = st.sidebar.text_input("Start Date", value="2018-01-01")
    initial_capital = st.sidebar.number_input("Initial Capital", min_value=1000.0, value=100000.0, step=1000.0)
    probability_threshold = st.sidebar.slider("Probability Threshold", 0.50, 0.80, 0.55, 0.01)
    top_n = st.sidebar.slider("Top N Assets", 1, 6, 3, 1)
    optimizer_method = st.sidebar.selectbox(
        "Optimizer Method",
        ["equal_weight", "risk_parity", "markowitz"],
        index=1,
    )
    execution_delay = st.sidebar.slider("Execution Delay (bars)", 1, 3, 1, 1)

    run_button = st.sidebar.button("Run Institutional Research")

    if not run_button:
        st.info("Configure the parameters and click 'Run Institutional Research'.")
        return

    with st.spinner("Running institutional pipeline..."):
        result = run_pipeline(
            start_date=start_date,
            initial_capital=initial_capital,
            probability_threshold=probability_threshold,
            top_n=top_n,
            optimizer_method=optimizer_method,
            execution_delay=execution_delay,
        )

    backtest_result = result["backtest_result"]
    metrics = result["metrics"]
    mc_result = result["mc_result"]
    wf_result = result["wf_result"]
    weights_df = result["weights_df"]

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

    st.subheader("Walk-Forward Results")
    st.dataframe(wf_result["fold_metrics"], use_container_width=True)

    st.subheader("Strategy Comparison Panel")
    st.plotly_chart(
        plot_strategy_comparison(metrics, mc_result, wf_result),
        use_container_width=True,
    )

    st.subheader("Institutional Report")
    st.json(result["report"])


if __name__ == "__main__":
    main()
