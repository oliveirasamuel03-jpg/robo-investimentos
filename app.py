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
    history_limit = 500 if fast_mode else 1200
    return load_data(start=start_date, history_limit=history_limit)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_build_feature_panel(prices: pd.DataFrame) -> pd.DataFrame:
    ml_config = MLEngineConfig()
    return build_feature_panel(prices, config=ml_config)


def _safe_reset_index_with_date(df: pd.DataFrame) -> pd.DataFrame:
    out = df.reset_index().copy()
    if "date" not in out.columns:
        first_col = out.columns[0]
        out = out.rename(columns={first_col: "date"})
    out["date"] = pd.to_datetime(out["date"])
    return out


def _build_dynamic_wf_config(
    feature_panel: pd.DataFrame,
    initial_capital: float,
    execution_delay: int,
    fast_mode: bool,
    holdout_ratio: float,
) -> WalkForwardConfig:
    unique_dates = pd.Index(sorted(pd.to_datetime(feature_panel["date"]).unique()))
    n_dates = len(unique_dates)

    if n_dates < 120:
        raise ValueError(
            f"Poucas datas disponíveis após limpeza ({n_dates}). "
            f"Tente uma data inicial mais antiga ou desligue o modo rápido."
        )

    if fast_mode:
        train_window = min(160, max(60, int(n_dates * 0.45)))
        test_window = min(20, max(10, int(n_dates * 0.08)))
        step_size = max(10, min(test_window, 15))
        embargo = 5
    else:
        train_window = min(252, max(120, int(n_dates * 0.50)))
        test_window = min(30, max(15, int(n_dates * 0.10)))
        step_size = max(10, min(test_window, 21))
        embargo = 5

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
        cash_threshold=cash_threshold,
        target_portfolio_vol=target_portfolio_vol,
    )

    wf_config = _build_dynamic_wf_config(
        feature_panel=feature_panel,
        initial_capital=initial_capital,
        execution_delay=execution_delay,
        fast_mode=fast_mode,
        holdout_ratio=holdout_ratio,
    )

    mc_config = MonteCarloConfig(
        n_simulations=200 if fast_mode else 500,
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
    filters_long = _safe_reset_index_with_date(filters_df)

    def model_factory():
        return EnsembleProbabilityModel(config=ml_config)

    def signal_builder(pred_df: pd.DataFrame) -> pd.DataFrame:
        weighted = generate_target_weights(pred_df, prices=prices, config=strategy_config).copy()
        weighted["date"] = pd.to_datetime(weighted["date"])

        weighted = weighted.merge(
            filters_long[["date", "regime", "vol_filter", "trade_allowed"]],
            on="date",
            how="left",
        )

        weighted["trade_allowed"] = weighted["trade_allowed"].fillna(False)
        weighted.loc[~weighted["trade_allowed"], "weight"] = 0.0
        weighted.loc[~weighted["trade_allowed"], "selected"] = False

        return weighted[["date", "asset", "weight"]].copy()

    progress.progress(50, text="Executando walk-forward validation...")
    wf_result = run_walk_forward_validation(
        prices=prices,
        feature_data=feature_panel,
        model_factory=model_factory,
        signal_builder=signal_builder,
        config=wf_config,
    )

    progress.progress(72, text="Calculando métricas institucionais...")

    research_backtest = wf_result["backtest_result"]
    holdout_backtest = wf_result["holdout_backtest"]

    research_weights = wf_result["target_weights"].copy()
    research_metrics = compute_all_metrics(
        returns=research_backtest["returns"]["return"],
        equity=research_backtest["equity_curve"]["equity"],
        weights=research_weights,
        initial_capital=initial_capital,
    )

    holdout_metrics = wf_result["holdout_metrics"]
    spy_holdout_metrics = wf_result["holdout_spy_metrics"]
    random_holdout_metrics = wf_result["holdout_random_metrics"]

    progress.progress(85, text="Rodando Monte Carlo no holdout...")
    mc_result = run_monte_carlo(
        returns=holdout_backtest["returns"]["return"],
        config=mc_config,
    )

    progress.progress(95, text="Gerando relatório institucional...")
    report = build_institutional_report(
        metrics=research_metrics,
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
        "research_backtest": research_backtest,
        "research_metrics": research_metrics,
        "holdout_backtest": holdout_backtest,
        "holdout_metrics": holdout_metrics,
        "holdout_spy_metrics": spy_holdout_metrics,
        "holdout_random_metrics": random_holdout_metrics,
        "mc_result": mc_result,
        "report": report,
        "filters_df": filters_df,
        "wf_config_used": {
            "train_window": wf_config.train_window,
            "test_window": wf_config.test_window,
            "step_size": wf_config.step_size,
            "embargo": wf_config.embargo,
            "holdout_ratio": wf_config.holdout_ratio,
        },
    }


def plot_equity_curve(equity_curve: pd.DataFrame, title: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity_curve.index, y=equity_curve["equity"], mode="lines", name=title))
    fig.update_layout(title=title, template="plotly_dark", height=420, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def plot_drawdown(drawdown: pd.Series, title: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown.values, mode="lines", fill="tozeroy", name=title))
    fig.update_layout(title=title, template="plotly_dark", height=320, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def plot_monte_carlo_histogram(mc_df: pd.DataFrame):
    fig = px.histogram(mc_df, x="sharpe", nbins=30, title="Distribuição de Sharpe - Monte Carlo (Holdout)")
    fig.update_layout(template="plotly_dark", height=360, margin=dict(l=20, r=20, t=50, b=20))
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


def plot_holdout_comparison(strategy_backtest, spy_backtest, random_backtest):
    df = pd.DataFrame(index=strategy_backtest["equity_curve"].index)
    df["Strategy"] = strategy_backtest["equity_curve"]["equity"]
    df["SPY"] = spy_backtest["equity_curve"]["equity"].reindex(df.index).ffill()
    df["Random"] = random_backtest["equity_curve"]["equity"].reindex(df.index).ffill()

    fig = go.Figure()
    for col in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines", name=col))

    fig.update_layout(
        title="Holdout Final: Strategy vs SPY vs Random",
        template="plotly_dark",
        height=420,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def main():
    inject_css()

    st.title("Invest Pro Bot - Sistema de Pesquisa e Negociação Quantitativa Institucional")
    st.caption("Robustez > rentabilidade | estatística > intuição | validação > pressupostos")

    st.sidebar.header("Configuração de pesquisa")

    start_date = st.sidebar.text_input("Data de início", value="2019-01-01")
    initial_capital = st.sidebar.number_input("Capital inicial", min_value=1000.0, value=10000.0, step=1000.0)
    probability_threshold = st.sidebar.slider("Limiar de probabilidade", 0.50, 0.85, 0.60, 0.01)
    top_n = st.sidebar.slider("Principais ativos N", 1, 6, 2, 1)
    execution_delay = st.sidebar.slider("Atraso de execução (barras)", 1, 3, 1, 1)
    fast_mode = st.sidebar.toggle("Modo rápido para Streamlit", value=True)

    st.sidebar.subheader("Filtros de mercado")
    use_regime_filter = st.sidebar.toggle("Filtro bull/bear", value=True)
    use_volatility_filter = st.sidebar.toggle("Filtro de volatilidade", value=False)
    vol_threshold = st.sidebar.slider("Limite de volatilidade", 0.010, 0.050, 0.025, 0.001)

    st.sidebar.subheader("Controles de convicção")
    cash_threshold = st.sidebar.slider("Threshold para ficar em caixa", 0.50, 0.85, 0.50, 0.01)
    target_portfolio_vol = st.sidebar.slider("Vol alvo do portfólio", 0.05, 0.30, 0.08, 0.01)

    st.sidebar.subheader("Validação final")
    holdout_ratio = st.sidebar.slider("Parcela final fora da amostra", 0.10, 0.40, 0.20, 0.05)

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
            execution_delay=execution_delay,
            fast_mode=fast_mode,
            use_regime_filter=use_regime_filter,
            use_volatility_filter=use_volatility_filter,
            vol_threshold=vol_threshold,
            cash_threshold=cash_threshold,
            target_portfolio_vol=target_portfolio_vol,
            holdout_ratio=holdout_ratio,
        )
    except Exception as e:
        st.error(f"Erro ao executar pipeline: {e}")
        return

    research_metrics = result["research_metrics"]
    holdout_metrics = result["holdout_metrics"]
    spy_holdout_metrics = result["holdout_spy_metrics"]
    random_holdout_metrics = result["holdout_random_metrics"]

    st.subheader("Métricas da pesquisa (walk-forward)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sharpe Pesquisa", f"{research_metrics.get('sharpe', 0.0):.2f}")
    c2.metric("Sortino Pesquisa", f"{research_metrics.get('sortino', 0.0):.2f}")
    c3.metric("Calmar Pesquisa", f"{research_metrics.get('calmar', 0.0):.2f}")
    c4.metric("Max DD Pesquisa", f"{research_metrics.get('max_drawdown', 0.0):.2%}")

    st.subheader("Métricas do holdout final")
    h1, h2, h3 = st.columns(3)
    h1.metric("Strategy Sharpe", f"{holdout_metrics.get('sharpe', 0.0):.2f}")
    h2.metric("SPY Sharpe", f"{spy_holdout_metrics.get('sharpe', 0.0):.2f}")
    h3.metric("Random Sharpe", f"{random_holdout_metrics.get('sharpe', 0.0):.2f}")

    st.caption(
        f"Walk-forward usado: train={result['wf_config_used']['train_window']} | "
        f"test={result['wf_config_used']['test_window']} | "
        f"step={result['wf_config_used']['step_size']} | "
        f"embargo={result['wf_config_used']['embargo']} | "
        f"holdout={result['wf_config_used']['holdout_ratio']:.0%}"
    )

    st.plotly_chart(
        plot_holdout_comparison(
            result["holdout_backtest"],
            result["wf_result"]["holdout_spy_backtest"],
            result["wf_result"]["holdout_random_backtest"],
        ),
        use_container_width=True,
    )

    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            plot_equity_curve(result["research_backtest"]["equity_curve"], "Equity Curve - Pesquisa"),
            use_container_width=True,
        )
    with right:
        st.plotly_chart(
            plot_equity_curve(result["holdout_backtest"]["equity_curve"], "Equity Curve - Holdout"),
            use_container_width=True,
        )

    left2, right2 = st.columns(2)
    with left2:
        st.plotly_chart(
            plot_drawdown(result["research_backtest"]["drawdown"], "Drawdown - Pesquisa"),
            use_container_width=True,
        )
    with right2:
        st.plotly_chart(
            plot_drawdown(result["holdout_backtest"]["drawdown"], "Drawdown - Holdout"),
            use_container_width=True,
        )

    st.plotly_chart(plot_monte_carlo_histogram(result["mc_result"]["simulations"]), use_container_width=True)

    st.subheader("Filtros de mercado")
    st.plotly_chart(plot_filter_states(result["filters_df"]), use_container_width=True)

    st.subheader("Resultados do Walk-Forward")
    st.dataframe(result["wf_result"]["fold_metrics"], use_container_width=True)

    with st.expander("Relatório institucional"):
        st.json(result["report"])


if __name__ == "__main__":
    main()
