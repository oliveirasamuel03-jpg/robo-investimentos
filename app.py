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

    # garantir pelo menos 1 fold válido
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

    strategy = StrategyConfig(
        probability_threshold=probability_threshold,
        top_n=top_n,
        max_weight_per_asset=0.40,
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

    def model_factory():
        return EnsembleProbabilityModel(MLEngineConfig())

    def signal_builder(pred: pd.DataFrame):
        w = generate_target_weights(pred, prices, strategy).copy()
        w["date"] = pd.to_datetime(w["date"])

        w = w.merge(filters[["date", "regime", "vol_filter", "trade_allowed"]], on="date", how="left")
        w["trade_allowed"] = w["trade_allowed"].fillna(False)
        w.loc[~w["trade_allowed"], "weight"] = 0.0

        return w[["date", "asset", "weight"]]

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
            "train_window": wf_config.train_window,
            "test_window": wf_config.test_window,
            "step_size": wf_config.step_size,
            "embargo": wf_config.embargo,
            "holdout_ratio": wf_config.holdout_ratio,
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

    fig.update_layout(
        template="plotly_dark",
        height=420,
        title="Holdout Final: Strategy vs SPY vs Random",
    )
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
    fig.update_layout(
        template="plotly_dark",
        height=320,
        title="Estados dos Filtros de Mercado",
        yaxis=dict(tickmode="array", tickvals=[0, 1]),
    )
    return fig


def main():
    inject_css()

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
        return

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
        return

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

    st.caption(
        f"Walk-forward usado: train={result['wf_config_used']['train_window']} | "
        f"test={result['wf_config_used']['test_window']} | "
        f"step={result['wf_config_used']['step_size']} | "
        f"embargo={result['wf_config_used']['embargo']} | "
        f"holdout={result['wf_config_used']['holdout_ratio']:.0%}"
    )

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


if __name__ == "__main__":
    main()
