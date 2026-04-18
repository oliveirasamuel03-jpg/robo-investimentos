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


@st.cache_data
def cached_load_data(start_date: str, fast_mode: bool):
    history_limit = 500 if fast_mode else 1500
    return load_data(start=start_date, history_limit=history_limit)


@st.cache_data
def cached_features(prices):
    return build_feature_panel(prices, MLEngineConfig())


def run_pipeline(
    start_date,
    capital,
    threshold,
    top_n,
    execution_delay,
    fast_mode,
    bull_filter,
    vol_filter,
    vol_limit,
    cash_threshold,
    target_vol,
    holdout_ratio,
):
    prices = cached_load_data(start_date, fast_mode)
    features = cached_features(prices)

    strategy = StrategyConfig(
        probability_threshold=threshold,
        top_n=top_n,
        cash_threshold=cash_threshold,
        target_portfolio_vol=target_vol,
    )

    wf_config = WalkForwardConfig(
        initial_capital=capital,
        execution_delay=execution_delay,
        holdout_ratio=holdout_ratio,
    )

    filters = combined_market_filter(
        prices,
        use_regime_filter=bull_filter,
        use_volatility_filter=vol_filter,
        vol_threshold=vol_limit,
    )

    filters = filters.reset_index().rename(columns={"index": "date"})

    def model_factory():
        return EnsembleProbabilityModel(MLEngineConfig())

    def signal_builder(pred):
        w = generate_target_weights(pred, prices, strategy)
        w["date"] = pd.to_datetime(w["date"])

        w = w.merge(filters, on="date", how="left")
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

    bt = wf["backtest_result"]
    holdout_bt = wf["holdout_backtest"]

    metrics = compute_all_metrics(
        bt["returns"]["return"],
        bt["equity_curve"]["equity"],
    )

    holdout_metrics = wf["holdout_metrics"]

    mc = run_monte_carlo(
        holdout_bt["returns"]["return"],
        MonteCarloConfig(),
    )

    report = build_institutional_report(metrics, mc, wf)
    save_report(report)

    return wf, bt, holdout_bt, metrics, holdout_metrics, mc


# ================= UI =================

st.title("Invest Pro Bot - Quant Institutional")

st.sidebar.header("Configuração")

start = st.sidebar.text_input("Data de início", "2019-01-01")
capital = st.sidebar.number_input("Capital", 1000.0, value=10000.0)

threshold = st.sidebar.slider("Limiar", 0.5, 0.8, 0.6)
top_n = st.sidebar.slider("Top N", 1, 5, 2)
delay = st.sidebar.slider("Delay", 1, 3, 1)

fast = st.sidebar.toggle("Modo rápido", True)

st.sidebar.subheader("Filtros")
bull = st.sidebar.toggle("Bull/Bear", True)
vol = st.sidebar.toggle("Volatilidade", False)
vol_limit = st.sidebar.slider("Vol limite", 0.01, 0.05, 0.03)

st.sidebar.subheader("Convicção")
cash = st.sidebar.slider("Cash threshold", 0.5, 0.8, 0.5)
target_vol = st.sidebar.slider("Vol alvo", 0.05, 0.2, 0.08)

st.sidebar.subheader("Validação FINAL")

# 🔥 AQUI ESTÁ A ALTERAÇÃO IMPORTANTE
holdout = st.sidebar.slider(
    "Parcela fora da amostra",
    0.10,
    0.60,  # ← AGORA VAI ATÉ 60%
    0.20,
    0.05
)

run = st.sidebar.button("Executar")

if not run:
    st.info("Configure e execute")
    st.stop()

try:
    wf, bt, holdout_bt, metrics, holdout_metrics, mc = run_pipeline(
        start,
        capital,
        threshold,
        top_n,
        delay,
        fast,
        bull,
        vol,
        vol_limit,
        cash,
        target_vol,
        holdout,
    )
except Exception as e:
    st.error(f"Erro: {e}")
    st.stop()

# ================= RESULTADOS =================

st.subheader("Pesquisa")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Sharpe", f"{metrics['sharpe']:.2f}")
c2.metric("Sortino", f"{metrics['sortino']:.2f}")
c3.metric("Calmar", f"{metrics['calmar']:.2f}")
c4.metric("Drawdown", f"{metrics['max_drawdown']:.2%}")

st.subheader("Holdout FINAL")

h1, h2, h3 = st.columns(3)
h1.metric("Strategy", f"{holdout_metrics['sharpe']:.2f}")
h2.metric("SPY", f"{wf['holdout_spy_metrics']['sharpe']:.2f}")
h3.metric("Random", f"{wf['holdout_random_metrics']['sharpe']:.2f}")

fig = go.Figure()
fig.add_trace(go.Scatter(x=holdout_bt["equity_curve"].index, y=holdout_bt["equity_curve"]["equity"], name="Strategy"))
fig.add_trace(go.Scatter(x=wf["holdout_spy_backtest"]["equity_curve"].index,
                         y=wf["holdout_spy_backtest"]["equity_curve"]["equity"], name="SPY"))
fig.add_trace(go.Scatter(x=wf["holdout_random_backtest"]["equity_curve"].index,
                         y=wf["holdout_random_backtest"]["equity_curve"]["equity"], name="Random"))

fig.update_layout(template="plotly_dark", height=400)
st.plotly_chart(fig, use_container_width=True)

st.plotly_chart(
    px.histogram(mc["simulations"], x="sharpe", title="Monte Carlo Sharpe"),
    use_container_width=True
)

st.dataframe(wf["fold_metrics"])
