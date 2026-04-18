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
            :root {
                --bg-main: #020611;
                --bg-soft: #071120;
                --card-bg: rgba(8, 16, 32, 0.88);
                --card-bg-2: rgba(10, 20, 38, 0.92);
                --line: rgba(0, 245, 255, 0.22);
                --cyan: #00f5ff;
                --blue: #3b82f6;
                --purple: #8b5cf6;
                --pink: #ff2bd6;
                --green: #19f5c1;
                --text: #eaf6ff;
                --muted: #8aa0b8;
                --warning: #ffb703;
            }

            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(0,245,255,0.12), transparent 28%),
                    radial-gradient(circle at top right, rgba(139,92,246,0.12), transparent 26%),
                    radial-gradient(circle at bottom center, rgba(255,43,214,0.08), transparent 22%),
                    linear-gradient(180deg, #01040b 0%, #020611 45%, #04101d 100%);
                color: var(--text);
            }

            [data-testid="stSidebar"] {
                background:
                    linear-gradient(180deg, rgba(4,10,18,0.98), rgba(5,12,24,0.99));
                border-right: 1px solid rgba(0,245,255,0.12);
            }

            .block-container {
                max-width: 1450px;
                padding-top: 1.0rem;
                padding-bottom: 2rem;
            }

            h1, h2, h3, h4, h5 {
                color: var(--text) !important;
                letter-spacing: -0.02em;
            }

            p, div, span, label {
                color: var(--text);
            }

            .hero-box {
                position: relative;
                overflow: hidden;
                padding: 1.6rem 1.8rem;
                border-radius: 22px;
                background:
                    linear-gradient(135deg, rgba(7,18,34,0.96), rgba(10,24,44,0.85));
                border: 1px solid rgba(0,245,255,0.15);
                box-shadow:
                    0 0 0 1px rgba(0,245,255,0.04),
                    0 0 30px rgba(0,245,255,0.08),
                    0 0 60px rgba(139,92,246,0.06);
                margin-bottom: 1rem;
            }

            .hero-box::before {
                content: "";
                position: absolute;
                top: -80px;
                right: -80px;
                width: 220px;
                height: 220px;
                background: radial-gradient(circle, rgba(0,245,255,0.20), transparent 60%);
                pointer-events: none;
            }

            .hero-box::after {
                content: "";
                position: absolute;
                bottom: -90px;
                left: -70px;
                width: 240px;
                height: 240px;
                background: radial-gradient(circle, rgba(255,43,214,0.12), transparent 60%);
                pointer-events: none;
            }

            .hero-title {
                font-size: 2.3rem;
                font-weight: 800;
                line-height: 1.05;
                margin-bottom: 0.45rem;
                color: #f7fbff;
            }

            .hero-sub {
                color: var(--muted);
                font-size: 0.98rem;
            }

            .section-card {
                background: linear-gradient(180deg, rgba(8,16,30,0.95), rgba(7,14,24,0.95));
                border: 1px solid rgba(0,245,255,0.10);
                border-radius: 18px;
                padding: 1rem 1rem 0.7rem 1rem;
                box-shadow:
                    0 0 0 1px rgba(0,245,255,0.03),
                    0 0 18px rgba(0,245,255,0.04);
                margin-bottom: 1rem;
            }

            .kpi-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                gap: 0.8rem;
                margin-bottom: 1rem;
            }

            .kpi-card {
                background:
                    linear-gradient(180deg, rgba(10,18,34,0.96), rgba(7,14,26,0.96));
                border: 1px solid rgba(0,245,255,0.12);
                border-radius: 18px;
                padding: 0.95rem 1rem;
                box-shadow:
                    inset 0 0 0 1px rgba(255,255,255,0.01),
                    0 0 16px rgba(0,245,255,0.05);
            }

            .kpi-label {
                color: var(--muted);
                font-size: 0.82rem;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                margin-bottom: 0.35rem;
            }

            .kpi-value {
                color: #ffffff;
                font-size: 1.65rem;
                font-weight: 800;
                text-shadow: 0 0 8px rgba(0,245,255,0.18);
            }

            .kpi-good {
                color: var(--green);
                text-shadow: 0 0 10px rgba(25,245,193,0.25);
            }

            .kpi-bad {
                color: #ff5f7e;
                text-shadow: 0 0 10px rgba(255,95,126,0.20);
            }

            .stButton > button {
                width: 100%;
                border-radius: 14px;
                border: 1px solid rgba(0,245,255,0.18);
                background: linear-gradient(135deg, rgba(12,22,40,0.98), rgba(9,16,30,0.98));
                color: #f2fbff;
                font-weight: 700;
                min-height: 2.9rem;
                box-shadow:
                    0 0 0 1px rgba(0,245,255,0.03),
                    0 0 14px rgba(0,245,255,0.06);
            }

            .stButton > button:hover {
                border-color: rgba(0,245,255,0.36);
                box-shadow:
                    0 0 0 1px rgba(0,245,255,0.05),
                    0 0 22px rgba(0,245,255,0.12);
            }

            .stDataFrame, [data-testid="stDataFrame"] {
                border-radius: 16px;
                overflow: hidden;
                border: 1px solid rgba(0,245,255,0.08);
            }

            .stAlert {
                border-radius: 14px;
                border: 1px solid rgba(0,245,255,0.08);
            }

            hr {
                border-color: rgba(0,245,255,0.08);
            }

            .tiny-note {
                color: var(--muted);
                font-size: 0.9rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero-box">
            <div class="hero-title">Invest Pro Bot</div>
            <div class="hero-sub">
                Sistema de Pesquisa e Negociação Quantitativa Institucional ·
                visual premium neon · robustez &gt; rentabilidade · validação &gt; suposição
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_row(items: list[tuple[str, str, str]]) -> None:
    cards = []
    for label, value, css_class in items:
        cards.append(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value {css_class}">{value}</div>
            </div>
            """
        )
    st.markdown(f'<div class="kpi-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


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
    fig.update_layout(
        template="plotly_dark",
        height=360,
        title=title,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
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
    fig.update_layout(
        template="plotly_dark",
        height=320,
        title=title,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
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
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
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
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
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
    fig.update_layout(
        template="plotly_dark",
        height=320,
        title="Paper Trading Equity",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def main():
    inject_css()
    ensure_paper_files()

    render_hero()

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
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.info("Configure os parâmetros e execute a pesquisa institucional.")
        st.markdown("</div>", unsafe_allow_html=True)
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
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.error(f"Erro: {e}")
            st.markdown("</div>", unsafe_allow_html=True)
            result = None

        if result is not None:
            wf = result["wf"]
            research_metrics = result["research_metrics"]
            holdout_metrics = result["holdout_metrics"]

            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("Pesquisa quantitativa")
            render_kpi_row(
                [
                    ("Sharpe Pesquisa", f"{research_metrics['sharpe']:.2f}", "kpi-good"),
                    ("Sortino Pesquisa", f"{research_metrics['sortino']:.2f}", "kpi-good"),
                    ("Calmar Pesquisa", f"{research_metrics['calmar']:.2f}", "kpi-good"),
                    ("Max DD Pesquisa", f"{research_metrics['max_drawdown']:.2%}", "kpi-bad"),
                ]
            )
            render_kpi_row(
                [
                    ("Strategy Sharpe", f"{holdout_metrics['sharpe']:.2f}", "kpi-good"),
                    ("SPY Sharpe", f"{wf['holdout_spy_metrics']['sharpe']:.2f}", "kpi-good"),
                    ("Random Sharpe", f"{wf['holdout_random_metrics']['sharpe']:.2f}", "kpi-bad"),
                ]
            )
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.plotly_chart(
                plot_holdout_comparison(
                    result["holdout_bt"],
                    wf["holdout_spy_backtest"],
                    wf["holdout_random_backtest"],
                ),
                use_container_width=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

            left, right = st.columns(2)
            with left:
                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                st.plotly_chart(plot_equity(result["research_bt"], "Equity Curve - Pesquisa"), use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            with right:
                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                st.plotly_chart(plot_equity(result["holdout_bt"], "Equity Curve - Holdout"), use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            left2, right2 = st.columns(2)
            with left2:
                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                st.plotly_chart(plot_drawdown(result["research_bt"], "Drawdown - Pesquisa"), use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            with right2:
                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                st.plotly_chart(plot_drawdown(result["holdout_bt"], "Drawdown - Holdout"), use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.plotly_chart(
                px.histogram(
                    result["mc"]["simulations"],
                    x="sharpe",
                    title="Distribuição de Sharpe - Monte Carlo (Holdout)",
                ),
                use_container_width=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

            bottom_left, bottom_right = st.columns(2)
            with bottom_left:
                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                st.plotly_chart(plot_filter_states(result["filters"]), use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            with bottom_right:
                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                st.subheader("Walk-forward folds")
                st.dataframe(wf["fold_metrics"], use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            with st.expander("Relatório institucional"):
                st.json(result["report"])
            st.markdown("</div>", unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
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

    render_kpi_row(
        [
            ("Cash", f"{state.get('cash', 0.0):,.2f}", ""),
            ("Equity", f"{state.get('equity', 0.0):,.2f}", "kpi-good"),
            ("Posições", f"{len(state.get('positions', {}))}", ""),
            ("Runs", f"{state.get('run_count', 0)}", ""),
        ]
    )

    if not paper_equity_df.empty:
        st.plotly_chart(plot_paper_equity(paper_equity_df), use_container_width=True)

    st.subheader("Relatório automático do paper trading")
    render_kpi_row(
        [
            ("Retorno %", f"{paper_report['return_pct']:.2f}%", "kpi-good" if paper_report["return_pct"] >= 0 else "kpi-bad"),
            ("Max Drawdown %", f"{paper_report['max_drawdown_pct']:.2f}%", "kpi-bad"),
            ("Trades", f"{paper_report['trades_count']}", ""),
            ("Trades fechados", f"{paper_report['closed_trades_count']}", ""),
        ]
    )
    render_kpi_row(
        [
            ("Win Rate %", f"{paper_report['win_rate_pct']:.2f}%", ""),
            ("Payoff", f"{paper_report['payoff_ratio']:.2f}", ""),
            ("PnL realizado", f"{paper_report['realized_pnl']:.2f}", "kpi-good" if paper_report["realized_pnl"] >= 0 else "kpi-bad"),
            ("Ativo mais operado", f"{paper_report['most_traded_asset'] or '-'}", ""),
        ]
    )

    left3, right3 = st.columns(2)

    with left3:
        st.subheader("Posições atuais")
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
        st.subheader("Últimos trades")
        if paper_trades:
            st.dataframe(pd.DataFrame(paper_trades[::-1]), use_container_width=True)
        else:
            st.info("Sem trades ainda.")

    st.subheader("Diagnóstico automático")
    if paper_report["diagnosis"]:
        for item in paper_report["diagnosis"]:
            st.write(f"• {item}")
    else:
        st.info("Ainda não há dados suficientes para diagnóstico.")

    with st.expander("Estado bruto do paper trading"):
        st.json(state)

    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
