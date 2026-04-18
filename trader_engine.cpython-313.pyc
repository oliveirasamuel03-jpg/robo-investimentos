from __future__ import annotations

import pandas as pd

from engines.quant_bridge import (
    EnsembleProbabilityModel,
    MLEngineConfig,
    MonteCarloConfig,
    RiskConfig,
    StrategyConfig,
    WalkForwardConfig,
    apply_portfolio_risk_overlay,
    build_feature_panel,
    build_institutional_report,
    combined_market_filter,
    compute_all_metrics,
    generate_target_weights,
    load_data,
    run_monte_carlo,
    run_walk_forward_validation,
    save_report,
)


def _safe_reset_index_with_date(df: pd.DataFrame) -> pd.DataFrame:
    out = df.reset_index().copy()
    if "date" not in out.columns:
        out = out.rename(columns={out.columns[0]: "date"})
    out["date"] = pd.to_datetime(out["date"])
    return out


def _build_dynamic_wf_config(feature_panel: pd.DataFrame, initial_capital: float, execution_delay: int, holdout_ratio: float) -> WalkForwardConfig:
    unique_dates = pd.Index(sorted(pd.to_datetime(feature_panel["date"]).unique()))
    n_dates = len(unique_dates)
    if n_dates < 120:
        raise ValueError(f"Poucas datas úteis após limpeza ({n_dates}).")
    holdout_size = max(20, int(n_dates * holdout_ratio))
    research_size = n_dates - holdout_size
    train_window = max(40, min(160, int(research_size * 0.55)))
    test_window = max(10, min(20, int(research_size * 0.12)))
    step_size = max(5, min(15, test_window))
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


def run_investment_pipeline(
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
    include_brazil_stocks: bool,
    include_us_stocks: bool,
    include_etfs: bool,
    include_fiis: bool,
    include_crypto: bool,
    include_grains: bool,
    custom_tickers: list[str],
):
    history_limit = 500 if fast_mode else 1500
    prices = load_data(
        start=start_date,
        history_limit=history_limit,
        include_brazil_stocks=include_brazil_stocks,
        include_us_stocks=include_us_stocks,
        include_etfs=include_etfs,
        include_fiis=include_fiis,
        include_crypto=include_crypto,
        include_grains=include_grains,
        custom_tickers=custom_tickers,
    )
    features = build_feature_panel(prices, MLEngineConfig())

    strategy_config = StrategyConfig(
        probability_threshold=probability_threshold,
        top_n=top_n,
        max_weight_per_asset=0.35,
        cash_threshold=cash_threshold,
        target_portfolio_vol=target_portfolio_vol,
    )

    wf_config = _build_dynamic_wf_config(features, initial_capital, execution_delay, holdout_ratio)
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
        weighted = generate_target_weights(pred_df, prices=prices, config=strategy_config).copy()
        weighted["date"] = pd.to_datetime(weighted["date"])
        weighted = weighted.merge(filters[["date", "regime", "vol_filter", "trade_allowed"]], on="date", how="left")
        weighted["trade_allowed"] = weighted["trade_allowed"].fillna(False)
        weighted.loc[~weighted["trade_allowed"], "weight"] = 0.0
        out_rows = []
        for _, group in weighted.groupby("date", sort=True):
            base = group.set_index("asset")["weight"]
            safe = apply_portfolio_risk_overlay(base, equity_curve=None, config=risk_config)
            tmp = group.copy()
            tmp["weight"] = tmp["asset"].map(safe).fillna(0.0)
            out_rows.append(tmp[["date", "asset", "weight"]])
        return pd.concat(out_rows, ignore_index=True)

    wf = run_walk_forward_validation(prices, features, model_factory, signal_builder, wf_config)
    research_bt = wf["backtest_result"]
    holdout_bt = wf["holdout_backtest"]

    research_metrics = compute_all_metrics(research_bt["returns"]["return"], research_bt["equity_curve"]["equity"])
    holdout_metrics = wf["holdout_metrics"]
    mc = run_monte_carlo(holdout_bt["returns"]["return"], MonteCarloConfig())
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
        "prices_columns": list(prices.columns),
    }
