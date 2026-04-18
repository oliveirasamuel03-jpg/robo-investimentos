from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from backtest_engine import backtest_portfolio


@dataclass
class WalkForwardConfig:
    train_window: int = 252 * 2
    test_window: int = 21
    step_size: int = 21
    embargo: int = 5
    min_assets: int = 3
    initial_capital: float = 100000.0
    fee_rate: float = 0.0005
    slippage_rate: float = 0.0005
    execution_delay: int = 1
    holdout_ratio: float = 0.20


def _safe_series(x: pd.Series) -> pd.Series:
    if x is None:
        return pd.Series(dtype=float)
    return x.replace([np.inf, -np.inf], np.nan).dropna()


def _annualized_sharpe(returns: pd.Series, periods_per_year: int = 252) -> float:
    r = _safe_series(returns)
    if len(r) < 2 or float(r.std(ddof=0)) == 0.0:
        return 0.0
    return float(np.sqrt(periods_per_year) * r.mean() / r.std(ddof=0))


def _annualized_sortino(returns: pd.Series, periods_per_year: int = 252) -> float:
    r = _safe_series(returns)
    if len(r) < 2:
        return 0.0
    downside = r[r < 0]
    if len(downside) == 0:
        return 0.0
    downside_std = float(downside.std(ddof=0))
    if downside_std == 0.0:
        return 0.0
    return float(np.sqrt(periods_per_year) * r.mean() / downside_std)


def _max_drawdown_from_equity(equity: pd.Series) -> float:
    eq = _safe_series(equity)
    if eq.empty:
        return 0.0
    peak = eq.cummax()
    dd = (eq / peak) - 1.0
    return float(dd.min())


def _calmar_ratio(returns: pd.Series, equity: pd.Series, periods_per_year: int = 252) -> float:
    r = _safe_series(returns)
    if r.empty:
        return 0.0
    annual_return = float(r.mean() * periods_per_year)
    mdd = abs(_max_drawdown_from_equity(equity))
    if mdd == 0.0:
        return 0.0
    return annual_return / mdd


def _build_target_matrix(
    predictions: pd.DataFrame,
    price_index: pd.Index,
    price_columns: pd.Index,
) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame(0.0, index=price_index, columns=price_columns)

    tmp = predictions.copy()
    tmp["date"] = pd.to_datetime(tmp["date"])
    tmp["asset"] = tmp["asset"].astype(str)
    tmp["weight"] = pd.to_numeric(tmp["weight"], errors="coerce").fillna(0.0)

    tmp = tmp.groupby(["date", "asset"], as_index=False)["weight"].sum()

    pivot = tmp.pivot_table(
        index="date",
        columns="asset",
        values="weight",
        aggfunc="sum",
        fill_value=0.0,
    )

    pivot = pivot.reindex(index=price_index, columns=price_columns).fillna(0.0)
    return pivot


def _prepare_feature_data(feature_data: pd.DataFrame) -> pd.DataFrame:
    data = feature_data.copy()
    data["date"] = pd.to_datetime(data["date"])
    data["asset"] = data["asset"].astype(str)

    non_key_cols = [c for c in data.columns if c not in {"date", "asset"}]
    agg_map = {c: "mean" for c in non_key_cols}

    data = (
        data.groupby(["date", "asset"], as_index=False)
        .agg(agg_map)
        .sort_values(["date", "asset"])
        .reset_index(drop=True)
    )
    return data


def _evaluate_backtest_result(backtest_result: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    returns = backtest_result["returns"]["return"]
    equity = backtest_result["equity_curve"]["equity"]

    return {
        "sharpe": _annualized_sharpe(returns),
        "sortino": _annualized_sortino(returns),
        "calmar": _calmar_ratio(returns, equity),
        "max_drawdown": _max_drawdown_from_equity(equity),
        "total_return": float((equity.iloc[-1] / equity.iloc[0]) - 1.0) if len(equity) > 1 else 0.0,
    }


def _run_random_baseline(
    prices: pd.DataFrame,
    dates: pd.Index,
    top_n: int,
    initial_capital: float,
    fee_rate: float,
    slippage_rate: float,
    execution_delay: int,
    seed: int = 42,
) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    assets = list(prices.columns)

    rows = []
    for dt in dates:
        if len(assets) == 0:
            continue
        n_select = min(top_n, len(assets))
        selected = rng.choice(assets, size=n_select, replace=False)
        w = 1.0 / n_select

        for asset in assets:
            rows.append(
                {
                    "date": dt,
                    "asset": asset,
                    "weight": w if asset in selected else 0.0,
                }
            )

    random_weights = pd.DataFrame(rows)
    weight_matrix = _build_target_matrix(
        random_weights,
        price_index=prices.loc[dates].index,
        price_columns=prices.columns,
    )

    result = backtest_portfolio(
        prices=prices.loc[dates],
        target_weights=weight_matrix,
        initial_capital=initial_capital,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
        execution_delay=execution_delay,
    )

    return result


def _run_spy_benchmark(
    prices: pd.DataFrame,
    dates: pd.Index,
    initial_capital: float,
    fee_rate: float,
    slippage_rate: float,
    execution_delay: int,
) -> Dict[str, object]:
    benchmark_asset = "SPY" if "SPY" in prices.columns else prices.columns[0]

    rows = []
    for dt in dates:
        for asset in prices.columns:
            rows.append(
                {
                    "date": dt,
                    "asset": asset,
                    "weight": 1.0 if asset == benchmark_asset else 0.0,
                }
            )

    bench_weights = pd.DataFrame(rows)
    weight_matrix = _build_target_matrix(
        bench_weights,
        price_index=prices.loc[dates].index,
        price_columns=prices.columns,
    )

    result = backtest_portfolio(
        prices=prices.loc[dates],
        target_weights=weight_matrix,
        initial_capital=initial_capital,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
        execution_delay=execution_delay,
    )

    return result


def run_walk_forward_validation(
    prices: pd.DataFrame,
    feature_data: pd.DataFrame,
    model_factory: Callable[[], object],
    signal_builder: Callable[[pd.DataFrame], pd.DataFrame],
    config: Optional[WalkForwardConfig] = None,
) -> Dict[str, object]:
    if config is None:
        config = WalkForwardConfig()

    if prices.empty:
        raise ValueError("prices is empty")
    if feature_data.empty:
        raise ValueError("feature_data is empty")

    required_cols = {"date", "asset", "target"}
    missing = required_cols - set(feature_data.columns)
    if missing:
        raise ValueError(f"feature_data missing required columns: {sorted(missing)}")

    prices = prices.copy()
    prices.index = pd.to_datetime(prices.index)
    prices.columns = prices.columns.astype(str)
    prices = prices.sort_index()

    data = _prepare_feature_data(feature_data)

    feature_cols = [c for c in data.columns if c not in {"date", "asset", "target"}]
    if not feature_cols:
        raise ValueError("No feature columns found in feature_data")

    unique_dates = pd.Index(sorted(data["date"].dropna().unique()))
    if len(unique_dates) < (config.train_window + config.test_window + config.embargo + 20):
        raise ValueError("Not enough dates for walk-forward validation")

    holdout_size = max(config.test_window, int(len(unique_dates) * config.holdout_ratio))
    if holdout_size >= len(unique_dates) - (config.train_window + config.embargo + 1):
        holdout_size = config.test_window

    research_dates = unique_dates[:-holdout_size]
    holdout_dates = unique_dates[-holdout_size:]

    if len(research_dates) < (config.train_window + config.test_window + config.embargo + 1):
        raise ValueError("Research window too short after holdout split")

    fold_summaries: List[Dict[str, object]] = []
    prediction_frames: List[pd.DataFrame] = []
    fold_backtests: List[pd.DataFrame] = []
    fold_number = 0

    for train_end_idx in range(
        config.train_window,
        len(research_dates) - config.test_window + 1,
        config.step_size,
    ):
        test_start_idx = train_end_idx + config.embargo
        test_end_idx = test_start_idx + config.test_window

        if test_end_idx > len(research_dates):
            break

        train_start_idx = train_end_idx - config.train_window

        train_dates = research_dates[train_start_idx:train_end_idx]
        test_dates = research_dates[test_start_idx:test_end_idx]

        train_df = data[data["date"].isin(train_dates)].copy()
        test_df = data[data["date"].isin(test_dates)].copy()

        if train_df.empty or test_df.empty:
            continue

        train_assets_per_day = train_df.groupby("date")["asset"].nunique()
        test_assets_per_day = test_df.groupby("date")["asset"].nunique()

        if train_assets_per_day.empty or test_assets_per_day.empty:
            continue
        if int(train_assets_per_day.min()) < config.min_assets:
            continue
        if int(test_assets_per_day.min()) < config.min_assets:
            continue

        X_train = train_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        y_train = train_df["target"].astype(int)

        X_test = test_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        y_test = test_df["target"].astype(int)

        if y_train.nunique() < 2:
            continue

        model = model_factory()
        model.fit(X_train, y_train)

        raw_proba = model.predict_proba(X_test)
        if isinstance(raw_proba, np.ndarray) and raw_proba.ndim == 2:
            proba = raw_proba[:, 1]
        else:
            proba = np.asarray(raw_proba).reshape(-1)

        pred_df = test_df[["date", "asset"]].copy()
        pred_df["probability"] = proba
        pred_df["target"] = y_test.values
        pred_df["fold"] = fold_number

        pred_df = (
            pred_df.groupby(["date", "asset"], as_index=False)
            .agg(
                probability=("probability", "mean"),
                target=("target", "max"),
                fold=("fold", "min"),
            )
        )

        weights_df = signal_builder(pred_df[["date", "asset", "probability"]].copy())
        if not {"date", "asset", "weight"}.issubset(weights_df.columns):
            raise ValueError("signal_builder must return columns ['date', 'asset', 'weight']")

        weights_df = weights_df.copy()
        weights_df["date"] = pd.to_datetime(weights_df["date"])
        weights_df["asset"] = weights_df["asset"].astype(str)
        weights_df["weight"] = pd.to_numeric(weights_df["weight"], errors="coerce").fillna(0.0)
        weights_df = weights_df.groupby(["date", "asset"], as_index=False)["weight"].sum()

        merged = pred_df.merge(
            weights_df[["date", "asset", "weight"]],
            on=["date", "asset"],
            how="left",
        )
        merged["weight"] = merged["weight"].fillna(0.0)

        prediction_frames.append(merged)

        fold_prices = prices.loc[
            (prices.index >= pd.to_datetime(test_dates[0]))
            & (prices.index <= pd.to_datetime(test_dates[-1]))
        ].copy()

        if fold_prices.empty:
            fold_number += 1
            continue

        fold_target_weights = _build_target_matrix(
            predictions=merged[["date", "asset", "weight"]],
            price_index=fold_prices.index,
            price_columns=fold_prices.columns,
        )

        backtest_result = backtest_portfolio(
            prices=fold_prices,
            target_weights=fold_target_weights,
            initial_capital=config.initial_capital,
            fee_rate=config.fee_rate,
            slippage_rate=config.slippage_rate,
            execution_delay=config.execution_delay,
        )

        fold_returns = backtest_result["returns"]["return"]
        fold_equity = backtest_result["equity_curve"]["equity"]
        gross_hit_rate = float((merged["target"] == (merged["probability"] >= 0.5).astype(int)).mean())

        fold_summary = {
            "fold": fold_number,
            "train_start": pd.to_datetime(train_dates[0]),
            "train_end": pd.to_datetime(train_dates[-1]),
            "test_start": pd.to_datetime(test_dates[0]),
            "test_end": pd.to_datetime(test_dates[-1]),
            "n_train_rows": int(len(train_df)),
            "n_test_rows": int(len(test_df)),
            "n_assets_test": int(test_df["asset"].nunique()),
            "hit_rate": gross_hit_rate,
            "sharpe": _annualized_sharpe(fold_returns),
            "sortino": _annualized_sortino(fold_returns),
            "calmar": _calmar_ratio(fold_returns, fold_equity),
            "max_drawdown": _max_drawdown_from_equity(fold_equity),
            "total_return": float((fold_equity.iloc[-1] / fold_equity.iloc[0]) - 1.0) if len(fold_equity) > 1 else 0.0,
        }
        fold_summaries.append(fold_summary)

        eq = backtest_result["equity_curve"].copy()
        eq["fold"] = fold_number
        fold_backtests.append(eq)

        fold_number += 1

    if not prediction_frames:
        raise ValueError("No valid walk-forward folds were generated")

    predictions_all = pd.concat(prediction_frames, ignore_index=True)
    predictions_all = (
        predictions_all.groupby(["date", "asset"], as_index=False)
        .agg(
            probability=("probability", "mean"),
            target=("target", "max"),
            fold=("fold", "min"),
            weight=("weight", "sum"),
        )
    )

    research_target_weights = _build_target_matrix(
        predictions=predictions_all[["date", "asset", "weight"]],
        price_index=prices.loc[research_dates].index,
        price_columns=prices.columns,
    )

    research_backtest = backtest_portfolio(
        prices=prices.loc[research_dates],
        target_weights=research_target_weights,
        initial_capital=config.initial_capital,
        fee_rate=config.fee_rate,
        slippage_rate=config.slippage_rate,
        execution_delay=config.execution_delay,
    )

    # Holdout final: train once on all research data, test only on holdout
    research_df = data[data["date"].isin(research_dates)].copy()
    holdout_df = data[data["date"].isin(holdout_dates)].copy()

    X_research = research_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_research = research_df["target"].astype(int)

    X_holdout = holdout_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_holdout = holdout_df["target"].astype(int)

    final_model = model_factory()
    final_model.fit(X_research, y_research)

    raw_holdout_proba = final_model.predict_proba(X_holdout)
    if isinstance(raw_holdout_proba, np.ndarray) and raw_holdout_proba.ndim == 2:
        holdout_proba = raw_holdout_proba[:, 1]
    else:
        holdout_proba = np.asarray(raw_holdout_proba).reshape(-1)

    holdout_pred = holdout_df[["date", "asset"]].copy()
    holdout_pred["probability"] = holdout_proba
    holdout_pred["target"] = y_holdout.values

    holdout_pred = (
        holdout_pred.groupby(["date", "asset"], as_index=False)
        .agg(
            probability=("probability", "mean"),
            target=("target", "max"),
        )
    )

    holdout_weights = signal_builder(holdout_pred[["date", "asset", "probability"]].copy())
    holdout_weights = holdout_weights.copy()
    holdout_weights["date"] = pd.to_datetime(holdout_weights["date"])
    holdout_weights["asset"] = holdout_weights["asset"].astype(str)
    holdout_weights["weight"] = pd.to_numeric(holdout_weights["weight"], errors="coerce").fillna(0.0)
    holdout_weights = holdout_weights.groupby(["date", "asset"], as_index=False)["weight"].sum()

    holdout_target_weights = _build_target_matrix(
        predictions=holdout_weights[["date", "asset", "weight"]],
        price_index=prices.loc[holdout_dates].index,
        price_columns=prices.columns,
    )

    holdout_backtest = backtest_portfolio(
        prices=prices.loc[holdout_dates],
        target_weights=holdout_target_weights,
        initial_capital=config.initial_capital,
        fee_rate=config.fee_rate,
        slippage_rate=config.slippage_rate,
        execution_delay=config.execution_delay,
    )

    # Benchmarks on holdout
    top_n_guess = 2
    random_backtest = _run_random_baseline(
        prices=prices,
        dates=holdout_dates,
        top_n=top_n_guess,
        initial_capital=config.initial_capital,
        fee_rate=config.fee_rate,
        slippage_rate=config.slippage_rate,
        execution_delay=config.execution_delay,
    )

    spy_backtest = _run_spy_benchmark(
        prices=prices,
        dates=holdout_dates,
        initial_capital=config.initial_capital,
        fee_rate=config.fee_rate,
        slippage_rate=config.slippage_rate,
        execution_delay=config.execution_delay,
    )

    fold_metrics = pd.DataFrame(fold_summaries)
    research_returns = research_backtest["returns"]["return"]
    research_equity = research_backtest["equity_curve"]["equity"]

    aggregate_metrics = {
        "folds": int(len(fold_metrics)),
        "mean_fold_sharpe": float(fold_metrics["sharpe"].mean()) if not fold_metrics.empty else 0.0,
        "median_fold_sharpe": float(fold_metrics["sharpe"].median()) if not fold_metrics.empty else 0.0,
        "mean_fold_sortino": float(fold_metrics["sortino"].mean()) if not fold_metrics.empty else 0.0,
        "mean_fold_calmar": float(fold_metrics["calmar"].mean()) if not fold_metrics.empty else 0.0,
        "mean_fold_hit_rate": float(fold_metrics["hit_rate"].mean()) if not fold_metrics.empty else 0.0,
        "research_sharpe": _annualized_sharpe(research_returns),
        "research_sortino": _annualized_sortino(research_returns),
        "research_calmar": _calmar_ratio(research_returns, research_equity),
        "research_max_drawdown": _max_drawdown_from_equity(research_equity),
        "research_total_return": float((research_equity.iloc[-1] / config.initial_capital) - 1.0) if len(research_equity) > 0 else 0.0,
    }

    degradation_analysis = {
        "best_fold_sharpe": float(fold_metrics["sharpe"].max()) if not fold_metrics.empty else 0.0,
        "worst_fold_sharpe": float(fold_metrics["sharpe"].min()) if not fold_metrics.empty else 0.0,
        "sharpe_stability": float(fold_metrics["sharpe"].std(ddof=0)) if len(fold_metrics) > 1 else 0.0,
        "return_stability": float(fold_metrics["total_return"].std(ddof=0)) if len(fold_metrics) > 1 else 0.0,
        "wfe_proxy": float(
            aggregate_metrics["research_sharpe"] / (aggregate_metrics["mean_fold_sharpe"] + 1e-12)
        ) if aggregate_metrics["mean_fold_sharpe"] != 0 else 0.0,
    }

    holdout_metrics = _evaluate_backtest_result(holdout_backtest)
    random_metrics = _evaluate_backtest_result(random_backtest)
    spy_metrics = _evaluate_backtest_result(spy_backtest)

    fold_backtest_curve = pd.concat(fold_backtests).sort_index() if fold_backtests else pd.DataFrame()

    return {
        "fold_metrics": fold_metrics,
        "aggregate_metrics": aggregate_metrics,
        "predictions": predictions_all,
        "target_weights": research_target_weights,
        "backtest_result": research_backtest,
        "degradation_analysis": degradation_analysis,
        "fold_backtest_curve": fold_backtest_curve,
        "research_dates": research_dates,
        "holdout_dates": holdout_dates,
        "holdout_backtest": holdout_backtest,
        "holdout_metrics": holdout_metrics,
        "holdout_random_backtest": random_backtest,
        "holdout_random_metrics": random_metrics,
        "holdout_spy_backtest": spy_backtest,
        "holdout_spy_metrics": spy_metrics,
    }
