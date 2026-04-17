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
    min_assets: int = 3
    initial_capital: float = 100000.0
    fee_rate: float = 0.0005
    slippage_rate: float = 0.0005
    execution_delay: int = 1


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

    pivot = predictions.pivot(index="date", columns="asset", values="weight")
    pivot = pivot.reindex(index=price_index, columns=price_columns).fillna(0.0)
    return pivot


def run_walk_forward_validation(
    prices: pd.DataFrame,
    feature_data: pd.DataFrame,
    model_factory: Callable[[], object],
    signal_builder: Callable[[pd.DataFrame], pd.DataFrame],
    config: Optional[WalkForwardConfig] = None,
) -> Dict[str, object]:
    """
    Walk-forward validation without look-ahead bias.

    Parameters
    ----------
    prices : pd.DataFrame
        Wide price matrix. Index=date, columns=assets.
    feature_data : pd.DataFrame
        Long dataset with columns at least:
        ['date', 'asset', ..., 'target']
    model_factory : callable
        Returns a fresh ML model instance for each fold.
        The model must implement:
            fit(X, y)
            predict_proba(X) -> probabilities for class 1 or shape (n,2)
    signal_builder : callable
        Function receiving predictions DataFrame with columns:
            ['date', 'asset', 'probability']
        and returning:
            ['date', 'asset', 'weight']
    config : WalkForwardConfig
        Walk-forward parameters.

    Returns
    -------
    dict with:
        fold_metrics
        aggregate_metrics
        predictions
        target_weights
        backtest_result
        degradation_analysis
    """
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

    data = feature_data.copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values(["date", "asset"]).reset_index(drop=True)

    prices = prices.copy()
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()

    feature_cols = [c for c in data.columns if c not in {"date", "asset", "target"}]
    if not feature_cols:
        raise ValueError("No feature columns found in feature_data")

    unique_dates = pd.Index(sorted(data["date"].dropna().unique()))
    if len(unique_dates) < (config.train_window + config.test_window + 1):
        raise ValueError("Not enough dates for walk-forward validation")

    fold_summaries: List[Dict[str, object]] = []
    prediction_frames: List[pd.DataFrame] = []
    fold_backtests: List[pd.DataFrame] = []

    fold_number = 0

    for train_end_idx in range(
        config.train_window,
        len(unique_dates) - config.test_window + 1,
        config.step_size,
    ):
        train_start_idx = train_end_idx - config.train_window
        test_end_idx = train_end_idx + config.test_window

        train_dates = unique_dates[train_start_idx:train_end_idx]
        test_dates = unique_dates[train_end_idx:test_end_idx]

        train_df = data[data["date"].isin(train_dates)].copy()
        test_df = data[data["date"].isin(test_dates)].copy()

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

        weights_df = signal_builder(pred_df[["date", "asset", "probability"]].copy())
        if not {"date", "asset", "weight"}.issubset(weights_df.columns):
            raise ValueError("signal_builder must return columns ['date', 'asset', 'weight']")

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
    fold_metrics = pd.DataFrame(fold_summaries)

    full_target_weights = _build_target_matrix(
        predictions=predictions_all[["date", "asset", "weight"]],
        price_index=prices.index,
        price_columns=prices.columns,
    )

    full_backtest = backtest_portfolio(
        prices=prices,
        target_weights=full_target_weights,
        initial_capital=config.initial_capital,
        fee_rate=config.fee_rate,
        slippage_rate=config.slippage_rate,
        execution_delay=config.execution_delay,
    )

    full_returns = full_backtest["returns"]["return"]
    full_equity = full_backtest["equity_curve"]["equity"]

    aggregate_metrics = {
        "folds": int(len(fold_metrics)),
        "mean_fold_sharpe": float(fold_metrics["sharpe"].mean()) if not fold_metrics.empty else 0.0,
        "median_fold_sharpe": float(fold_metrics["sharpe"].median()) if not fold_metrics.empty else 0.0,
        "mean_fold_sortino": float(fold_metrics["sortino"].mean()) if not fold_metrics.empty else 0.0,
        "mean_fold_calmar": float(fold_metrics["calmar"].mean()) if not fold_metrics.empty else 0.0,
        "mean_fold_hit_rate": float(fold_metrics["hit_rate"].mean()) if not fold_metrics.empty else 0.0,
        "full_period_sharpe": _annualized_sharpe(full_returns),
        "full_period_sortino": _annualized_sortino(full_returns),
        "full_period_calmar": _calmar_ratio(full_returns, full_equity),
        "full_period_max_drawdown": _max_drawdown_from_equity(full_equity),
        "full_period_total_return": float((full_equity.iloc[-1] / config.initial_capital) - 1.0) if len(full_equity) > 0 else 0.0,
    }

    degradation_analysis = {
        "best_fold_sharpe": float(fold_metrics["sharpe"].max()) if not fold_metrics.empty else 0.0,
        "worst_fold_sharpe": float(fold_metrics["sharpe"].min()) if not fold_metrics.empty else 0.0,
        "sharpe_stability": float(fold_metrics["sharpe"].std(ddof=0)) if len(fold_metrics) > 1 else 0.0,
        "return_stability": float(fold_metrics["total_return"].std(ddof=0)) if len(fold_metrics) > 1 else 0.0,
        "wfe_proxy": float(
            aggregate_metrics["full_period_sharpe"] / (aggregate_metrics["mean_fold_sharpe"] + 1e-12)
        ) if aggregate_metrics["mean_fold_sharpe"] != 0 else 0.0,
    }

    fold_backtest_curve = pd.concat(fold_backtests).sort_index() if fold_backtests else pd.DataFrame()

    return {
        "fold_metrics": fold_metrics,
        "aggregate_metrics": aggregate_metrics,
        "predictions": predictions_all,
        "target_weights": full_target_weights,
        "backtest_result": full_backtest,
        "degradation_analysis": degradation_analysis,
        "fold_backtest_curve": fold_backtest_curve,
    }
