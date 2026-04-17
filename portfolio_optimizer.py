from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize


OptimizerMethod = Literal["equal_weight", "risk_parity", "markowitz"]


@dataclass
class OptimizerConfig:
    method: OptimizerMethod = "risk_parity"
    lookback_window: int = 60
    max_weight: float = 0.40
    min_weight: float = 0.0
    target_gross_exposure: float = 1.0
    regularization: float = 1e-4
    annualization_factor: int = 252


def _clean_returns(returns: pd.DataFrame) -> pd.DataFrame:
    if returns.empty:
        return returns.copy()
    out = returns.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    out = out.fillna(0.0)
    return out


def _normalize_weights(
    weights: pd.Series,
    max_weight: float,
    min_weight: float,
    target_gross_exposure: float,
) -> pd.Series:
    weights = weights.clip(lower=min_weight, upper=max_weight)

    gross = float(weights.abs().sum())
    if gross <= 0:
        return pd.Series(0.0, index=weights.index)

    weights = weights / gross * target_gross_exposure
    return weights


def equal_weight_optimizer(
    selected_assets: list[str],
    config: Optional[OptimizerConfig] = None,
) -> pd.Series:
    if config is None:
        config = OptimizerConfig(method="equal_weight")

    if not selected_assets:
        return pd.Series(dtype=float)

    weights = pd.Series(1.0 / len(selected_assets), index=selected_assets, dtype=float)
    return _normalize_weights(
        weights,
        max_weight=config.max_weight,
        min_weight=config.min_weight,
        target_gross_exposure=config.target_gross_exposure,
    )


def risk_parity_optimizer(
    returns_window: pd.DataFrame,
    selected_assets: Optional[list[str]] = None,
    config: Optional[OptimizerConfig] = None,
) -> pd.Series:
    if config is None:
        config = OptimizerConfig(method="risk_parity")

    returns_window = _clean_returns(returns_window)

    if selected_assets is not None:
        selected_assets = [a for a in selected_assets if a in returns_window.columns]
        returns_window = returns_window[selected_assets]

    if returns_window.empty or returns_window.shape[1] == 0:
        return pd.Series(dtype=float)

    vol = returns_window.std(ddof=0)
    vol = vol.replace(0.0, np.nan).dropna()

    if vol.empty:
        return pd.Series(dtype=float)

    inv_vol = 1.0 / vol
    weights = inv_vol / inv_vol.sum()

    return _normalize_weights(
        weights,
        max_weight=config.max_weight,
        min_weight=config.min_weight,
        target_gross_exposure=config.target_gross_exposure,
    )


def _markowitz_objective(
    weights: np.ndarray,
    mu: np.ndarray,
    cov: np.ndarray,
    regularization: float,
) -> float:
    portfolio_return = float(weights @ mu)
    portfolio_variance = float(weights @ cov @ weights)
    penalty = regularization * float(np.sum(weights ** 2))

    # maximize return/risk proxy by minimizing negative utility
    return -(portfolio_return - 0.5 * portfolio_variance) + penalty


def markowitz_optimizer(
    returns_window: pd.DataFrame,
    selected_assets: Optional[list[str]] = None,
    config: Optional[OptimizerConfig] = None,
) -> pd.Series:
    if config is None:
        config = OptimizerConfig(method="markowitz")

    returns_window = _clean_returns(returns_window)

    if selected_assets is not None:
        selected_assets = [a for a in selected_assets if a in returns_window.columns]
        returns_window = returns_window[selected_assets]

    if returns_window.empty or returns_window.shape[1] == 0:
        return pd.Series(dtype=float)

    assets = list(returns_window.columns)
    n = len(assets)

    mu = returns_window.mean().values * config.annualization_factor
    cov = returns_window.cov().values * config.annualization_factor

    cov = cov + np.eye(n) * config.regularization

    x0 = np.repeat(config.target_gross_exposure / n, n)

    bounds = [(config.min_weight, config.max_weight) for _ in range(n)]
    constraints = [
        {
            "type": "eq",
            "fun": lambda w: np.sum(w) - config.target_gross_exposure,
        }
    ]

    result = minimize(
        fun=_markowitz_objective,
        x0=x0,
        args=(mu, cov, config.regularization),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    if not result.success:
        weights = pd.Series(x0, index=assets, dtype=float)
    else:
        weights = pd.Series(result.x, index=assets, dtype=float)

    return _normalize_weights(
        weights,
        max_weight=config.max_weight,
        min_weight=config.min_weight,
        target_gross_exposure=config.target_gross_exposure,
    )


def optimize_weights(
    returns_window: pd.DataFrame,
    selected_assets: list[str],
    config: Optional[OptimizerConfig] = None,
) -> pd.Series:
    if config is None:
        config = OptimizerConfig()

    if not selected_assets:
        return pd.Series(dtype=float)

    if config.method == "equal_weight":
        return equal_weight_optimizer(selected_assets, config=config)

    if config.method == "risk_parity":
        return risk_parity_optimizer(
            returns_window=returns_window,
            selected_assets=selected_assets,
            config=config,
        )

    if config.method == "markowitz":
        return markowitz_optimizer(
            returns_window=returns_window,
            selected_assets=selected_assets,
            config=config,
        )

    raise ValueError(f"Unknown optimizer method: {config.method}")


def build_rolling_optimized_weights(
    prices: pd.DataFrame,
    selected_signals: pd.DataFrame,
    config: Optional[OptimizerConfig] = None,
) -> pd.DataFrame:
    """
    Build optimized daily weight matrix from selected assets.

    Parameters
    ----------
    prices : pd.DataFrame
        Wide price matrix, date x asset.
    selected_signals : pd.DataFrame
        Long-form table with at least:
            - date
            - asset
            - selected (bool)
        Optional:
            - probability
            - weight
    config : OptimizerConfig

    Returns
    -------
    pd.DataFrame
        Wide matrix date x asset of optimized weights.
    """
    if config is None:
        config = OptimizerConfig()

    if prices.empty:
        raise ValueError("prices is empty")

    required_cols = {"date", "asset", "selected"}
    missing = required_cols - set(selected_signals.columns)
    if missing:
        raise ValueError(f"selected_signals missing required columns: {sorted(missing)}")

    prices = prices.sort_index().copy()
    prices.index = pd.to_datetime(prices.index)

    asset_returns = prices.pct_change().fillna(0.0)

    df = selected_signals.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "asset"]).reset_index(drop=True)

    weight_rows = []

    for dt in prices.index:
        signal_day = df[df["date"] == dt].copy()

        if signal_day.empty:
            row = {"date": dt}
            for asset in prices.columns:
                row[asset] = 0.0
            weight_rows.append(row)
            continue

        selected_assets = signal_day.loc[signal_day["selected"], "asset"].tolist()

        if not selected_assets:
            row = {"date": dt}
            for asset in prices.columns:
                row[asset] = 0.0
            weight_rows.append(row)
            continue

        history = asset_returns.loc[:dt].tail(config.lookback_window)

        if history.empty or len(history) < 5:
            weights = equal_weight_optimizer(selected_assets, config=config)
        else:
            weights = optimize_weights(
                returns_window=history,
                selected_assets=selected_assets,
                config=config,
            )

        row = {"date": dt}
        for asset in prices.columns:
            row[asset] = float(weights.get(asset, 0.0))
        weight_rows.append(row)

    weight_matrix = pd.DataFrame(weight_rows).set_index("date").sort_index()
    return weight_matrix
