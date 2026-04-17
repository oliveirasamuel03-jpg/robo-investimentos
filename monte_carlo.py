from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class MonteCarloConfig:
    n_simulations: int = 1000
    block_size: int = 5
    random_state: int = 42
    periods_per_year: int = 252


def _to_array(returns) -> np.ndarray:
    if isinstance(returns, pd.Series):
        r = returns.values
    else:
        r = np.array(returns)

    r = r[~np.isnan(r)]
    return r


def _sharpe(returns: np.ndarray, periods_per_year: int) -> float:
    if len(returns) < 2:
        return 0.0
    std = returns.std()
    if std == 0:
        return 0.0
    return np.sqrt(periods_per_year) * returns.mean() / std


def _equity_curve(returns: np.ndarray) -> np.ndarray:
    return np.cumprod(1 + returns)


def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0
    return dd.min()


def bootstrap_simulation(returns: np.ndarray, size: int) -> np.ndarray:
    return np.random.choice(returns, size=size, replace=True)


def shuffle_simulation(returns: np.ndarray) -> np.ndarray:
    shuffled = returns.copy()
    np.random.shuffle(shuffled)
    return shuffled


def block_bootstrap_simulation(returns: np.ndarray, size: int, block_size: int) -> np.ndarray:
    blocks = []
    n = len(returns)

    while len(blocks) * block_size < size:
        start = np.random.randint(0, n - block_size)
        block = returns[start:start + block_size]
        blocks.append(block)

    return np.concatenate(blocks)[:size]


def run_monte_carlo(
    returns,
    config: MonteCarloConfig | None = None,
) -> Dict:
    if config is None:
        config = MonteCarloConfig()

    np.random.seed(config.random_state)

    r = _to_array(returns)

    if len(r) < 10:
        raise ValueError("Not enough data for Monte Carlo simulation")

    results = []

    for _ in range(config.n_simulations):

        sim_type = np.random.choice(["bootstrap", "shuffle", "block"])

        if sim_type == "bootstrap":
            sim_returns = bootstrap_simulation(r, len(r))

        elif sim_type == "shuffle":
            sim_returns = shuffle_simulation(r)

        else:
            sim_returns = block_bootstrap_simulation(r, len(r), config.block_size)

        equity = _equity_curve(sim_returns)

        total_return = equity[-1] - 1.0
        sharpe = _sharpe(sim_returns, config.periods_per_year)
        mdd = _max_drawdown(equity)

        results.append({
            "total_return": total_return,
            "sharpe": sharpe,
            "max_drawdown": mdd,
        })

    df = pd.DataFrame(results)

    prob_positive = float((df["total_return"] > 0).mean())

    sharpe_mean = float(df["sharpe"].mean())
    sharpe_std = float(df["sharpe"].std())

    ci_lower = float(df["total_return"].quantile(0.05))
    ci_upper = float(df["total_return"].quantile(0.95))

    robustness_score = prob_positive * max(sharpe_mean, 0)

    return {
        "simulations": df,
        "probability_positive": prob_positive,
        "sharpe_mean": sharpe_mean,
        "sharpe_std": sharpe_std,
        "confidence_interval_5_95": (ci_lower, ci_upper),
        "robustness_score": robustness_score,
    }
