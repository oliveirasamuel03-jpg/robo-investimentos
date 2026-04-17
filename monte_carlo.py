from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class MonteCarloConfig:
    n_simulations: int = 200
    block_size: int = 5
    random_state: int = 42
    periods_per_year: int = 252


def _to_array(returns) -> np.ndarray:
    if isinstance(returns, pd.Series):
        r = returns.values
    else:
        r = np.array(returns)

    r = r.astype(float)
    r = r[~np.isnan(r)]
    return r


def _sharpe(returns: np.ndarray, periods_per_year: int) -> float:
    if len(returns) < 2:
        return 0.0
    std = returns.std()
    if std == 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * returns.mean() / std)


def _equity_curve(returns: np.ndarray) -> np.ndarray:
    return np.cumprod(1.0 + returns)


def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0
    return float(dd.min())


def bootstrap_simulation(returns: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    idx = rng.integers(0, len(returns), size=size)
    return returns[idx]


def shuffle_simulation(returns: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    shuffled = returns.copy()
    rng.shuffle(shuffled)
    return shuffled


def block_bootstrap_simulation(
    returns: np.ndarray,
    size: int,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n = len(returns)
    if n <= block_size:
        return bootstrap_simulation(returns, size, rng)

    blocks = []
    while sum(len(b) for b in blocks) < size:
        start = int(rng.integers(0, n - block_size + 1))
        blocks.append(returns[start:start + block_size])

    return np.concatenate(blocks)[:size]


def run_monte_carlo(
    returns,
    config: MonteCarloConfig | None = None,
) -> Dict:
    if config is None:
        config = MonteCarloConfig()

    r = _to_array(returns)

    if len(r) < 20:
        raise ValueError("Dados insuficientes para Monte Carlo")

    rng = np.random.default_rng(config.random_state)
    results = []

    sim_types = ["bootstrap", "shuffle", "block"]

    for _ in range(config.n_simulations):
        sim_type = sim_types[int(rng.integers(0, len(sim_types)))]

        if sim_type == "bootstrap":
            sim_returns = bootstrap_simulation(r, len(r), rng)
        elif sim_type == "shuffle":
            sim_returns = shuffle_simulation(r, rng)
        else:
            sim_returns = block_bootstrap_simulation(r, len(r), config.block_size, rng)

        equity = _equity_curve(sim_returns)
        total_return = float(equity[-1] - 1.0)
        sharpe = _sharpe(sim_returns, config.periods_per_year)
        mdd = _max_drawdown(equity)

        results.append(
            {
                "simulation_type": sim_type,
                "total_return": total_return,
                "sharpe": sharpe,
                "max_drawdown": mdd,
            }
        )

    df = pd.DataFrame(results)

    prob_positive = float((df["total_return"] > 0).mean())
    sharpe_mean = float(df["sharpe"].mean())
    sharpe_std = float(df["sharpe"].std(ddof=0))
    ci_lower = float(df["total_return"].quantile(0.05))
    ci_upper = float(df["total_return"].quantile(0.95))
    robustness_score = float(prob_positive * max(sharpe_mean, 0.0))

    return {
        "simulations": df,
        "probability_positive": prob_positive,
        "sharpe_mean": sharpe_mean,
        "sharpe_std": sharpe_std,
        "confidence_interval_5_95": (ci_lower, ci_upper),
        "robustness_score": robustness_score,
    }
