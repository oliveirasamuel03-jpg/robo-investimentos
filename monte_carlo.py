from dataclasses import dataclass
import numpy as np


@dataclass
class MonteCarloConfig:
    n_simulations: int = 100
    horizon: int = 50


def run_monte_carlo(returns, config: MonteCarloConfig):
    if len(returns) == 0:
        return {
            "mean_final": 0,
            "min_final": 0,
            "max_final": 0,
        }

    simulations = []

    for _ in range(config.n_simulations):
        sim = np.random.choice(returns, size=config.horizon, replace=True)
        simulations.append(np.prod(1 + sim) - 1)

    simulations = np.array(simulations)

    return {
        "mean_final": float(np.mean(simulations)),
        "min_final": float(np.min(simulations)),
        "max_final": float(np.max(simulations)),
    }
