from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class RiskConfig:
    max_weight_per_asset: float = 0.35
    max_gross_exposure: float = 1.00
    max_net_exposure: float = 1.00
    max_drawdown_circuit_breaker: float = 0.12
    recovery_exposure: float = 0.35
    min_nav_fraction: float = 0.70


def apply_weight_caps(
    weights: pd.Series,
    config: RiskConfig | None = None,
) -> pd.Series:
    if config is None:
        config = RiskConfig()

    w = weights.astype(float).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    w = w.clip(lower=0.0, upper=config.max_weight_per_asset)

    gross = float(w.abs().sum())
    if gross > config.max_gross_exposure and gross > 0:
        w = w / gross * config.max_gross_exposure

    net = float(w.sum())
    if net > config.max_net_exposure and net > 0:
        w = w / net * config.max_net_exposure

    return w


def compute_drawdown(equity_curve: pd.Series) -> pd.Series:
    eq = equity_curve.astype(float)
    peak = eq.cummax()
    return eq / peak - 1.0


def circuit_breaker_scale(
    equity_curve: pd.Series,
    config: RiskConfig | None = None,
) -> float:
    if config is None:
        config = RiskConfig()

    if equity_curve.empty:
        return 1.0

    dd = compute_drawdown(equity_curve).iloc[-1]
    if dd <= -config.max_drawdown_circuit_breaker:
        return config.recovery_exposure

    peak = float(equity_curve.cummax().iloc[-1])
    nav = float(equity_curve.iloc[-1])

    if peak <= 0:
        return 1.0

    nav_fraction = nav / peak
    if nav_fraction < config.min_nav_fraction:
        return config.recovery_exposure

    return 1.0


def apply_portfolio_risk_overlay(
    weights: pd.Series,
    equity_curve: pd.Series | None = None,
    config: RiskConfig | None = None,
) -> pd.Series:
    if config is None:
        config = RiskConfig()

    w = apply_weight_caps(weights, config)

    if equity_curve is not None and len(equity_curve) > 0:
        scale = circuit_breaker_scale(equity_curve, config)
        w = w * scale

    return w
