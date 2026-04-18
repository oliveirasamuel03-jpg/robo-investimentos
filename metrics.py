from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


def _to_series(values) -> pd.Series:
    if isinstance(values, pd.Series):
        s = values.copy()
    elif isinstance(values, pd.DataFrame):
        if values.shape[1] != 1:
            raise ValueError("Expected a single-column DataFrame or a Series.")
        s = values.iloc[:, 0].copy()
    else:
        s = pd.Series(values)

    s = pd.to_numeric(s, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    return s


def annualized_return(returns, periods_per_year: int = 252) -> float:
    r = _to_series(returns)
    if r.empty:
        return 0.0
    compounded = float((1.0 + r).prod())
    n_periods = len(r)
    if n_periods == 0 or compounded <= 0:
        return 0.0
    return float(compounded ** (periods_per_year / n_periods) - 1.0)


def annualized_volatility(returns, periods_per_year: int = 252) -> float:
    r = _to_series(returns)
    if len(r) < 2:
        return 0.0
    return float(r.std(ddof=0) * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    r = _to_series(returns)
    if len(r) < 2:
        return 0.0

    rf_per_period = risk_free_rate / periods_per_year
    excess = r - rf_per_period
    vol = excess.std(ddof=0)

    if vol == 0 or np.isnan(vol):
        return 0.0

    return float(np.sqrt(periods_per_year) * excess.mean() / vol)


def sortino_ratio(
    returns,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    r = _to_series(returns)
    if len(r) < 2:
        return 0.0

    rf_per_period = risk_free_rate / periods_per_year
    excess = r - rf_per_period
    downside = excess[excess < 0]

    if len(downside) == 0:
        return 0.0

    downside_std = downside.std(ddof=0)
    if downside_std == 0 or np.isnan(downside_std):
        return 0.0

    return float(np.sqrt(periods_per_year) * excess.mean() / downside_std)


def equity_curve_from_returns(
    returns,
    initial_capital: float = 100000.0,
) -> pd.Series:
    r = _to_series(returns)
    if r.empty:
        return pd.Series(dtype=float)

    equity = initial_capital * (1.0 + r).cumprod()
    equity.name = "equity"
    return equity


def drawdown_series(equity) -> pd.Series:
    eq = _to_series(equity)
    if eq.empty:
        return pd.Series(dtype=float)

    running_peak = eq.cummax()
    dd = eq / running_peak - 1.0
    dd.name = "drawdown"
    return dd


def max_drawdown(equity) -> float:
    dd = drawdown_series(equity)
    if dd.empty:
        return 0.0
    return float(dd.min())


def calmar_ratio(
    returns,
    equity: Optional[pd.Series] = None,
    periods_per_year: int = 252,
) -> float:
    r = _to_series(returns)
    if r.empty:
        return 0.0

    if equity is None:
        equity = equity_curve_from_returns(r)

    eq = _to_series(equity)
    if eq.empty:
        return 0.0

    ann_ret = annualized_return(r, periods_per_year=periods_per_year)
    mdd = abs(max_drawdown(eq))

    if mdd == 0:
        return 0.0

    return float(ann_ret / mdd)


def total_return(returns=None, equity=None, initial_capital: float = 100000.0) -> float:
    if equity is not None:
        eq = _to_series(equity)
        if eq.empty:
            return 0.0
        return float(eq.iloc[-1] / eq.iloc[0] - 1.0) if len(eq) > 1 else 0.0

    r = _to_series(returns)
    if r.empty:
        return 0.0

    eq = equity_curve_from_returns(r, initial_capital=initial_capital)
    return float(eq.iloc[-1] / initial_capital - 1.0)


def win_rate(returns) -> float:
    r = _to_series(returns)
    if r.empty:
        return 0.0
    return float((r > 0).mean())


def profit_factor(returns) -> float:
    r = _to_series(returns)
    if r.empty:
        return 0.0

    gross_profit = float(r[r > 0].sum())
    gross_loss = float(-r[r < 0].sum())

    if gross_loss == 0:
        return 0.0 if gross_profit == 0 else np.inf

    return float(gross_profit / gross_loss)


def expectancy(returns) -> float:
    r = _to_series(returns)
    if r.empty:
        return 0.0
    return float(r.mean())


def hit_ratio(returns) -> float:
    return win_rate(returns)


def payoff_ratio(returns) -> float:
    r = _to_series(returns)
    if r.empty:
        return 0.0

    avg_win = float(r[r > 0].mean()) if (r > 0).any() else 0.0
    avg_loss = float(-r[r < 0].mean()) if (r < 0).any() else 0.0

    if avg_loss == 0:
        return 0.0 if avg_win == 0 else np.inf

    return float(avg_win / avg_loss)


def turnover_from_weights(weights: pd.DataFrame) -> pd.Series:
    if weights.empty:
        return pd.Series(dtype=float)

    w = weights.copy().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    turnover = w.diff().abs().sum(axis=1).fillna(0.0)
    turnover.name = "turnover"
    return turnover


def compute_all_metrics(
    returns,
    equity: Optional[pd.Series] = None,
    weights: Optional[pd.DataFrame] = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    initial_capital: float = 100000.0,
) -> Dict[str, float]:
    r = _to_series(returns)

    if equity is None:
        equity = equity_curve_from_returns(r, initial_capital=initial_capital)

    eq = _to_series(equity)

    metrics = {
        "total_return": total_return(returns=r, equity=eq, initial_capital=initial_capital),
        "annualized_return": annualized_return(r, periods_per_year=periods_per_year),
        "volatility": annualized_volatility(r, periods_per_year=periods_per_year),
        "sharpe": sharpe_ratio(r, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year),
        "sortino": sortino_ratio(r, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year),
        "calmar": calmar_ratio(r, equity=eq, periods_per_year=periods_per_year),
        "max_drawdown": max_drawdown(eq),
        "win_rate": win_rate(r),
        "profit_factor": profit_factor(r),
        "expectancy": expectancy(r),
        "hit_ratio": hit_ratio(r),
        "payoff_ratio": payoff_ratio(r),
    }

    if weights is not None and not weights.empty:
        turnover = turnover_from_weights(weights)
        metrics["average_turnover"] = float(turnover.mean()) if not turnover.empty else 0.0
        metrics["max_turnover"] = float(turnover.max()) if not turnover.empty else 0.0
    else:
        metrics["average_turnover"] = 0.0
        metrics["max_turnover"] = 0.0

    return metrics
