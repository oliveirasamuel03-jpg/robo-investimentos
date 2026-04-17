import numpy as np

def sharpe(returns):
    return np.sqrt(252) * returns.mean() / (returns.std() + 1e-9)

def sortino(returns):
    downside = returns[returns < 0].std()
    return np.sqrt(252) * returns.mean() / (downside + 1e-9)

def max_drawdown(equity):
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return dd.min()

def calmar(returns, equity):
    dd = abs(max_drawdown(equity))
    return (returns.mean() * 252) / (dd + 1e-9)