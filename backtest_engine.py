import numpy as np
import pandas as pd

def backtest(prices, signals, fee=0.001, slippage=0.001):
    returns = prices.pct_change().fillna(0)

    portfolio = []
    position = 0

    for i in range(1, len(prices)):
        signal = signals[i-1]

        position = signal - fee - slippage
        ret = position * returns.iloc[i].mean()

        portfolio.append(ret)

    equity = np.cumsum(portfolio)
    return pd.Series(equity)