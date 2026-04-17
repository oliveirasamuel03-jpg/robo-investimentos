from __future__ import annotations

import pandas as pd
import yfinance as yf

DEFAULT_TICKERS = [
    "AAPL",
    "MSFT",
    "TSLA",
    "NVDA",
    "SPY",
    "QQQ",
    "VALE3.SA",
    "PETR4.SA",
    "ITUB4.SA",
]


def load_data(
    start: str = "2019-01-01",
    tickers=None,
    history_limit: int | None = 600,
) -> pd.DataFrame:
    if tickers is None:
        tickers = DEFAULT_TICKERS

    data = yf.download(
        tickers=tickers,
        start=start,
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    if data.empty:
        raise ValueError("Nenhum dado retornado do yfinance")

    if isinstance(data.columns, pd.MultiIndex):
        if "Close" not in data.columns.get_level_values(0):
            raise ValueError("Os dados baixados não contêm preços de fechamento")
        prices = data["Close"].copy()
    else:
        prices = data[["Close"]].copy()
        if len(tickers) == 1:
            prices.columns = [tickers[0]]
        else:
            prices.columns = ["Close"]

    prices = prices.sort_index().ffill().dropna(how="all")

    valid_cols = [c for c in prices.columns if prices[c].notna().sum() > 100]
    prices = prices[valid_cols]

    if prices.empty:
        raise ValueError("Sem séries suficientes após limpeza")

    if history_limit is not None and len(prices) > history_limit:
        prices = prices.tail(history_limit)

    return prices
