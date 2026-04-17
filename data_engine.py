import yfinance as yf
import pandas as pd

ASSETS = [
    "AAPL", "MSFT", "TSLA", "NVDA",
    "SPY", "QQQ",
    "VALE3.SA", "PETR4.SA", "ITUB4.SA"
]

def load_data(start="2018-01-01"):
    data = {}
    for asset in ASSETS:
        df = yf.download(asset, start=start, auto_adjust=True)
        df = df[['Close']].dropna()
        df.columns = [asset]
        data[asset] = df
    prices = pd.concat(data.values(), axis=1).dropna()
    return prices