import pandas as pd
import numpy as np


def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def generate_signals(df: pd.DataFrame):
    df = df.copy()

    # garantir coluna close
    if "close" not in df.columns and "Close" in df.columns:
        df["close"] = df["Close"]

    # indicadores
    df["ma50"] = df["close"].rolling(50).mean()
    df["rsi"] = calculate_rsi(df["close"])
    df["volatility"] = df["close"].pct_change().rolling(10).std()

    signals = []

    for i in range(len(df)):
        if i < 50:
            signals.append(0)
            continue

        price = df["close"].iloc[i]
        ma50 = df["ma50"].iloc[i]
        rsi = df["rsi"].iloc[i]
        vol = df["volatility"].iloc[i]

        # filtro de volatilidade
        if vol < 0.005:
            signals.append(0)
            continue

        # COMPRA
        if price > ma50 and 40 < rsi < 60:
            signals.append(1)

        # VENDA
        elif rsi > 70 or price < ma50:
            signals.append(0)

        else:
            signals.append(0)

    return pd.Series(signals)
