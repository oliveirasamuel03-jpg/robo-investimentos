import pandas as pd
import numpy as np


def backtest(df: pd.DataFrame, signals):
    df = df.copy()

    # garante que signals é 1D
    if isinstance(signals, (pd.DataFrame, np.ndarray)):
        signals = np.array(signals)

        # se vier matriz (probabilidades), pega a coluna de compra
        if signals.ndim == 2:
            signals = signals[:, -1]

    signals = pd.Series(signals).reset_index(drop=True)

    # alinhar tamanho
    df = df.iloc[-len(signals):].copy()
    df = df.reset_index(drop=True)

    df["signal"] = signals

    # garantir coluna close
    if "close" not in df.columns and "Close" in df.columns:
        df["close"] = df["Close"]

    df["return"] = df["close"].pct_change().fillna(0)

    df["strategy"] = df["signal"].shift(1).fillna(0) * df["return"]

    df["equity"] = (1 + df["strategy"]).cumprod()

    results = {
        "total_return": float(df["equity"].iloc[-1] - 1),
        "trades": int((df["signal"].diff().abs() > 0).sum()),
    }

    return results
