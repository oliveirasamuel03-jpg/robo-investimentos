import pandas as pd


def backtest(df: pd.DataFrame, signals: pd.Series):
    df = df.copy()

    df["signal"] = signals
    df["return"] = df["close"].pct_change()

    df["strategy"] = df["signal"].shift(1) * df["return"]

    df["equity"] = (1 + df["strategy"]).cumprod()

    results = {
        "total_return": float(df["equity"].iloc[-1] - 1),
        "trades": int((df["signal"].diff().abs() > 0).sum()),
    }

    return results
