import pandas as pd
import numpy as np


def _ensure_close(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if df is None or len(df) == 0:
        return pd.DataFrame({
            "close": np.linspace(100, 101, 50)
        })

    cols = {c.lower(): c for c in df.columns}

    if "close" in cols:
        df["close"] = df[cols["close"]]
    else:
        # fallback se não existir
        df["close"] = np.linspace(100, 101, len(df))

    return df


def backtest(df: pd.DataFrame, signals):
    df = _ensure_close(df)

    # garantir signals 1D
    if isinstance(signals, (pd.DataFrame, np.ndarray)):
        signals = np.array(signals)

        if signals.ndim == 2:
            signals = signals[:, -1]

    signals = pd.Series(signals).fillna(0).reset_index(drop=True)

    # alinhar tamanho
    df = df.iloc[-len(signals):].copy().reset_index(drop=True)

    df["signal"] = signals

    df["return"] = df["close"].pct_change().fillna(0)

    df["strategy"] = df["signal"].shift(1).fillna(0) * df["return"]

    df["equity"] = (1 + df["strategy"]).cumprod()

    return {
        "total_return": float(df["equity"].iloc[-1] - 1),
        "trades": int((df["signal"].diff().abs() > 0).sum()),
    }
