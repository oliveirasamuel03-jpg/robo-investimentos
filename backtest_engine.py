from __future__ import annotations

import numpy as np
import pandas as pd


def _ensure_close(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if df is None or len(df) == 0:
        return pd.DataFrame({
            "close": np.linspace(100, 101, 50)
        })

    cols = {str(c).lower(): c for c in df.columns}

    if "close" in cols:
        df["close"] = df[cols["close"]]
    elif "adj close" in cols:
        df["close"] = df[cols["adj close"]]
    else:
        df["close"] = np.linspace(100, 101, len(df))

    return df


def _normalize_signals(signals, target_len: int) -> pd.Series:
    if isinstance(signals, pd.DataFrame):
        arr = signals.to_numpy()
    elif isinstance(signals, pd.Series):
        arr = signals.to_numpy()
    else:
        arr = np.array(signals)

    if arr.ndim == 2:
        arr = arr[:, -1]

    arr = np.asarray(arr).reshape(-1)

    if len(arr) == 0:
        arr = np.zeros(target_len)

    s = pd.Series(arr).fillna(0).reset_index(drop=True)

    if len(s) > target_len:
        s = s.iloc[-target_len:].reset_index(drop=True)
    elif len(s) < target_len:
        pad = pd.Series(np.zeros(target_len - len(s)))
        s = pd.concat([pad, s], ignore_index=True)

    return s.astype(float)


def backtest(df: pd.DataFrame, signals):
    df = _ensure_close(df).copy().reset_index(drop=True)

    signals = _normalize_signals(signals, len(df))

    df["signal"] = signals
    df["return"] = df["close"].pct_change().fillna(0.0)
    df["strategy"] = df["signal"].shift(1).fillna(0.0) * df["return"]
    df["equity"] = (1.0 + df["strategy"]).cumprod()

    total_return = float(df["equity"].iloc[-1] - 1.0) if len(df) else 0.0
    trades = int((df["signal"].diff().abs().fillna(0) > 0).sum()) if len(df) else 0

    return {
        "total_return": total_return,
        "trades": trades,
    }


def backtest_portfolio(prices: pd.DataFrame, signals):
    """
    Compatível com walk_forward.py
    """
    prices = _ensure_close(prices).copy().reset_index(drop=True)
    signals = _normalize_signals(signals, len(prices))

    prices["signal"] = signals
    prices["return"] = prices["close"].pct_change().fillna(0.0)
    prices["strategy"] = prices["signal"].shift(1).fillna(0.0) * prices["return"]
    prices["equity"] = (1.0 + prices["strategy"]).cumprod()

    equity_curve = prices["equity"].tolist()

    metrics = {
        "total_return": float(prices["equity"].iloc[-1] - 1.0) if len(prices) else 0.0,
        "trades": int((prices["signal"].diff().abs().fillna(0) > 0).sum()) if len(prices) else 0,
    }

    return {
        "equity_curve": equity_curve,
        "metrics": metrics,
    }
