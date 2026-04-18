from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf

from ml_engine import (
    MLEngineConfig,
    EnsembleProbabilityModel,
    build_feature_panel,
    create_labels,
)
from backtest_engine import backtest


def load_prices(symbol: str = "AAPL", period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    df = yf.download(
        tickers=symbol,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.reset_index()

    rename_map = {
        "Date": "datetime",
        "Datetime": "datetime",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }
    df = df.rename(columns=rename_map)

    if "close" not in df.columns:
        return pd.DataFrame()

    if "volume" not in df.columns:
        df["volume"] = 0.0

    keep_cols = [c for c in ["datetime", "open", "high", "low", "close", "volume"] if c in df.columns]
    return df[keep_cols].copy()


def _build_fallback_prices(n: int = 120) -> pd.DataFrame:
    base = np.linspace(100, 110, n)
    return pd.DataFrame(
        {
            "datetime": pd.date_range(end=pd.Timestamp.utcnow(), periods=n, freq="D"),
            "open": base,
            "high": base * 1.01,
            "low": base * 0.99,
            "close": base,
            "volume": np.zeros(n),
        }
    )


def _build_equity_curve(total_return: float, steps: int = 50) -> list[float]:
    start = 1.0
    end = 1.0 + float(total_return)
    return np.linspace(start, end, steps).tolist()


def run_paper_trading_demo(symbol: str = "AAPL") -> dict:
    prices = load_prices(symbol=symbol)

    if prices.empty:
        prices = _build_fallback_prices()

    config = MLEngineConfig()
    X = build_feature_panel(prices, config)
    y = create_labels(prices)

    if len(X) == 0:
        prices = _build_fallback_prices()
        X = build_feature_panel(prices, config)
        y = create_labels(prices)

    min_len = min(len(X), len(y))
    X = X.iloc[:min_len].copy()
    y = y.iloc[:min_len].copy()

    model = EnsembleProbabilityModel()
    model.fit(X, y)

    probs = model.predict_proba(X)
    signals = (probs[:, 1] > 0.5).astype(int)

    bt = backtest(prices.iloc[-len(signals):].copy(), signals)

    equity_curve = _build_equity_curve(bt.get("total_return", 0.0), steps=max(len(signals), 50))

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=equity_curve, mode="lines", name="Equity Curve"))
    fig.update_layout(
        template="plotly_dark",
        title=f"Paper Trading Demo - {symbol}",
        xaxis_title="Passos",
        yaxis_title="Equity",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=420,
    )

    return {
        "symbol": symbol,
        "prices_rows": int(len(prices)),
        "features_rows": int(len(X)),
        "signals_count": int(len(signals)),
        "signals": signals.tolist(),
        "probabilities": probs[:, 1].tolist(),
        "backtest": bt,
        "equity_curve": equity_curve,
        "figure": fig,
    }
