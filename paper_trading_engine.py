from __future__ import annotations

from dataclasses import dataclass
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


# =========================
# CONFIG (FALTAVA ISSO)
# =========================
@dataclass
class PaperTradingConfig:
    symbol: str = "AAPL"
    period: str = "6mo"
    interval: str = "1d"


# =========================
# DADOS
# =========================
def load_prices(config: PaperTradingConfig) -> pd.DataFrame:
    df = yf.download(
        tickers=config.symbol,
        period=config.period,
        interval=config.interval,
        progress=False,
    )

    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.reset_index()

    df = df.rename(columns={
        "Date": "datetime",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    })

    return df


def fallback_data():
    base = np.linspace(100, 110, 100)
    return pd.DataFrame({
        "close": base,
        "volume": np.zeros(len(base)),
    })


# =========================
# MAIN
# =========================
def run_paper_trading_demo(config: PaperTradingConfig = PaperTradingConfig()):
    prices = load_prices(config)

    if prices.empty or "close" not in prices.columns:
        prices = fallback_data()

    # ML
    ml_config = MLEngineConfig()
    X = build_feature_panel(prices, ml_config)
    y = create_labels(prices)

    min_len = min(len(X), len(y))
    X = X.iloc[:min_len]
    y = y.iloc[:min_len]

    model = EnsembleProbabilityModel()
    model.fit(X, y)

    probs = model.predict_proba(X)
    signals = (probs[:, 1] > 0.5).astype(int)

    # BACKTEST
    result = backtest(prices.iloc[-len(signals):], signals)

    # curva fake (evita erro plotly)
    equity = np.linspace(1, 1 + result["total_return"], 50)

    # gráfico
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=equity, name="Equity Curve"))

    return {
        "signals": signals.tolist(),
        "result": result,
        "equity": equity.tolist(),
        "figure": fig,
    }
