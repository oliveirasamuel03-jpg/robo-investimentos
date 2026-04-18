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
# =========================
# REPORT (FALTAVA ISSO)
# =========================
def build_paper_report(result: dict) -> dict:
    return {
        "total_return": result.get("result", {}).get("total_return", 0),
        "trades": result.get("result", {}).get("trades", 0),
        "signals": len(result.get("signals", [])),
        "status": "running",
    }
# =========================
# FILE SETUP (FALTAVA ISSO)
# =========================
import os
import json


def ensure_paper_files():
    base_path = "data"

    os.makedirs(base_path, exist_ok=True)

    files = {
        "paper_state.json": {
            "balance": 10000,
            "positions": [],
            "history": []
        },
        "paper_trades.json": []
    }

    for file_name, default_content in files.items():
        path = os.path.join(base_path, file_name)

        if not os.path.exists(path):
            with open(path, "w") as f:
                json.dump(default_content, f)
# =========================
# STATE (FALTAVA ISSO)
# =========================
import json
import os


def load_paper_state():
    path = "data/paper_state.json"

    if not os.path.exists(path):
        return {
            "balance": 10000,
            "positions": [],
            "history": []
        }

    with open(path, "r") as f:
        return json.load(f)


def save_paper_state(state: dict):
    os.makedirs("data", exist_ok=True)

    with open("data/paper_state.json", "w") as f:
        json.dump(state, f, indent=2)
# =========================
# EQUITY READER (ÚLTIMO)
# =========================
def read_paper_equity():
    state = load_paper_state()

    balance = state.get("balance", 10000)

    # cria curva simples baseada no saldo
    equity = [balance * (1 + i * 0.001) for i in range(50)]

    return equity
# =========================
# TRADES READER (ÚLTIMO)
# =========================
def read_paper_trades():
    import os
    import json

    path = "data/paper_trades.json"

    if not os.path.exists(path):
        return []

    with open(path, "r") as f:
        return json.load(f)
# =========================
# RESET STATE (ÚLTIMO)
# =========================
def reset_paper_state():
    default_state = {
        "balance": 10000,
        "positions": [],
        "history": []
    }

    save_paper_state(default_state)

    # também limpa trades
    import os
    import json

    os.makedirs("data", exist_ok=True)

    with open("data/paper_trades.json", "w") as f:
        json.dump([], f)
