from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any

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


DATA_DIR = "data"
PAPER_STATE_FILE = os.path.join(DATA_DIR, "paper_state.json")
PAPER_TRADES_FILE = os.path.join(DATA_DIR, "paper_trades.json")


@dataclass
class PaperTradingConfig:
    symbol: str = "AAPL"
    period: str = "6mo"
    interval: str = "1d"


def ensure_paper_files() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(PAPER_STATE_FILE):
        save_paper_state(
            {
                "balance": 10000.0,
                "positions": [],
                "history": [],
            }
        )

    if not os.path.exists(PAPER_TRADES_FILE):
        with open(PAPER_TRADES_FILE, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2)


def load_paper_state() -> dict:
    ensure_paper_files()

    with open(PAPER_STATE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_paper_state(state: dict) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)

    with open(PAPER_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def reset_paper_state() -> dict:
    state = {
        "balance": 10000.0,
        "positions": [],
        "history": [],
    }
    save_paper_state(state)

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(PAPER_TRADES_FILE, "w", encoding="utf-8") as f:
        json.dump([], f, indent=2)

    return state


def read_paper_trades() -> list[dict]:
    ensure_paper_files()

    with open(PAPER_TRADES_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    return []


def _append_paper_trade(trade: dict[str, Any]) -> None:
    trades = read_paper_trades()
    trades.append(trade)

    with open(PAPER_TRADES_FILE, "w", encoding="utf-8") as f:
        json.dump(trades, f, indent=2)


def read_paper_equity() -> list[float]:
    state = load_paper_state()
    balance = float(state.get("balance", 10000.0))
    history = state.get("history", [])

    if not history:
        return [balance]

    curve = [10000.0]
    running = 10000.0

    for item in history:
        total_return = float(item.get("total_return", 0.0))
        running = running * (1 + total_return)
        curve.append(running)

    return curve


def load_prices(config: PaperTradingConfig) -> pd.DataFrame:
    try:
        df = yf.download(
            tickers=config.symbol,
            period=config.period,
            interval=config.interval,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
    except Exception:
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.reset_index()

    df = df.rename(
        columns={
            "Date": "datetime",
            "Datetime": "datetime",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )

    if "close" not in df.columns:
        return pd.DataFrame()

    if "volume" not in df.columns:
        df["volume"] = 0.0

    keep = [c for c in ["datetime", "open", "high", "low", "close", "volume"] if c in df.columns]
    return df[keep].copy()


def fallback_data(rows: int = 120) -> pd.DataFrame:
    base = np.linspace(100, 110, rows)
    return pd.DataFrame(
        {
            "datetime": pd.date_range(end=pd.Timestamp.utcnow(), periods=rows, freq="D"),
            "open": base,
            "high": base * 1.01,
            "low": base * 0.99,
            "close": base,
            "volume": np.zeros(rows),
        }
    )


def _safe_prices(config: PaperTradingConfig) -> pd.DataFrame:
    prices = load_prices(config)

    if prices.empty or "close" not in prices.columns:
        prices = fallback_data()

    return prices


def _build_equity_curve(total_return: float, steps: int = 50) -> list[float]:
    start = 1.0
    end = 1.0 + float(total_return)
    return np.linspace(start, end, steps).tolist()


def build_paper_report(result: dict) -> dict:
    result_block = result.get("result", {})
    signals = result.get("signals", [])

    return {
        "total_return": float(result_block.get("total_return", 0.0)),
        "trades": int(result_block.get("trades", 0)),
        "signals": int(len(signals)),
        "status": "running",
    }


def run_paper_trading_demo(config: PaperTradingConfig = PaperTradingConfig()) -> dict:
    prices = _safe_prices(config)

    ml_config = MLEngineConfig()
    X = build_feature_panel(prices, ml_config)
    y = create_labels(prices)

    min_len = min(len(X), len(y))
    if min_len <= 0:
        prices = fallback_data()
        X = build_feature_panel(prices, ml_config)
        y = create_labels(prices)
        min_len = min(len(X), len(y))

    X = X.iloc[:min_len].copy()
    y = y.iloc[:min_len].copy()

    model = EnsembleProbabilityModel()
    model.fit(X, y)

    probs = model.predict_proba(X)
    signals = (probs[:, 1] > 0.5).astype(int)

    result = backtest(prices.iloc[-len(signals):].copy(), signals)

    equity = _build_equity_curve(result.get("total_return", 0.0), steps=max(len(signals), 50))

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=equity, mode="lines", name="Equity Curve"))

    return {
        "signals": signals.tolist(),
        "result": result,
        "equity": equity,
        "figure": fig,
        "symbol": config.symbol,
    }


def run_paper_cycle() -> dict:
    ensure_paper_files()

    state = load_paper_state()
    result = run_paper_trading_demo()
    report = build_paper_report(result)

    state.setdefault("history", []).append(report)
    save_paper_state(state)

    trades = read_paper_trades()
    equity = read_paper_equity()

    return {
        "state": state,
        "trades": trades,
        "equity": equity,
        "result": result,
        "report": report,
    }
