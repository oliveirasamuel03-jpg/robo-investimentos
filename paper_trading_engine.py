from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf

from core.config import RUNTIME_DIR, ensure_app_directories
from core.persistence import (
    append_json_row,
    database_enabled,
    load_json_rows,
    load_json_state,
    replace_json_rows,
    save_json_state,
)


PAPER_STATE_FILE = RUNTIME_DIR / "paper_state.json"
PAPER_TRADES_FILE = RUNTIME_DIR / "paper_trades.json"
PAPER_STATE_NAMESPACE = "paper_state"
PAPER_TRADES_NAMESPACE = "paper_trades"


@dataclass
class PaperTradingConfig:
    initial_capital: float = 10000.0
    custom_tickers: list[str] = field(default_factory=lambda: ["AAPL"])
    period: str = "6mo"
    interval: str = "1h"
    ticket_value: float = 100.0
    min_trade_notional: float = 100.0
    max_open_positions: int = 3
    holding_minutes: int = 60
    history_limit: int = 500
    allow_new_entries: bool = True


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().isoformat()


def _default_state(initial_capital: float) -> dict[str, Any]:
    capital = round(float(initial_capital), 2)
    return {
        "initial_capital": capital,
        "cash": capital,
        "equity": capital,
        "realized_pnl": 0.0,
        "positions": {},
        "last_prices": {},
        "history": [],
        "run_count": 0,
        "updated_at": "",
        "last_trade_at": "",
    }


def ensure_paper_files(config: PaperTradingConfig | None = None) -> None:
    ensure_app_directories()

    initial_capital = float(config.initial_capital) if config else 10000.0

    if database_enabled():
        load_json_state(PAPER_STATE_NAMESPACE, lambda: _default_state(initial_capital), PAPER_STATE_FILE)
        return

    if not PAPER_STATE_FILE.exists():
        save_paper_state(_default_state(initial_capital))

    if not PAPER_TRADES_FILE.exists():
        PAPER_TRADES_FILE.write_text("[]", encoding="utf-8")


def load_paper_state(config: PaperTradingConfig | None = None) -> dict[str, Any]:
    ensure_paper_files(config)
    state = load_json_state(
        PAPER_STATE_NAMESPACE,
        lambda: _default_state(float(config.initial_capital) if config else 10000.0),
        PAPER_STATE_FILE,
    )
    merged = _default_state(float(config.initial_capital) if config else state.get("initial_capital", 10000.0))
    merged.update(state)
    merged["positions"] = merged.get("positions", {}) or {}
    merged["last_prices"] = merged.get("last_prices", {}) or {}
    merged["history"] = merged.get("history", []) or []
    return merged


def save_paper_state(state: dict[str, Any]) -> None:
    ensure_app_directories()
    save_json_state(PAPER_STATE_NAMESPACE, state, PAPER_STATE_FILE)


def reset_paper_state(config: PaperTradingConfig | None = None) -> dict[str, Any]:
    initial_capital = float(config.initial_capital) if config else 10000.0
    state = _default_state(initial_capital)
    save_paper_state(state)
    replace_json_rows(PAPER_TRADES_NAMESPACE, [], PAPER_TRADES_FILE)
    return state


def read_paper_trades(limit: int | None = None) -> list[dict[str, Any]]:
    ensure_paper_files()
    trades = load_json_rows(PAPER_TRADES_NAMESPACE, PAPER_TRADES_FILE, limit=limit)
    if limit is None:
        return trades
    return trades[-int(limit):]


def _append_paper_trade(trade: dict[str, Any]) -> None:
    append_json_row(PAPER_TRADES_NAMESPACE, trade, PAPER_TRADES_FILE)


def read_paper_equity(limit: int | None = None) -> pd.DataFrame:
    state = load_paper_state()
    history = state.get("history", []) or []

    if not history:
        df = pd.DataFrame(
            [
                {
                    "timestamp": state.get("updated_at") or _utc_now_iso(),
                    "cash": float(state.get("cash", 0.0)),
                    "equity": float(state.get("equity", state.get("cash", 0.0))),
                }
            ]
        )
    else:
        df = pd.DataFrame(history)

    if limit is not None and not df.empty:
        df = df.tail(int(limit)).reset_index(drop=True)

    return df


def _fallback_seed(symbol: str) -> int:
    return sum(ord(ch) for ch in symbol) % 17


def fallback_data(symbol: str, rows: int = 240) -> pd.DataFrame:
    seed = _fallback_seed(symbol)
    idx = np.arange(rows, dtype=float)
    base = 100 + seed + idx * 0.12 + np.sin(idx / 8 + seed) * 2.4
    close = np.maximum(base, 1.0)
    open_ = close + np.sin(idx / 5 + seed) * 0.35
    high = np.maximum(open_, close) + 0.6
    low = np.minimum(open_, close) - 0.6
    volume = 1000 + ((idx + seed) % 30) * 25

    return pd.DataFrame(
        {
            "datetime": pd.date_range(end=pd.Timestamp.utcnow(), periods=rows, freq="h"),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def load_prices(symbol: str, config: PaperTradingConfig) -> pd.DataFrame:
    try:
        df = yf.download(
            tickers=symbol,
            period=config.period,
            interval=config.interval,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
    except Exception:
        df = pd.DataFrame()

    if df is None or df.empty:
        return fallback_data(symbol, rows=max(180, config.history_limit))

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.reset_index().rename(
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
        return fallback_data(symbol, rows=max(180, config.history_limit))

    if "volume" not in df.columns:
        df["volume"] = 0.0

    keep = [c for c in ["datetime", "open", "high", "low", "close", "volume"] if c in df.columns]
    return df[keep].tail(config.history_limit).copy()


def _calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50.0)


def _enrich_indicators(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["close"] = pd.to_numeric(data["close"], errors="coerce")
    data["ma50"] = data["close"].rolling(50).mean()
    data["rsi"] = _calculate_rsi(data["close"])
    data["volatility"] = data["close"].pct_change().rolling(12).std().fillna(0.0)
    data["momentum"] = data["close"].pct_change(3).fillna(0.0)
    return data.dropna(subset=["close"]).reset_index(drop=True)


def _normalize_symbol_list(config: PaperTradingConfig) -> list[str]:
    symbols = [str(symbol).strip().upper() for symbol in (config.custom_tickers or []) if str(symbol).strip()]
    return symbols or ["AAPL"]


def _compute_buy_score(latest: pd.Series) -> float:
    price = float(latest["close"])
    ma50 = float(latest.get("ma50", price))
    rsi = float(latest.get("rsi", 50.0))
    vol = float(latest.get("volatility", 0.0))
    momentum = float(latest.get("momentum", 0.0))

    trend_score = 1.0 if price >= ma50 else max(0.0, 1.0 - ((ma50 - price) / max(ma50, 1e-9)) * 8)
    rsi_score = max(0.0, 1.0 - abs(rsi - 52.0) / 30.0)
    vol_score = min(max(vol / 0.015, 0.0), 1.0)
    momentum_score = min(max((momentum + 0.03) / 0.06, 0.0), 1.0)

    return round(0.45 * trend_score + 0.30 * rsi_score + 0.15 * vol_score + 0.10 * momentum_score, 4)


def _should_buy(latest: pd.Series) -> tuple[bool, float]:
    score = _compute_buy_score(latest)
    price = float(latest["close"])
    ma50 = float(latest.get("ma50", price))
    rsi = float(latest.get("rsi", 50.0))
    vol = float(latest.get("volatility", 0.0))

    setup_ok = (
        (price >= ma50 * 0.99 and 35.0 <= rsi <= 67.0)
        or (price >= ma50 and 30.0 <= rsi <= 72.0 and vol >= 0.001)
    )
    return bool(setup_ok and score >= 0.55), score


def _is_position_expired(opened_at: str | None, holding_minutes: int) -> bool:
    if not opened_at:
        return False
    try:
        opened = datetime.fromisoformat(opened_at)
    except ValueError:
        return False
    return _utc_now() >= opened + timedelta(minutes=int(holding_minutes))


def _should_sell(position: dict[str, Any], latest: pd.Series, holding_minutes: int) -> tuple[bool, str]:
    price = float(latest["close"])
    ma50 = float(latest.get("ma50", price))
    rsi = float(latest.get("rsi", 50.0))

    if rsi >= 70.0:
        return True, "rsi alto"
    if price < ma50:
        return True, "perdeu tendencia"
    if _is_position_expired(position.get("opened_at"), holding_minutes):
        return True, "holding expirado"
    return False, ""


def _portfolio_equity(state: dict[str, Any]) -> float:
    cash = float(state.get("cash", 0.0))
    equity = cash

    for asset, position in (state.get("positions", {}) or {}).items():
        last_price = float(state.get("last_prices", {}).get(asset, position.get("last_price", 0.0)))
        equity += float(position.get("quantity", 0.0)) * last_price

    return round(equity, 2)


def build_paper_report(
    result: dict[str, Any] | None = None,
    initial_capital: float | None = None,
) -> dict[str, Any]:
    state = load_paper_state()
    initial = float(initial_capital or state.get("initial_capital", 10000.0))
    equity = float(state.get("equity", state.get("cash", initial)))
    net_profit = round(equity - initial, 2)
    total_return = (net_profit / initial) if initial > 0 else 0.0
    trades = read_paper_trades()
    signals = len(result.get("signals", [])) if isinstance(result, dict) else 0

    return {
        "status": "running",
        "cash": round(float(state.get("cash", 0.0)), 2),
        "equity": equity,
        "realized_pnl": round(float(state.get("realized_pnl", 0.0)), 2),
        "net_profit": net_profit,
        "total_return": total_return,
        "trades": len(trades),
        "trades_count": len(trades),
        "signals": signals,
        "open_positions": len(state.get("positions", {}) or {}),
        "run_count": int(state.get("run_count", 0)),
    }


def run_paper_trading_demo(config: PaperTradingConfig = PaperTradingConfig()) -> dict[str, Any]:
    symbols = _normalize_symbol_list(config)
    first_symbol = symbols[0]
    prices = _enrich_indicators(load_prices(first_symbol, config))
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=prices["datetime"], y=prices["close"], mode="lines", name=first_symbol))

    latest = prices.iloc[-1]
    should_buy, score = _should_buy(latest)

    return {
        "symbol": first_symbol,
        "signals": [{"asset": first_symbol, "score": score, "buy": should_buy}],
        "result": {"total_return": 0.0, "trades": 0},
        "equity": prices["close"].tail(50).tolist(),
        "figure": figure,
    }


def run_paper_cycle(config: PaperTradingConfig = PaperTradingConfig()) -> dict[str, Any]:
    ensure_paper_files(config)
    state = load_paper_state(config)

    if int(state.get("run_count", 0)) == 0 and not state.get("positions") and float(state.get("cash", 0.0)) <= 0:
        state["cash"] = round(float(config.initial_capital), 2)

    symbols = _normalize_symbol_list(config)
    market_data: dict[str, pd.DataFrame] = {}

    for symbol in symbols:
        data = _enrich_indicators(load_prices(symbol, config))
        if data.empty:
            continue
        market_data[symbol] = data
        state["last_prices"][symbol] = round(float(data["close"].iloc[-1]), 6)

    trades_executed: list[dict[str, Any]] = []
    signals: list[dict[str, Any]] = []

    open_positions = dict(state.get("positions", {}) or {})
    for asset, position in list(open_positions.items()):
        data = market_data.get(asset)
        if data is None or data.empty:
            continue

        latest = data.iloc[-1]
        state["last_prices"][asset] = round(float(latest["close"]), 6)
        should_sell, reason = _should_sell(position, latest, config.holding_minutes)
        if not should_sell:
            position["last_price"] = round(float(latest["close"]), 6)
            position["updated_at"] = _utc_now_iso()
            state["positions"][asset] = position
            continue

        quantity = float(position.get("quantity", 0.0))
        price = float(latest["close"])
        gross_value = round(quantity * price, 2)
        avg_price = float(position.get("avg_price", price))
        realized = round((price - avg_price) * quantity, 2)
        state["cash"] = round(float(state.get("cash", 0.0)) + gross_value, 2)
        state["realized_pnl"] = round(float(state.get("realized_pnl", 0.0)) + realized, 2)
        state["last_trade_at"] = _utc_now_iso()
        del state["positions"][asset]

        trade = {
            "timestamp": _utc_now_iso(),
            "asset": asset,
            "side": "SELL",
            "quantity": round(quantity, 6),
            "price": round(price, 6),
            "gross_value": gross_value,
            "cost": 0.0,
            "cash_after": round(float(state["cash"]), 2),
            "realized_pnl": realized,
            "reason": reason,
        }
        trades_executed.append(trade)
        _append_paper_trade(trade)

    slots_left = max(int(config.max_open_positions) - len(state.get("positions", {}) or {}), 0)
    candidates: list[dict[str, Any]] = []

    for asset, data in market_data.items():
        if asset in state.get("positions", {}):
            continue

        latest = data.iloc[-1]
        should_buy, score = _should_buy(latest)
        signal = {
            "asset": asset,
            "price": round(float(latest["close"]), 6),
            "rsi": round(float(latest.get("rsi", 50.0)), 2),
            "score": score,
            "buy": should_buy,
        }
        signals.append(signal)

        if should_buy:
            candidates.append(signal)

    candidates.sort(key=lambda item: item["score"], reverse=True)

    for candidate in candidates[:slots_left]:
        if not config.allow_new_entries:
            break

        cash = float(state.get("cash", 0.0))
        notional = min(float(config.ticket_value), cash)
        if notional < float(config.min_trade_notional):
            continue

        asset = str(candidate["asset"])
        price = float(candidate["price"])
        quantity = round(notional / price, 6)
        if quantity <= 0:
            continue

        state["cash"] = round(cash - notional, 2)
        state["positions"][asset] = {
            "quantity": quantity,
            "avg_price": round(price, 6),
            "last_price": round(price, 6),
            "opened_at": _utc_now_iso(),
            "updated_at": _utc_now_iso(),
        }
        state["last_trade_at"] = _utc_now_iso()

        trade = {
            "timestamp": _utc_now_iso(),
            "asset": asset,
            "side": "BUY",
            "quantity": quantity,
            "price": round(price, 6),
            "gross_value": round(notional, 2),
            "cost": 0.0,
            "cash_after": round(float(state["cash"]), 2),
            "realized_pnl": 0.0,
            "reason": f"score {candidate['score']:.2f}",
        }
        trades_executed.append(trade)
        _append_paper_trade(trade)

    state["run_count"] = int(state.get("run_count", 0)) + 1
    state["updated_at"] = _utc_now_iso()
    state["equity"] = _portfolio_equity(state)

    history = state.get("history", []) or []
    history.append(
        {
            "timestamp": state["updated_at"],
            "cash": round(float(state["cash"]), 2),
            "equity": round(float(state["equity"]), 2),
            "open_positions": len(state.get("positions", {}) or {}),
            "total_return": round(
                (float(state["equity"]) - float(state.get("initial_capital", config.initial_capital)))
                / max(float(state.get("initial_capital", config.initial_capital)), 1e-9),
                6,
            ),
        }
    )
    state["history"] = history[-max(int(config.history_limit), 1) :]
    save_paper_state(state)

    report = build_paper_report({"signals": signals}, initial_capital=float(config.initial_capital))

    return {
        "state": state,
        "trades": read_paper_trades(limit=200),
        "equity": read_paper_equity(limit=200),
        "report": report,
        "signals": signals,
        "trades_executed": len(trades_executed),
    }
