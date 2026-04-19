import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf

from core.config import (
    LEGACY_BOT_LOG_FILE,
    LEGACY_BOT_STATE_FILE,
    LEGACY_RUNTIME_CONFIG_FILE,
    ensure_app_directories,
)

CONFIG_PATH = LEGACY_RUNTIME_CONFIG_FILE
STATE_PATH = LEGACY_BOT_STATE_FILE
LOG_PATH = LEGACY_BOT_LOG_FILE

DEFAULT_TICKERS = [
    "VALE3.SA",
    "PETR4.SA",
    "ITUB4.SA",
    "BBDC4.SA",
    "BBAS3.SA",
    "WEGE3.SA",
]


@dataclass
class StrategyConfig:
    capital_inicial: float = 1000.0
    capital_por_operacao: float = 250.0
    media_curta: int = 9
    media_longa: int = 21
    media_tendencia: int = 50
    periodo_rsi: int = 14
    rsi_compra_max: float = 68.0
    rsi_venda_min: float = 75.0
    stop_loss_pct: float = 3.0
    taxa_operacao_pct: float = 0.1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_RUNTIME = {
    "enabled": False,
    "mode": "paper",
    "interval_minutes": 5,
    "capital_total": 1000.0,
    "capital_por_operacao": 250.0,
    "backtest_start": "2020-01-01",
    "tickers": DEFAULT_TICKERS,
    "strategy": StrategyConfig().to_dict(),
}

DEFAULT_STATE = {
    "cash": 1000.0,
    "positions": {},
    "trade_history": [],
    "last_run_at": None,
    "run_count": 0,
}


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def ensure_runtime_files() -> None:
    ensure_app_directories()
    if not CONFIG_PATH.exists():
        CONFIG_PATH.write_text(json.dumps(DEFAULT_RUNTIME, indent=2), encoding="utf-8")
    if not STATE_PATH.exists():
        STATE_PATH.write_text(json.dumps(DEFAULT_STATE, indent=2), encoding="utf-8")
    if not LOG_PATH.exists():
        LOG_PATH.write_text("", encoding="utf-8")


def load_json(path: Path, fallback: dict) -> dict:
    if not path.exists():
        return fallback.copy()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return fallback.copy()
    return payload if isinstance(payload, dict) else fallback.copy()


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")


def load_runtime_config() -> dict:
    ensure_runtime_files()
    runtime = load_json(CONFIG_PATH, DEFAULT_RUNTIME)
    runtime.setdefault("strategy", StrategyConfig().to_dict())
    runtime.setdefault("tickers", DEFAULT_TICKERS)
    runtime.setdefault("mode", "paper")
    runtime.setdefault("enabled", False)
    return runtime


def save_runtime_config(runtime: dict) -> None:
    save_json(CONFIG_PATH, runtime)


def load_bot_state() -> dict:
    ensure_runtime_files()
    state = load_json(STATE_PATH, DEFAULT_STATE)
    state.setdefault("cash", float(DEFAULT_STATE["cash"]))
    state.setdefault("positions", {})
    state.setdefault("trade_history", [])
    state.setdefault("run_count", 0)
    return state


def save_bot_state(state: dict) -> None:
    save_json(STATE_PATH, state)


def reset_bot_state(capital_total: float) -> dict:
    state = {
        "cash": float(capital_total),
        "positions": {},
        "trade_history": [],
        "last_run_at": None,
        "run_count": 0,
    }
    save_bot_state(state)
    append_log("INFO", "Conta paper reiniciada.", {"capital_total": capital_total})
    return state


def append_log(level: str, message: str, context: dict | None = None) -> None:
    payload = {
        "timestamp": now_iso(),
        "level": level,
        "message": message,
        "context": context or {},
    }
    with LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(payload, ensure_ascii=True) + "\n")


def read_logs(limit: int = 50) -> list[dict]:
    ensure_runtime_files()
    lines = LOG_PATH.read_text(encoding="utf-8").splitlines()
    rows: list[dict] = []
    for line in lines[-limit:]:
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data.columns = [str(col).title() for col in data.columns]
    return data


def download_ticker_data(ticker: str, start_date: date) -> pd.DataFrame:
    df = yf.download(
        ticker,
        start=start_date.isoformat(),
        end=(date.today() + timedelta(days=1)).isoformat(),
        progress=False,
        auto_adjust=False,
    )
    df = normalize_columns(df)
    if df.empty:
        raise ValueError(f"Sem dados para {ticker}")
    return df


def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, pd.NA)
    return (100 - (100 / (1 + rs))).fillna(50)


def enrich_indicators(df: pd.DataFrame, strategy: StrategyConfig) -> pd.DataFrame:
    data = df.copy()
    close = data["Close"]
    data["media_curta"] = close.rolling(strategy.media_curta).mean()
    data["media_longa"] = close.rolling(strategy.media_longa).mean()
    data["media_tendencia"] = close.rolling(strategy.media_tendencia).mean()
    data["rsi"] = calc_rsi(close, strategy.periodo_rsi)
    data["sinal"] = 0
    data.loc[data["media_curta"] > data["media_longa"], "sinal"] = 1
    data.loc[data["media_curta"] < data["media_longa"], "sinal"] = -1
    return data.dropna().copy()


def max_drawdown(equity_curve: pd.Series) -> float:
    running_peak = equity_curve.cummax()
    drawdown = (equity_curve / running_peak) - 1
    return float(drawdown.min()) if not drawdown.empty else 0.0


def run_backtest_for_ticker(ticker: str, strategy: StrategyConfig, start_date: date) -> dict:
    data = enrich_indicators(download_ticker_data(ticker, start_date), strategy)
    capital = float(strategy.capital_inicial)
    quantity = 0.0
    buy_price = 0.0
    fee_factor = 1 - (strategy.taxa_operacao_pct / 100)
    stop_factor = 1 - (strategy.stop_loss_pct / 100)
    equity = []

    for _, row in data.iterrows():
        price = float(row["Close"])
        signal = int(row["sinal"])
        rsi = float(row["rsi"])
        trend = float(row["media_tendencia"])

        if signal == 1 and capital > 0 and rsi <= strategy.rsi_compra_max and price >= trend:
            quantity = (capital * fee_factor) / price
            capital = 0.0
            buy_price = price
        elif quantity > 0 and (signal == -1 or rsi >= strategy.rsi_venda_min or price <= buy_price * stop_factor):
            capital = quantity * price * fee_factor
            quantity = 0.0
            buy_price = 0.0

        equity.append(capital + quantity * price)

    equity_curve = pd.Series(equity, index=data.index, name="equity")
    final_value = float(equity_curve.iloc[-1])
    summary = {
        "Ativo": ticker,
        "Valor final": final_value,
        "Lucro": final_value - strategy.capital_inicial,
        "Retorno %": ((final_value / strategy.capital_inicial) - 1) * 100,
        "Drawdown %": max_drawdown(equity_curve) * 100,
    }
    return {"summary": summary, "data": data, "equity": equity_curve}


def make_trade_record(ticker: str, side: str, quantity: float, price: float, reason: str) -> dict:
    return {
        "timestamp": now_iso(),
        "ticker": ticker,
        "side": side,
        "quantity": round(quantity, 6),
        "price": round(price, 4),
        "gross_value": round(quantity * price, 2),
        "reason": reason,
    }


def _position_snapshot(quantity: float, avg_price: float, stop_price: float, last_price: float, last_rsi: float) -> dict:
    return {
        "quantity": round(quantity, 6),
        "avg_price": round(avg_price, 4),
        "stop_price": round(stop_price, 4),
        "last_price": round(last_price, 4),
        "last_rsi": round(last_rsi, 2),
        "updated_at": now_iso(),
    }


def evaluate_ticker(state: dict, runtime: dict, ticker: str, data: pd.DataFrame, strategy: StrategyConfig) -> list[dict]:
    latest = data.iloc[-1]
    price = float(latest["Close"])
    signal = int(latest["sinal"])
    rsi = float(latest["rsi"])
    trend = float(latest["media_tendencia"])
    fee_factor = 1 - (strategy.taxa_operacao_pct / 100)
    max_capital = min(float(runtime["capital_por_operacao"]), float(state["cash"]))
    trades = []

    if ticker in state["positions"]:
        position = state["positions"][ticker]
        stop_price = float(position["stop_price"])
        sell_signal = signal == -1 or rsi >= strategy.rsi_venda_min or price <= stop_price
        if sell_signal:
            quantity = float(position["quantity"])
            proceeds = quantity * price * fee_factor
            state["cash"] = round(float(state["cash"]) + proceeds, 2)
            trades.append(make_trade_record(ticker, "SELL", quantity, price, "saida tecnica"))
            del state["positions"][ticker]
        else:
            state["positions"][ticker] = _position_snapshot(
                quantity=float(position["quantity"]),
                avg_price=float(position["avg_price"]),
                stop_price=float(position["stop_price"]),
                last_price=price,
                last_rsi=rsi,
            )
        return trades

    should_buy = signal == 1 and rsi <= strategy.rsi_compra_max and price >= trend
    if should_buy and max_capital >= 50:
        quantity = (max_capital * fee_factor) / price
        if quantity > 0 and not math.isclose(quantity, 0.0):
            state["cash"] = round(float(state["cash"]) - max_capital, 2)
            stop_price = price * (1 - (strategy.stop_loss_pct / 100))
            state["positions"][ticker] = _position_snapshot(quantity, price, stop_price, price, rsi)
            trades.append(make_trade_record(ticker, "BUY", quantity, price, "entrada tecnica"))
    return trades


def scan_market() -> dict:
    ensure_runtime_files()
    runtime = load_runtime_config()
    state = load_bot_state()
    strategy_data = dict(runtime["strategy"])
    strategy_data["capital_inicial"] = float(runtime["capital_total"])
    strategy_data["capital_por_operacao"] = float(runtime["capital_por_operacao"])
    strategy = StrategyConfig(**strategy_data)

    if runtime.get("mode") != "paper":
        append_log("ERROR", "Modo real ainda nao implementado. Nenhuma ordem enviada.")
        return {"status": "blocked", "errors": ["real mode not implemented"], "trades": []}

    if float(state["cash"]) <= 0 and not state["positions"]:
        state["cash"] = float(runtime["capital_total"])

    trades = []
    errors = []
    start_date = date.fromisoformat(runtime["backtest_start"])

    for ticker in runtime["tickers"]:
        try:
            data = enrich_indicators(download_ticker_data(ticker, start_date), strategy)
            ticker_trades = evaluate_ticker(state, runtime, ticker, data, strategy)
            trades.extend(ticker_trades)
        except Exception as exc:
            errors.append(f"{ticker}: {exc}")

    state["trade_history"].extend(trades)
    state["trade_history"] = state["trade_history"][-300:]
    state["last_run_at"] = now_iso()
    state["run_count"] = int(state.get("run_count", 0)) + 1
    save_bot_state(state)

    if trades:
        for trade in trades:
            append_log("INFO", f"{trade['side']} {trade['ticker']} por {trade['price']}", trade)
    if errors:
        for error in errors:
            append_log("ERROR", error)
    if not trades and not errors:
        append_log("INFO", "Ciclo executado sem novas ordens.")

    return {
        "status": "ok" if not errors else "partial",
        "trades": trades,
        "errors": errors,
        "positions": state["positions"],
        "cash": state["cash"],
        "last_run_at": state["last_run_at"],
    }


def run_forever(sleep_seconds: int | None = None) -> None:
    ensure_runtime_files()
    append_log("INFO", "Runner iniciado.")
    while True:
        runtime = load_runtime_config()
        if runtime.get("enabled"):
            scan_market()
        else:
            append_log("INFO", "Runner acordou, mas o robo esta pausado.")

        wait_seconds = sleep_seconds or int(runtime.get("interval_minutes", 5)) * 60
        time.sleep(max(wait_seconds, 30))
