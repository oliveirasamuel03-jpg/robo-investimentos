from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime

import pandas as pd

from core.config import BOT_LOG_FILE, BOT_STATE_FILE, INVESTOR_ORDERS_FILE, STORAGE_DIR, TRADER_ORDERS_FILE

DEFAULT_STATE = {
    "wallet_value": 10000.0,
    "cash": 10000.0,
    "bot_status": "PAUSED",
    "bot_mode": "Automático",
    "reserve_cash_pct": 0.10,
    "start_date": "2019-01-01",
    "probability_threshold": 0.60,
    "top_n": 2,
    "use_regime_filter": True,
    "use_volatility_filter": False,
    "vol_threshold": 0.03,
    "cash_threshold": 0.50,
    "target_portfolio_vol": 0.08,
    "include_brazil_stocks": True,
    "include_us_stocks": True,
    "include_etfs": True,
    "include_fiis": True,
    "include_crypto": True,
    "include_grains": True,
    "custom_tickers": [],
    "realized_pnl": 0.0,
    "positions": [],
    "last_action": "Nenhuma ação recente",
    "last_run_at": "",
    "next_run_at": "",
    "worker_status": "offline",
    "worker_heartbeat": "",
    "trader": {
        "enabled": True,
        "ticket_value": 100.0,
        "holding_minutes": 60,
        "max_open_positions": 3,
        "watchlist": ["BTC-USD", "ETH-USD", "VALE3.SA", "PETR4.SA", "AAPL", "KC=F"],
    },
    "investment": {
        "enabled": True,
        "budget": 1000.0,
        "rebalance_days": 7,
        "max_positions": 5,
        "watchlist": ["SPY", "QQQ", "WEGE3.SA", "HGLG11.SA", "IVVB11.SA"],
        "last_report": {},
    },
}


def _ensure_csv(file_path, columns: list[str]) -> None:
    if not file_path.exists():
        pd.DataFrame(columns=columns).to_csv(file_path, index=False)


def ensure_storage() -> None:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    if not BOT_STATE_FILE.exists():
        save_bot_state(deepcopy(DEFAULT_STATE))

    _ensure_csv(
        TRADER_ORDERS_FILE,
        ["timestamp", "asset", "side", "quantity", "price", "gross_value", "cost", "cash_after", "source"],
    )
    _ensure_csv(
        INVESTOR_ORDERS_FILE,
        ["timestamp", "metric", "value", "notes"],
    )
    _ensure_csv(
        BOT_LOG_FILE,
        ["timestamp", "level", "message"],
    )


def _merge_missing_keys(current: dict, default: dict) -> dict:
    for key, value in default.items():
        if key not in current:
            current[key] = deepcopy(value)
        elif isinstance(value, dict) and isinstance(current.get(key), dict):
            current[key] = _merge_missing_keys(current[key], value)
    return current


def load_bot_state() -> dict:
    ensure_storage()
    with open(BOT_STATE_FILE, "r", encoding="utf-8") as f:
        state = json.load(f)

    state = _merge_missing_keys(state, deepcopy(DEFAULT_STATE))
    return state


def save_bot_state(state: dict) -> None:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    with open(BOT_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def reset_state() -> dict:
    state = deepcopy(DEFAULT_STATE)
    save_bot_state(state)
    return state


def append_csv_row(file_path, row: dict) -> None:
    ensure_storage()
    df = pd.read_csv(file_path)
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(file_path, index=False)


def log_event(level: str, message: str) -> None:
    append_csv_row(
        BOT_LOG_FILE,
        {
            "timestamp": datetime.utcnow().isoformat(),
            "level": level,
            "message": message,
        },
    )


def update_worker_heartbeat(status: str = "online") -> None:
    state = load_bot_state()
    state["worker_status"] = status
    state["worker_heartbeat"] = datetime.utcnow().isoformat()
    save_bot_state(state)
