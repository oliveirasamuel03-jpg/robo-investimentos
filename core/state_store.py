from __future__ import annotations

from copy import deepcopy
from datetime import datetime

import pandas as pd

from core.config import (
    BOT_LOG_COLUMNS,
    BOT_LOG_FILE,
    BOT_STATE_FILE,
    TRADER_REPORTS_COLUMNS,
    TRADER_REPORTS_FILE,
    TRADER_ORDERS_COLUMNS,
    TRADER_ORDERS_FILE,
    ensure_app_directories,
)
from core.persistence import (
    append_table_row,
    database_enabled,
    load_json_state,
    read_table,
    replace_table,
    save_json_state,
)
from core.trader_profiles import DEFAULT_TRADER_PROFILE


DEFAULT_STATE = {
    "wallet_value": 10000.0,
    "cash": 10000.0,
    "bot_status": "PAUSED",
    "bot_mode": "Automatico",
    "realized_pnl": 0.0,
    "positions": [],
    "last_action": "Nenhuma acao recente",
    "last_run_at": "",
    "next_run_at": "",
    "worker_status": "offline",
    "worker_heartbeat": "",
    "security": {
        "real_mode_enabled": False,
        "real_mode_enabled_by": "",
        "real_mode_enabled_at": "",
    },
    "trader": {
        "enabled": True,
        "profile": DEFAULT_TRADER_PROFILE,
        "ticket_value": 100.0,
        "holding_minutes": 60,
        "max_open_positions": 3,
        "watchlist": ["BTC-USD", "ETH-USD", "VALE3.SA", "PETR4.SA", "AAPL", "KC=F"],
    },
}


def _ensure_csv(file_path, columns: list[str]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    if not file_path.exists():
        pd.DataFrame(columns=columns).to_csv(file_path, index=False)


def ensure_storage() -> None:
    ensure_app_directories()
    if database_enabled():
        load_json_state("bot_state", lambda: deepcopy(DEFAULT_STATE), BOT_STATE_FILE)
        return

    if not BOT_STATE_FILE.exists():
        save_bot_state(deepcopy(DEFAULT_STATE))

    _ensure_csv(TRADER_ORDERS_FILE, TRADER_ORDERS_COLUMNS)
    _ensure_csv(TRADER_REPORTS_FILE, TRADER_REPORTS_COLUMNS)
    _ensure_csv(BOT_LOG_FILE, BOT_LOG_COLUMNS)


def _merge_missing_keys(current: dict, default: dict) -> dict:
    for key, value in default.items():
        if key not in current:
            current[key] = deepcopy(value)
        elif isinstance(value, dict) and isinstance(current.get(key), dict):
            current[key] = _merge_missing_keys(current[key], value)
    return current


def load_bot_state() -> dict:
    ensure_storage()
    state = load_json_state("bot_state", lambda: deepcopy(DEFAULT_STATE), BOT_STATE_FILE)
    state = _merge_missing_keys(state, deepcopy(DEFAULT_STATE))
    return state


def save_bot_state(state: dict) -> None:
    ensure_app_directories()
    save_json_state("bot_state", state, BOT_STATE_FILE)


def reset_state() -> dict:
    state = deepcopy(DEFAULT_STATE)
    save_bot_state(state)
    return state


def append_csv_row(file_path, row: dict) -> None:
    ensure_storage()
    columns_map = {
        str(TRADER_ORDERS_FILE): TRADER_ORDERS_COLUMNS,
        str(TRADER_REPORTS_FILE): TRADER_REPORTS_COLUMNS,
        str(BOT_LOG_FILE): BOT_LOG_COLUMNS,
    }
    append_table_row(file_path, row, columns=columns_map.get(str(file_path)))


def read_storage_table(file_path, columns: list[str] | None = None) -> pd.DataFrame:
    ensure_storage()
    return read_table(file_path, columns=columns)


def replace_storage_table(file_path, rows: list[dict], columns: list[str] | None = None) -> None:
    ensure_storage()
    replace_table(file_path, rows, columns=columns)


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
