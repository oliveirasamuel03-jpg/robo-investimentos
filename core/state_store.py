from __future__ import annotations

from copy import deepcopy
from datetime import datetime

import pandas as pd

from core.config import (
    BOT_LOG_COLUMNS,
    BOT_LOG_FILE,
    BOT_STATE_FILE,
    BROKER_MODE,
    BROKER_PROVIDER,
    MARKET_DATA_PROVIDER,
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
    "market_data": {
        "provider": MARKET_DATA_PROVIDER,
        "status": "unknown",
        "last_sync_at": "",
        "last_success_at": "",
        "last_error": "",
        "last_source": "",
        "source_breakdown": {},
        "symbols": [],
        "requested_by": "",
        "contexts": {},
    },
    "broker": {
        "provider": BROKER_PROVIDER,
        "mode": BROKER_MODE,
        "status": "paper",
        "last_sync_at": "",
        "last_error": "",
        "account_id": "",
    },
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


def update_market_data_status(status_payload: dict | None) -> dict:
    state = load_bot_state()
    market_state = state.get("market_data", {}) or {}
    payload = status_payload or {}
    context_name = str(payload.get("requested_by") or "runtime")
    contexts = market_state.get("contexts", {}) or {}
    context_state = dict(contexts.get(context_name, {}) or {})

    if payload.get("provider"):
        context_state["provider"] = str(payload.get("provider"))
    if payload.get("status"):
        context_state["status"] = str(payload.get("status"))
    if payload.get("last_sync_at"):
        context_state["last_sync_at"] = str(payload.get("last_sync_at"))
    if payload.get("last_source"):
        context_state["last_source"] = str(payload.get("last_source"))
    if isinstance(payload.get("source_breakdown"), dict):
        context_state["source_breakdown"] = dict(payload.get("source_breakdown") or {})
    if payload.get("symbols") is not None:
        context_state["symbols"] = [str(symbol).upper() for symbol in (payload.get("symbols") or [])]
    context_state["requested_by"] = context_name

    source_breakdown = context_state.get("source_breakdown", {}) or {}
    if int(source_breakdown.get("market", 0) or 0) > 0 or int(source_breakdown.get("cached", 0) or 0) > 0:
        context_state["last_success_at"] = context_state.get("last_sync_at", "")
        context_state["last_error"] = ""
    elif payload.get("last_error"):
        context_state["last_error"] = str(payload.get("last_error"))

    contexts[context_name] = context_state
    market_state["contexts"] = contexts

    should_promote_to_top_level = context_name == "worker_cycle" or not market_state.get("requested_by")
    if should_promote_to_top_level:
        for key in (
            "provider",
            "status",
            "last_sync_at",
            "last_success_at",
            "last_error",
            "last_source",
            "source_breakdown",
            "symbols",
            "requested_by",
        ):
            if key in context_state:
                market_state[key] = context_state.get(key)

    state["market_data"] = market_state
    save_bot_state(state)
    return context_state


def update_broker_status(status_payload: dict | None) -> dict:
    state = load_bot_state()
    broker_state = state.get("broker", {}) or {}
    payload = status_payload or {}

    for key in ("provider", "mode", "status", "last_sync_at", "last_error", "account_id"):
        if payload.get(key) is not None:
            broker_state[key] = payload.get(key)

    state["broker"] = broker_state
    save_bot_state(state)
    return broker_state
