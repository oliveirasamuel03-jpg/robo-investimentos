from __future__ import annotations

from copy import deepcopy
from datetime import datetime

import pandas as pd

from core.config import (
    ALERT_EMAIL_ENABLED,
    ALERT_EMAIL_PROVIDER,
    BOT_LOG_COLUMNS,
    BOT_LOG_FILE,
    BOT_STATE_FILE,
    BROKER_MODE,
    BROKER_PROVIDER,
    MARKET_DATA_PROVIDER,
    PRODUCTION_MODE,
    RETENTION_ARCHIVE_TRADER_ORDERS,
    RETENTION_DAYS,
    RETENTION_ENABLED,
    RETENTION_RUN_INTERVAL_HOURS,
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
        "fallback_since_at": "",
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
        "requested_by": "",
        "configured_mode": BROKER_MODE,
        "effective_mode": BROKER_MODE,
        "base_url": "",
        "api_key_configured": False,
        "api_secret_configured": False,
        "execution_enabled": False,
        "can_submit_orders": False,
        "warning": "",
    },
    "security": {
        "real_mode_enabled": False,
        "real_mode_enabled_by": "",
        "real_mode_enabled_at": "",
    },
    "production": {
        "enabled": PRODUCTION_MODE,
        "alert_email_enabled": ALERT_EMAIL_ENABLED,
        "alert_provider": ALERT_EMAIL_PROVIDER,
        "heartbeat_age_seconds": None,
        "last_execution_at": "",
        "last_success_at": "",
        "feed_status": "unknown",
        "broker_status": "paper",
        "consecutive_errors": 0,
        "health_level": "healthy",
        "health_reason": "healthy",
        "health_message": "Sistema saudavel. Broker em modo simulado (paper). Nenhuma ordem real sera enviada.",
        "last_health_at": "",
        "last_error": "",
        "last_error_at": "",
        "last_exception": "",
        "fallback_since_at": "",
        "fallback_age_minutes": 0,
        "last_alert_sent_at": "",
        "last_alert_type": "",
        "last_alert_subject": "",
        "last_alert_provider": "",
        "last_alert_error": "",
        "next_alert_eligible_at": "",
        "last_recovery_email_at": "",
    },
    "retention": {
        "enabled": RETENTION_ENABLED,
        "retention_days": RETENTION_DAYS,
        "run_interval_hours": RETENTION_RUN_INTERVAL_HOURS,
        "archive_trader_orders": RETENTION_ARCHIVE_TRADER_ORDERS,
        "last_run_at": "",
        "last_success_at": "",
        "last_error": "",
        "last_error_at": "",
        "last_summary": {},
        "archive_catalog": {
            "reports": [],
            "logs": [],
            "orders": [],
            "weekly_reports": [],
        },
        "weekly_reports_index": [],
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
    production_state = state.get("production", {}) or {}
    production_state["enabled"] = PRODUCTION_MODE
    production_state["alert_email_enabled"] = ALERT_EMAIL_ENABLED
    production_state["alert_provider"] = ALERT_EMAIL_PROVIDER
    state["production"] = production_state
    retention_state = state.get("retention", {}) or {}
    retention_state["enabled"] = RETENTION_ENABLED
    retention_state["retention_days"] = RETENTION_DAYS
    retention_state["run_interval_hours"] = RETENTION_RUN_INTERVAL_HOURS
    retention_state["archive_trader_orders"] = RETENTION_ARCHIVE_TRADER_ORDERS
    state["retention"] = retention_state
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
        context_state["fallback_since_at"] = ""
    elif payload.get("last_error"):
        context_state["last_error"] = str(payload.get("last_error"))
        context_state["fallback_since_at"] = str(
            context_state.get("fallback_since_at") or context_state.get("last_sync_at") or ""
        )

    if str(context_state.get("last_source") or "").lower() != "fallback" and str(context_state.get("status") or "").lower() != "error":
        context_state["fallback_since_at"] = ""

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
            "fallback_since_at",
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

    for key in (
        "provider",
        "mode",
        "status",
        "last_sync_at",
        "last_error",
        "account_id",
        "requested_by",
        "configured_mode",
        "effective_mode",
        "base_url",
        "api_key_configured",
        "api_secret_configured",
        "execution_enabled",
        "can_submit_orders",
        "warning",
    ):
        if payload.get(key) is not None:
            broker_state[key] = payload.get(key)

    state["broker"] = broker_state
    save_bot_state(state)
    return broker_state


def update_production_status(status_payload: dict | None) -> dict:
    state = load_bot_state()
    production_state = state.get("production", {}) or {}
    payload = status_payload or {}

    for key, value in payload.items():
        if value is not None:
            production_state[key] = value

    state["production"] = production_state
    save_bot_state(state)
    return production_state


def update_retention_status(status_payload: dict | None) -> dict:
    state = load_bot_state()
    retention_state = state.get("retention", {}) or {}
    payload = status_payload or {}

    for key, value in payload.items():
        if value is not None:
            retention_state[key] = value

    state["retention"] = retention_state
    save_bot_state(state)
    return retention_state
