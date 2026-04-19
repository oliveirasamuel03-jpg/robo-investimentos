from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None


BASE_DIR = Path(__file__).resolve().parents[1]
ENV_FILE = BASE_DIR / ".env"

if load_dotenv is not None and ENV_FILE.exists():
    load_dotenv(dotenv_path=ENV_FILE, override=False)


def _env_str(name: str, default: str = "") -> str:
    return str(os.getenv(name, default) or "").strip()


def env_flag(name: str, default: bool = False) -> bool:
    raw = _env_str(name)
    if not raw:
        return default
    return raw.lower() in {"1", "true", "yes", "y", "on"}


def env_int(name: str, default: int) -> int:
    raw = _env_str(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _resolve_storage_dir() -> Path:
    raw = _env_str("ROBO_STORAGE_DIR")
    if not raw:
        return BASE_DIR / "storage"

    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        return candidate
    return (BASE_DIR / candidate).resolve()


APP_TITLE = _env_str("APP_TITLE", "Invest Pro Desk") or "Invest Pro Desk"
APP_ENV = _env_str("APP_ENV", _env_str("ENVIRONMENT", "development")) or "development"

STORAGE_DIR = _resolve_storage_dir()
RUNTIME_DIR = STORAGE_DIR / "runtime"
CACHE_DIR = STORAGE_DIR / "cache"
REPORTS_DIR = STORAGE_DIR / "reports"

BOT_STATE_FILE = RUNTIME_DIR / "bot_state.json"
TRADER_ORDERS_FILE = RUNTIME_DIR / "trader_orders.csv"
TRADER_REPORTS_FILE = RUNTIME_DIR / "trade_reports.csv"
INVESTOR_ORDERS_FILE = RUNTIME_DIR / "investor_orders.csv"
BOT_LOG_FILE = RUNTIME_DIR / "bot_log.csv"
AUTH_USERS_FILE = RUNTIME_DIR / "auth_users.json"

LEGACY_RUNTIME_CONFIG_FILE = RUNTIME_DIR / "legacy_runtime_config.json"
LEGACY_BOT_STATE_FILE = RUNTIME_DIR / "legacy_bot_state.json"
LEGACY_BOT_LOG_FILE = RUNTIME_DIR / "legacy_bot_logs.jsonl"

TRADER_ORDERS_COLUMNS = [
    "timestamp",
    "asset",
    "side",
    "quantity",
    "price",
    "gross_value",
    "cost",
    "cash_after",
    "source",
]

TRADER_REPORTS_COLUMNS = [
    "trade_id",
    "opened_at",
    "closed_at",
    "duration_minutes",
    "asset",
    "profile",
    "entry_price",
    "exit_price",
    "quantity",
    "gross_entry_value",
    "gross_exit_value",
    "realized_pnl",
    "realized_pnl_pct",
    "entry_score",
    "exit_reason",
    "rsi_entry",
    "rsi_exit",
    "atr_pct_entry",
    "atr_pct_exit",
    "ma20_entry",
    "ma50_entry",
    "ma20_exit",
    "ma50_exit",
    "peak_price",
    "trailing_stop_final",
    "holding_minutes_limit",
    "status_final",
]

INVESTOR_ORDERS_COLUMNS = [
    "timestamp",
    "metric",
    "value",
    "notes",
]

BOT_LOG_COLUMNS = [
    "timestamp",
    "level",
    "message",
]

MIN_TICKET = 10.0
MAX_TICKET = 10000.0
MIN_HOLDING_MINUTES = 1
MAX_HOLDING_MINUTES = 2880


def ensure_app_directories() -> None:
    for directory in (STORAGE_DIR, RUNTIME_DIR, CACHE_DIR, REPORTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


ensure_app_directories()
