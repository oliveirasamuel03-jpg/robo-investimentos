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


APP_TITLE = _env_str("APP_TITLE", "Trade Ops Desk") or "Trade Ops Desk"
APP_ENV = _env_str("APP_ENV", _env_str("ENVIRONMENT", "development")) or "development"
PRODUCTION_MODE = env_flag("PRODUCTION_MODE", False)
MARKET_DATA_PROVIDER = _env_str("MARKET_DATA_PROVIDER", "yahoo") or "yahoo"
MARKET_DATA_CACHE_TTL_SECONDS = max(30, env_int("MARKET_DATA_CACHE_TTL_SECONDS", 300))
MARKET_DATA_HISTORY_LIMIT = max(120, env_int("MARKET_DATA_HISTORY_LIMIT", 500))
BROKER_PROVIDER = _env_str("BROKER_PROVIDER", "paper") or "paper"
BROKER_MODE = _env_str("BROKER_MODE", "paper") or "paper"
BROKER_ACCOUNT_ID = _env_str("BROKER_ACCOUNT_ID", "")
BROKER_BASE_URL = _env_str("BROKER_BASE_URL", "")
BROKER_API_KEY = _env_str("BROKER_API_KEY", "")
BROKER_API_SECRET = _env_str("BROKER_API_SECRET", "")
BROKER_HTTP_HEALTHCHECK = env_flag("BROKER_HTTP_HEALTHCHECK", False)
ALERT_EMAIL_ENABLED = env_flag("ALERT_EMAIL_ENABLED", False)
ALERT_EMAIL_TO = _env_str("ALERT_EMAIL_TO", "oliveirasamuel03@gmail.com")
SMTP_HOST = _env_str("SMTP_HOST", "")
SMTP_PORT = max(1, env_int("SMTP_PORT", 587))
SMTP_USERNAME = _env_str("SMTP_USERNAME", "")
SMTP_PASSWORD = _env_str("SMTP_PASSWORD", "")
SMTP_USE_TLS = env_flag("SMTP_USE_TLS", True)
SMTP_TIMEOUT_SECONDS = max(5, env_int("SMTP_TIMEOUT_SECONDS", 15))
ALERT_HEARTBEAT_MAX_DELAY_SECONDS = max(60, env_int("ALERT_HEARTBEAT_MAX_DELAY_SECONDS", 180))
ALERT_MAX_CONSECUTIVE_ERRORS = max(1, env_int("ALERT_MAX_CONSECUTIVE_ERRORS", 3))
ALERT_FEED_FALLBACK_MAX_MINUTES = max(1, env_int("ALERT_FEED_FALLBACK_MAX_MINUTES", 15))
ALERT_COOLDOWN_MINUTES = max(1, env_int("ALERT_COOLDOWN_MINUTES", 30))
ALERT_SEND_RECOVERY_EMAIL = env_flag("ALERT_SEND_RECOVERY_EMAIL", True)

STORAGE_DIR = _resolve_storage_dir()
RUNTIME_DIR = STORAGE_DIR / "runtime"
CACHE_DIR = STORAGE_DIR / "cache"
REPORTS_DIR = STORAGE_DIR / "reports"

BOT_STATE_FILE = RUNTIME_DIR / "bot_state.json"
TRADER_ORDERS_FILE = RUNTIME_DIR / "trader_orders.csv"
TRADER_REPORTS_FILE = RUNTIME_DIR / "trade_reports.csv"
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
