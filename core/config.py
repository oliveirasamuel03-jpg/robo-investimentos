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


def env_float(name: str, default: float) -> float:
    raw = _env_str(name)
    if not raw:
        return default
    try:
        return float(raw)
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
SERVICE_NAME = _env_str("SERVICE_NAME", "") or _env_str("RAILWAY_SERVICE_NAME", "")
RAILWAY_SERVICE_NAME = _env_str("RAILWAY_SERVICE_NAME", "")
RAILWAY_GIT_COMMIT_SHA = _env_str("RAILWAY_GIT_COMMIT_SHA", "")
RAILWAY_DEPLOYMENT_ID = _env_str("RAILWAY_DEPLOYMENT_ID", "")
APP_SOURCE_COMMIT_SHA = _env_str("APP_SOURCE_COMMIT_SHA", _env_str("SOURCE_COMMIT_SHA", ""))
BUILD_TIMESTAMP = _env_str("BUILD_TIMESTAMP", _env_str("RAILWAY_BUILD_TIMESTAMP", ""))
STATE_SCHEMA_VERSION = _env_str("STATE_SCHEMA_VERSION", "worker-audit-v1")
MARKET_DATA_DIAGNOSTIC_VERSION = _env_str("MARKET_DATA_DIAGNOSTIC_VERSION", "td-request-trace-v1")
_raw_market_data_build = (
    _env_str("MARKET_DATA_BUILD_LABEL", "")
    or RAILWAY_GIT_COMMIT_SHA
    or RAILWAY_DEPLOYMENT_ID
    or APP_ENV
)
MARKET_DATA_BUILD_LABEL = _raw_market_data_build[:12] if _raw_market_data_build else APP_ENV
PRODUCTION_MODE = env_flag("PRODUCTION_MODE", False)
MARKET_DATA_PROVIDER = _env_str("MARKET_DATA_PROVIDER", "twelvedata") or "twelvedata"
MARKET_DATA_FALLBACK_PROVIDER = _env_str("MARKET_DATA_FALLBACK_PROVIDER", "yahoo") or "yahoo"
TWELVEDATA_API_KEY = _env_str("TWELVEDATA_API_KEY", "")
TWELVEDATA_API_BASE = _env_str("TWELVEDATA_API_BASE", "https://api.twelvedata.com") or "https://api.twelvedata.com"
TWELVEDATA_MIN_CACHE_TTL_SECONDS = max(300, env_int("TWELVEDATA_MIN_CACHE_TTL_SECONDS", 900))
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
ALERT_EMAIL_PROVIDER = _env_str("ALERT_EMAIL_PROVIDER", "smtp") or "smtp"
ALERT_EMAIL_TO = _env_str("ALERT_EMAIL_TO", "oliveirasamuel03@gmail.com")
ALERT_EMAIL_FROM = _env_str("ALERT_EMAIL_FROM", "")
SMTP_HOST = _env_str("SMTP_HOST", "")
SMTP_PORT = max(1, env_int("SMTP_PORT", 587))
SMTP_USERNAME = _env_str("SMTP_USERNAME", "")
SMTP_PASSWORD = _env_str("SMTP_PASSWORD", "")
SMTP_USE_TLS = env_flag("SMTP_USE_TLS", True)
SMTP_TIMEOUT_SECONDS = max(5, env_int("SMTP_TIMEOUT_SECONDS", 15))
RESEND_API_KEY = _env_str("RESEND_API_KEY", "")
RESEND_API_BASE = _env_str("RESEND_API_BASE", "https://api.resend.com/emails") or "https://api.resend.com/emails"
REPORT_EMAIL_ENABLED = env_flag("REPORT_EMAIL_ENABLED", ALERT_EMAIL_ENABLED)
REPORT_EMAIL_PROVIDER = _env_str("REPORT_EMAIL_PROVIDER", ALERT_EMAIL_PROVIDER) or ALERT_EMAIL_PROVIDER
REPORT_EMAIL_TO = _env_str("REPORT_EMAIL_TO", ALERT_EMAIL_TO)
REPORT_EMAIL_FROM = _env_str("REPORT_EMAIL_FROM", ALERT_EMAIL_FROM)
REPORT_EMAIL_SMTP_HOST = _env_str("REPORT_EMAIL_SMTP_HOST", SMTP_HOST)
REPORT_EMAIL_SMTP_PORT = max(1, env_int("REPORT_EMAIL_SMTP_PORT", SMTP_PORT))
REPORT_EMAIL_SMTP_USERNAME = _env_str("REPORT_EMAIL_SMTP_USERNAME", SMTP_USERNAME)
REPORT_EMAIL_SMTP_PASSWORD = _env_str("REPORT_EMAIL_SMTP_PASSWORD", SMTP_PASSWORD)
REPORT_EMAIL_USE_TLS = env_flag("REPORT_EMAIL_USE_TLS", SMTP_USE_TLS)
REPORT_EMAIL_TIMEOUT_SECONDS = max(5, env_int("REPORT_EMAIL_TIMEOUT_SECONDS", SMTP_TIMEOUT_SECONDS))
REPORT_EMAIL_RESEND_API_KEY = _env_str("REPORT_EMAIL_RESEND_API_KEY", RESEND_API_KEY)
REPORT_EMAIL_RESEND_API_BASE = (
    _env_str("REPORT_EMAIL_RESEND_API_BASE", RESEND_API_BASE) or RESEND_API_BASE
)
REPORT_EMAIL_DAILY_ENABLED = env_flag("REPORT_EMAIL_DAILY_ENABLED", REPORT_EMAIL_ENABLED)
REPORT_EMAIL_WEEKLY_ENABLED = env_flag("REPORT_EMAIL_WEEKLY_ENABLED", REPORT_EMAIL_ENABLED)
REPORT_EMAIL_10DAY_ENABLED = env_flag("REPORT_EMAIL_10DAY_ENABLED", REPORT_EMAIL_ENABLED)
REPORT_EMAIL_FINAL_ENABLED = env_flag("REPORT_EMAIL_FINAL_ENABLED", REPORT_EMAIL_ENABLED)
ALERT_HEARTBEAT_MAX_DELAY_SECONDS = max(60, env_int("ALERT_HEARTBEAT_MAX_DELAY_SECONDS", 180))
ALERT_MAX_CONSECUTIVE_ERRORS = max(1, env_int("ALERT_MAX_CONSECUTIVE_ERRORS", 3))
ALERT_FEED_FALLBACK_MAX_MINUTES = max(1, env_int("ALERT_FEED_FALLBACK_MAX_MINUTES", 15))
ALERT_COOLDOWN_MINUTES = max(1, env_int("ALERT_COOLDOWN_MINUTES", 30))
ALERT_SEND_RECOVERY_EMAIL = env_flag("ALERT_SEND_RECOVERY_EMAIL", True)
RETENTION_ENABLED = env_flag("RETENTION_ENABLED", True)
RETENTION_DAYS = max(7, env_int("RETENTION_DAYS", 60))
RETENTION_RUN_INTERVAL_HOURS = max(6, env_int("RETENTION_RUN_INTERVAL_HOURS", 24))
RETENTION_ARCHIVE_TRADER_ORDERS = env_flag("RETENTION_ARCHIVE_TRADER_ORDERS", False)
WEEKLY_REPORT_RUNTIME_WEEKS = max(2, env_int("WEEKLY_REPORT_RUNTIME_WEEKS", 8))

STORAGE_DIR = _resolve_storage_dir()
RUNTIME_DIR = STORAGE_DIR / "runtime"
CACHE_DIR = STORAGE_DIR / "cache"
REPORTS_DIR = STORAGE_DIR / "reports"
ARCHIVE_DIR = STORAGE_DIR / "archive"
ARCHIVE_REPORTS_DIR = ARCHIVE_DIR / "reports"
ARCHIVE_LOGS_DIR = ARCHIVE_DIR / "logs"
ARCHIVE_WEEKLY_REPORTS_DIR = ARCHIVE_DIR / "weekly_reports"
WEEKLY_REPORTS_DIR = RUNTIME_DIR / "weekly_reports"

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
LEGACY_VALIDATION_INITIAL_CAPITAL_BRL = 10000.0
VALIDATION_INITIAL_CAPITAL_BRL = 1000.0
VALIDATION_DEFAULT_ENTRY_AMOUNT_BRL = 100.0
VALIDATION_DEFAULT_MAX_OPEN_POSITIONS = 2
DAILY_LOSS_LIMIT_BRL_DEFAULT = max(1.0, env_float("DAILY_LOSS_LIMIT_BRL", 100.0))
VALIDATION_MODE_DISPLAY = "swing_10d_crypto"
VALIDATION_TRADING_MODE = "paper"
VALIDATION_LIVE_TRADING_ENABLED = False

SWING_VALIDATION_RECOMMENDED_WATCHLIST = [
    "BTC-USD",
    "ETH-USD",
    "BNB-USD",
    "SOL-USD",
    "LINK-USD",
]

LEGACY_MIXED_DEFAULT_WATCHLIST = [
    "BTC-USD",
    "ETH-USD",
    "VALE3.SA",
    "PETR4.SA",
    "AAPL",
    "KC=F",
]

SWING_VALIDATION_WATCHLIST_DETAILS = [
    {
        "asset": "BTC-USD",
        "papel": "Referencia principal do mercado",
        "motivo": "Maior liquidez e melhor leitura estrutural para swing.",
        "risco": "Pode andar mais devagar em alguns ciclos.",
        "perfil": "Consistencia / tendencia",
    },
    {
        "asset": "ETH-USD",
        "papel": "Segundo ativo-base da watchlist",
        "motivo": "Alta liquidez e relevancia ampla no mercado cripto.",
        "risco": "Pode ampliar movimentos do BTC em momentos de estresse.",
        "perfil": "Consistencia / tendencia com mais amplitude",
    },
    {
        "asset": "BNB-USD",
        "papel": "Ativo intermediario entre estabilidade e oportunidade",
        "motivo": "Boa liquidez e comportamento tecnico relativamente organizado.",
        "risco": "Pode sofrer impacto especifico do ecossistema relacionado.",
        "perfil": "Consistencia moderada / tendencia",
    },
    {
        "asset": "SOL-USD",
        "papel": "Ativo de maior oportunidade",
        "motivo": "Costuma oferecer swings mais fortes e legiveis.",
        "risco": "Volatilidade mais alta e ruido maior.",
        "perfil": "Volatilidade / tendencia",
    },
    {
        "asset": "LINK-USD",
        "papel": "Diversificacao tatica da watchlist",
        "motivo": "Boa leitura tecnica em muitos ciclos e liquidez adequada.",
        "risco": "Pode perder forca em periodos mais seletivos.",
        "perfil": "Tendencia / oportunidade moderada",
    },
]

SWING_VALIDATION_DISCOURAGED_ASSET_NOTES = [
    "Memecoins e altcoins de baixa liquidez nao sao recomendadas nesta fase.",
    "Ativos com feed inconsistente ou comportamento excessivamente caotico devem ficar fora da watchlist padrao.",
    "A lista foi reduzida intencionalmente para priorizar qualidade de sinal, estabilidade operacional e comparabilidade dos relatorios.",
]


def ensure_app_directories() -> None:
    for directory in (
        STORAGE_DIR,
        RUNTIME_DIR,
        CACHE_DIR,
        REPORTS_DIR,
        ARCHIVE_DIR,
        ARCHIVE_REPORTS_DIR,
        ARCHIVE_LOGS_DIR,
        ARCHIVE_WEEKLY_REPORTS_DIR,
        WEEKLY_REPORTS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


ensure_app_directories()
