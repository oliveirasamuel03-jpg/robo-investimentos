from __future__ import annotations

from pathlib import Path

APP_TITLE = "Invest Pro Desk"

BASE_DIR = Path(__file__).resolve().parents[1]
STORAGE_DIR = BASE_DIR / "storage"

BOT_STATE_FILE = STORAGE_DIR / "bot_state.json"
TRADER_ORDERS_FILE = STORAGE_DIR / "trader_orders.csv"
INVESTOR_ORDERS_FILE = STORAGE_DIR / "investor_orders.csv"
BOT_LOG_FILE = STORAGE_DIR / "bot_log.csv"
RISK_EVENTS_FILE = STORAGE_DIR / "risk_events.json"
DAILY_RISK_STATUS_FILE = STORAGE_DIR / "daily_risk_status.json"

MIN_TICKET = 10.0
MAX_TICKET = 10000.0
MIN_HOLDING_MINUTES = 1
MAX_HOLDING_MINUTES = 2880

DEFAULT_RISK_LIMITS = {
    "max_risk_per_trade_pct": 0.01,
    "max_daily_loss_pct": 0.02,
    "max_drawdown_pct": 0.10,
    "max_open_positions": 3,
    "max_asset_exposure_pct": 0.35,
    "kill_switch_enabled": True,
}
