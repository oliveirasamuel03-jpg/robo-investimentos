from __future__ import annotations

from pathlib import Path

APP_TITLE = "Invest Pro Desk"

BASE_DIR = Path(__file__).resolve().parents[1]
STORAGE_DIR = BASE_DIR / "storage"

BOT_STATE_FILE = STORAGE_DIR / "bot_state.json"
TRADER_ORDERS_FILE = STORAGE_DIR / "trader_orders.csv"
INVESTOR_ORDERS_FILE = STORAGE_DIR / "investor_orders.csv"
BOT_LOG_FILE = STORAGE_DIR / "bot_log.csv"

MIN_TICKET = 10.0
MAX_TICKET = 10000.0
MIN_HOLDING_MINUTES = 1
MAX_HOLDING_MINUTES = 2880
