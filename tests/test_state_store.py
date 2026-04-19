from __future__ import annotations

import pandas as pd

from tests.conftest import load_module


def test_state_store_bootstraps_files_and_defaults(isolated_storage):
    config = load_module("core.config")
    state_store = load_module("core.state_store")

    state_store.ensure_storage()
    state = state_store.load_bot_state()

    assert config.BOT_STATE_FILE.exists()
    assert config.TRADER_ORDERS_FILE.exists()
    assert config.INVESTOR_ORDERS_FILE.exists()
    assert config.BOT_LOG_FILE.exists()
    assert state["wallet_value"] > 0
    assert isinstance(state["positions"], list)

    logs = state_store.read_storage_table(config.BOT_LOG_FILE, columns=config.BOT_LOG_COLUMNS)
    assert isinstance(logs, pd.DataFrame)
    assert list(logs.columns) == config.BOT_LOG_COLUMNS
