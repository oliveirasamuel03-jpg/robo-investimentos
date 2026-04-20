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
    assert config.TRADER_REPORTS_FILE.exists()
    assert config.BOT_LOG_FILE.exists()
    assert state["wallet_value"] == config.VALIDATION_INITIAL_CAPITAL_BRL
    assert isinstance(state["positions"], list)
    assert state["trader"]["profile"] == "Equilibrado"
    assert state["trader"]["ticket_value"] == config.VALIDATION_DEFAULT_ENTRY_AMOUNT_BRL
    assert state["trader"]["max_open_positions"] == config.VALIDATION_DEFAULT_MAX_OPEN_POSITIONS
    assert state["market_data"]["provider"] == config.MARKET_DATA_PROVIDER
    assert state["broker"]["provider"] == config.BROKER_PROVIDER
    assert state["production"]["health_level"] == "healthy"
    assert state["retention"]["retention_days"] == config.RETENTION_DAYS
    assert state["retention"]["weekly_reports_index"] == []
    assert state["validation"]["validation_mode"] == "swing_10d"
    assert state["validation"]["validation_mode_label"] == config.VALIDATION_MODE_DISPLAY
    assert state["validation"]["trading_mode"] == config.VALIDATION_TRADING_MODE
    assert state["validation"]["final_email_sent"] is False
    assert state["trader"]["watchlist"] == config.SWING_VALIDATION_RECOMMENDED_WATCHLIST

    logs = state_store.read_storage_table(config.BOT_LOG_FILE, columns=config.BOT_LOG_COLUMNS)
    assert isinstance(logs, pd.DataFrame)
    assert list(logs.columns) == config.BOT_LOG_COLUMNS


def test_state_store_does_not_reset_db_state_when_local_file_is_missing(isolated_storage, monkeypatch):
    state_store = load_module("core.state_store")

    load_calls = []

    monkeypatch.setattr(state_store, "database_enabled", lambda: True)
    monkeypatch.setattr(
        state_store,
        "load_json_state",
        lambda namespace, default_factory, path: load_calls.append((namespace, path)) or {"bot_status": "RUNNING"},
    )

    def fail_save(state: dict) -> None:
        raise AssertionError("save_bot_state should not be called when DATABASE_URL is active and local file is absent")

    monkeypatch.setattr(state_store, "save_bot_state", fail_save)

    state_store.ensure_storage()

    assert load_calls


def test_update_market_data_status_tracks_last_success_and_error(isolated_storage):
    state_store = load_module("core.state_store")

    success = state_store.update_market_data_status(
        {
            "provider": "yahoo",
            "status": "healthy",
            "last_sync_at": "2026-04-20T01:00:00+00:00",
            "last_source": "market",
            "source_breakdown": {"market": 2, "cached": 0, "fallback": 0, "unknown": 0},
            "symbols": ["AAPL", "MSFT"],
            "requested_by": "worker_cycle",
        }
    )
    root_after_success = state_store.load_bot_state()["market_data"]
    assert success["last_success_at"] == "2026-04-20T01:00:00+00:00"
    assert success["last_error"] == ""
    assert root_after_success["requested_by"] == "worker_cycle"
    assert root_after_success["status"] == "healthy"

    degraded = state_store.update_market_data_status(
        {
            "provider": "yahoo",
            "status": "error",
            "last_sync_at": "2026-04-20T01:05:00+00:00",
            "last_source": "fallback",
            "last_error": "rate limit",
            "source_breakdown": {"market": 0, "cached": 0, "fallback": 2, "unknown": 0},
            "symbols": ["AAPL", "MSFT"],
            "requested_by": "trader_chart",
        }
    )
    root_after_chart = state_store.load_bot_state()["market_data"]
    assert degraded["last_success_at"] == "2026-04-20T01:00:00+00:00"
    assert degraded["last_error"] == "rate limit"
    assert root_after_chart["requested_by"] == "worker_cycle"
    assert root_after_chart["status"] == "healthy"
    assert root_after_chart["contexts"]["trader_chart"]["status"] == "error"


def test_update_broker_status_tracks_readiness_fields(isolated_storage):
    state_store = load_module("core.state_store")

    broker_state = state_store.update_broker_status(
        {
            "provider": "alpaca",
            "mode": "live",
            "status": "guarded",
            "last_sync_at": "2026-04-20T04:00:00+00:00",
            "last_error": "",
            "account_id": "***1234",
            "requested_by": "admin_panel",
            "configured_mode": "live",
            "effective_mode": "live",
            "base_url": "https://api.alpaca.markets",
            "api_key_configured": True,
            "api_secret_configured": True,
            "execution_enabled": False,
            "can_submit_orders": False,
            "warning": "guarded",
        }
    )

    assert broker_state["provider"] == "alpaca"
    assert broker_state["status"] == "guarded"
    assert broker_state["api_key_configured"] is True
    assert broker_state["execution_enabled"] is False


def test_update_production_status_tracks_health_fields(isolated_storage):
    state_store = load_module("core.state_store")

    production_state = state_store.update_production_status(
        {
            "health_level": "warning",
            "health_message": "Heartbeat atrasado.",
            "consecutive_errors": 2,
            "last_alert_sent_at": "2026-04-20T04:00:00+00:00",
        }
    )

    assert production_state["health_level"] == "warning"
    assert production_state["consecutive_errors"] == 2
    assert production_state["last_alert_sent_at"] == "2026-04-20T04:00:00+00:00"


def test_update_validation_status_tracks_cycle_fields(isolated_storage):
    state_store = load_module("core.state_store")

    validation_state = state_store.update_validation_status(
        {
            "validation_day_number": 7,
            "validation_phase": "Ajuste controlado",
            "final_validation_grade": "APROVADO_COM_AJUSTES",
        }
    )

    assert validation_state["validation_day_number"] == 7
    assert validation_state["validation_phase"] == "Ajuste controlado"
    assert validation_state["final_validation_grade"] == "APROVADO_COM_AJUSTES"


def test_state_store_migrates_legacy_default_watchlist_to_crypto_only_default(isolated_storage):
    config = load_module("core.config")
    state_store = load_module("core.state_store")

    state = state_store.load_bot_state()
    state["trader"]["watchlist"] = list(config.LEGACY_MIXED_DEFAULT_WATCHLIST)
    state_store.save_bot_state(state)

    reloaded = state_store.load_bot_state()

    assert reloaded["trader"]["watchlist"] == config.SWING_VALIDATION_RECOMMENDED_WATCHLIST


def test_state_store_preserves_custom_watchlist_outside_recommended_default(isolated_storage):
    state_store = load_module("core.state_store")

    state = state_store.load_bot_state()
    state["trader"]["watchlist"] = ["BTC-USD", "ETH-USD", "DOGE-USD"]
    state_store.save_bot_state(state)

    reloaded = state_store.load_bot_state()

    assert reloaded["trader"]["watchlist"] == ["BTC-USD", "ETH-USD", "DOGE-USD"]
