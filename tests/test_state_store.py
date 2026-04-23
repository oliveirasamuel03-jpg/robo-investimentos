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
    assert state["market_data"]["configured_provider"] == config.MARKET_DATA_PROVIDER
    assert state["market_data"]["fallback_provider"] == config.MARKET_DATA_FALLBACK_PROVIDER
    assert state["market_data"]["feed_status"] == "UNKNOWN"
    assert state["market_data"]["status_legacy"] == "unknown"
    assert state["market_data"]["provider_diagnostics"] == {}
    assert state["broker"]["provider"] == config.BROKER_PROVIDER
    assert state["production"]["health_level"] == "healthy"
    assert state["email_reporting"]["enabled"] == config.REPORT_EMAIL_ENABLED
    assert state["email_reporting"]["destination"] == config.REPORT_EMAIL_TO
    assert state["email_reporting"]["last_email_delivery_status"] == ""
    assert state["risk"]["daily_loss_block_active"] is False
    assert state["risk"]["daily_loss_limit_brl"] == config.DAILY_LOSS_LIMIT_BRL_DEFAULT
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
            "provider_diagnostics": {"twelvedata": {"api_key_present": True, "request_attempted": True}},
            "symbols": ["AAPL", "MSFT"],
            "requested_by": "worker_cycle",
        }
    )
    root_after_success = state_store.load_bot_state()["market_data"]
    assert success["last_success_at"] == "2026-04-20T01:00:00+00:00"
    assert success["last_error"] == ""
    assert root_after_success["requested_by"] == "worker_cycle"
    assert root_after_success["status"] == "healthy"
    assert root_after_success["status_legacy"] == "healthy"
    assert root_after_success["feed_status"] == "LIVE"
    assert root_after_success["provider_diagnostics"]["twelvedata"]["api_key_present"] is True

    degraded = state_store.update_market_data_status(
        {
            "provider": "yahoo",
            "status": "error",
            "last_sync_at": "2026-04-20T01:05:00+00:00",
            "last_source": "fallback",
            "last_error": "rate limit",
            "source_breakdown": {"market": 0, "cached": 0, "fallback": 2, "unknown": 0},
            "provider_diagnostics": {"twelvedata": {"api_key_present": True, "request_attempted": True}},
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
    assert root_after_chart["contexts"]["trader_chart"]["status_legacy"] == "error"
    assert root_after_chart["contexts"]["trader_chart"]["feed_status"] == "FALLBACK"


def test_update_market_data_status_logs_provider_switch(isolated_storage):
    config = load_module("core.config")
    state_store = load_module("core.state_store")

    state_store.update_market_data_status(
        {
            "provider": "twelvedata",
            "configured_provider": "twelvedata",
            "fallback_provider": "yahoo",
            "provider_chain": ["twelvedata", "yahoo"],
            "status": "healthy",
            "last_sync_at": "2026-04-20T01:00:00+00:00",
            "last_source": "market",
            "source_breakdown": {"market": 1, "cached": 0, "fallback": 0, "unknown": 0},
            "provider_breakdown": {"twelvedata": 1, "yahoo": 0, "synthetic": 0, "mixed": 0, "unknown": 0},
            "symbols": ["BTC-USD"],
            "requested_by": "worker_cycle",
        }
    )

    state_store.update_market_data_status(
        {
            "provider": "yahoo",
            "configured_provider": "twelvedata",
            "fallback_provider": "yahoo",
            "provider_chain": ["twelvedata", "yahoo"],
            "status": "healthy",
            "last_sync_at": "2026-04-20T01:05:00+00:00",
            "last_source": "market",
            "source_breakdown": {"market": 1, "cached": 0, "fallback": 0, "unknown": 0},
            "provider_breakdown": {"twelvedata": 0, "yahoo": 1, "synthetic": 0, "mixed": 0, "unknown": 0},
            "symbols": ["BTC-USD"],
            "requested_by": "worker_cycle",
        }
    )

    logs = state_store.read_storage_table(config.BOT_LOG_FILE, columns=config.BOT_LOG_COLUMNS)

    assert logs["message"].str.contains("Provider de dados alterado em worker_cycle").any()
    assert logs["message"].str.contains("Twelve Data -> Yahoo").any()


def test_persist_worker_cycle_state_writes_shared_runtime_snapshot(isolated_storage):
    state_store = load_module("core.state_store")

    state_store.persist_worker_cycle_state(
        last_action="Rodada concluida sem novas operacoes",
        next_run_delta_seconds=60,
        worker_status="online",
        runtime_started_at="2026-04-22T03:00:00+00:00",
        process_role="worker",
        market_data_payload={
            "requested_by": "worker_cycle",
            "provider": "twelvedata",
            "provider_effective": "twelvedata",
            "configured_provider": "twelvedata",
            "fallback_provider": "yahoo",
            "status": "healthy",
            "feed_status": "LIVE",
            "last_sync_at": "2026-04-22T03:01:00+00:00",
            "last_source": "market",
            "source_breakdown": {"market": 5, "cached": 0, "fallback": 0, "unknown": 0},
            "requested_symbols": ["BTC-USD", "ETH-USD"],
            "build_active": "abc123",
            "git_sha": "abc123def456",
            "source_commit_sha": "33cb320",
            "build_timestamp": "2026-04-22T02:55:00+00:00",
            "service_name": "work",
            "api_key_present": True,
            "request_prepared": True,
            "request_attempted": True,
            "response_received": True,
            "response_status_code": 200,
            "last_stage": "success",
            "last_error": "",
        },
    )

    state = state_store.load_bot_state()
    market_state = state["market_data"]

    assert state["worker_status"] == "online"
    assert state["worker_heartbeat"]
    assert state["last_run_at"]
    assert market_state["requested_by"] == "worker_cycle"
    assert market_state["state_writer"] == "worker"
    assert market_state["process_role"] == "worker"
    assert market_state["runtime_started_at"] == "2026-04-22T03:00:00+00:00"
    assert market_state["response_status_code"] == 200
    assert market_state["provider_effective"] == "twelvedata"
    assert market_state["requested_symbols"] == ["BTC-USD", "ETH-USD"]
    assert market_state["ui_audit_probe"] == "worker_state_v2"
    assert market_state["state_build_sha"] == "abc123def456"
    assert market_state["source_commit_sha"] == "33cb320"


def test_persist_worker_cycle_state_uses_optional_source_commit_sha_env(isolated_storage, monkeypatch):
    monkeypatch.setenv("APP_SOURCE_COMMIT_SHA", "branch-tip-123")
    state_store = load_module("core.state_store")

    state_store.persist_worker_cycle_state(
        last_action="Robo pausado. Aguardando ativacao.",
        next_run_delta_seconds=5,
        worker_status="online",
        runtime_started_at="2026-04-23T00:00:00+00:00",
        process_role="worker",
        market_data_payload={"requested_by": "worker_cycle"},
    )

    state = state_store.load_bot_state()

    assert state["market_data"]["source_commit_sha"] == "branch-tip-123"


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


def test_update_email_reporting_status_tracks_delivery_fields(isolated_storage):
    state_store = load_module("core.state_store")

    email_state = state_store.update_email_reporting_status(
        {
            "last_daily_report_email_date": "2026-04-23",
            "last_email_delivery_status": "sent",
            "last_email_delivery_reason": "sent",
            "last_email_delivery_attempt_ts": "2026-04-23T12:00:00+00:00",
            "last_email_delivery_success_ts": "2026-04-23T12:00:02+00:00",
            "last_email_delivery_report_type": "daily",
        }
    )

    assert email_state["last_daily_report_email_date"] == "2026-04-23"
    assert email_state["last_email_delivery_status"] == "sent"
    assert email_state["last_email_delivery_report_type"] == "daily"


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
