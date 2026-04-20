from __future__ import annotations

from tests.conftest import load_module


def test_archive_old_trade_reports_is_idempotent(isolated_storage):
    config = load_module("core.config")
    state_store = load_module("core.state_store")
    retention = load_module("core.retention")

    state_store.replace_storage_table(
        config.TRADER_REPORTS_FILE,
        [
            {
                "trade_id": "old-trade",
                "opened_at": "2026-01-10T10:00:00+00:00",
                "closed_at": "2026-01-10T11:00:00+00:00",
                "duration_minutes": 60,
                "asset": "AAPL",
                "profile": "Equilibrado",
                "entry_price": 100,
                "exit_price": 101,
                "quantity": 1,
                "gross_entry_value": 100,
                "gross_exit_value": 101,
                "realized_pnl": 1,
                "realized_pnl_pct": 0.01,
                "entry_score": 0.8,
                "exit_reason": "holding expirado",
                "rsi_entry": 50,
                "rsi_exit": 55,
                "atr_pct_entry": 0.01,
                "atr_pct_exit": 0.01,
                "ma20_entry": 100,
                "ma50_entry": 98,
                "ma20_exit": 101,
                "ma50_exit": 99,
                "peak_price": 102,
                "trailing_stop_final": 99,
                "holding_minutes_limit": 60,
                "status_final": "gain",
            },
            {
                "trade_id": "recent-trade",
                "opened_at": "2026-04-18T10:00:00+00:00",
                "closed_at": "2026-04-18T11:00:00+00:00",
                "duration_minutes": 60,
                "asset": "MSFT",
                "profile": "Equilibrado",
                "entry_price": 100,
                "exit_price": 99,
                "quantity": 1,
                "gross_entry_value": 100,
                "gross_exit_value": 99,
                "realized_pnl": -1,
                "realized_pnl_pct": -0.01,
                "entry_score": 0.7,
                "exit_reason": "stop",
                "rsi_entry": 45,
                "rsi_exit": 40,
                "atr_pct_entry": 0.01,
                "atr_pct_exit": 0.02,
                "ma20_entry": 100,
                "ma50_entry": 98,
                "ma20_exit": 98,
                "ma50_exit": 97,
                "peak_price": 100,
                "trailing_stop_final": 98,
                "holding_minutes_limit": 60,
                "status_final": "loss",
            },
        ],
        columns=config.TRADER_REPORTS_COLUMNS,
    )

    first = retention.archive_old_trade_reports(
        retention_days=60,
        now=retention._utc_now(retention.datetime.fromisoformat("2026-04-20T00:00:00+00:00")),
    )
    runtime_after_first = state_store.read_storage_table(config.TRADER_REPORTS_FILE, columns=config.TRADER_REPORTS_COLUMNS)
    archive_after_first = state_store.read_storage_table(first["files_updated"][0], columns=config.TRADER_REPORTS_COLUMNS)

    assert first["archived_rows"] == 1
    assert len(runtime_after_first) == 1
    assert len(archive_after_first) == 1
    assert archive_after_first.iloc[0]["trade_id"] == "old-trade"

    second = retention.archive_old_trade_reports(
        retention_days=60,
        now=retention._utc_now(retention.datetime.fromisoformat("2026-04-20T00:00:00+00:00")),
    )
    runtime_after_second = state_store.read_storage_table(config.TRADER_REPORTS_FILE, columns=config.TRADER_REPORTS_COLUMNS)
    archive_after_second = state_store.read_storage_table(first["files_updated"][0], columns=config.TRADER_REPORTS_COLUMNS)

    assert second["archived_rows"] == 0
    assert len(runtime_after_second) == 1
    assert len(archive_after_second) == 1


def test_run_retention_job_generates_weekly_validation_summary(isolated_storage):
    config = load_module("core.config")
    state_store = load_module("core.state_store")
    retention = load_module("core.retention")

    state_store.replace_storage_table(
        config.TRADER_REPORTS_FILE,
        [
            {
                "trade_id": "week-trade-1",
                "opened_at": "2026-04-13T10:00:00+00:00",
                "closed_at": "2026-04-13T10:10:00+00:00",
                "duration_minutes": 10,
                "asset": "AAPL",
                "profile": "Equilibrado",
                "entry_price": 100,
                "exit_price": 102,
                "quantity": 1,
                "gross_entry_value": 100,
                "gross_exit_value": 102,
                "realized_pnl": 2,
                "realized_pnl_pct": 0.02,
                "entry_score": 0.8,
                "exit_reason": "target",
                "rsi_entry": 50,
                "rsi_exit": 65,
                "atr_pct_entry": 0.01,
                "atr_pct_exit": 0.02,
                "ma20_entry": 100,
                "ma50_entry": 98,
                "ma20_exit": 103,
                "ma50_exit": 100,
                "peak_price": 103,
                "trailing_stop_final": 101,
                "holding_minutes_limit": 60,
                "status_final": "gain",
            },
            {
                "trade_id": "week-trade-2",
                "opened_at": "2026-04-14T10:00:00+00:00",
                "closed_at": "2026-04-14T11:30:00+00:00",
                "duration_minutes": 90,
                "asset": "MSFT",
                "profile": "Equilibrado",
                "entry_price": 100,
                "exit_price": 97,
                "quantity": 1,
                "gross_entry_value": 100,
                "gross_exit_value": 97,
                "realized_pnl": -3,
                "realized_pnl_pct": -0.03,
                "entry_score": 0.7,
                "exit_reason": "holding expirado",
                "rsi_entry": 45,
                "rsi_exit": 35,
                "atr_pct_entry": 0.01,
                "atr_pct_exit": 0.02,
                "ma20_entry": 100,
                "ma50_entry": 98,
                "ma20_exit": 97,
                "ma50_exit": 96,
                "peak_price": 101,
                "trailing_stop_final": 96,
                "holding_minutes_limit": 60,
                "status_final": "loss",
            },
        ],
        columns=config.TRADER_REPORTS_COLUMNS,
    )
    state_store.replace_storage_table(
        config.BOT_LOG_FILE,
        [
            {"timestamp": "2026-04-13T10:10:00+00:00", "level": "INFO", "message": "[cycle_health] health_level=warning;feed_status=error;broker_status=paper;worker_status=online;consecutive_errors=0;fallback=1"},
            {"timestamp": "2026-04-13T10:11:00+00:00", "level": "INFO", "message": "[cycle_health] health_level=healthy;feed_status=healthy;broker_status=paper;worker_status=online;consecutive_errors=0;fallback=0"},
            {"timestamp": "2026-04-14T11:30:00+00:00", "level": "ERROR", "message": "Erro operacional de teste"},
        ],
        columns=config.BOT_LOG_COLUMNS,
    )

    summary = retention.run_retention_job(now=retention.datetime.fromisoformat("2026-04-20T00:00:00+00:00"))
    weekly_index = summary["weekly_reports_index"]

    assert weekly_index
    selected = next(item for item in weekly_index if item["week_key"] == "2026_W16")
    weekly_summary = retention.load_weekly_summary(selected)
    weekly_rows = retention.read_weekly_report_rows(selected)

    assert len(weekly_rows) == 2
    assert weekly_summary["trades_count"] == 2
    assert weekly_summary["operational_errors"] == 1
    assert weekly_summary["fallback_cycle_pct"] == 50.0
    assert weekly_summary["suggestions"]
