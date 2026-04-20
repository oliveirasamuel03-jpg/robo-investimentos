from __future__ import annotations

from tests.conftest import load_module


def test_apply_swing_validation_overrides_promotes_daily_swing_config(isolated_storage):
    profiles = load_module("core.trader_profiles")
    swing_validation = load_module("core.swing_validation")

    base = profiles.get_trader_profile_config(
        "Equilibrado",
        base_ticket_value=100.0,
        base_holding_minutes=60,
        base_max_open_positions=3,
    )
    swing = swing_validation.apply_swing_validation_overrides("Equilibrado", base)

    assert swing["interval"] == "1d"
    assert swing["period"] == "2y"
    assert swing["holding_minutes"] >= 6 * 24 * 60
    assert swing["min_signal_score"] >= 0.74
    assert swing["reentry_cooldown_minutes"] >= 24 * 60


def test_refresh_swing_validation_cycle_generates_final_verdict(isolated_storage):
    config = load_module("core.config")
    state_store = load_module("core.state_store")
    swing_validation = load_module("core.swing_validation")

    state = state_store.load_bot_state()
    state["validation"]["validation_started_at"] = "2026-04-10T00:00:00+00:00"
    state["validation"]["signal_counters"] = {
        "signals_total": 8,
        "signals_approved": 3,
        "signals_rejected": 5,
        "entries_against_trend": 0,
        "rejections": {
            "against_trend": 0,
            "weak_score": 3,
            "rsi_filter": 1,
            "atr_filter": 0,
            "structure_filter": 0,
            "momentum_filter": 1,
            "feed_unreliable": 0,
        },
        "assets_observed": {"AAPL": 4, "MSFT": 4},
        "assets_approved": {"AAPL": 2, "MSFT": 1},
    }
    state["production"]["consecutive_errors"] = 0
    state_store.save_bot_state(state)

    state_store.replace_storage_table(
        config.TRADER_REPORTS_FILE,
        [
            {
                "trade_id": "swing-1",
                "opened_at": "2026-04-12T00:00:00+00:00",
                "closed_at": "2026-04-16T00:00:00+00:00",
                "duration_minutes": 5760,
                "asset": "AAPL",
                "profile": "Equilibrado",
                "entry_price": 100,
                "exit_price": 104,
                "quantity": 1,
                "gross_entry_value": 100,
                "gross_exit_value": 104,
                "realized_pnl": 4,
                "realized_pnl_pct": 0.04,
                "entry_score": 0.82,
                "exit_reason": "holding expirado",
                "rsi_entry": 50,
                "rsi_exit": 66,
                "atr_pct_entry": 0.01,
                "atr_pct_exit": 0.012,
                "ma20_entry": 100,
                "ma50_entry": 98,
                "ma20_exit": 103,
                "ma50_exit": 100,
                "peak_price": 105,
                "trailing_stop_final": 101,
                "holding_minutes_limit": 8640,
                "status_final": "gain",
            },
            {
                "trade_id": "swing-2",
                "opened_at": "2026-04-13T00:00:00+00:00",
                "closed_at": "2026-04-18T00:00:00+00:00",
                "duration_minutes": 7200,
                "asset": "MSFT",
                "profile": "Equilibrado",
                "entry_price": 100,
                "exit_price": 102,
                "quantity": 1,
                "gross_entry_value": 100,
                "gross_exit_value": 102,
                "realized_pnl": 2,
                "realized_pnl_pct": 0.02,
                "entry_score": 0.79,
                "exit_reason": "trailing stop",
                "rsi_entry": 49,
                "rsi_exit": 60,
                "atr_pct_entry": 0.012,
                "atr_pct_exit": 0.011,
                "ma20_entry": 100,
                "ma50_entry": 97,
                "ma20_exit": 102,
                "ma50_exit": 99,
                "peak_price": 103,
                "trailing_stop_final": 100,
                "holding_minutes_limit": 8640,
                "status_final": "gain",
            },
        ],
        columns=config.TRADER_REPORTS_COLUMNS,
    )
    state_store.replace_storage_table(
        config.TRADER_ORDERS_FILE,
        [
            {"timestamp": "2026-04-12T00:00:00+00:00", "asset": "AAPL", "side": "BUY", "quantity": 1, "price": 100, "gross_value": 100, "cost": 0, "cash_after": 9900, "source": "paper"},
            {"timestamp": "2026-04-16T00:00:00+00:00", "asset": "AAPL", "side": "SELL", "quantity": 1, "price": 104, "gross_value": 104, "cost": 0, "cash_after": 10004, "source": "paper"},
            {"timestamp": "2026-04-13T00:00:00+00:00", "asset": "MSFT", "side": "BUY", "quantity": 1, "price": 100, "gross_value": 100, "cost": 0, "cash_after": 9904, "source": "paper"},
            {"timestamp": "2026-04-18T00:00:00+00:00", "asset": "MSFT", "side": "SELL", "quantity": 1, "price": 102, "gross_value": 102, "cost": 0, "cash_after": 10006, "source": "paper"},
        ],
        columns=config.TRADER_ORDERS_COLUMNS,
    )
    state_store.replace_storage_table(
        config.BOT_LOG_FILE,
        [
            {"timestamp": "2026-04-12T00:00:00+00:00", "level": "INFO", "message": "[cycle_health] health_level=healthy;feed_status=healthy;broker_status=paper;worker_status=online;consecutive_errors=0;fallback=0"},
            {"timestamp": "2026-04-13T00:00:00+00:00", "level": "INFO", "message": "[cycle_health] health_level=healthy;feed_status=healthy;broker_status=paper;worker_status=online;consecutive_errors=0;fallback=0"},
        ],
        columns=config.BOT_LOG_COLUMNS,
    )

    report = swing_validation.refresh_swing_validation_cycle(
        now=swing_validation._utc_now(swing_validation.datetime.fromisoformat("2026-04-20T00:00:00+00:00"))
    )

    assert report["validation_day_number"] == 10
    assert report["validation_phase"] == "Consolidacao"
    assert report["final_validation_grade"] in {"APROVADO", "APROVADO_COM_AJUSTES"}
    assert report["metrics"]["trades_closed"] == 2
    assert report["performance"]["pnl_total"] > 0
