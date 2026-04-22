from __future__ import annotations

from datetime import datetime, timezone

from tests.conftest import load_module


def test_calculate_daily_realized_pnl_uses_closed_sells_only(isolated_storage):
    daily_risk = load_module("core.daily_risk")

    trades = [
        {"timestamp": "2026-04-21T01:00:00+00:00", "side": "BUY", "realized_pnl": 0.0},
        {"timestamp": "2026-04-21T02:00:00+00:00", "side": "SELL", "realized_pnl": -30.0},
        {"timestamp": "2026-04-21T03:00:00+00:00", "side": "SELL", "realized_pnl": 10.0},
        {"timestamp": "2026-04-20T03:00:00+00:00", "side": "SELL", "realized_pnl": -999.0},
    ]

    realized = daily_risk.calculate_daily_realized_pnl(trades, day_key="2026-04-21")

    assert realized == -20.0


def test_evaluate_daily_loss_guard_blocks_new_entries_when_limit_is_consumed(isolated_storage):
    daily_risk = load_module("core.daily_risk")

    now = datetime(2026, 4, 21, 12, 0, tzinfo=timezone.utc)
    previous_state = {"daily_loss_limit_brl": 100.0}
    trades = [
        {"timestamp": "2026-04-21T09:00:00+00:00", "side": "SELL", "realized_pnl": -70.0},
        {"timestamp": "2026-04-21T11:00:00+00:00", "side": "SELL", "realized_pnl": -35.0},
    ]

    result = daily_risk.evaluate_daily_loss_guard(previous_state, trades, now=now)

    assert result["daily_loss_day_key"] == "2026-04-21"
    assert result["daily_realized_pnl_brl"] == -105.0
    assert result["daily_loss_consumed_brl"] == 105.0
    assert result["daily_loss_block_active"] is True
    assert result["daily_loss_remaining_brl"] == 0.0
    assert result["last_transition"] == "blocked"
    assert "Limite de perda diaria atingido" in result["daily_loss_block_reason"]
    assert result["daily_loss_blocked_at"] == now.isoformat()


def test_evaluate_daily_loss_guard_resets_block_on_utc_day_rollover(isolated_storage):
    daily_risk = load_module("core.daily_risk")

    now = datetime(2026, 4, 22, 1, 0, tzinfo=timezone.utc)
    previous_state = {
        "daily_loss_limit_brl": 100.0,
        "daily_loss_day_key": "2026-04-21",
        "daily_loss_block_active": True,
        "daily_loss_blocked_at": "2026-04-21T19:00:00+00:00",
    }

    result = daily_risk.evaluate_daily_loss_guard(previous_state, trades=[], now=now)

    assert result["daily_loss_day_key"] == "2026-04-22"
    assert result["daily_loss_consumed_brl"] == 0.0
    assert result["daily_loss_block_active"] is False
    assert result["daily_loss_blocked_at"] == ""
    assert result["last_transition"] == "reset_day"
    assert result["daily_loss_reset_at"] == now.isoformat()
