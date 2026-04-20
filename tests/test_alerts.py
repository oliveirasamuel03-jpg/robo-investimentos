from __future__ import annotations

from tests.conftest import load_module


def test_should_send_alert_respects_disabled_mode(isolated_storage):
    state_store = load_module("core.state_store")
    alerts = load_module("core.alerts")

    state = state_store.load_bot_state()
    result = alerts.should_send_alert(state, alert_type="manual_test")

    assert result["allowed"] is False


def test_should_send_alert_respects_cooldown(isolated_storage):
    state_store = load_module("core.state_store")
    alerts = load_module("core.alerts")

    state = state_store.load_bot_state()
    state["production"]["last_alert_sent_at"] = "2026-04-20T03:00:00+00:00"
    state_store.save_bot_state(state)

    result = alerts.should_send_alert(
        state_store.load_bot_state(),
        alert_type="heartbeat_delayed",
        now=alerts.parse_iso_datetime("2026-04-20T03:10:00+00:00"),
        force=False,
    )

    assert result["allowed"] is False
    assert result["reason"] == "cooldown_active"
