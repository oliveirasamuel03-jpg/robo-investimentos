from __future__ import annotations

from tests.conftest import load_module


def test_evaluate_production_health_marks_healthy_with_recent_heartbeat(isolated_storage):
    state_store = load_module("core.state_store")
    production_monitor = load_module("core.production_monitor")

    state = state_store.load_bot_state()
    state["worker_status"] = "online"
    state["worker_heartbeat"] = "2026-04-20T03:00:00+00:00"
    state["market_data"]["status"] = "healthy"
    state["market_data"]["last_source"] = "market"
    state["broker"]["status"] = "paper"
    state["production"]["last_success_at"] = "2026-04-20T03:00:00+00:00"
    state_store.save_bot_state(state)

    health = production_monitor.evaluate_production_health(
        state_store.load_bot_state(),
        now=production_monitor.parse_iso_datetime("2026-04-20T03:01:00+00:00"),
    )

    assert health["health_level"] == "healthy"
    assert health["heartbeat_age_seconds"] == 60
    assert health["broker_status"] == "paper"


def test_evaluate_production_health_marks_critical_after_consecutive_errors(isolated_storage):
    state_store = load_module("core.state_store")
    production_monitor = load_module("core.production_monitor")

    state = state_store.load_bot_state()
    state["worker_status"] = "error"
    state["worker_heartbeat"] = "2026-04-20T03:00:00+00:00"
    state["market_data"]["status"] = "error"
    state["market_data"]["last_source"] = "fallback"
    state["market_data"]["fallback_since_at"] = "2026-04-20T02:30:00+00:00"
    state["production"]["consecutive_errors"] = 3
    state_store.save_bot_state(state)

    health = production_monitor.evaluate_production_health(
        state_store.load_bot_state(),
        now=production_monitor.parse_iso_datetime("2026-04-20T03:10:00+00:00"),
    )

    assert health["health_level"] == "critical"
    assert health["health_reason"] in {"worker_error", "consecutive_errors", "heartbeat_delayed", "feed_fallback"}
    assert health["fallback_age_minutes"] >= 40
