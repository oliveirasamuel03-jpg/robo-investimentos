from __future__ import annotations

from datetime import datetime, timedelta, timezone

from conftest import load_module


def _load_macro(monkeypatch, enabled: bool = True):
    monkeypatch.setenv("MACRO_ALERTS_ENABLED", "true" if enabled else "false")
    return load_module("core.macro_alerts")


def test_no_event_keeps_macro_alert_inactive(isolated_storage, monkeypatch):
    macro_alerts = _load_macro(monkeypatch, enabled=True)
    now = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)

    alert = macro_alerts.evaluate_macro_alert(events=[], now=now)

    assert alert["macro_alert_active"] is False
    assert alert["macro_alert_window_status"] == "INACTIVE"
    assert alert["macro_alert_blocks_new_entries"] is False


def test_pre_in_and_post_event_windows_activate(isolated_storage, monkeypatch):
    macro_alerts = _load_macro(monkeypatch, enabled=True)
    now = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)

    pre_alert = macro_alerts.evaluate_macro_alert(
        events=[{"title": "FOMC", "currency": "USD", "datetime": (now + timedelta(minutes=30)).isoformat(), "impact_level": "HIGH"}],
        now=now,
        pre_minutes=60,
        event_window_minutes=30,
        post_minutes=60,
    )
    in_alert = macro_alerts.evaluate_macro_alert(
        events=[{"title": "FOMC", "currency": "USD", "datetime": (now - timedelta(minutes=5)).isoformat(), "impact_level": "HIGH"}],
        now=now,
        pre_minutes=60,
        event_window_minutes=30,
        post_minutes=60,
    )
    post_alert = macro_alerts.evaluate_macro_alert(
        events=[{"title": "FOMC", "currency": "USD", "datetime": (now - timedelta(minutes=45)).isoformat(), "impact_level": "HIGH"}],
        now=now,
        pre_minutes=60,
        event_window_minutes=30,
        post_minutes=60,
    )

    assert pre_alert["macro_alert_window_status"] == "PRE_EVENT"
    assert in_alert["macro_alert_window_status"] == "IN_EVENT"
    assert post_alert["macro_alert_window_status"] == "POST_EVENT"


def test_high_impact_event_reduces_confidence(isolated_storage, monkeypatch):
    macro_alerts = _load_macro(monkeypatch, enabled=True)
    alert = {
        **macro_alerts.default_macro_alert_state(enabled=True),
        "macro_alert_active": True,
        "macro_alert_level": "HIGH",
        "macro_alert_window_status": "PRE_EVENT",
        "macro_alert_penalty": 0.04,
    }

    result = macro_alerts.apply_macro_risk_filter(
        {
            "score": 0.82,
            "effective_min_signal_score": 0.74,
            "trend_alignment": "aligned",
            "trend_ok": True,
        },
        alert,
    )

    assert result["adjusted_score"] < 0.82
    assert result["effective_min_signal_score"] > 0.74


def test_fragile_setup_blocks_during_high_macro_window(isolated_storage, monkeypatch):
    macro_alerts = _load_macro(monkeypatch, enabled=True)
    alert = {
        **macro_alerts.default_macro_alert_state(enabled=True),
        "macro_alert_active": True,
        "macro_alert_level": "HIGH",
        "macro_alert_window_status": "IN_EVENT",
        "macro_alert_penalty": 0.08,
        "macro_alert_blocks_new_entries": True,
    }

    result = macro_alerts.apply_macro_risk_filter(
        {
            "score": 0.78,
            "effective_min_signal_score": 0.74,
            "trend_alignment": "aligned",
            "trend_ok": True,
            "breakout_ok": True,
            "momentum_ok": True,
        },
        alert,
    )

    assert result["setup_family"] == "BREAKOUT_CONFIRMED"
    assert result["blocked"] is True
    assert result["reason_code"] == "macro_alert_guard"


def test_strong_trend_setup_can_remain_eligible_under_macro_alert(isolated_storage, monkeypatch):
    macro_alerts = _load_macro(monkeypatch, enabled=True)
    alert = {
        **macro_alerts.default_macro_alert_state(enabled=True),
        "macro_alert_active": True,
        "macro_alert_level": "HIGH",
        "macro_alert_window_status": "IN_EVENT",
        "macro_alert_penalty": 0.08,
        "macro_alert_blocks_new_entries": True,
    }

    result = macro_alerts.apply_macro_risk_filter(
        {
            "score": 0.95,
            "effective_min_signal_score": 0.74,
            "trend_alignment": "aligned",
            "trend_ok": True,
            "breakout_ok": False,
            "momentum_ok": False,
        },
        alert,
    )

    assert result["setup_family"] == "TREND_FOLLOWING_SWING"
    assert result["blocked"] is False
    assert result["adjusted_score"] >= result["effective_min_signal_score"]


def test_fallback_feed_with_high_macro_alert_blocks_entry(isolated_storage, monkeypatch):
    macro_alerts = _load_macro(monkeypatch, enabled=True)
    alert = {
        **macro_alerts.default_macro_alert_state(enabled=True),
        "macro_alert_active": True,
        "macro_alert_level": "HIGH",
        "macro_alert_window_status": "PRE_EVENT",
        "macro_alert_penalty": 0.04,
    }

    result = macro_alerts.apply_macro_risk_filter(
        {
            "score": 0.95,
            "effective_min_signal_score": 0.74,
            "trend_alignment": "aligned",
            "trend_ok": True,
            "data_source": "fallback",
        },
        alert,
    )

    assert result["blocked"] is True
    assert result["reason_code"] == "macro_alert_guard"


def test_invalid_event_data_is_ignored_safely(isolated_storage, monkeypatch):
    macro_alerts = _load_macro(monkeypatch, enabled=True)

    alert = macro_alerts.evaluate_macro_alert(
        events=[{"title": "Invalid event", "datetime": "not-a-date", "impact_level": "HIGH"}],
        now=datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc),
    )

    assert alert["macro_alert_active"] is False
    assert alert["macro_alert_window_status"] == "INACTIVE"


def test_old_state_loads_with_macro_defaults(isolated_storage, monkeypatch):
    monkeypatch.setenv("MACRO_ALERTS_ENABLED", "false")
    state_store = load_module("core.state_store")

    state = state_store.load_bot_state()

    assert "macro_alert" in state
    assert state["macro_alert"]["macro_alert_active"] is False
    assert state["macro_alert"]["macro_alert_window_status"] == "INACTIVE"
