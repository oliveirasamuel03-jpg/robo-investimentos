from __future__ import annotations

from datetime import datetime, timedelta, timezone

from tests.conftest import load_module


def _load_external_signals(monkeypatch, *, enabled: bool = True, sources: str = "tradingview"):
    monkeypatch.setenv("EXTERNAL_SIGNAL_WEBHOOK_ENABLED", "true" if enabled else "false")
    monkeypatch.setenv("EXTERNAL_SIGNAL_SECRET", "secret-token")
    monkeypatch.setenv("EXTERNAL_SIGNAL_ALLOWED_SOURCES", sources)
    monkeypatch.setenv("EXTERNAL_SIGNAL_ALLOWED_TIMEFRAMES", "30s,1m,5m,15m,1h,1d")
    monkeypatch.setenv("EXTERNAL_SIGNAL_MAX_AGE_SECONDS", "300")
    monkeypatch.setenv("EXTERNAL_SIGNAL_DEDUPE_SECONDS", "300")
    return load_module("core.external_signals")


def _payload(**overrides):
    now = datetime(2026, 4, 24, 12, 0, tzinfo=timezone.utc)
    payload = {
        "token": "secret-token",
        "source": "tradingview",
        "strategy": "audit_signal",
        "symbol": "BTC-USD",
        "timeframe": "15m",
        "side": "BUY",
        "alert_price": 65000.0,
        "score": 0.72,
        "ts": now.isoformat(),
        "extra": {"note": "audit only"},
    }
    payload.update(overrides)
    return payload


def test_webhook_disabled_returns_disabled_status(isolated_storage, monkeypatch):
    external_signals = _load_external_signals(monkeypatch, enabled=False)

    event = external_signals.validate_external_signal_payload(
        _payload(),
        now=datetime(2026, 4, 24, 12, 1, tzinfo=timezone.utc),
    )

    assert event["status"] == "DISABLED"
    assert event["trade_approved"] is False
    assert event["execution_authority"] is False


def test_test_panel_disabled_by_default_config(isolated_storage, monkeypatch):
    monkeypatch.delenv("EXTERNAL_SIGNAL_TEST_PANEL_ENABLED", raising=False)
    config = load_module("core.config")
    state_store = load_module("core.state_store")

    state = state_store.load_bot_state()

    assert config.EXTERNAL_SIGNAL_TEST_PANEL_ENABLED is False
    assert state["external_signal"]["test_panel_enabled"] is False


def test_invalid_token_rejected(isolated_storage, monkeypatch):
    external_signals = _load_external_signals(monkeypatch)

    event = external_signals.validate_external_signal_payload(
        _payload(token="wrong"),
        now=datetime(2026, 4, 24, 12, 1, tzinfo=timezone.utc),
    )

    assert event["status"] == "REJECTED"
    assert event["reason"] == "invalid_token"


def test_missing_required_fields_rejected(isolated_storage, monkeypatch):
    external_signals = _load_external_signals(monkeypatch)
    payload = _payload()
    payload.pop("symbol")

    event = external_signals.validate_external_signal_payload(
        payload,
        now=datetime(2026, 4, 24, 12, 1, tzinfo=timezone.utc),
    )

    assert event["status"] == "REJECTED"
    assert "missing_required_fields" in event["reason"]


def test_expired_event_rejected(isolated_storage, monkeypatch):
    external_signals = _load_external_signals(monkeypatch)

    event = external_signals.validate_external_signal_payload(
        _payload(ts=datetime(2026, 4, 24, 11, 50, tzinfo=timezone.utc).isoformat()),
        now=datetime(2026, 4, 24, 12, 1, tzinfo=timezone.utc),
    )

    assert event["status"] == "EXPIRED"
    assert event["reason"] == "signal_expired"


def test_invalid_timestamp_rejected(isolated_storage, monkeypatch):
    external_signals = _load_external_signals(monkeypatch)

    event = external_signals.validate_external_signal_payload(
        _payload(ts="not-a-date"),
        now=datetime(2026, 4, 24, 12, 1, tzinfo=timezone.utc),
    )

    assert event["status"] == "REJECTED"
    assert event["reason"] == "invalid_timestamp"


def test_symbol_outside_watchlist_rejected(isolated_storage, monkeypatch):
    external_signals = _load_external_signals(monkeypatch)

    event = external_signals.validate_external_signal_payload(
        _payload(symbol="DOGE-USD"),
        now=datetime(2026, 4, 24, 12, 1, tzinfo=timezone.utc),
    )

    assert event["status"] == "REJECTED"
    assert event["reason"] == "symbol_not_in_watchlist"


def test_invalid_timeframe_rejected(isolated_storage, monkeypatch):
    external_signals = _load_external_signals(monkeypatch)

    event = external_signals.validate_external_signal_payload(
        _payload(timeframe="2m"),
        now=datetime(2026, 4, 24, 12, 1, tzinfo=timezone.utc),
    )

    assert event["status"] == "REJECTED"
    assert event["reason"] == "timeframe_not_allowed"


def test_invalid_side_rejected(isolated_storage, monkeypatch):
    external_signals = _load_external_signals(monkeypatch)

    event = external_signals.validate_external_signal_payload(
        _payload(side="HOLD"),
        now=datetime(2026, 4, 24, 12, 1, tzinfo=timezone.utc),
    )

    assert event["status"] == "REJECTED"
    assert event["reason"] == "side_not_allowed"


def test_disallowed_source_rejected(isolated_storage, monkeypatch):
    external_signals = _load_external_signals(monkeypatch, sources="other")

    event = external_signals.validate_external_signal_payload(
        _payload(),
        now=datetime(2026, 4, 24, 12, 1, tzinfo=timezone.utc),
    )

    assert event["status"] == "REJECTED"
    assert event["reason"] == "source_not_allowed"


def test_valid_signal_accepted_for_audit_only(isolated_storage, monkeypatch):
    external_signals = _load_external_signals(monkeypatch)

    event = external_signals.validate_external_signal_payload(
        _payload(score=72),
        now=datetime(2026, 4, 24, 12, 1, tzinfo=timezone.utc),
    )

    assert event["status"] == "ACCEPTED_FOR_AUDIT"
    assert event["score"] == 0.72
    assert event["audit_only"] is True
    assert event["trade_approved"] is False
    assert event["execution_authority"] is False


def test_sensitive_keys_are_redacted(isolated_storage, monkeypatch):
    external_signals = _load_external_signals(monkeypatch)

    safe = external_signals.safe_payload(
        {
            "token": "secret-token",
            "source": "tradingview",
            "extra": {"api_key": "abc", "nested": {"secret": "hidden"}},
        }
    )

    assert safe["token"] == "[redacted]"
    assert safe["extra"]["api_key"] == "[redacted]"
    assert safe["extra"]["nested"]["secret"] == "[redacted]"


def test_duplicate_signal_rejected(isolated_storage, monkeypatch):
    external_signals = _load_external_signals(monkeypatch)
    now = datetime(2026, 4, 24, 12, 1, tzinfo=timezone.utc)

    first = external_signals.process_external_signal_payload(_payload(), now=now, persist=True)
    second_state = load_module("core.state_store").load_bot_state()
    second = external_signals.validate_external_signal_payload(
        _payload(),
        state=second_state,
        now=now + timedelta(seconds=30),
    )

    assert first["event"]["status"] == "ACCEPTED_FOR_AUDIT"
    assert second["status"] == "DUPLICATE"
    assert second["reason"] == "duplicate_signal"


def test_recent_events_are_capped(isolated_storage, monkeypatch):
    external_signals = _load_external_signals(monkeypatch)
    previous_state = {
        "external_signal": {
            "recent_events": [
                {"status": "REJECTED", "dedupe_key": str(index), "received_at": f"2026-04-24T12:{index:02d}:00+00:00"}
                for index in range(25)
            ]
        }
    }
    event = {
        "status": "ACCEPTED_FOR_AUDIT",
        "reason": "accepted_for_audit_only_no_trade_authority",
        "received_at": "2026-04-24T12:30:00+00:00",
        "dedupe_key": "latest",
    }

    update = external_signals.build_external_signal_state_update(event, previous_state=previous_state)

    assert len(update["recent_events"]) == 20
    assert update["recent_events"][-1]["dedupe_key"] == "latest"


def test_recent_events_display_helper_handles_empty_state(isolated_storage, monkeypatch):
    external_signals = _load_external_signals(monkeypatch)

    assert external_signals.format_external_signal_events_for_display({}, limit=10) == []


def test_recent_events_display_helper_hides_raw_payload(isolated_storage, monkeypatch):
    external_signals = _load_external_signals(monkeypatch)
    rows = external_signals.format_external_signal_events_for_display(
        {
            "recent_events": [
                {
                    "received_at": "2026-04-24T12:00:00+00:00",
                    "source": "tradingview",
                    "strategy": "audit_signal",
                    "symbol": "BTC-USD",
                    "side": "BUY",
                    "timeframe": "15m",
                    "score": 0.5,
                    "status": "ACCEPTED_FOR_AUDIT",
                    "reason": "accepted",
                    "raw_payload_safe": {"token": "[redacted]"},
                }
            ]
        },
        limit=10,
    )

    assert rows[0]["symbol"] == "BTC-USD"
    assert "raw_payload_safe" not in rows[0]


def test_old_state_loads_without_external_signal_keys(isolated_storage, monkeypatch):
    state_store = load_module("core.state_store")

    state = state_store.load_bot_state()
    state.pop("external_signal", None)
    state_store.save_bot_state(state)
    reloaded = state_store.load_bot_state()

    assert reloaded["external_signal"]["last_status"] == "DISABLED"
    assert reloaded["external_signal"]["audit_only"] is True


def test_old_state_loads_without_recent_events(isolated_storage, monkeypatch):
    state_store = load_module("core.state_store")

    state = state_store.load_bot_state()
    state["external_signal"].pop("recent_events", None)
    state_store.save_bot_state(state)
    reloaded = state_store.load_bot_state()

    assert reloaded["external_signal"]["recent_events"] == []
