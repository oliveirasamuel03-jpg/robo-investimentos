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


def test_old_state_loads_without_external_signal_keys(isolated_storage, monkeypatch):
    state_store = load_module("core.state_store")

    state = state_store.load_bot_state()
    state.pop("external_signal", None)
    state_store.save_bot_state(state)
    reloaded = state_store.load_bot_state()

    assert reloaded["external_signal"]["last_status"] == "DISABLED"
    assert reloaded["external_signal"]["audit_only"] is True
