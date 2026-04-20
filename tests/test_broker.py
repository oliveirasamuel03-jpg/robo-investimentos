from __future__ import annotations

from tests.conftest import load_module


def test_probe_broker_status_defaults_to_paper(isolated_storage):
    broker = load_module("core.broker")

    status = broker.probe_broker_status({}, requested_by="worker_cycle")

    assert status["provider"] == "paper"
    assert status["status"] == "paper"
    assert status["mode"] == "paper"
    assert status["can_submit_orders"] is False
    assert status["execution_enabled"] is False


def test_probe_broker_status_requires_credentials_for_non_paper(isolated_storage, monkeypatch):
    monkeypatch.setenv("BROKER_PROVIDER", "alpaca")
    monkeypatch.setenv("BROKER_MODE", "live")
    monkeypatch.delenv("BROKER_API_KEY", raising=False)
    monkeypatch.delenv("BROKER_API_SECRET", raising=False)

    broker = load_module("core.broker")
    status = broker.probe_broker_status({"real_mode_enabled": True}, requested_by="worker_cycle")

    assert status["provider"] == "alpaca"
    assert status["status"] == "missing_credentials"
    assert status["can_submit_orders"] is False


def test_probe_broker_status_keeps_live_guarded_until_real_mode(isolated_storage, monkeypatch):
    monkeypatch.setenv("BROKER_PROVIDER", "alpaca")
    monkeypatch.setenv("BROKER_MODE", "live")
    monkeypatch.setenv("BROKER_API_KEY", "test-key")
    monkeypatch.setenv("BROKER_API_SECRET", "test-secret")

    broker = load_module("core.broker")
    status = broker.probe_broker_status({"real_mode_enabled": False}, requested_by="admin_panel")

    assert status["provider"] == "alpaca"
    assert status["status"] == "guarded"
    assert status["effective_mode"] == "live"
    assert status["can_submit_orders"] is False
