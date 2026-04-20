from __future__ import annotations

import json

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


def test_should_send_alert_requires_resend_config(isolated_storage, monkeypatch):
    monkeypatch.setenv("PRODUCTION_MODE", "true")
    monkeypatch.setenv("ALERT_EMAIL_ENABLED", "true")
    monkeypatch.setenv("ALERT_EMAIL_PROVIDER", "resend")
    monkeypatch.delenv("RESEND_API_KEY", raising=False)
    monkeypatch.delenv("ALERT_EMAIL_FROM", raising=False)

    alerts = load_module("core.alerts")
    state_store = load_module("core.state_store")

    result = alerts.should_send_alert(state_store.load_bot_state(), alert_type="manual_test")

    assert result["allowed"] is False
    assert result["reason"] == "resend_not_configured"


def test_force_still_requires_provider_configuration(isolated_storage, monkeypatch):
    monkeypatch.setenv("PRODUCTION_MODE", "true")
    monkeypatch.setenv("ALERT_EMAIL_ENABLED", "true")
    monkeypatch.setenv("ALERT_EMAIL_PROVIDER", "resend")
    monkeypatch.delenv("RESEND_API_KEY", raising=False)
    monkeypatch.delenv("ALERT_EMAIL_FROM", raising=False)

    alerts = load_module("core.alerts")
    state_store = load_module("core.state_store")

    result = alerts.should_send_alert(
        state_store.load_bot_state(),
        alert_type="manual_test",
        force=True,
    )

    assert result["allowed"] is False
    assert result["reason"] == "resend_not_configured"


def test_send_email_alert_uses_resend_provider(isolated_storage, monkeypatch):
    monkeypatch.setenv("PRODUCTION_MODE", "true")
    monkeypatch.setenv("ALERT_EMAIL_ENABLED", "true")
    monkeypatch.setenv("ALERT_EMAIL_PROVIDER", "resend")
    monkeypatch.setenv("ALERT_EMAIL_FROM", "Trade Ops Desk <alerts@example.com>")
    monkeypatch.setenv("ALERT_EMAIL_TO", "oliveirasamuel03@gmail.com")
    monkeypatch.setenv("RESEND_API_KEY", "re_test_key")
    monkeypatch.setenv("RESEND_API_BASE", "https://api.resend.com/emails")

    alerts = load_module("core.alerts")
    state_store = load_module("core.state_store")

    calls = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps({"id": "email_123"}).encode("utf-8")

    def fake_urlopen(request, timeout=0):
        calls["url"] = request.full_url
        calls["auth"] = request.headers.get("Authorization")
        calls["user_agent"] = request.headers.get("User-agent")
        calls["body"] = request.data.decode("utf-8")
        return FakeResponse()

    monkeypatch.setattr(alerts, "urlopen", fake_urlopen)

    result = alerts.send_email_alert(
        "Teste Resend",
        "Corpo de teste",
        alert_type="manual_test",
        state=state_store.load_bot_state(),
        force=True,
    )

    assert result["sent"] is True
    assert result["provider"] == "resend"
    assert calls["url"] == "https://api.resend.com/emails"
    assert "Bearer re_test_key" in str(calls["auth"])
    assert calls["user_agent"]
    assert "Teste Resend" in calls["body"]


def test_send_final_validation_email_uses_generic_alert_path(isolated_storage, monkeypatch):
    alerts = load_module("core.alerts")

    captured = {}

    def fake_send_email_alert(subject, body, **kwargs):
        captured["subject"] = subject
        captured["body"] = body
        captured["kwargs"] = kwargs
        return {"sent": True, "reason": "sent", "provider": "resend"}

    monkeypatch.setattr(alerts, "send_email_alert", fake_send_email_alert)

    result = alerts.send_final_validation_email(
        {
            "validation_mode": "swing_10d",
            "validation_started_at": "2026-04-20T00:00:00+00:00",
            "validation_window_end_at": "2026-04-30T00:00:00+00:00",
            "validation_status": "completed",
            "verdict_message": "Aprovado para continuar em paper trading com a configuracao atual.",
            "final_validation_reason": "Ciclo estavel e consistente.",
            "metrics": {
                "trades_closed": 8,
                "avg_duration_minutes": 4320,
                "fallback_cycle_pct": 10.0,
                "operational_errors": 1,
            },
            "performance": {
                "win_rate": 0.5,
                "payoff": 1.4,
                "pnl_total": 120.0,
            },
            "best_assets": [{"asset": "AAPL"}],
            "worst_assets": [{"asset": "MSFT"}],
            "errors": ["Feed instavel em parte do ciclo."],
            "successes": ["Boa relacao risco/retorno."],
        }
    )

    assert result["sent"] is True
    assert captured["subject"] == "Resultado do Robo - Validacao Swing 10 Dias"
    assert "Trades fechados: 8" in captured["body"]
    assert captured["kwargs"]["alert_type"] == "final_validation"
