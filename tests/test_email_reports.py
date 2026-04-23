from __future__ import annotations

from datetime import datetime, timezone

import pytest

from tests.conftest import load_module


def _configure_report_email_env(monkeypatch, *, daily=True, weekly=False, ten_day=False, final=False, smtp_ready=True):
    monkeypatch.setenv("REPORT_EMAIL_ENABLED", "true")
    monkeypatch.setenv("REPORT_EMAIL_PROVIDER", "smtp")
    monkeypatch.setenv("REPORT_EMAIL_TO", "ops@example.com")
    monkeypatch.setenv("REPORT_EMAIL_FROM", "desk@example.com")
    monkeypatch.setenv("REPORT_EMAIL_DAILY_ENABLED", "true" if daily else "false")
    monkeypatch.setenv("REPORT_EMAIL_WEEKLY_ENABLED", "true" if weekly else "false")
    monkeypatch.setenv("REPORT_EMAIL_10DAY_ENABLED", "true" if ten_day else "false")
    monkeypatch.setenv("REPORT_EMAIL_FINAL_ENABLED", "true" if final else "false")
    if smtp_ready:
        monkeypatch.setenv("REPORT_EMAIL_SMTP_HOST", "smtp.example.com")
        monkeypatch.setenv("REPORT_EMAIL_SMTP_PORT", "587")
        monkeypatch.setenv("REPORT_EMAIL_SMTP_USERNAME", "user")
        monkeypatch.setenv("REPORT_EMAIL_SMTP_PASSWORD", "secret")
        monkeypatch.setenv("REPORT_EMAIL_USE_TLS", "true")
    else:
        monkeypatch.delenv("REPORT_EMAIL_SMTP_HOST", raising=False)
        monkeypatch.delenv("REPORT_EMAIL_SMTP_PORT", raising=False)
        monkeypatch.delenv("REPORT_EMAIL_SMTP_USERNAME", raising=False)
        monkeypatch.delenv("REPORT_EMAIL_SMTP_PASSWORD", raising=False)


def _sample_validation_report(day_number: int) -> dict:
    final_grade = "APROVADO_COM_AJUSTES" if day_number >= 10 else ""
    return {
        "validation_day_number": day_number,
        "validation_phase": "Validacao",
        "validation_status": "completed" if day_number >= 10 else "running",
        "validation_started_at": "2026-04-01T00:00:00+00:00",
        "timeframe_label": "Diario (1D)",
        "phase_conclusion": "Fase em andamento com leitura parcial.",
        "final_validation_grade": final_grade,
        "final_validation_reason": "Manter PAPER com ajustes controlados." if final_grade else "",
        "verdict_message": "Aprovado com ajustes." if final_grade else "",
        "metrics": {
            "signals_total": 12,
            "signals_approved": 2,
            "signals_rejected": 10,
            "trades_opened": 1,
            "trades_closed": 1,
            "fallback_cycle_pct": 0.0,
            "operational_errors": 0,
            "max_drawdown_pct": 0.04,
        },
        "performance": {
            "pnl_total": 25.0,
            "win_rate": 0.5,
            "payoff": 1.3,
            "avg_win": 10.0,
            "avg_loss": -7.0,
        },
        "consistency": {
            "signal_approval_rate": 0.166,
            "watchlist_phase_aligned": True,
            "fine_tuning_ready": True,
            "validation_reading_message": "Base minima disponivel para ajustes finos.",
            "signal_quality_message": "Filtro seletivo, mas com feed confiavel.",
        },
        "rejection_quality": {
            "top_reason": "score_below_minimum",
            "top_layer": "strategy",
            "top_strategy": "trend_pullback_breakout",
            "has_minimum_sample": True,
            "top_reasons": [
                {
                    "reason_code": "score_below_minimum",
                    "human_reason": "Score ajustado ficou abaixo do minimo exigido.",
                    "count": 6,
                    "pct": 0.6,
                }
            ],
        },
        "successes": ["Feed confiavel e watchlist coerente."],
        "errors": ["Amostra ainda curta para calibracao fina."],
    }


def test_daily_report_email_sends_once_only_per_day(isolated_storage, monkeypatch):
    _configure_report_email_env(monkeypatch, daily=True, weekly=False, ten_day=False, final=False)
    state_store = load_module("core.state_store")
    email_reports = load_module("core.email_reports")

    captured: list[tuple[str, str]] = []

    monkeypatch.setattr(
        email_reports,
        "_send_report_email",
        lambda subject, body: captured.append((subject, body)) or {"sent": True, "reason": "sent", "provider": "smtp"},
    )

    now = datetime(2026, 4, 23, 12, 0, tzinfo=timezone.utc)
    report = _sample_validation_report(4)

    first = email_reports.process_report_email_delivery(validation_report=report, now=now)
    second = email_reports.process_report_email_delivery(validation_report=report, now=now)
    state = state_store.load_bot_state()

    assert len(first["sent"]) == 1
    assert not second["sent"]
    assert captured[0][0] == "[PAPER] Daily Trading Report - 2026-04-23"
    assert captured[0][1].startswith("[PAPER MODE]")
    assert state["email_reporting"]["last_daily_report_email_date"] == "2026-04-23"


def test_weekly_report_email_sends_once_only_per_week(isolated_storage, monkeypatch):
    _configure_report_email_env(monkeypatch, daily=False, weekly=True, ten_day=False, final=False)
    state_store = load_module("core.state_store")
    email_reports = load_module("core.email_reports")

    captured: list[tuple[str, str]] = []
    monkeypatch.setattr(
        email_reports,
        "_send_report_email",
        lambda subject, body: captured.append((subject, body)) or {"sent": True, "reason": "sent", "provider": "smtp"},
    )

    now = datetime(2026, 4, 23, 12, 0, tzinfo=timezone.utc)
    report = _sample_validation_report(6)

    first = email_reports.process_report_email_delivery(validation_report=report, now=now)
    second = email_reports.process_report_email_delivery(validation_report=report, now=now)
    state = state_store.load_bot_state()

    assert len(first["sent"]) == 1
    assert not second["sent"]
    assert captured[0][0] == "[PAPER] Weekly Trading Report - 2026-W17"
    assert captured[0][1].startswith("[PAPER MODE]")
    assert state["email_reporting"]["last_weekly_report_email_week"] == "2026-W17"


@pytest.mark.parametrize(
    ("day_number", "expected_subject"),
    [
        (10, "[PAPER] Evaluation Block 1 Report - Day 10"),
        (20, "[PAPER] Evaluation Block 2 Report - Day 20"),
        (30, "[PAPER] Evaluation Block 3 Report - Day 30"),
    ],
)
def test_ten_day_report_email_sends_for_block_markers(isolated_storage, monkeypatch, day_number, expected_subject):
    _configure_report_email_env(monkeypatch, daily=False, weekly=False, ten_day=True, final=False)
    state_store = load_module("core.state_store")
    email_reports = load_module("core.email_reports")

    captured: list[tuple[str, str]] = []
    monkeypatch.setattr(
        email_reports,
        "_send_report_email",
        lambda subject, body: captured.append((subject, body)) or {"sent": True, "reason": "sent", "provider": "smtp"},
    )

    now = datetime(2026, 4, 23, 12, 0, tzinfo=timezone.utc)
    report = _sample_validation_report(day_number)

    result = email_reports.process_report_email_delivery(validation_report=report, now=now)
    state = state_store.load_bot_state()

    assert len(result["sent"]) == 1
    assert captured[0][0] == expected_subject
    assert captured[0][1].startswith("[PAPER MODE]")
    assert state["email_reporting"]["last_10day_report_email_block"] == str(((day_number - 1) // 10) + 1)


def test_final_report_email_sends_once_only(isolated_storage, monkeypatch):
    _configure_report_email_env(monkeypatch, daily=False, weekly=False, ten_day=False, final=True)
    state_store = load_module("core.state_store")
    email_reports = load_module("core.email_reports")

    captured: list[tuple[str, str]] = []
    monkeypatch.setattr(
        email_reports,
        "_send_report_email",
        lambda subject, body: captured.append((subject, body)) or {"sent": True, "reason": "sent", "provider": "smtp"},
    )

    now = datetime(2026, 4, 10, 12, 0, tzinfo=timezone.utc)
    report = _sample_validation_report(10)
    report["final_validation_grade"] = "APROVADO"

    first = email_reports.process_report_email_delivery(validation_report=report, now=now)
    second = email_reports.process_report_email_delivery(validation_report=report, now=now)
    state = state_store.load_bot_state()

    assert len(first["sent"]) == 1
    assert not second["sent"]
    assert captured[0][0] == "[PAPER] Final 10-Day Phase Report"
    assert captured[0][1].startswith("[PAPER MODE]")
    assert state["email_reporting"]["last_final_report_email_ts"]


def test_final_report_path_is_reachable_in_active_10_day_cycle(isolated_storage, monkeypatch):
    _configure_report_email_env(monkeypatch, daily=False, weekly=False, ten_day=False, final=True)
    email_reports = load_module("core.email_reports")

    report = _sample_validation_report(10)

    assert email_reports.active_final_model_label() == "10-day final"
    assert email_reports.final_report_path_reachable(report) is True


def test_missing_smtp_config_fails_safely(isolated_storage, monkeypatch):
    _configure_report_email_env(monkeypatch, daily=True, weekly=False, ten_day=False, final=False, smtp_ready=False)
    state_store = load_module("core.state_store")
    email_reports = load_module("core.email_reports")

    result = email_reports.process_report_email_delivery(
        validation_report=_sample_validation_report(4),
        now=datetime(2026, 4, 23, 12, 0, tzinfo=timezone.utc),
    )
    state = state_store.load_bot_state()

    assert not result["sent"]
    assert result["skipped"] == ["smtp_not_configured"]
    assert state["email_reporting"]["last_email_delivery_status"] == "misconfigured"


def test_email_delivery_failure_does_not_raise_and_persists_metadata(isolated_storage, monkeypatch):
    _configure_report_email_env(monkeypatch, daily=True, weekly=False, ten_day=False, final=False)
    state_store = load_module("core.state_store")
    email_reports = load_module("core.email_reports")

    monkeypatch.setattr(
        email_reports,
        "_send_report_email",
        lambda subject, body: {"sent": False, "reason": "smtp_failed", "provider": "smtp"},
    )

    result = email_reports.process_report_email_delivery(
        validation_report=_sample_validation_report(4),
        now=datetime(2026, 4, 23, 12, 0, tzinfo=timezone.utc),
    )
    state = state_store.load_bot_state()

    assert not result["sent"]
    assert len(result["failed"]) == 1
    assert state["email_reporting"]["last_email_delivery_status"] == "failed"
    assert state["email_reporting"]["last_email_delivery_reason"] == "smtp_failed"


def test_email_reporting_status_supports_old_state_shape(isolated_storage, monkeypatch):
    _configure_report_email_env(monkeypatch, daily=True, weekly=True, ten_day=True, final=True)
    email_reports = load_module("core.email_reports")

    status = email_reports.build_email_reporting_status({"production": {}, "validation": {}})

    assert status["destination"] == "ops@example.com"
    assert status["last_delivery_status"] == ""
    assert status["enabled"] is True
