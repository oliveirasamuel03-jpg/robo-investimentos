from __future__ import annotations

import pandas as pd

from tests.conftest import load_module


def _engine():
    return load_module("paper_trading_engine")


def _latest(breakout_20: float = -0.043) -> pd.Series:
    return pd.Series({"breakout_20": breakout_20})


def _evaluation(reasons: list[str] | None = None) -> dict:
    return {
        "rejection_reasons": reasons or ["breakout_not_confirmed"],
        "trend_ok": True,
        "score_ok": True,
        "rsi_ok": True,
        "atr_ok": True,
        "pullback_ok": True,
        "momentum_short_ok": True,
        "momentum_medium_ok": True,
    }


def _call(engine, **overrides):
    config = engine.PaperTradingConfig(min_signal_score=0.74, breakout_min=-0.04)
    payload = {
        "latest": _latest(),
        "evaluation": _evaluation(),
        "adjusted_score": 0.75,
        "effective_min_signal_score": 0.74,
        "data_source": "market",
        "provider_effective": "twelvedata",
        "context_status": "FAVORAVEL",
        "context_blocked": False,
        "macro_alert_active": False,
        "daily_loss_block_active": False,
        "slots_left": 1,
        "config": config,
    }
    payload.update(overrides)
    return engine._evaluate_phase2_secondary_confirmation_fine_tune(**payload)


def test_phase2_fine_tune_allows_only_marginal_secondary_confirmation_relaxation(isolated_storage):
    engine = _engine()

    result = _call(engine)

    assert result["evaluated"] is True
    assert result["applied"] is True
    assert result["target"] == engine.PHASE2_FINE_TUNE_TARGET


def test_phase2_fine_tune_preserves_old_behavior_when_disabled(isolated_storage, monkeypatch):
    engine = _engine()
    monkeypatch.setattr(engine, "PHASE2_FINE_TUNE_ENABLED", False)

    result = _call(engine)

    assert result["evaluated"] is False
    assert result["applied"] is False


def test_phase2_fine_tune_does_not_apply_outside_paper(isolated_storage, monkeypatch):
    engine = _engine()
    monkeypatch.setattr(engine, "BROKER_MODE", "live")

    result = _call(engine)

    assert result["applied"] is False
    assert result["guard_reason"] == "paper_mode_not_confirmed"


def test_phase2_fine_tune_does_not_apply_with_fallback_feed(isolated_storage):
    engine = _engine()

    result = _call(engine, data_source="fallback")

    assert result["applied"] is False
    assert result["guard_reason"] == "feed_not_live_or_provider_unknown"


def test_phase2_fine_tune_does_not_apply_with_unknown_provider(isolated_storage):
    engine = _engine()

    result = _call(engine, provider_effective="unknown")

    assert result["applied"] is False
    assert result["guard_reason"] == "feed_not_live_or_provider_unknown"


def test_phase2_fine_tune_does_not_apply_with_critical_context(isolated_storage):
    engine = _engine()

    result = _call(engine, context_status="CRITICO")

    assert result["applied"] is False
    assert result["guard_reason"] == "context_not_safe"


def test_phase2_fine_tune_does_not_ignore_daily_loss_guard(isolated_storage):
    engine = _engine()

    result = _call(engine, daily_loss_block_active=True)

    assert result["applied"] is False
    assert result["guard_reason"] == "daily_loss_guard_active"


def test_phase2_fine_tune_does_not_ignore_position_limit(isolated_storage):
    engine = _engine()

    result = _call(engine, slots_left=0)

    assert result["applied"] is False
    assert result["guard_reason"] == "position_limit_reached"


def test_phase2_fine_tune_does_not_apply_when_breakout_is_too_far(isolated_storage):
    engine = _engine()

    result = _call(engine, latest=_latest(breakout_20=-0.047))

    assert result["applied"] is False
    assert result["guard_reason"] == "breakout_too_far_from_threshold"
