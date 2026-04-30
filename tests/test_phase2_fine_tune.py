from __future__ import annotations

import pandas as pd

from tests.conftest import load_module


def _engine():
    return load_module("paper_trading_engine")


def _latest(breakout_20: float = -0.043, momentum: float = -0.007, momentum_8: float = -0.024) -> pd.Series:
    return pd.Series({"breakout_20": breakout_20, "momentum": momentum, "momentum_8": momentum_8})


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
        "trend_alignment": "aligned",
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


def _call_phase2_1(engine, **overrides):
    config = engine.PaperTradingConfig(min_signal_score=0.74, breakout_min=-0.04)
    payload = {
        "latest": _latest(),
        "evaluation": _evaluation(["breakout_not_confirmed", "confidence_too_low", "score_below_minimum"]),
        "adjusted_score": 0.728,
        "effective_min_signal_score": 0.74,
        "data_source": "market",
        "provider_effective": "twelvedata",
        "context_status": "FAVORAVEL",
        "context_blocked": False,
        "macro_alert_active": False,
        "daily_loss_block_active": False,
        "slots_left": 1,
        "fallback_current_count": 0,
        "setup_name": engine.DEFAULT_STRATEGY_NAME,
        "config": config,
    }
    payload.update(overrides)
    return engine._evaluate_phase2_1_multi_minor_confirmation_fine_tune(**payload)


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


def test_phase2_1_applies_for_small_combined_secondary_failures(isolated_storage):
    engine = _engine()

    result = _call_phase2_1(engine)

    assert result["evaluated"] is True
    assert result["applied"] is True
    assert result["score_gap"] == 0.012
    assert result["allowed_reasons"] == ["breakout_not_confirmed", "confidence_too_low", "score_below_minimum"]


def test_phase2_1_blocks_outside_paper(isolated_storage, monkeypatch):
    engine = _engine()
    monkeypatch.setattr(engine, "BROKER_MODE", "live")

    result = _call_phase2_1(engine)

    assert result["applied"] is False
    assert result["guard_reason"] == "paper_mode_not_confirmed"


def test_phase2_1_blocks_fallback_feed(isolated_storage):
    engine = _engine()

    result = _call_phase2_1(engine, data_source="fallback")

    assert result["applied"] is False
    assert result["guard_reason"] == "feed_or_provider_not_safe"


def test_phase2_1_blocks_current_cycle_fallback_count(isolated_storage):
    engine = _engine()

    result = _call_phase2_1(engine, fallback_current_count=1)

    assert result["applied"] is False
    assert result["guard_reason"] == "feed_or_provider_not_safe"


def test_phase2_1_blocks_unknown_provider(isolated_storage):
    engine = _engine()

    result = _call_phase2_1(engine, provider_effective="unknown")

    assert result["applied"] is False
    assert result["guard_reason"] == "feed_or_provider_not_safe"


def test_phase2_1_blocks_critical_context(isolated_storage):
    engine = _engine()

    result = _call_phase2_1(engine, context_status="CRITICO")

    assert result["applied"] is False
    assert result["guard_reason"] == "context_not_safe"


def test_phase2_1_blocks_macro_alert(isolated_storage):
    engine = _engine()

    result = _call_phase2_1(engine, macro_alert_active=True)

    assert result["applied"] is False
    assert result["guard_reason"] == "macro_alert_active"


def test_phase2_1_does_not_ignore_daily_loss_guard(isolated_storage):
    engine = _engine()

    result = _call_phase2_1(engine, daily_loss_block_active=True)

    assert result["applied"] is False
    assert result["guard_reason"] == "daily_loss_guard_active"


def test_phase2_1_does_not_ignore_position_limit(isolated_storage):
    engine = _engine()

    result = _call_phase2_1(engine, slots_left=0)

    assert result["applied"] is False
    assert result["guard_reason"] == "position_limit_reached"


def test_phase2_1_blocks_score_too_far_below_minimum(isolated_storage):
    engine = _engine()

    result = _call_phase2_1(engine, adjusted_score=0.72)

    assert result["applied"] is False
    assert result["guard_reason"] == "score_gap_too_large"


def test_phase2_1_blocks_strong_primary_failures(isolated_storage):
    engine = _engine()

    result = _call_phase2_1(
        engine,
        evaluation=_evaluation(["breakout_not_confirmed", "confidence_too_low", "trend_not_confirmed"]),
    )

    assert result["applied"] is False
    assert result["guard_reason"] == "primary_or_critical_rejection_present"
    assert result["blocked_reasons"] == ["trend_not_confirmed"]


def test_phase2_1_blocks_rsi_out_of_range(isolated_storage):
    engine = _engine()

    result = _call_phase2_1(
        engine,
        evaluation=_evaluation(["breakout_not_confirmed", "confidence_too_low", "reversal_not_eligible"]),
    )

    assert result["applied"] is False
    assert result["guard_reason"] == "primary_or_critical_rejection_present"
    assert result["blocked_reasons"] == ["reversal_not_eligible"]
