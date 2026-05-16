from __future__ import annotations

from copy import deepcopy

from tests.conftest import load_module


SAFE_RECOMMENDATIONS = {
    "continue_paper_with_controlled_plan",
    "prepare_micro_adjustment_study",
    "observe_more_before_adjustment",
    "no_threshold_change_recommended",
    "no_real_money_recommended",
    "keep_current_strategy_until_next_validation",
    "insufficient_data",
}


def _module():
    return load_module("core.post_10d_calibration_plan")


def _build(**kwargs) -> dict:
    return _module().build_post_10d_calibration_plan(**kwargs)


def _state(**overrides) -> dict:
    state = {
        "worker_status": "online",
        "broker": {"provider": "paper", "mode": "paper"},
        "validation": {"trading_mode": "PAPER", "live_trading_enabled": False},
        "production": {"heartbeat_age_seconds": 0, "consecutive_errors": 0, "worker_online": True},
        "market_data": {
            "feed_status": "LIVE",
            "provider": "twelvedata",
            "provider_effective": "twelvedata",
            "source_breakdown": {"market": 5, "fallback": 0, "unknown": 0},
            "symbols": ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "LINK-USD"],
            "state_writer": "worker",
            "state_written_at": "2026-05-16T00:00:00+00:00",
        },
        "wallet_value": 10000.0,
        "positions": [],
        "history": [],
        "realized_pnl": 0.0,
        "trader": {"ticket_value": 100.0, "max_open_positions": 3, "profile": "Equilibrado"},
    }
    state.update(overrides)
    return state


def _report(**overrides) -> dict:
    report = {
        "validation_day_number": 10,
        "validation_status": "completed",
        "final_validation_grade": "APROVADO_COM_AJUSTES",
        "final_validation_reason": "Continuar em PAPER com ajustes pequenos.",
        "metrics": {
            "signals_total": 30,
            "signals_approved": 0,
            "signals_rejected": 30,
            "operational_errors": 0,
            "heartbeat_age_seconds": 0,
            "consecutive_errors": 0,
            "max_drawdown_pct": 0.0,
        },
        "performance": {"payoff": 1.37, "pnl_total": 0.0},
        "strategy_bottleneck": {
            "dominant_bottleneck": "SCORE_BELOW_MIN",
            "dominant_setup": "trend_pullback_breakout",
            "dominant_asset": "BTC-USD",
        },
        "calibration_preview": {
            "near_approved_count": 0,
            "safe_conditions_met_count": 0,
            "recommendation": "observe_more",
        },
        "feed_scope_reconciliation": {
            "current_feed_is_clean": True,
            "fallback_blocker_scope": "NONE",
            "fallback_scope_status": "NONE",
            "current_fallback_count": 0,
        },
        "setup_blocker_taxonomy_audit": {
            "top_symbol": "BTC-USD",
            "top_setup": "trend_pullback_breakout",
            "normalized_primary_reason": "BOS_MISSING",
            "taxonomy_status": "NO_SETUP_WITH_BOS_MISSING",
        },
        "bos_confirmation_quality_audit": {
            "top_symbol": "BTC-USD",
            "bos_quality_status": "PIVOT_FORMING_BUT_BOS_MISSING",
            "bos_failure_reason": "h1_bos_missing",
        },
        "h1_confirmation_after_h4_bos_audit": {
            "top_symbol": "ETH-USD",
            "h1_confirmation_status": "H1_INSUFFICIENT_DATA_AFTER_H4_BOS",
        },
        "no_setup_eligible_decomposition": {
            "top_reason_bucket": "STRUCTURE_CONFIRMED_BUT_SETUP_NOT_ELIGIBLE",
            "top_real_blocker": "NO_SETUP_ELIGIBLE",
        },
    }
    report.update(overrides)
    return report


def test_aprovado_com_ressalvas_builds_controlled_plan():
    result = _build(state=_state(), validation_report=_report())

    assert result["mode"] == "PLANNING_ONLY"
    assert result["diagnostic_mode"] == "DIAGNOSTIC_ONLY"
    assert result["safety_mode"] == "SHADOW_ONLY"
    assert result["final_classification"] == "APROVADO COM RESSALVAS"
    assert result["plan_status"] == "OPERATIONAL_APPROVED_STRATEGY_WITH_RESSALVAS"
    assert result["recommended_next_phase"] == "FASE 2.6B - Controlled Micro-Adjustment Study"
    assert result["recommendation"] == "prepare_micro_adjustment_study"


def test_final_classification_with_underscore_is_normalized_to_ressalvas():
    result = _build(
        state=_state(),
        validation_report=_report(final_classification="APROVADO_COM_RESSALVAS", final_validation_grade=""),
    )

    assert result["final_classification"] == "APROVADO COM RESSALVAS"
    assert result["plan_status"] == "OPERATIONAL_APPROVED_STRATEGY_WITH_RESSALVAS"


def test_worker_and_feed_healthy_are_approved():
    result = _build(state=_state(), validation_report=_report())

    assert result["operational_status"] == "APPROVED"
    assert result["feed_status"] == "APPROVED"
    assert result["provider_effective"] == "twelvedata"
    assert result["paper_mode_confirmed"] is True


def test_zero_approval_blocks_real_money_and_threshold_change():
    result = _build(state=_state(), validation_report=_report())

    assert "strategy_too_selective" in result["caution_findings"]
    assert "start_real_money" in result["blocked_actions"]
    assert "lower_global_min_signal_score_now" in result["blocked_actions"]
    assert result["should_start_real_money"] is False
    assert result["should_change_threshold_now"] is False
    assert result["threshold_change_allowed_now"] is False


def test_calibration_preview_without_safe_near_approved_blocks_threshold_change():
    report = _report(calibration_preview={"near_approved_count": 0, "safe_conditions_met_count": 0})
    result = _build(state=_state(), validation_report=report)

    assert "insufficient_safe_near_approved_sample" in result["caution_findings"]
    assert "min_signal_score_global" in result["proposed_no_change_items"]
    assert result["threshold_change_allowed_now"] is False


def test_clean_current_feed_does_not_become_current_fallback_bottleneck():
    result = _build(state=_state(), validation_report=_report())

    assert "fallback_not_current_blocker" in result["caution_findings"]
    assert result["dominant_bottleneck"] == "SCORE_BELOW_MIN"
    assert result["feed_status"] == "APPROVED"


def test_accumulated_fallback_does_not_override_current_clean_feed_scope():
    state = _state(
        market_data={
            "feed_status": "LIVE",
            "provider": "twelvedata",
            "provider_effective": "twelvedata",
            "source_breakdown": {"market": 5, "fallback": 344, "unknown": 0},
            "symbols": ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "LINK-USD"],
            "state_writer": "worker",
        }
    )

    result = _build(state=state, validation_report=_report())

    assert result["feed_status"] == "APPROVED"
    assert "fallback_not_current_blocker" in result["caution_findings"]


def test_proposed_micro_adjustments_are_future_studies_only():
    result = _build(state=_state(), validation_report=_report())

    assert result["proposed_micro_adjustments"]
    for item in result["proposed_micro_adjustments"]:
        assert item["allowed_now"] is False
        assert item["requires_next_phase"] is True
        assert "no_global_threshold_change" in item["safety_constraints"]


def test_required_blocked_actions_are_present():
    result = _build(state=_state(), validation_report=_report())

    required = {
        "start_real_money",
        "lower_global_min_signal_score_now",
        "bypass_guards",
        "remove_no_setup_eligible",
        "remove_reversal_not_eligible",
        "convert_shadow_to_real_signal",
        "use_bos_as_direct_trigger",
        "use_fibonacci_as_direct_trigger",
        "increase_risk_now",
        "change_profile_to_aggressive_now",
    }
    assert required.issubset(set(result["blocked_actions"]))


def test_safety_flags_are_hard_blocked():
    result = _build(state=_state(), validation_report=_report())

    assert result["should_continue_paper"] is True
    assert result["should_start_real_money"] is False
    assert result["should_change_threshold_now"] is False
    assert result["should_change_profile_now"] is False
    assert result["real_trade_allowed"] is False
    assert result["capital_change_allowed"] is False
    assert result["strategy_change_allowed_now"] is False
    assert result["recommendation"] in SAFE_RECOMMENDATIONS


def test_does_not_mutate_operational_state_or_thresholds():
    state = _state(
        score_real=0.74,
        min_signal_score=0.74,
        thresholds={"entry": 0.74},
        positions=[{"asset": "BTC-USD"}],
        history=[{"trade": "old"}],
        realized_pnl=12.3,
    )
    before = deepcopy(state)

    result = _build(state=state, validation_report=_report())

    assert state == before
    assert result["should_start_real_money"] is False
    assert result["should_change_threshold_now"] is False


def test_old_state_without_plan_is_compatible_default():
    default_state = _module().default_post_10d_calibration_plan_state()

    assert default_state["enabled"] is True
    assert default_state["mode"] == "PLANNING_ONLY"
    assert default_state["plan_status"] == "INSUFFICIENT_DATA_FOR_PLAN"
    assert default_state["should_continue_paper"] is True
    assert default_state["should_start_real_money"] is False
    assert default_state["should_change_threshold_now"] is False
    assert default_state["recommendation"] == "insufficient_data"


def test_insufficient_data_generates_safe_plan_default():
    result = _build()

    assert result["plan_status"] == "INSUFFICIENT_DATA_FOR_PLAN"
    assert result["recommendation"] == "insufficient_data"
    assert result["should_continue_paper"] is True
    assert result["should_start_real_money"] is False
    assert result["blocked_actions"]
