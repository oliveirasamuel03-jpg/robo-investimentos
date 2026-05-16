from __future__ import annotations

from copy import deepcopy

from tests.conftest import load_module


SAFE_RECOMMENDATIONS = {
    "observe_more_before_adjustment",
    "prepare_single_micro_adjustment_phase",
    "study_secondary_confirmation_only",
    "study_breakout_confirmation_only",
    "study_real_rule_mapping_only",
    "study_h1_after_h4_bos_only",
    "study_pullback_quality_only",
    "no_threshold_change_recommended",
    "no_profile_change_recommended",
    "no_real_money_recommended",
    "insufficient_data",
}


def _module():
    return load_module("core.controlled_micro_adjustment_study")


def _build(**kwargs) -> dict:
    return _module().build_controlled_micro_adjustment_study(**kwargs)


def _state(**overrides) -> dict:
    state = {
        "worker_status": "online",
        "broker": {"mode": "paper", "provider": "paper"},
        "validation": {"trading_mode": "PAPER", "live_trading_enabled": False},
        "market_context": {"market_context_status": "NEUTRO", "context_score": 50.0},
        "market_data": {
            "feed_status": "LIVE",
            "provider_effective": "twelvedata",
            "source_breakdown": {"market": 5, "fallback": 0, "unknown": 0},
            "symbols": ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "LINK-USD"],
        },
        "positions": [],
        "history": [],
        "wallet_value": 10000.0,
        "realized_pnl": 0.0,
        "trader": {"ticket_value": 100.0, "max_open_positions": 3},
    }
    state.update(overrides)
    return state


def _report(**overrides) -> dict:
    report = {
        "post_10d_calibration_plan": {
            "mode": "PLANNING_ONLY",
            "final_classification": "APROVADO COM RESSALVAS",
            "plan_status": "OPERATIONAL_APPROVED_STRATEGY_WITH_RESSALVAS",
            "dominant_bottleneck": "SCORE_BELOW_MIN",
            "dominant_setup": "trend_pullback_breakout",
            "should_continue_paper": True,
            "should_start_real_money": False,
            "should_change_threshold_now": False,
        },
        "calibration_preview": {
            "near_approved_count": 0,
            "best_score_seen": 0.55,
            "min_score_current": 0.80,
            "preview_score_floor": 0.76,
            "recommendation": "observe_more",
        },
        "strategy_bottleneck": {
            "dominant_bottleneck": "SCORE_BELOW_MIN",
            "dominant_setup": "trend_pullback_breakout",
            "secondary_confirmation_weak_count": 5,
        },
        "feed_scope_reconciliation": {
            "current_feed_is_clean": True,
            "current_fallback_count": 0,
            "fallback_blocker_scope": "NONE",
        },
        "setup_blocker_taxonomy_audit": {
            "normalized_primary_reason": "BOS_MISSING",
            "taxonomy_status": "NO_SETUP_WITH_BOS_MISSING",
        },
        "no_setup_eligible_decomposition": {},
        "strategy_decision_bridge_trace": {},
        "bos_confirmation_quality_audit": {},
        "h1_confirmation_after_h4_bos_audit": {},
    }
    report.update(overrides)
    return report


def test_context_desfavoravel_blocks_application_now():
    result = _build(
        state=_state(market_context={"market_context_status": "DESFAVORAVEL", "context_score": 33.0}),
        validation_report=_report(),
    )

    assert result["study_status"] == "CONTEXT_NOT_SAFE_FOR_ADJUSTMENT"
    assert result["market_context_status"] == "DESFAVORAVEL"
    assert result["context_allows_adjustment_now"] is False
    assert result["should_apply_micro_adjustment_now"] is False
    assert result["recommendation"] == "observe_more_before_adjustment"


def test_near_approved_zero_blocks_threshold_change():
    result = _build(state=_state(), validation_report=_report())

    assert result["near_approved_count"] == 0
    assert result["should_change_threshold_now"] is False
    assert "lower_global_min_signal_score_now" in result["blocked_actions"]


def test_aprovado_com_ressalvas_plan_generates_2_6b_study():
    result = _build(state=_state(), validation_report=_report())

    assert result["mode"] == "STUDY_ONLY"
    assert result["diagnostic_mode"] == "DIAGNOSTIC_ONLY"
    assert result["safety_mode"] == "SHADOW_ONLY"
    assert result["source_phase"] == "FASE 2.6A"
    assert result["recommended_next_phase"] == "FASE_2_6C_ONLY_IF_CONDITIONS_PASS"


def test_structure_confirmed_no_setup_prioritizes_real_rule_mapping_study():
    result = _build(
        state=_state(),
        validation_report=_report(
            no_setup_eligible_decomposition={
                "top_reason_bucket": "STRUCTURE_CONFIRMED_BUT_SETUP_NOT_ELIGIBLE",
                "top_real_blocker": "NO_SETUP_ELIGIBLE",
                "structure_confirmed_but_no_setup_count": 2,
            },
            strategy_decision_bridge_trace={
                "top_real_blocker": "NO_SETUP_ELIGIBLE",
                "top_bridge_status": "STRUCTURE_CONFIRMED",
            },
        ),
    )

    assert result["selected_candidate_adjustment"] == "real_rule_mapping_study"
    assert result["selected_candidate_allowed_now"] is False


def test_h4_bos_confirmed_h1_pending_prioritizes_h1_mapping_study():
    result = _build(
        state=_state(),
        validation_report=_report(
            h1_confirmation_after_h4_bos_audit={
                "h4_bos_state": "BOS_RETEST_CONFIRMED",
                "h1_bos_state": "INSUFFICIENT_DATA",
                "h1_confirmation_status": "H1_INSUFFICIENT_DATA_AFTER_H4_BOS",
                "h4_bos_confirmed_count": 1,
                "h1_missing_confirmation_count": 1,
            },
        ),
    )

    assert result["selected_candidate_adjustment"] == "h1_after_h4_bos_mapping_study"
    assert result["selected_candidate_allowed_now"] is False


def test_score_below_min_secondary_lists_secondary_confirmation_candidate():
    result = _build(
        state=_state(),
        validation_report=_report(
            calibration_preview={"near_approved_count": 2, "best_score_seen": 0.77, "min_score_current": 0.80},
            strategy_bottleneck={
                "dominant_bottleneck": "SCORE_BELOW_MIN_SECONDARY_CONFIRMATION_WEAK",
                "dominant_setup": "trend_pullback_breakout",
                "secondary_confirmation_weak_count": 5,
            },
        ),
    )

    candidate = next(item for item in result["candidate_adjustments"] if item["id"] == "secondary_confirmation_micro_adjustment_study")
    assert candidate["allowed_now"] is False
    assert candidate["requires_next_phase"] is True
    assert candidate["can_change_threshold"] is False


def test_candidate_adjustments_are_always_future_studies_only():
    result = _build(state=_state(), validation_report=_report())

    assert result["candidate_adjustments"]
    for item in result["candidate_adjustments"]:
        assert item["allowed_now"] is False
        assert item["requires_next_phase"] is True
        assert item["can_change_threshold"] is False
        assert item["can_change_profile"] is False
        assert item["can_affect_real_trade"] is False


def test_safety_flags_are_hard_blocked():
    result = _build(state=_state(), validation_report=_report())

    assert result["should_continue_paper"] is True
    assert result["should_start_real_money"] is False
    assert result["should_change_threshold_now"] is False
    assert result["should_change_profile_now"] is False
    assert result["should_apply_micro_adjustment_now"] is False
    assert result["selected_candidate_allowed_now"] is False


def test_blocked_actions_contain_required_items():
    result = _build(state=_state(), validation_report=_report())

    required = {
        "apply_micro_adjustment_now",
        "start_real_money",
        "lower_global_min_signal_score_now",
        "change_profile_to_aggressive_now",
        "bypass_guards",
        "remove_no_setup_eligible",
        "remove_reversal_not_eligible",
        "convert_shadow_to_real_signal",
        "use_bos_as_direct_trigger",
        "use_fibonacci_as_direct_trigger",
        "use_h4_bos_as_direct_trigger",
        "use_h1_confirmation_as_direct_trigger",
        "increase_risk_now",
        "increase_ticket_now",
        "increase_max_open_positions_now",
    }
    assert required.issubset(set(result["blocked_actions"]))


def test_recommendations_belong_to_safe_whitelist():
    result = _build(state=_state(), validation_report=_report())

    assert result["recommendation"] in SAFE_RECOMMENDATIONS


def test_does_not_mutate_operational_state():
    state = _state(
        score_real=0.80,
        min_signal_score=0.80,
        thresholds={"entry": 0.80},
        positions=[{"asset": "BTC-USD", "status": "OPEN"}],
        history=[{"id": 1}],
        wallet_value=12345.67,
    )
    before = deepcopy(state)

    result = _build(state=state, validation_report=_report())

    assert state == before
    assert result["should_apply_micro_adjustment_now"] is False


def test_old_state_without_study_is_compatible_default():
    default_state = _module().default_controlled_micro_adjustment_study_state()

    assert default_state["enabled"] is True
    assert default_state["mode"] == "STUDY_ONLY"
    assert default_state["study_status"] == "INSUFFICIENT_DATA_FOR_MICRO_ADJUSTMENT"
    assert default_state["should_continue_paper"] is True
    assert default_state["should_start_real_money"] is False
    assert default_state["should_change_threshold_now"] is False
    assert default_state["should_apply_micro_adjustment_now"] is False
    assert default_state["recommendation"] == "insufficient_data"


def test_insufficient_data_generates_safe_default():
    result = _build()

    assert result["study_status"] == "INSUFFICIENT_DATA_FOR_MICRO_ADJUSTMENT"
    assert result["recommendation"] == "insufficient_data"
    assert result["should_continue_paper"] is True
    assert result["should_start_real_money"] is False
    assert result["should_apply_micro_adjustment_now"] is False
