from __future__ import annotations

from tests.conftest import load_module


SAFE_RECOMMENDATIONS = {
    "observe_more",
    "study_routing_map",
    "study_real_rule_mapping",
    "study_pullback_quality",
    "study_breakout_confirmation",
    "study_multitf_conflict",
    "keep_blocked_until_structure_confirms",
    "keep_blocked_until_pullback_reaction",
    "keep_blocked_until_real_setup_maps_structure",
    "no_strategy_change_recommended",
    "insufficient_data",
}


def _feed_clean(scope: str = "NONE") -> dict:
    return {
        "fallback_scope_status": "NO_CURRENT_FALLBACK" if scope == "NONE" else "ACCUMULATED_ONLY_FALLBACK",
        "fallback_blocker_scope": scope,
        "current_feed_is_clean": True,
    }


def test_trend_setup_reversal_blocker_is_traced_without_operational_authority(isolated_storage):
    module = load_module("core.reversal_blocker_routing_audit")

    result = module.build_reversal_blocker_routing_audit(
        signals=[
            {
                "asset": "BTC-USD",
                "strategy_name": "trend_pullback_breakout",
                "score": 0.73,
                "effective_min_signal_score": 0.74,
                "rejection_reasons": ["reversal_not_eligible"],
            }
        ],
        feed_scope_reconciliation=_feed_clean(),
    )
    candidate = result["candidates"][0]

    assert result["mode"] == "DIAGNOSTIC_ONLY"
    assert result["safety_mode"] == "SHADOW_ONLY"
    assert candidate["route_status"] == "REVERSAL_BLOCKER_ON_TREND_SETUP"
    assert result["should_keep_blocked"] is True
    assert candidate["should_keep_blocked"] is True
    assert result["safe_to_change_strategy_now"] is False
    assert candidate["safe_to_change_strategy_now"] is False
    assert result["recommendation"] in SAFE_RECOMMENDATIONS


def test_btc_like_multitf_conflict_explains_reversal_risk(isolated_storage):
    module = load_module("core.reversal_blocker_routing_audit")

    result = module.build_reversal_blocker_routing_audit(
        signals=[
            {
                "asset": "BTC-USD",
                "strategy_name": "trend_pullback_breakout",
                "score": 0.72,
                "effective_min_signal_score": 0.74,
                "rejection_reasons": ["reversal_not_eligible"],
            }
        ],
        strategy_decision_bridge_trace={
            "recent_candidates": [
                {
                    "symbol": "BTC-USD",
                    "real_strategy": "trend_pullback_breakout",
                    "real_score": 0.72,
                    "min_score": 0.74,
                    "primary_real_blocker": "REVERSAL_NOT_ELIGIBLE",
                    "real_rejection_reason": "reversal_not_eligible",
                    "multi_tf_alignment_status": "CONFLICT",
                    "bos_state_4h": "NO_BOS",
                    "pivot_state_4h": "PIVOT_INVALIDATED",
                }
            ]
        },
        multi_timeframe_swing_audit={
            "recent_candidates": [
                {
                    "symbol": "BTC-USD",
                    "daily_bias": "UP",
                    "h4_structure": "SIDEWAYS",
                    "h1_confirmation": "DOWN",
                    "alignment_status": "CONFLICT",
                }
            ]
        },
        bos_pivot_trace_audit={
            "recent_candidates": [
                {
                    "symbol": "BTC-USD",
                    "timeframe": "4h",
                    "bos_state": "NO_BOS",
                    "pivot_state": "PIVOT_INVALIDATED",
                }
            ]
        },
        feed_scope_reconciliation=_feed_clean(),
    )
    candidate = result["candidates"][0]

    assert candidate["route_status"] == "LEGITIMATE_REVERSAL_RISK_BLOCK"
    assert candidate["alternative_bucket"] == "SHOULD_BE_MULTITF_CONFLICT"
    assert result["recommendation"] == "study_multitf_conflict"
    assert candidate["should_keep_blocked"] is True


def test_inactive_reversal_pattern_in_trend_setup_marks_mixed_routing(isolated_storage):
    module = load_module("core.reversal_blocker_routing_audit")

    result = module.build_reversal_blocker_routing_audit(
        signals=[
            {
                "asset": "SOL-USD",
                "strategy_name": "trend_pullback_breakout",
                "score": 0.735,
                "effective_min_signal_score": 0.74,
                "rejection_reasons": ["reversal_not_eligible"],
            }
        ],
        strategy_decision_bridge_trace={
            "recent_candidates": [
                {
                    "symbol": "SOL-USD",
                    "primary_real_blocker": "REVERSAL_NOT_ELIGIBLE",
                    "reversal_pattern_active": False,
                }
            ]
        },
        feed_scope_reconciliation=_feed_clean(),
    )
    candidate = result["candidates"][0]

    assert candidate["route_status"] == "REVERSAL_PATTERN_NOT_ACTIVE_BUT_BLOCKER_PRESENT"
    assert candidate["alternative_bucket"] == "MIXED_TREND_REVERSAL_ROUTING"
    assert candidate["suggested_future_study"] == "study_routing_map"


def test_breakout_not_confirmed_suggests_breakout_bucket(isolated_storage):
    module = load_module("core.reversal_blocker_routing_audit")

    result = module.build_reversal_blocker_routing_audit(
        signals=[
            {
                "asset": "ETH-USD",
                "strategy_name": "trend_pullback_breakout",
                "score": 0.7349,
                "effective_min_signal_score": 0.74,
                "rejection_reasons": ["reversal_not_eligible", "breakout_not_confirmed"],
            }
        ],
        strategy_decision_bridge_trace={
            "recent_candidates": [
                {
                    "symbol": "ETH-USD",
                    "primary_real_blocker": "REVERSAL_NOT_ELIGIBLE",
                    "secondary_real_blocker": "BREAKOUT_NOT_CONFIRMED",
                }
            ]
        },
        feed_scope_reconciliation=_feed_clean(),
    )
    candidate = result["candidates"][0]

    assert candidate["route_status"] == "REVERSAL_BLOCKER_ON_TREND_SETUP"
    assert candidate["alternative_bucket"] == "SHOULD_BE_BREAKOUT_NOT_CONFIRMED"
    assert candidate["suggested_future_study"] == "study_breakout_confirmation"


def test_score_far_below_min_suggests_score_primary_bucket(isolated_storage):
    module = load_module("core.reversal_blocker_routing_audit")

    result = module.build_reversal_blocker_routing_audit(
        signals=[
            {
                "asset": "LINK-USD",
                "strategy_name": "trend_pullback_breakout",
                "score": 0.68,
                "effective_min_signal_score": 0.74,
                "rejection_reasons": ["reversal_not_eligible", "score_below_minimum"],
            }
        ],
        feed_scope_reconciliation=_feed_clean(),
    )
    candidate = result["candidates"][0]

    assert candidate["alternative_bucket"] == "SHOULD_BE_SCORE_BELOW_MIN_PRIMARY"
    assert candidate["score_gap"] > 0.02
    assert candidate["should_keep_blocked"] is True


def test_clean_or_historical_fallback_never_becomes_current_feed_blocker(isolated_storage):
    module = load_module("core.reversal_blocker_routing_audit")

    result = module.build_reversal_blocker_routing_audit(
        signals=[
            {
                "asset": "BNB-USD",
                "strategy_name": "trend_pullback_breakout",
                "score": 0.73,
                "effective_min_signal_score": 0.74,
                "rejection_reasons": ["reversal_not_eligible"],
            }
        ],
        feed_scope_reconciliation=_feed_clean("ACCUMULATED"),
    )

    assert result["current_feed_is_clean"] is True
    assert result["fallback_blocker_scope"] == "ACCUMULATED"
    assert result["fallback_blocker_scope"] != "CURRENT_CYCLE"
    assert result["candidates"][0]["fallback_blocker_scope"] != "CURRENT_CYCLE"


def test_insufficient_data_returns_safe_default(isolated_storage):
    module = load_module("core.reversal_blocker_routing_audit")

    result = module.build_reversal_blocker_routing_audit(signals=[], feed_scope_reconciliation=_feed_clean())

    assert result["status"] == "INSUFFICIENT_DATA"
    assert result["top_route_status"] == "INSUFFICIENT_DATA_FOR_ROUTING"
    assert result["recommendation"] == "insufficient_data"
    assert result["should_keep_blocked"] is True
    assert result["safe_to_change_strategy_now"] is False


def test_diagnostic_does_not_mutate_signal_or_emit_operational_fields(isolated_storage):
    module = load_module("core.reversal_blocker_routing_audit")
    signal = {
        "asset": "BTC-USD",
        "strategy_name": "trend_pullback_breakout",
        "score": 0.73,
        "effective_min_signal_score": 0.74,
        "rejection_reasons": ["reversal_not_eligible"],
        "should_buy": False,
    }
    before = dict(signal)

    result = module.build_reversal_blocker_routing_audit(
        signals=[signal],
        feed_scope_reconciliation=_feed_clean(),
    )

    assert signal == before
    assert result["should_keep_blocked"] is True
    assert result["safe_to_change_strategy_now"] is False
    assert "position" not in result
    assert "pnl" not in result
    assert "history" not in result
