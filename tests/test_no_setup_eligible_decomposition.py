from __future__ import annotations

from tests.conftest import load_module


SAFE_RECOMMENDATIONS = {
    "observe_more",
    "study_real_rule_mapping",
    "study_pullback_quality",
    "study_breakout_confirmation",
    "study_trend_pullback_selectivity",
    "candidate_for_future_shadow_calibration",
    "keep_blocked_until_pullback_reaction",
    "keep_blocked_until_real_setup_maps_structure",
    "no_threshold_change_recommended",
    "insufficient_data",
}


def _feed_clean() -> dict:
    return {
        "fallback_scope_status": "NO_CURRENT_FALLBACK",
        "fallback_blocker_scope": "NONE",
        "current_feed_is_clean": True,
    }


def _bnb_like_payload() -> dict:
    return {
        "signals": [
            {
                "asset": "BNB-USD",
                "strategy_name": "trend_pullback_breakout",
                "score": 0.76,
                "effective_min_signal_score": 0.74,
                "rejection_reasons": ["no_setup_eligible"],
            }
        ],
        "strategy_decision_bridge_trace": {
            "recent_candidates": [
                {
                    "symbol": "BNB-USD",
                    "real_strategy": "trend_pullback_breakout",
                    "real_score": 0.76,
                    "min_score": 0.74,
                    "primary_real_blocker": "NO_SETUP_ELIGIBLE",
                    "secondary_real_blocker": "",
                    "real_rejection_reason": "no_setup_eligible",
                    "multi_tf_alignment_status": "STRONG_ALIGNMENT",
                    "bos_state_4h": "BOS_RETEST_CONFIRMED",
                    "pivot_state_4h": "PIVOT_TRIGGERED",
                    "should_keep_blocked": True,
                }
            ]
        },
        "multi_timeframe_swing_audit": {
            "recent_candidates": [
                {
                    "symbol": "BNB-USD",
                    "daily_bias": "UP",
                    "h4_structure": "UP",
                    "h1_confirmation": "UP",
                    "alignment_status": "STRONG_ALIGNMENT",
                }
            ]
        },
        "bos_pivot_trace_audit": {
            "recent_candidates": [
                {
                    "symbol": "BNB-USD",
                    "timeframe": "4h",
                    "bos_state": "BOS_RETEST_CONFIRMED",
                    "pivot_state": "PIVOT_TRIGGERED",
                }
            ]
        },
        "market_structure_audit": {
            "market_structure_best_candidates": [
                {
                    "symbol": "BNB-USD",
                    "market_structure_score": 0.82,
                    "current_fib_zone": "BREAKOUT_ZONE",
                }
            ]
        },
        "feed_scope_reconciliation": _feed_clean(),
    }


def test_bnb_like_structure_confirmed_no_setup_is_diagnostic_only(isolated_storage):
    module = load_module("core.no_setup_eligible_decomposition")

    result = module.build_no_setup_eligible_decomposition(**_bnb_like_payload())
    candidate = result["candidates"][0]

    assert result["mode"] == "DIAGNOSTIC_ONLY"
    assert result["safety_mode"] == "SHADOW_ONLY"
    assert result["top_symbol"] == "BNB-USD"
    assert result["top_reason_bucket"] == "STRUCTURE_CONFIRMED_BUT_SETUP_NOT_ELIGIBLE"
    assert result["structure_confirmed_but_no_setup_count"] == 1
    assert result["should_keep_blocked"] is True
    assert candidate["should_keep_blocked"] is True
    assert candidate["safe_to_change_threshold_now"] is False
    assert result["recommendation"] in SAFE_RECOMMENDATIONS


def test_eth_like_near_min_breakout_missing_is_traced(isolated_storage):
    module = load_module("core.no_setup_eligible_decomposition")

    result = module.build_no_setup_eligible_decomposition(
        signals=[
            {
                "asset": "ETH-USD",
                "strategy_name": "trend_pullback_breakout",
                "score": 0.7349,
                "effective_min_signal_score": 0.74,
                "rejection_reasons": ["score_below_minimum", "breakout_not_confirmed"],
            }
        ],
        strategy_decision_bridge_trace={
            "recent_candidates": [
                {
                    "symbol": "ETH-USD",
                    "real_score": 0.7349,
                    "min_score": 0.74,
                    "primary_real_blocker": "SCORE_BELOW_MIN",
                    "secondary_real_blocker": "BREAKOUT_NOT_CONFIRMED",
                    "real_rejection_reason": "score_below_minimum, breakout_not_confirmed",
                }
            ]
        },
        feed_scope_reconciliation=_feed_clean(),
    )

    candidate = result["candidates"][0]
    assert candidate["symbol"] == "ETH-USD"
    assert candidate["reason_bucket"] == "SCORE_NEAR_MIN_BUT_BREAKOUT_MISSING"
    assert candidate["score_gap"] <= 0.01
    assert candidate["suggested_future_study"] == "study_breakout_confirmation"
    assert candidate["should_keep_blocked"] is True


def test_feed_clean_never_becomes_current_fallback_bucket(isolated_storage):
    module = load_module("core.no_setup_eligible_decomposition")

    result = module.build_no_setup_eligible_decomposition(**_bnb_like_payload())
    candidate = result["candidates"][0]

    assert result["current_feed_is_clean"] is True
    assert result["fallback_blocker_scope"] != "CURRENT_CYCLE"
    assert candidate["feed_is_not_current_blocker"] is True
    assert "FALLBACK" not in candidate["reason_bucket"]


def test_historical_fallback_does_not_override_current_feed_clean(isolated_storage):
    module = load_module("core.no_setup_eligible_decomposition")
    payload = _bnb_like_payload()
    payload["feed_scope_reconciliation"] = {
        "fallback_scope_status": "ACCUMULATED_ONLY_FALLBACK",
        "fallback_blocker_scope": "ACCUMULATED",
        "current_feed_is_clean": True,
    }

    result = module.build_no_setup_eligible_decomposition(**payload)

    assert result["fallback_blocker_scope"] == "ACCUMULATED"
    assert result["current_feed_is_clean"] is True
    assert result["top_reason_bucket"] == "STRUCTURE_CONFIRMED_BUT_SETUP_NOT_ELIGIBLE"


def test_insufficient_data_returns_safe_default(isolated_storage):
    module = load_module("core.no_setup_eligible_decomposition")

    result = module.build_no_setup_eligible_decomposition(signals=[], feed_scope_reconciliation=_feed_clean())

    assert result["status"] == "INSUFFICIENT_DATA"
    assert result["top_reason_bucket"] == "INSUFFICIENT_DATA_FOR_DECOMPOSITION"
    assert result["should_keep_blocked"] is True
    assert result["recommendation"] == "insufficient_data"


def test_diagnostic_does_not_mutate_operational_fields(isolated_storage):
    module = load_module("core.no_setup_eligible_decomposition")
    signal = {
        "asset": "BNB-USD",
        "score": 0.76,
        "effective_min_signal_score": 0.74,
        "rejection_reasons": ["no_setup_eligible"],
        "buy": False,
    }

    before = dict(signal)
    result = module.build_no_setup_eligible_decomposition(
        signals=[signal],
        feed_scope_reconciliation=_feed_clean(),
    )

    assert signal == before
    assert result["should_keep_blocked"] is True
    assert result["candidates"][0]["safe_to_change_threshold_now"] is False
    assert "position" not in result
    assert "pnl" not in result
    assert "history" not in result
