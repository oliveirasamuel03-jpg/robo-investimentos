from __future__ import annotations

from copy import deepcopy

from conftest import load_module


SAFE_RECOMMENDATIONS = {
    "observe_more",
    "study_taxonomy_mapping",
    "study_real_rule_mapping",
    "study_bos_confirmation",
    "study_pullback_quality",
    "study_breakout_confirmation",
    "study_multitf_conflict",
    "keep_blocked_until_bos_confirms",
    "keep_blocked_until_pullback_reaction",
    "keep_blocked_until_real_setup_maps_structure",
    "no_threshold_change_recommended",
    "no_strategy_change_recommended",
    "insufficient_data",
}

FORBIDDEN_MESSAGE_FRAGMENTS = (
    "entrada aprovada",
    "pode comprar",
    "reduza score",
    "ignore blocker",
    "remova reversal",
    "opere agora",
)


def _module():
    return load_module("core.setup_blocker_taxonomy_audit")


def _feed_scope(scope: str = "NONE") -> dict:
    return {
        "enabled": True,
        "mode": "DIAGNOSTIC_ONLY",
        "current_feed_status": "LIVE",
        "current_fallback_count": 0,
        "current_live_count": 5,
        "current_feed_is_clean": True,
        "fallback_scope_status": scope,
        "fallback_blocker_scope": scope,
    }


def _build(**kwargs) -> dict:
    return _module().build_setup_blocker_taxonomy_audit(**kwargs)


def test_trend_no_setup_with_pivot_without_bos_is_taxonomized_btc_like():
    result = _build(
        signals=[
            {
                "asset": "BTC-USD",
                "strategy_name": "trend_pullback_breakout",
                "score": 0.6441,
                "effective_min_signal_score": 0.74,
                "rejection_reasons": ["no_setup_eligible"],
            }
        ],
        feed_scope_reconciliation=_feed_scope(),
        no_setup_eligible_decomposition={
            "candidates": [
                {
                    "symbol": "BTC-USD",
                    "setup": "trend_pullback_breakout",
                    "score": 0.6441,
                    "min_score": 0.74,
                    "primary_real_blocker": "NO_SETUP_ELIGIBLE",
                    "reason_bucket": "PIVOT_TRIGGERED_BUT_REAL_SETUP_MISSING",
                    "pivot_state": "PIVOT_TRIGGERED",
                    "bos_state": "NO_BOS",
                }
            ]
        },
    )

    assert result["mode"] == "DIAGNOSTIC_ONLY"
    assert result["safety_mode"] == "SHADOW_ONLY"
    assert result["top_symbol"] == "BTC-USD"
    assert result["taxonomy_status"] == "NO_SETUP_WITH_PIVOT_BUT_NO_BOS"
    assert result["normalized_primary_reason"] == "BOS_MISSING"
    assert result["should_keep_blocked"] is True
    assert result["safe_to_change_strategy_now"] is False
    assert result["safe_to_change_threshold_now"] is False


def test_reversal_not_eligible_on_trend_setup_stays_context_when_reversal_inactive():
    result = _build(
        reversal_blocker_routing_audit={
            "candidates": [
                {
                    "symbol": "BTC-USD",
                    "setup": "trend_pullback_breakout",
                    "primary_real_blocker": "REVERSAL_NOT_ELIGIBLE",
                    "route_status": "REVERSAL_PATTERN_NOT_ACTIVE_BUT_BLOCKER_PRESENT",
                    "alternative_bucket": "MIXED_TREND_REVERSAL_ROUTING",
                }
            ]
        },
        feed_scope_reconciliation=_feed_scope(),
    )

    assert result["taxonomy_status"] in {
        "REVERSAL_CONTEXT_ON_TREND_SETUP",
        "MIXED_TREND_REVERSAL_TAXONOMY",
    }
    assert result["normalized_primary_reason"] != "REVERSAL_NOT_ELIGIBLE"
    assert result["normalized_secondary_reason"] == "REVERSAL_RISK_CONTEXT"
    assert result["reversal_as_context_count"] >= 1
    assert result["should_keep_blocked"] is True


def test_score_gap_high_with_score_below_min_stays_primary_score_blocker():
    result = _build(
        signals=[
            {
                "asset": "SOL-USD",
                "strategy_name": "trend_pullback_breakout",
                "score": 0.62,
                "effective_min_signal_score": 0.74,
                "rejection_reasons": ["score_below_minimum"],
            }
        ],
        feed_scope_reconciliation=_feed_scope(),
    )

    assert result["taxonomy_status"] == "SCORE_BELOW_MIN_PRIMARY"
    assert result["normalized_primary_reason"] == "SCORE_BELOW_MIN"
    assert result["recommendation"] == "no_threshold_change_recommended"


def test_breakout_not_confirmed_is_normalized_as_breakout_reason_eth_like():
    result = _build(
        signals=[
            {
                "asset": "ETH-USD",
                "strategy_name": "trend_pullback_breakout",
                "score": 0.7349,
                "effective_min_signal_score": 0.74,
                "rejection_reasons": ["breakout_not_confirmed"],
            }
        ],
        feed_scope_reconciliation=_feed_scope(),
    )

    assert result["taxonomy_status"] == "NO_SETUP_WITH_BREAKOUT_NOT_CONFIRMED"
    assert result["normalized_primary_reason"] == "BREAKOUT_NOT_CONFIRMED"
    assert result["should_keep_blocked"] is True


def test_multitf_conflict_becomes_primary_context():
    result = _build(
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
        feed_scope_reconciliation=_feed_scope(),
    )

    assert result["taxonomy_status"] == "MULTITF_CONFLICT_PRIMARY"
    assert result["normalized_primary_reason"] == "MULTITF_CONFLICT"


def test_fib_structure_good_without_bos_is_not_operational_bnb_like():
    result = _build(
        bos_pivot_trace_audit={
            "recent_candidates": [
                {
                    "symbol": "BNB-USD",
                    "timeframe": "4h",
                    "bos_state": "NO_BOS",
                    "pivot_state": "PIVOT_TRIGGERED",
                }
            ]
        },
        market_structure_audit={
            "market_structure_best_candidates": [
                {
                    "symbol": "BNB-USD",
                    "current_fib_zone": "BREAKOUT_ZONE",
                    "market_structure_score": 0.81,
                }
            ]
        },
        feed_scope_reconciliation=_feed_scope(),
    )

    assert result["taxonomy_status"] == "FIB_STRUCTURE_NOT_OPERATIONAL"
    assert result["normalized_primary_reason"] == "BOS_MISSING"
    assert result["should_keep_blocked"] is True


def test_clean_feed_and_historical_fallback_never_become_current_primary_blocker():
    result = _build(
        signals=[
            {
                "asset": "LINK-USD",
                "strategy_name": "trend_pullback_breakout",
                "rejection_reasons": ["fallback_historical"],
            }
        ],
        feed_scope_reconciliation=_feed_scope("ACCUMULATED"),
    )

    assert result["current_feed_is_clean"] is True
    assert result["fallback_blocker_scope"] != "CURRENT_CYCLE"
    assert result["normalized_primary_reason"] == "FEED_NOT_CURRENT_BLOCKER"


def test_recommendations_and_messages_are_safe():
    result = _build(
        signals=[
            {
                "asset": "ETH-USD",
                "strategy_name": "trend_pullback_breakout",
                "score": 0.7349,
                "effective_min_signal_score": 0.74,
                "rejection_reasons": ["breakout_not_confirmed"],
            }
        ],
        feed_scope_reconciliation=_feed_scope(),
    )

    assert result["recommendation"] in SAFE_RECOMMENDATIONS
    for row in result["candidates"]:
        assert row["suggested_future_study"] in SAFE_RECOMMENDATIONS
        message = str(row["suggested_ui_message"]).lower()
        assert not any(fragment in message for fragment in FORBIDDEN_MESSAGE_FRAGMENTS)
        assert row["should_keep_blocked"] is True
        assert row["safe_to_change_strategy_now"] is False
        assert row["safe_to_change_threshold_now"] is False


def test_audit_does_not_mutate_source_signal_or_emit_operational_state():
    source_signal = {
        "asset": "ETH-USD",
        "strategy_name": "trend_pullback_breakout",
        "score": 0.7349,
        "effective_min_signal_score": 0.74,
        "rejection_reasons": ["breakout_not_confirmed"],
    }
    original = deepcopy(source_signal)

    result = _build(signals=[source_signal], feed_scope_reconciliation=_feed_scope())

    assert source_signal == original
    assert "open_positions" not in result
    assert "pnl" not in result
    assert "wallet" not in result
    assert result["should_keep_blocked"] is True


def test_insufficient_data_returns_safe_default():
    result = _build(feed_scope_reconciliation=_feed_scope())

    assert result["taxonomy_status"] == "INSUFFICIENT_DATA_FOR_TAXONOMY"
    assert result["recommendation"] == "insufficient_data"
    assert result["should_keep_blocked"] is True
    assert result["safe_to_change_strategy_now"] is False
    assert result["safe_to_change_threshold_now"] is False
    assert result["candidates"] == []
