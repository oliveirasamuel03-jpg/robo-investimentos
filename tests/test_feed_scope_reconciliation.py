from __future__ import annotations

from tests.conftest import load_module


def _live_status() -> dict:
    return {
        "feed_status": "LIVE",
        "provider_effective": "twelvedata",
        "symbols": ["BTC-USD", "ETH-USD", "BNB-USD"],
        "live_symbols": ["BTC-USD", "ETH-USD", "BNB-USD"],
        "fallback_symbols": [],
        "unknown_symbols": [],
        "source_breakdown": {"market": 3, "cached": 0, "fallback": 0, "unknown": 0},
    }


def test_live_feed_with_accumulated_fallback_is_not_current_scope(isolated_storage):
    module = load_module("core.feed_scope_reconciliation")

    result = module.build_feed_scope_reconciliation(
        market_data_status=_live_status(),
        validation_state={
            "fallback_rejection_current_cycle_count": 0,
            "fallback_rejection_accumulated_count": 344,
            "strategy_rejection_accumulated_count": 20,
            "current_cycle_rejection_reason": "score_below_minimum",
            "accumulated_rejection_reason": "fallback_blocked",
        },
    )

    assert result["current_feed_is_clean"] is True
    assert result["current_cycle_fallback_count"] == 0
    assert result["fallback_scope_status"] == "ACCUMULATED_ONLY_FALLBACK"
    assert result["fallback_blocker_scope"] == "ACCUMULATED"
    assert result["fallback_blocker_scope"] != "CURRENT_CYCLE"


def test_current_feed_fallback_generates_current_cycle_scope(isolated_storage):
    module = load_module("core.feed_scope_reconciliation")

    result = module.build_feed_scope_reconciliation(
        market_data_status={
            "feed_status": "FALLBACK",
            "provider_effective": "twelvedata",
            "symbols": ["BTC-USD"],
            "live_symbols": [],
            "fallback_symbols": ["BTC-USD"],
            "source_breakdown": {"market": 0, "cached": 0, "fallback": 1, "unknown": 0},
        },
        validation_state={"fallback_rejection_current_cycle_count": 1},
    )

    assert result["current_feed_is_clean"] is False
    assert result["fallback_scope_status"] == "CURRENT_CYCLE_FALLBACK"
    assert result["fallback_blocker_scope"] == "CURRENT_CYCLE"


def test_visual_only_fallback_stays_visual_scope(isolated_storage):
    module = load_module("core.feed_scope_reconciliation")

    result = module.build_feed_scope_reconciliation(
        market_data_status=_live_status(),
        visual_chart_status={
            "feed_status": "FALLBACK",
            "symbols": ["BTC-USD"],
            "fallback_symbols": ["BTC-USD"],
            "source_breakdown": {"market": 0, "cached": 0, "fallback": 1, "unknown": 0},
        },
    )

    assert result["fallback_scope_status"] == "VISUAL_ONLY_FALLBACK"
    assert result["fallback_blocker_scope"] == "VISUAL_CHART_ONLY"


def test_old_candidate_fallback_is_candidate_old_scope(isolated_storage):
    module = load_module("core.feed_scope_reconciliation")

    result = module.build_feed_scope_reconciliation(
        market_data_status=_live_status(),
        shadow_decision_simulator={
            "shadow_accumulated_recent_candidates": [
                {
                    "symbol": "ETH-USD",
                    "feed_status": "FALLBACK",
                    "count_scope": "accumulated_recent",
                    "already_seen": True,
                }
            ]
        },
    )

    assert result["fallback_scope_status"] == "CANDIDATE_LEVEL_OLD_FALLBACK"
    assert result["fallback_blocker_scope"] == "CANDIDATE_OLD"


def test_shadow_simulator_keeps_current_dominant_clean_when_fallback_is_accumulated(isolated_storage):
    module = load_module("core.shadow_decision_simulator")
    result = module.build_shadow_decision_simulator(
        signals=[
            {
                "asset": "BNB-USD",
                "signal_timestamp": "2026-05-12T00:00:00+00:00",
                "score": 0.72,
                "effective_min_signal_score": 0.74,
                "buy": False,
                "data_source": "market",
                "provider_effective": "twelvedata",
                "context_status": "FAVORAVEL",
                "rejection_reasons": ["score_below_minimum"],
            }
        ],
        state={
            "broker": {"mode": "paper"},
            "validation": {"trading_mode": "paper", "fallback_rejection_accumulated_count": 9},
            "shadow_decision_simulator": {
                "shadow_accumulated_recent_candidates": [
                    {
                        "symbol": "ETH-USD",
                        "shadow_candidate_key": "ETH-USD|old|0.700000",
                        "candidate_class": "UNSAFE_REJECTION",
                        "why_not_safe": "blocked_by_fallback",
                        "feed_status": "FALLBACK",
                    }
                ],
                "shadow_accumulated_raw_received_count": 1,
            },
            "positions": [],
            "trader": {"max_open_positions": 1},
        },
        market_data={},
        market_data_status=_live_status(),
    )

    assert result["fallback_blocker_scope"] == "ACCUMULATED"
    assert result["fallback_current_count"] == 0
    assert result["dominant_exclusion_current_scope"] != "blocked_by_fallback"


def test_strategy_bridge_recommends_accumulated_only_for_clean_current_feed(isolated_storage):
    module = load_module("core.strategy_decision_bridge_trace")
    feed_scope = {
        "fallback_scope_status": "ACCUMULATED_ONLY_FALLBACK",
        "fallback_blocker_scope": "ACCUMULATED",
        "current_feed_is_clean": True,
        "notes": "Fallback is accumulated/historical and does not represent the current clean worker feed.",
    }

    result = module.build_strategy_decision_bridge_trace(
        signals=[
            {
                "asset": "BNB-USD",
                "score": 0.72,
                "effective_min_signal_score": 0.74,
                "buy": False,
                "data_source": "market",
                "rejection_reasons": ["score_below_minimum"],
            }
        ],
        shadow_decision_simulator={},
        multi_timeframe_swing_audit={},
        bos_pivot_trace_audit={},
        market_structure_audit={},
        fib_alignment_audit={},
        market_data_status=_live_status(),
        validation_state={"fallback_rejection_accumulated_count": 12},
        paper_state={"positions": []},
        feed_scope_reconciliation=feed_scope,
    )

    candidate = result["recent_candidates"][0]
    assert result["recommendation"] == "accumulated_fallback_only"
    assert candidate["fallback_blocker_scope"] == "ACCUMULATED"
    assert candidate["fallback_is_current_cycle"] is False
    assert candidate["should_keep_blocked"] is True
    assert result["recommendation"] != "reconcile_feed_scope"
