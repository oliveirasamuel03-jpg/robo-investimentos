from __future__ import annotations

import pandas as pd

from tests.conftest import load_module


def _state(**overrides):
    state = {
        "broker": {"provider": "paper"},
        "validation": {"trading_mode": "paper"},
        "risk": {"daily_loss_block_active": False},
        "positions": {},
        "trader": {"max_open_positions": 2},
    }
    state.update(overrides)
    return state


def _signal(**overrides):
    payload = {
        "asset": "BNB-USD",
        "strategy_name": "trend_pullback_breakout",
        "signal_key": "BNB-USD|2026-05-05T10:00:00+00:00|market",
        "signal_timestamp": "2026-05-05T10:00:00+00:00",
        "price": 100.0,
        "score": 0.735,
        "effective_min_signal_score": 0.74,
        "base_min_signal_score": 0.74,
        "buy": False,
        "data_source": "market",
        "provider_effective": "twelvedata",
        "context_status": "NEUTRO",
        "macro_alert_active": False,
        "macro_alert_level": "LOW",
        "rejection_reasons": ["breakout_not_confirmed"],
    }
    payload.update(overrides)
    return payload


def _market_data(price: float = 100.0):
    return {"BNB-USD": pd.DataFrame({"close": [price], "data_source": ["market"]})}


def _structure(**overrides):
    row = {
        "symbol": "BNB-USD",
        "market_structure_score": 0.72,
        "current_fib_zone": "MEDIUM_ZONE",
        "pivot_detected": True,
        "bos_detected": True,
    }
    row.update(overrides)
    return {"market_structure_best_candidates": [row]}


def _fib(**overrides):
    payload = {
        "fib_alignment_top_symbol": "BNB-USD",
        "fib_alignment_score": 0.82,
        "fib_alignment_status": "strong_alignment",
    }
    payload.update(overrides)
    return payload


def test_small_secondary_rejection_can_be_shadow_would_enter_without_real_trade_fields():
    module = load_module("core.shadow_decision_simulator")

    result = module.build_shadow_decision_simulator(
        signals=[_signal()],
        state=_state(),
        market_data=_market_data(),
        market_structure_audit=_structure(),
        fib_alignment_audit=_fib(),
    )

    candidate = result["shadow_recent_candidates"][0]
    assert result["shadow_would_enter_count"] == 1
    assert result["preview_near_approved_count"] == 1
    assert result["shadow_raw_near_approved_count"] == 1
    assert candidate["shadow_would_enter"] is True
    assert candidate["raw_near_approved"] is True
    assert candidate["safe_candidate"] is True
    assert candidate["outcome_label"] == "STILL_PENDING"
    assert "trade_approved" not in result
    assert "positions" not in result
    assert "wallet_value" not in result


def test_no_setup_eligible_blocks_shadow_entry():
    module = load_module("core.shadow_decision_simulator")

    result = module.build_shadow_decision_simulator(
        signals=[_signal(rejection_reasons=["no_setup_eligible"], score=0.735)],
        state=_state(),
        market_data=_market_data(),
        market_structure_audit=_structure(),
        fib_alignment_audit=_fib(),
    )

    candidate = result["shadow_recent_candidates"][0]
    assert candidate["candidate_class"] == "STRUCTURE_MISSING"
    assert candidate["raw_near_approved"] is True
    assert candidate["why_not_safe"] == "blocked_by_no_setup_eligible"
    assert candidate["shadow_would_enter"] is False
    assert result["preview_near_approved_count"] == 1
    assert result["shadow_raw_near_approved_count"] == 1
    assert result["shadow_structure_missing_count"] == 1


def test_trend_not_confirmed_blocks_shadow_entry():
    module = load_module("core.shadow_decision_simulator")

    result = module.build_shadow_decision_simulator(
        signals=[_signal(rejection_reasons=["trend_not_confirmed"], score=0.735)],
        state=_state(),
        market_data=_market_data(),
        market_structure_audit=_structure(),
        fib_alignment_audit=_fib(),
    )

    candidate = result["shadow_recent_candidates"][0]
    assert candidate["shadow_would_enter"] is False
    assert candidate["why_not_safe"] == "blocked_by_trend_not_confirmed"


def test_fallback_blocks_shadow_entry():
    module = load_module("core.shadow_decision_simulator")

    result = module.build_shadow_decision_simulator(
        signals=[_signal(data_source="fallback", rejection_reasons=["fallback_blocked"])],
        state=_state(),
        market_data=_market_data(),
        market_structure_audit=_structure(),
        fib_alignment_audit=_fib(),
    )

    assert result["shadow_recent_candidates"][0]["candidate_class"] == "UNSAFE_REJECTION"
    assert result["shadow_recent_candidates"][0]["shadow_would_enter"] is False


def test_critical_context_blocks_shadow_entry():
    module = load_module("core.shadow_decision_simulator")

    result = module.build_shadow_decision_simulator(
        signals=[_signal(context_status="CRITICO", rejection_reasons=["context_blocked"])],
        state=_state(),
        market_data=_market_data(),
        market_structure_audit=_structure(),
        fib_alignment_audit=_fib(),
    )

    candidate = result["shadow_recent_candidates"][0]
    assert candidate["shadow_would_enter"] is False
    assert candidate["why_not_safe"] == "blocked_by_context"


def test_partial_alignment_without_pivot_or_bos_does_not_release_alone():
    module = load_module("core.shadow_decision_simulator")

    result = module.build_shadow_decision_simulator(
        signals=[_signal()],
        state=_state(),
        market_data=_market_data(),
        market_structure_audit=_structure(pivot_detected=False, bos_detected=False),
        fib_alignment_audit=_fib(fib_alignment_score=0.65, fib_alignment_status="partial_alignment"),
    )

    candidate = result["shadow_recent_candidates"][0]
    assert candidate["candidate_class"] == "CONFIRMATION_MISSING"
    assert candidate["why_not_safe"] == "blocked_by_missing_pivot_bos"
    assert candidate["shadow_would_enter"] is False


def test_shadow_outcome_is_updated_separately_from_official_state():
    module = load_module("core.shadow_decision_simulator")
    first = module.build_shadow_decision_simulator(
        signals=[_signal()],
        state=_state(),
        market_data=_market_data(100.0),
        market_structure_audit=_structure(),
        fib_alignment_audit=_fib(),
    )
    next_state = _state(shadow_decision_simulator=first, wallet_value=1000.0, history=[])

    second = module.build_shadow_decision_simulator(
        signals=[],
        state=next_state,
        market_data=_market_data(105.0),
        market_structure_audit=_structure(),
        fib_alignment_audit=_fib(),
    )

    assert second["shadow_would_win_count"] == 1
    assert next_state["wallet_value"] == 1000.0
    assert next_state["history"] == []


def test_reversal_not_eligible_is_traceable_and_blocks_shadow_entry():
    module = load_module("core.shadow_decision_simulator")

    result = module.build_shadow_decision_simulator(
        signals=[_signal(rejection_reasons=["reversal_not_eligible"], score=0.735)],
        state=_state(),
        market_data=_market_data(),
        market_structure_audit=_structure(),
        fib_alignment_audit=_fib(),
    )

    candidate = result["shadow_recent_candidates"][0]
    assert candidate["raw_near_approved"] is True
    assert candidate["shadow_would_enter"] is False
    assert candidate["why_not_safe"] == "blocked_by_reversal_not_eligible"


def test_small_secondary_gap_can_be_marginal_without_becoming_safe():
    module = load_module("core.shadow_decision_simulator")

    result = module.build_shadow_decision_simulator(
        signals=[_signal(score=0.715, rejection_reasons=["confidence_too_low"])],
        state=_state(),
        market_data=_market_data(),
        market_structure_audit=_structure(),
        fib_alignment_audit=_fib(),
    )

    candidate = result["shadow_recent_candidates"][0]
    assert candidate["raw_near_approved"] is True
    assert candidate["candidate_class"] == "MARGINAL_NEAR_APPROVED"
    assert candidate["safe_candidate"] is False
    assert candidate["shadow_would_enter"] is False


def test_unsafe_rejection_counts_as_analyzed_and_classified():
    module = load_module("core.shadow_decision_simulator")

    result = module.build_shadow_decision_simulator(
        signals=[_signal(data_source="fallback", rejection_reasons=["fallback_blocked"])],
        state=_state(),
        market_data=_market_data(),
        market_structure_audit=_structure(),
        fib_alignment_audit=_fib(),
    )

    candidate = result["shadow_recent_candidates"][0]
    assert candidate["candidate_class"] == "UNSAFE_REJECTION"
    assert candidate["analyzed_by_shadow"] is True
    assert candidate["classified_by_shadow"] is True
    assert result["shadow_current_cycle_analyzed_count"] == 1
    assert result["shadow_current_cycle_classified_count"] == 1
    assert result["shadow_current_cycle_unsafe_count"] == 1
    assert result["shadow_candidates_analyzed_count"] == 1
    assert result["shadow_counter_warning"] is False


def test_duplicate_candidates_are_ignored_without_zeroing_accumulated_analysis():
    module = load_module("core.shadow_decision_simulator")
    signal = _signal(data_source="fallback", rejection_reasons=["fallback_blocked"])

    first = module.build_shadow_decision_simulator(
        signals=[signal],
        state=_state(),
        market_data=_market_data(),
        market_structure_audit=_structure(),
        fib_alignment_audit=_fib(),
    )
    second = module.build_shadow_decision_simulator(
        signals=[signal],
        state=_state(shadow_decision_simulator=first),
        market_data=_market_data(),
        market_structure_audit=_structure(),
        fib_alignment_audit=_fib(),
    )

    candidate = second["shadow_recent_candidates"][0]
    assert second["shadow_current_cycle_received_count"] == 1
    assert second["shadow_candidates_unique_count"] == 0
    assert second["shadow_current_cycle_new_unique_count"] == 0
    assert second["shadow_current_cycle_duplicate_count"] == 1
    assert second["shadow_current_cycle_already_analyzed_count"] == 1
    assert second["shadow_current_cycle_ignored_count"] == 1
    assert second["shadow_current_cycle_analyzed_count"] == 0
    assert second["shadow_current_cycle_analyzed_new_count"] == 0
    assert second["shadow_accumulated_analyzed_count"] == 1
    assert second["shadow_accumulated_analyzed_unique_count"] == 1
    assert second["shadow_candidates_analyzed_count"] == 1
    assert second["shadow_duplicate_ratio"] == 1.0
    assert second["shadow_current_cycle_candidates"] == []
    assert candidate["duplicate_candidate"] is True
    assert candidate["count_scope"] == "accumulated_recent"
    assert candidate["already_seen"] is True
    assert candidate["analyzed_this_cycle"] is False
    assert candidate["analyzed_previously"] is True
    assert candidate["shadow_trace_status"] == "duplicate_existing_shadow_candidate"


def test_counter_subsets_stay_inside_classified_current_cycle():
    module = load_module("core.shadow_decision_simulator")

    result = module.build_shadow_decision_simulator(
        signals=[
            _signal(signal_key="BNB-USD|safe", signal_timestamp="2026-05-05T10:00:00+00:00"),
            _signal(
                signal_key="BNB-USD|marginal",
                signal_timestamp="2026-05-05T10:01:00+00:00",
                score=0.715,
                rejection_reasons=["confidence_too_low"],
            ),
            _signal(
                signal_key="BNB-USD|unsafe",
                signal_timestamp="2026-05-05T10:02:00+00:00",
                data_source="fallback",
                rejection_reasons=["fallback_blocked"],
            ),
        ],
        state=_state(),
        market_data=_market_data(),
        market_structure_audit=_structure(),
        fib_alignment_audit=_fib(),
    )

    classified = result["shadow_current_cycle_classified_count"]
    subset_total = (
        result["shadow_current_cycle_safe_near_approved_count"]
        + result["shadow_current_cycle_marginal_near_approved_count"]
        + result["shadow_current_cycle_unsafe_count"]
    )
    assert result["shadow_current_cycle_received_count"] >= result["shadow_candidates_unique_count"]
    assert result["shadow_current_cycle_received_count"] == (
        result["shadow_current_cycle_new_unique_count"] + result["shadow_current_cycle_duplicate_count"]
    )
    assert result["shadow_current_cycle_ignored_count"] <= result["shadow_current_cycle_received_count"]
    assert result["shadow_current_cycle_analyzed_count"] == classified
    assert result["shadow_current_cycle_analyzed_new_count"] == classified
    assert subset_total <= classified
    assert result["shadow_counter_warning"] is False
    assert result["shadow_scope_warning"] is False


def test_all_duplicate_cycle_has_zero_new_analysis_with_accumulated_recent_table():
    module = load_module("core.shadow_decision_simulator")
    signals = [
        _signal(
            signal_key=f"BNB-USD|duplicate-{idx}",
            signal_timestamp=f"2026-05-05T10:0{idx}:00+00:00",
            data_source="fallback",
            rejection_reasons=["fallback_blocked"],
        )
        for idx in range(5)
    ]

    first = module.build_shadow_decision_simulator(
        signals=signals,
        state=_state(),
        market_data=_market_data(),
        market_structure_audit=_structure(),
        fib_alignment_audit=_fib(),
    )
    second = module.build_shadow_decision_simulator(
        signals=signals,
        state=_state(shadow_decision_simulator=first),
        market_data=_market_data(),
        market_structure_audit=_structure(),
        fib_alignment_audit=_fib(),
    )

    assert second["shadow_current_cycle_received_count"] == 5
    assert second["shadow_current_cycle_new_unique_count"] == 0
    assert second["shadow_current_cycle_duplicate_count"] == 5
    assert second["shadow_current_cycle_analyzed_new_count"] == 0
    assert second["shadow_current_cycle_classified_new_count"] == 0
    assert second["shadow_current_cycle_unsafe_new_count"] == 0
    assert second["shadow_current_cycle_already_analyzed_count"] == 5
    assert second["shadow_accumulated_unique_candidates_count"] == 5
    assert second["shadow_accumulated_analyzed_unique_count"] == 5
    assert second["shadow_accumulated_unsafe_unique_count"] == 5
    assert second["shadow_accumulated_raw_received_count"] == 10
    assert second["shadow_duplicate_ratio"] == 1.0
    assert second["shadow_raw_to_unique_ratio"] == 2.0
    assert second["shadow_counter_health_status"] == "all_duplicates_current_cycle"
    assert second["shadow_current_cycle_candidates"] == []
    assert len(second["shadow_accumulated_recent_candidates"]) == 5
    assert all(row["count_scope"] == "accumulated_recent" for row in second["shadow_accumulated_recent_candidates"])
    assert all(row["duplicate_candidate"] is True for row in second["shadow_accumulated_recent_candidates"])
    assert "positions" not in second
    assert "wallet_value" not in second
