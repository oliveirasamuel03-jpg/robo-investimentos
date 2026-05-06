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
