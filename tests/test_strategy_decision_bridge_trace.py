from __future__ import annotations

from tests.conftest import load_module


def _signal(*, reason: str = "score_below_minimum", score: float = 0.72, min_score: float = 0.74) -> dict:
    return {
        "asset": "BNB-USD",
        "strategy_name": "trend_pullback_breakout",
        "score": score,
        "effective_min_signal_score": min_score,
        "buy": False,
        "data_source": "market",
        "context_status": "FAVORAVEL",
        "rejection_reasons": [reason],
    }


def _bos_trace() -> dict:
    return {
        "enabled": True,
        "mode": "SHADOW_ONLY",
        "top_symbol": "BNB-USD",
        "recent_candidates": [
            {
                "symbol": "BNB-USD",
                "timeframe": "4h",
                "bos_state": "BOS_RETEST_CONFIRMED",
                "pivot_state": "PIVOT_TRIGGERED",
                "relationship_to_higher_tf": "BOTH_CONFIRMED",
                "should_keep_blocked": True,
            },
            {
                "symbol": "BNB-USD",
                "timeframe": "1h",
                "bos_state": "BOS_BY_CLOSE_CONFIRMED",
                "pivot_state": "PIVOT_CONFIRMED",
                "relationship_to_higher_tf": "BOTH_CONFIRMED",
                "should_keep_blocked": True,
            },
        ],
    }


def _mtf(status: str = "PARTIAL_ALIGNMENT", missing: list[str] | None = None) -> dict:
    return {
        "top_symbol": "BNB-USD",
        "top_alignment_status": status,
        "recent_candidates": [
            {
                "symbol": "BNB-USD",
                "alignment_status": status,
                "alignment_score": 0.62,
                "missing_for_setup": missing or [],
            }
        ],
    }


def _build(**overrides):
    module = load_module("core.strategy_decision_bridge_trace")
    payload = {
        "signals": [_signal()],
        "shadow_decision_simulator": {"shadow_recent_candidates": []},
        "multi_timeframe_swing_audit": _mtf(),
        "bos_pivot_trace_audit": _bos_trace(),
        "market_structure_audit": {
            "market_structure_best_candidates": [
                {
                    "symbol": "BNB-USD",
                    "market_structure_score": 0.72,
                    "structure_confirms_trend_pullback": True,
                    "current_fib_zone": "MEDIUM_ZONE",
                }
            ]
        },
        "market_data_status": {
            "feed_status": "LIVE",
            "provider_effective": "twelvedata",
            "symbols": ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "LINK-USD"],
            "live_symbols": ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "LINK-USD"],
            "fallback_symbols": [],
            "source_breakdown": {"market": 5, "cached": 0, "fallback": 0, "unknown": 0},
        },
        "validation_state": {},
        "paper_state": {"positions": {}, "market_context": {"market_context_status": "FAVORAVEL"}},
        "daily_loss_block_active": False,
        "slots_left": 1,
    }
    payload.update(overrides)
    return module.build_strategy_decision_bridge_trace(**payload)


def test_confirmed_structure_with_score_below_is_real_blocked_shadow_only():
    result = _build()
    candidate = result["recent_candidates"][0]

    assert result["mode"] == "SHADOW_ONLY"
    assert candidate["decision_bridge_status"] == "STRUCTURE_CONFIRMED_BUT_REAL_BLOCKED"
    assert candidate["primary_real_blocker"] == "SCORE_BELOW_MIN"
    assert candidate["should_keep_blocked"] is True
    assert candidate["shadow_would_enter"] is False


def test_confirmed_structure_with_no_setup_reconciles_real_setup_missing():
    result = _build(signals=[_signal(reason="no_setup_eligible")])
    candidate = result["recent_candidates"][0]

    assert candidate["reconciliation_status"] == "BOS_CONFIRMED_BUT_REAL_SETUP_MISSING"
    assert candidate["primary_real_blocker"] == "NO_SETUP_ELIGIBLE"
    assert candidate["should_keep_blocked"] is True


def test_fallback_current_false_but_accumulated_true_marks_scope_mismatch():
    result = _build(validation_state={"fallback_rejection_accumulated_count": 4})
    candidate = result["recent_candidates"][0]

    assert candidate["fallback_current"] is False
    assert candidate["fallback_accumulated"] is True
    assert candidate["fallback_blocker_scope"] == "ACCUMULATED"
    assert candidate["reconciliation_status"] == "FALLBACK_SCOPE_MISMATCH"


def test_multi_tf_insufficient_with_bos_confirmed_is_reconciled():
    result = _build(multi_timeframe_swing_audit=_mtf("INSUFFICIENT_DATA", ["h4_bos_missing"]))
    candidate = result["recent_candidates"][0]

    assert candidate["multi_tf_alignment_status"] == "INSUFFICIENT_DATA"
    assert candidate["bos_state_4h"] == "BOS_RETEST_CONFIRMED"
    assert candidate["reconciliation_status"] == "STRUCTURE_CONFIRMED_BUT_MULTI_TF_INSUFFICIENT"


def test_bridge_never_changes_official_trading_fields():
    result = _build()

    assert result["shadow_only"] is True
    assert "positions" not in result
    assert "wallet" not in result
    assert "realized_pnl" not in result
    assert "min_signal_score" not in result
    assert all(row["should_keep_blocked"] is True for row in result["recent_candidates"])
