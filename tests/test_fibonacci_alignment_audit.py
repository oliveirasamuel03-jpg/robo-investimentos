from __future__ import annotations

from tests.conftest import load_module


def _top_row(**overrides):
    base = {
        "symbol": "ETH-USD",
        "market_structure_minimum_sample_met": True,
        "relevant_swing_low": 100.0,
        "relevant_swing_high": 130.0,
        "current_fib_zone": "MEDIUM_ZONE",
        "pivot_detected": True,
        "bos_detected": True,
        "false_breakout_risk": False,
        "is_in_pullback_zone": True,
        "is_in_reaction_zone": True,
        "structure_confirms_trend_pullback": True,
        "structure_confirms_breakout": True,
        "structure_direction": "UP",
        "market_structure_why_no_candidate": "",
    }
    base.update(overrides)
    return base


def _market_structure_audit(row: dict):
    return {
        "market_structure_audit_enabled": True,
        "market_structure_top_symbol": row.get("symbol", "ETH-USD"),
        "market_structure_best_candidates": [row],
    }


def test_alignment_is_strong_when_all_key_rules_match():
    module = load_module("core.fibonacci_alignment_audit")

    result = module.build_fibonacci_alignment_audit(_market_structure_audit(_top_row()))

    assert result["fib_alignment_status"] == "strong_alignment"
    assert result["fib_alignment_score"] >= 0.80
    assert result["fib_alignment_recommendation"] == "video_pdf_alignment_strong"


def test_alignment_is_partial_when_zone_matches_but_pivot_and_bos_are_missing():
    module = load_module("core.fibonacci_alignment_audit")

    row = _top_row(pivot_detected=False, bos_detected=False, is_in_reaction_zone=False, structure_confirms_breakout=False)
    result = module.build_fibonacci_alignment_audit(_market_structure_audit(row))

    assert result["fib_alignment_status"] in {"partial_alignment", "weak_alignment"}
    assert result["fib_alignment_recommendation"] in {
        "fib_zone_matches_but_confirmation_missing",
        "pivot_bos_missing",
    }
    assert result["fib_alignment_pivot_status"] == "divergent"
    assert result["fib_alignment_bos_status"] == "divergent"


def test_alignment_diverges_when_anchors_are_invalid():
    module = load_module("core.fibonacci_alignment_audit")

    row = _top_row(relevant_swing_low=None, relevant_swing_high=None, current_fib_zone="INCONCLUSIVE")
    result = module.build_fibonacci_alignment_audit(_market_structure_audit(row))

    assert result["fib_alignment_recommendation"] == "anchors_need_review"
    assert result["fib_alignment_anchor_low_status"] in {"divergent", "insufficient"}
    assert result["fib_alignment_anchor_high_status"] in {"divergent", "insufficient"}


def test_alignment_handles_insufficient_data():
    module = load_module("core.fibonacci_alignment_audit")

    row = _top_row(market_structure_minimum_sample_met=False)
    result = module.build_fibonacci_alignment_audit(_market_structure_audit(row))

    assert result["fib_alignment_status"] == "insufficient_data"
    assert result["fib_alignment_recommendation"] == "insufficient_data"


def test_alignment_remains_shadow_only_without_trade_authority():
    module = load_module("core.fibonacci_alignment_audit")

    result = module.build_fibonacci_alignment_audit(_market_structure_audit(_top_row()))

    assert result["fib_alignment_mode"] == "SHADOW_ONLY"
    assert "trade_approved" not in result
    assert "broker" not in result
    assert "order" not in result
