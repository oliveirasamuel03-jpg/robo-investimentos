from __future__ import annotations

from copy import deepcopy

from tests.conftest import load_module


def _base_signal(**overrides):
    signal = {
        "asset": "BTC-USD",
        "score": 0.731,
        "raw_score": 0.736,
        "base_min_signal_score": 0.74,
        "effective_min_signal_score": 0.74,
        "buy": False,
        "data_source": "market",
        "provider_effective": "twelvedata",
        "context_status": "FAVORAVEL",
        "macro_alert_active": False,
        "trend_ok": True,
        "score_ok": False,
        "rsi_ok": True,
        "atr_ok": True,
        "pullback_ok": True,
        "breakout_ok": False,
        "momentum_short_ok": False,
        "momentum_medium_ok": False,
        "structure_ok": True,
        "momentum_ok": False,
        "trend_alignment": "aligned",
        "rsi": 48.0,
        "momentum": -0.002,
        "momentum_8": -0.004,
        "breakout_20": -0.002,
        "rejection_reasons": ["score_below_minimum", "breakout_not_confirmed", "confidence_too_low"],
    }
    signal.update(overrides)
    return signal


def test_structural_audit_is_shadow_only_and_does_not_mutate_signal():
    audit_module = load_module("core.strategy_structure_audit")
    signal = _base_signal()
    original = deepcopy(signal)

    audit = audit_module.build_strategy_structure_audit([signal])

    assert signal == original
    assert audit["structural_audit_mode"] == "SHADOW_ONLY"
    assert audit["structural_audit_candidates"] >= 1
    assert audit["structural_audit_top_score"] == signal["score"]
    assert "trade_approved" not in audit
    assert "open_position" not in audit
    assert "order" not in audit


def test_structural_audit_does_not_create_shadow_candidates_in_fallback():
    audit_module = load_module("core.strategy_structure_audit")
    signal = _base_signal(data_source="fallback", rejection_reasons=["fallback_blocked", "score_below_minimum"])

    audit = audit_module.build_strategy_structure_audit([signal])

    assert audit["structural_audit_candidates"] == 0
    assert all(row["shadow_candidates"] == 0 for row in audit["structural_audit_setup_comparison"])
    assert any("feed_not_live" in candidate["risk_blockers"] for candidate in audit["structural_audit_recent_candidates"])


def test_structural_audit_setup_comparison_has_expected_structure():
    audit_module = load_module("core.strategy_structure_audit")

    audit = audit_module.build_strategy_structure_audit([_base_signal(asset="ETH-USD"), _base_signal(asset="SOL-USD", score=0.69)])
    comparison = audit["structural_audit_setup_comparison"]

    assert {row["setup"] for row in comparison} == {
        "trend_pullback_breakout",
        "breakout_confirmed",
        "reversal_v_pattern",
    }
    assert set(audit["structural_audit_total_candidates_by_setup"]) == {
        "trend_pullback_breakout",
        "breakout_confirmed",
        "reversal_v_pattern",
    }
    assert "trend_pullback_breakout" in audit["structural_audit_best_candidate_by_setup"]
    for row in comparison:
        assert "shadow_candidates" in row
        assert "best_symbol" in row
        assert "average_gap" in row
        assert "dominant_blocker" in row
        assert "recommendation" in row


def test_structural_audit_recent_candidates_are_capped():
    audit_module = load_module("core.strategy_structure_audit")
    signals = [_base_signal(asset=f"ASSET-{idx}", score=0.735 - (idx * 0.001)) for idx in range(25)]

    audit = audit_module.build_strategy_structure_audit(signals)

    assert len(audit["structural_audit_recent_candidates"]) <= 15
