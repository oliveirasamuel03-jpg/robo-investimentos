from __future__ import annotations

from copy import deepcopy

from conftest import load_module


SAFE_RECOMMENDATIONS = {
    "observe_more",
    "study_bos_confirmation",
    "study_bos_close_quality",
    "study_bos_retest_quality",
    "study_multitf_bos_confirmation",
    "study_pivot_to_bos_mapping",
    "study_false_breakout_risk",
    "keep_blocked_until_bos_confirms",
    "keep_blocked_until_retest_confirms",
    "keep_blocked_until_h4_confirms",
    "keep_blocked_until_h1_confirms",
    "no_threshold_change_recommended",
    "no_strategy_change_recommended",
    "insufficient_data",
}

FORBIDDEN_MESSAGE_FRAGMENTS = (
    "entrada aprovada",
    "pode comprar",
    "opere agora",
    "reduza score",
    "ignore bos",
    "bos suficiente para entrada real",
)


def _module():
    return load_module("core.bos_confirmation_quality_audit")


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
    return _module().build_bos_confirmation_quality_audit(**kwargs)


def _row(**kwargs) -> dict:
    base = {
        "symbol": "BTC-USD",
        "timeframe": "4h",
        "setup": "trend_pullback_breakout",
        "bos_state": "NO_BOS",
        "pivot_state": "PIVOT_FORMING",
        "bos_level": 100.0,
        "last_close": 99.5,
        "close_distance_to_bos_pct": 0.005,
        "wick_crossed_level": False,
        "close_above_or_below_level": False,
        "close_confirmed_level": False,
        "retest_detected": False,
        "retest_hold": False,
    }
    base.update(kwargs)
    return base


def test_pivot_triggered_without_bos_stays_blocked():
    result = _build(
        bos_pivot_trace_audit={
            "recent_candidates": [
                _row(pivot_state="PIVOT_TRIGGERED", bos_state="NO_BOS", timeframe="4h"),
                _row(pivot_state="PIVOT_TRIGGERED", bos_state="NO_BOS", timeframe="1h"),
            ]
        },
        feed_scope_reconciliation=_feed_scope(),
    )

    assert result["bos_quality_status"] == "PIVOT_TRIGGERED_BUT_BOS_MISSING"
    assert result["should_keep_blocked"] is True
    assert result["safe_to_change_strategy_now"] is False
    assert result["safe_to_change_threshold_now"] is False


def test_pivot_forming_without_bos_matches_btc_like():
    result = _build(
        signals=[
            {
                "asset": "BTC-USD",
                "strategy_name": "trend_pullback_breakout",
                "score": 0.6441,
                "effective_min_signal_score": 0.74,
            }
        ],
        bos_pivot_trace_audit={
            "recent_candidates": [
                _row(pivot_state="PIVOT_FORMING", bos_state="NO_BOS", timeframe="4h"),
                _row(pivot_state="PIVOT_FORMING", bos_state="NO_BOS", timeframe="1h"),
            ]
        },
        setup_blocker_taxonomy_audit={
            "candidates": [
                {
                    "symbol": "BTC-USD",
                    "setup": "trend_pullback_breakout",
                    "normalized_primary_reason": "BOS_MISSING",
                }
            ]
        },
        feed_scope_reconciliation=_feed_scope(),
    )

    assert result["top_symbol"] == "BTC-USD"
    assert result["bos_quality_status"] == "PIVOT_FORMING_BUT_BOS_MISSING"
    assert result["bos_failure_reason"] == "h1_bos_missing"
    assert result["current_feed_is_clean"] is True


def test_wick_cross_without_close_is_not_confirmed_bos():
    result = _build(
        bos_pivot_trace_audit={
            "recent_candidates": [
                _row(
                    bos_state="BOS_BY_WICK_ONLY",
                    wick_crossed_level=True,
                    close_above_or_below_level=False,
                    close_confirmed_level=False,
                )
            ]
        },
        feed_scope_reconciliation=_feed_scope(),
    )

    assert result["bos_quality_status"] == "BOS_BY_WICK_ONLY"
    assert result["bos_failure_reason"] == "wick_cross_without_close"
    assert result["should_keep_blocked"] is True


def test_weak_close_is_classified_as_close_quality_issue():
    result = _build(
        bos_pivot_trace_audit={
            "recent_candidates": [
                _row(
                    bos_state="BOS_BY_CLOSE_WEAK",
                    close_above_or_below_level=True,
                    close_confirmed_level=False,
                    close_distance_to_bos_pct=0.0002,
                )
            ]
        },
        feed_scope_reconciliation=_feed_scope(),
    )

    assert result["bos_quality_status"] == "BOS_BY_CLOSE_WEAK"
    assert result["bos_failure_reason"] == "close_distance_too_small"


def test_failed_breakout_is_marked_as_failed_bos():
    result = _build(
        bos_pivot_trace_audit={
            "recent_candidates": [
                _row(
                    bos_state="BOS_FAILED",
                    false_breakout_risk="HIGH",
                    why_bos_not_confirmed="breakout_returned_inside_structure",
                )
            ]
        },
        feed_scope_reconciliation=_feed_scope(),
    )

    assert result["bos_quality_status"] == "BOS_FAILED"
    assert result["bos_failure_reason"] == "close_back_inside_structure"


def test_h1_bos_without_h4_requires_h4_confirmation():
    result = _build(
        bos_pivot_trace_audit={
            "recent_candidates": [
                _row(timeframe="1h", bos_state="BOS_BY_CLOSE_CONFIRMED", close_confirmed_level=True),
                _row(timeframe="4h", bos_state="NO_BOS", close_confirmed_level=False),
            ]
        },
        feed_scope_reconciliation=_feed_scope(),
    )

    assert result["bos_quality_status"] == "BOS_MISSING_H4_CONFIRMATION"
    assert result["bos_failure_reason"] == "h1_only_without_h4"


def test_h4_bos_without_h1_requires_h1_confirmation():
    result = _build(
        bos_pivot_trace_audit={
            "recent_candidates": [
                _row(timeframe="4h", bos_state="BOS_BY_CLOSE_CONFIRMED", close_confirmed_level=True),
                _row(timeframe="1h", bos_state="NO_BOS", close_confirmed_level=False),
            ]
        },
        feed_scope_reconciliation=_feed_scope(),
    )

    assert result["bos_quality_status"] == "BOS_MISSING_H1_CONFIRMATION"
    assert result["bos_failure_reason"] == "h4_only_without_h1_confirmation"


def test_taxonomy_bos_missing_feeds_audit_without_operational_mutation():
    signal = {
        "asset": "ETH-USD",
        "strategy_name": "trend_pullback_breakout",
        "score": 0.71,
        "effective_min_signal_score": 0.74,
    }
    before = deepcopy(signal)

    result = _build(
        signals=[signal],
        setup_blocker_taxonomy_audit={
            "candidates": [
                {
                    "symbol": "ETH-USD",
                    "setup": "trend_pullback_breakout",
                    "normalized_primary_reason": "BOS_MISSING",
                }
            ]
        },
        feed_scope_reconciliation=_feed_scope(),
    )

    assert result["bos_quality_status"] in {"STRUCTURE_LEVEL_NOT_CLEAR", "BOS_MISSING_CLOSE_CONFIRMATION"}
    assert signal == before
    assert result["should_keep_blocked"] is True


def test_clean_feed_never_becomes_current_feed_problem():
    result = _build(
        bos_pivot_trace_audit={"recent_candidates": [_row()]},
        feed_scope_reconciliation=_feed_scope("ACCUMULATED"),
    )

    assert result["current_feed_is_clean"] is True
    assert result["fallback_blocker_scope"] != "CURRENT_CYCLE"
    assert "FEED" not in result["bos_quality_status"]


def test_safety_fields_and_recommendations_are_locked_down():
    result = _build(
        bos_pivot_trace_audit={"recent_candidates": [_row()]},
        feed_scope_reconciliation=_feed_scope(),
    )

    assert result["mode"] == "DIAGNOSTIC_ONLY"
    assert result["safety_mode"] == "SHADOW_ONLY"
    assert result["should_keep_blocked"] is True
    assert result["safe_to_change_strategy_now"] is False
    assert result["safe_to_change_threshold_now"] is False
    assert result["recommendation"] in SAFE_RECOMMENDATIONS
    for candidate in result["candidates"]:
        assert candidate["recommendation"] in SAFE_RECOMMENDATIONS
        message = str(candidate.get("suggested_ui_message") or "").lower()
        assert not any(fragment in message for fragment in FORBIDDEN_MESSAGE_FRAGMENTS)


def test_insufficient_data_returns_safe_default():
    result = _build(feed_scope_reconciliation=_feed_scope())

    assert result["bos_quality_status"] == "INSUFFICIENT_DATA_FOR_BOS_QUALITY"
    assert result["recommendation"] == "insufficient_data"
    assert result["should_keep_blocked"] is True
