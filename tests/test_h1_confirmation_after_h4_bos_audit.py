from __future__ import annotations

from copy import deepcopy

from tests.conftest import load_module


SAFE_RECOMMENDATIONS = {
    "observe_more",
    "study_h1_confirmation",
    "study_h1_retest_quality",
    "study_h1_after_h4_bos_mapping",
    "study_multitf_confirmation",
    "study_entry_timing_risk",
    "keep_blocked_until_h1_confirms",
    "keep_blocked_until_h1_retest_confirms",
    "keep_blocked_until_h1_structure_is_clear",
    "keep_blocked_until_multitf_confirms",
    "no_threshold_change_recommended",
    "no_strategy_change_recommended",
    "insufficient_data",
}

FORBIDDEN_MESSAGE_FRAGMENTS = (
    "entrada aprovada",
    "pode comprar",
    "opere agora",
    "reduza score",
    "ignore 1h",
    "h4 e suficiente para entrada real",
)


def _module():
    return load_module("core.h1_confirmation_after_h4_bos_audit")


def _build(**kwargs) -> dict:
    return _module().build_h1_confirmation_after_h4_bos_audit(**kwargs)


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


def _row(timeframe: str = "4h", **kwargs) -> dict:
    base = {
        "symbol": "ETH-USD",
        "timeframe": timeframe,
        "setup": "trend_pullback_breakout",
        "bos_state": "NO_BOS",
        "pivot_state": "PIVOT_TRIGGERED",
        "retest_detected": False,
        "retest_hold": False,
        "relationship_to_higher_tf": "H4_STRUCTURE_MISSING",
    }
    base.update(kwargs)
    return base


def _mtf(**kwargs) -> dict:
    base = {
        "symbol": "ETH-USD",
        "daily_bias": "UP",
        "h4_structure": "UP",
        "h1_confirmation": "UP",
        "alignment_status": "PARTIAL_ALIGNMENT",
        "timeframe_diagnostics": [
            {"timeframe": "4h", "data_quality": "ok"},
            {"timeframe": "1h", "data_quality": "ok"},
        ],
    }
    base.update(kwargs)
    return base


def _base_kwargs(h1_bos: str = "NO_BOS", **overrides) -> dict:
    kwargs = {
        "signals": [
            {
                "asset": "ETH-USD",
                "strategy_name": "trend_pullback_breakout",
                "score": 0.7349,
                "effective_min_signal_score": 0.74,
            }
        ],
        "bos_pivot_trace_audit": {
            "recent_candidates": [
                _row(
                    "4h",
                    bos_state="BOS_RETEST_CONFIRMED",
                    pivot_state="PIVOT_TRIGGERED",
                    retest_detected=True,
                    retest_hold=True,
                    relationship_to_higher_tf="H4_CONFIRMS_H1",
                ),
                _row("1h", bos_state=h1_bos, pivot_state="NO_PIVOT"),
            ]
        },
        "multi_timeframe_swing_audit": {"recent_candidates": [_mtf()]},
        "feed_scope_reconciliation": _feed_scope(),
    }
    kwargs.update(overrides)
    return kwargs


def _top(result: dict) -> dict:
    return (result.get("candidates") or [{}])[0]


def test_eth_like_h4_retest_confirmed_with_h1_insufficient_data_stays_blocked():
    result = _build(**_base_kwargs(h1_bos="INSUFFICIENT_DATA"))

    assert result["top_symbol"] == "ETH-USD"
    assert result["h4_bos_state"] == "BOS_RETEST_CONFIRMED"
    assert result["h1_bos_state"] == "INSUFFICIENT_DATA"
    assert result["h1_confirmation_status"] == "H1_INSUFFICIENT_DATA_AFTER_H4_BOS"
    assert result["h1_failure_reason"] == "h1_insufficient_data"
    assert result["recommendation"] == "keep_blocked_until_h1_confirms"
    assert result["should_keep_blocked"] is True


def test_h4_confirmed_with_h1_no_bos_requires_h1_confirmation():
    result = _build(**_base_kwargs(h1_bos="NO_BOS"))

    assert result["h1_confirmation_status"] == "H1_NO_BOS_AFTER_H4_BOS"
    assert result["h1_failure_reason"] == "h1_no_bos"
    assert _top(result)["h1_entry_timing_risk"] == "ENTRY_TOO_EARLY_RISK"


def test_h4_confirmed_with_h1_against_h4_is_conflict():
    result = _build(
        **_base_kwargs(
            h1_bos="NO_BOS",
            multi_timeframe_swing_audit={"recent_candidates": [_mtf(h4_structure="UP", h1_confirmation="DOWN")]},
        )
    )

    assert result["h1_confirmation_status"] == "H1_CONFLICTS_WITH_H4_BOS"
    assert result["h1_failure_reason"] == "h1_trend_conflict"
    assert result["h1_h4_alignment"] == "CONFLICT"


def test_h4_confirmed_with_h1_sideways_is_sideways_after_h4_bos():
    result = _build(
        **_base_kwargs(
            h1_bos="NO_BOS",
            multi_timeframe_swing_audit={"recent_candidates": [_mtf(h4_structure="UP", h1_confirmation="SIDEWAYS")]},
        )
    )

    assert result["h1_confirmation_status"] == "H1_SIDEWAYS_AFTER_H4_BOS"
    assert result["h1_failure_reason"] == "h1_sideways"


def test_h4_confirmed_with_h1_pivot_forming_waits_for_activation():
    result = _build(
        **_base_kwargs(
            h1_bos="NO_BOS",
            bos_pivot_trace_audit={
                "recent_candidates": [
                    _row("4h", bos_state="BOS_RETEST_CONFIRMED", pivot_state="PIVOT_TRIGGERED", retest_hold=True),
                    _row("1h", bos_state="NO_BOS", pivot_state="PIVOT_FORMING"),
                ]
            },
        )
    )

    assert result["h1_confirmation_status"] == "H1_PIVOT_FORMING_AFTER_H4_BOS"
    assert result["h1_failure_reason"] == "h1_pivot_forming_only"


def test_h4_confirmed_with_h1_bos_without_retest_is_pending():
    result = _build(
        **_base_kwargs(
            h1_bos="BOS_BY_CLOSE_CONFIRMED",
            bos_pivot_trace_audit={
                "recent_candidates": [
                    _row("4h", bos_state="BOS_RETEST_CONFIRMED", pivot_state="PIVOT_TRIGGERED", retest_hold=True),
                    _row(
                        "1h",
                        bos_state="BOS_BY_CLOSE_CONFIRMED",
                        pivot_state="PIVOT_TRIGGERED",
                        retest_detected=False,
                        retest_hold=False,
                    ),
                ]
            },
        )
    )

    assert result["h1_confirmation_status"] == "H1_RETEST_PENDING_AFTER_H4_BOS"
    assert result["h1_failure_reason"] == "h1_retest_pending"


def test_h4_confirmed_with_h1_confirmed_remains_shadow_only_and_blocked():
    result = _build(
        **_base_kwargs(
            h1_bos="BOS_RETEST_CONFIRMED",
            bos_pivot_trace_audit={
                "recent_candidates": [
                    _row("4h", bos_state="BOS_RETEST_CONFIRMED", pivot_state="PIVOT_TRIGGERED", retest_hold=True),
                    _row("1h", bos_state="BOS_RETEST_CONFIRMED", pivot_state="PIVOT_TRIGGERED", retest_hold=True),
                ]
            },
        )
    )

    assert result["h1_confirmation_status"] == "H1_CONFIRMED_AFTER_H4_BOS"
    assert result["should_keep_blocked"] is True
    assert result["safe_to_change_strategy_now"] is False
    assert result["safe_to_change_threshold_now"] is False
    assert _top(result)["shadow_only"] is True


def test_clean_feed_and_historical_fallback_do_not_become_current_feed_problem():
    result = _build(
        **_base_kwargs(
            feed_scope_reconciliation=_feed_scope("ACCUMULATED"),
        )
    )

    assert result["current_feed_is_clean"] is True
    assert result["fallback_blocker_scope"] == "ACCUMULATED"
    assert result["h1_failure_reason"] != "fallback"


def test_safety_whitelist_messages_and_no_operational_mutation():
    signal = {
        "asset": "ETH-USD",
        "strategy_name": "trend_pullback_breakout",
        "score": 0.7349,
        "effective_min_signal_score": 0.74,
        "should_buy": False,
        "positions": [],
        "pnl": 0.0,
    }
    before = deepcopy(signal)

    result = _build(**_base_kwargs(signals=[signal], h1_bos="BOS_RETEST_CONFIRMED"))

    assert signal == before
    assert result["recommendation"] in SAFE_RECOMMENDATIONS
    assert result["should_keep_blocked"] is True
    assert result["safe_to_change_strategy_now"] is False
    assert result["safe_to_change_threshold_now"] is False
    for candidate in result["candidates"]:
        assert candidate["recommendation"] in SAFE_RECOMMENDATIONS
        assert candidate["should_keep_blocked"] is True
        assert candidate["safe_to_change_strategy_now"] is False
        assert candidate["safe_to_change_threshold_now"] is False
        lowered = (candidate.get("suggested_ui_message") or "").lower()
        assert not any(fragment in lowered for fragment in FORBIDDEN_MESSAGE_FRAGMENTS)


def test_insufficient_trace_data_returns_safe_default():
    result = _build(feed_scope_reconciliation=_feed_scope())

    assert result["status"] == "INSUFFICIENT_DATA"
    assert result["h1_confirmation_status"] == "INSUFFICIENT_DATA_FOR_H1_CONFIRMATION"
    assert result["recommendation"] == "insufficient_data"
    assert result["should_keep_blocked"] is True
    assert result["candidates"] == []
