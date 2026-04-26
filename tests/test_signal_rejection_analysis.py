from __future__ import annotations

from tests.conftest import load_module


def test_build_signal_rejection_events_maps_feed_and_strategy_reasons(isolated_storage):
    rejection_analysis = load_module("core.signal_rejection_analysis")

    events = rejection_analysis.build_signal_rejection_events(
        {
            "asset": "BTC-USD",
            "signal_key": "BTC-USD|2026-04-23T00:00:00+00:00|fallback",
            "signal_timestamp": "2026-04-23T00:00:00+00:00",
            "data_source": "fallback",
            "strategy_name": rejection_analysis.DEFAULT_STRATEGY_NAME,
            "rejection_reasons": [
                "score_below_minimum",
                "fallback_blocked",
                "score_below_minimum",
            ],
        }
    )

    assert len(events) == 2
    assert {event["reason_code"] for event in events} == {"score_below_minimum", "fallback_blocked"}
    assert {event["layer"] for event in events} == {"strategy", "feed"}


def test_refresh_swing_validation_cycle_persists_rejection_breakdowns(isolated_storage):
    state_store = load_module("core.state_store")
    swing_validation = load_module("core.swing_validation")

    report = swing_validation.refresh_swing_validation_cycle(
        cycle_result={
            "signals": [],
            "validation_cycle": {
                "signals_rejected": 2,
                "rejection_events": [
                    {
                        "reason_code": "score_below_minimum",
                        "human_reason": "Score ajustado ficou abaixo do minimo exigido.",
                        "layer": "strategy",
                        "strategy_name": "trend_pullback_breakout",
                        "symbol": "BTC-USD",
                        "timestamp": "2026-04-23T00:00:00+00:00",
                        "event_key": "BTC-USD|2026-04-23T00:00:00+00:00|score_below_minimum",
                    },
                    {
                        "reason_code": "fallback_blocked",
                        "human_reason": "Ativo caiu em fallback sintetico e o sinal foi bloqueado.",
                        "layer": "feed",
                        "strategy_name": "feed_guard",
                        "symbol": "ETH-USD",
                        "timestamp": "2026-04-23T00:00:00+00:00",
                        "event_key": "ETH-USD|2026-04-23T00:00:00+00:00|fallback_blocked",
                    },
                ],
            },
        }
    )

    state = state_store.load_bot_state()
    validation_state = state.get("validation", {}) or {}

    assert validation_state["rejection_reason_breakdown"]["score_below_minimum"] == 1
    assert validation_state["rejection_reason_breakdown"]["fallback_blocked"] == 1
    assert validation_state["rejection_layer_breakdown"]["strategy"] == 1
    assert validation_state["rejection_layer_breakdown"]["feed"] == 1
    assert report["metrics"]["rejection_top_reason"] in {"score_below_minimum", "fallback_blocked"}
    assert isinstance(report["rejection_quality"]["top_reasons"], list)
    assert "feed_rejection_consistency" in report


def test_feed_live_with_accumulated_fallback_marks_possible_stale_label(isolated_storage):
    rejection_analysis = load_module("core.signal_rejection_analysis")

    diagnostic = rejection_analysis.build_feed_rejection_consistency_diagnostic(
        feed_quality={
            "feed_status": "LIVE",
            "provider_effective": "twelvedata",
            "live_count": 5,
            "fallback_count": 0,
            "total_symbols": 5,
        },
        current_cycle_summary={
            "total_rejection_events": 3,
            "rejected_by_reason": {"score_below_minimum": 3},
            "rejected_by_layer": {"strategy": 3},
            "top_rejection_reason": "score_below_minimum",
            "top_rejection_layer": "strategy",
        },
        accumulated_summary={
            "total_rejection_events": 20,
            "reason_breakdown": {"fallback_blocked": 12, "score_below_minimum": 8},
            "layer_breakdown": {"feed": 12, "strategy": 8},
            "top_reason": "fallback_blocked",
            "top_layer": "feed",
        },
    )

    assert diagnostic["is_current_feed_live"] is True
    assert diagnostic["is_fallback_rejection_current"] is False
    assert diagnostic["possible_stale_fallback_label"] is True
    assert diagnostic["dominant_rejection_scope"] == "current_cycle"


def test_current_fallback_blocks_are_marked_as_feed_issue(isolated_storage):
    rejection_analysis = load_module("core.signal_rejection_analysis")

    diagnostic = rejection_analysis.build_feed_rejection_consistency_diagnostic(
        feed_quality={
            "feed_status": "FALLBACK",
            "provider_effective": "synthetic",
            "live_count": 0,
            "fallback_count": 5,
            "total_symbols": 5,
        },
        current_cycle_summary={
            "total_rejection_events": 5,
            "rejected_by_reason": {"fallback_blocked": 5},
            "rejected_by_layer": {"feed": 5},
            "top_rejection_reason": "fallback_blocked",
            "top_rejection_layer": "feed",
        },
        accumulated_summary={},
    )

    assert diagnostic["is_current_feed_live"] is False
    assert diagnostic["is_fallback_rejection_current"] is True
    assert diagnostic["possible_stale_fallback_label"] is False
    assert diagnostic["dominant_rejection_scope"] == "current_cycle"


def test_current_provider_unknown_is_marked_as_feed_issue(isolated_storage):
    rejection_analysis = load_module("core.signal_rejection_analysis")

    diagnostic = rejection_analysis.build_feed_rejection_consistency_diagnostic(
        feed_quality={
            "feed_status": "DELAYED",
            "provider_effective": "unknown",
            "live_count": 0,
            "fallback_count": 0,
            "total_symbols": 5,
        },
        current_cycle_summary={
            "total_rejection_events": 4,
            "rejected_by_reason": {"provider_unknown": 4},
            "rejected_by_layer": {"feed": 4},
            "top_rejection_reason": "provider_unknown",
            "top_rejection_layer": "feed",
        },
        accumulated_summary={},
    )

    assert diagnostic["is_feed_rejection_current"] is True
    assert diagnostic["is_fallback_rejection_current"] is False
    assert diagnostic["possible_stale_fallback_label"] is False
    assert "feed" in diagnostic["diagnostic_note"]


def test_strategy_dominant_with_live_feed_marks_strategy_bottleneck(isolated_storage):
    rejection_analysis = load_module("core.signal_rejection_analysis")

    diagnostic = rejection_analysis.build_feed_rejection_consistency_diagnostic(
        feed_quality={
            "feed_status": "LIVE",
            "provider_effective": "twelvedata",
            "live_count": 5,
            "fallback_count": 0,
            "total_symbols": 5,
        },
        current_cycle_summary={
            "total_rejection_events": 7,
            "rejected_by_reason": {"trend_not_confirmed": 7},
            "rejected_by_layer": {"strategy": 7},
            "top_rejection_reason": "trend_not_confirmed",
            "top_rejection_layer": "strategy",
        },
        accumulated_summary={},
    )

    assert diagnostic["dominant_rejection_layer"] == "strategy"
    assert diagnostic["strategy_rejection_current_cycle_count"] == 7
    assert "estrategico" in diagnostic["diagnostic_note"]
