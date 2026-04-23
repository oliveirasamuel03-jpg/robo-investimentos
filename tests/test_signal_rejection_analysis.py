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
