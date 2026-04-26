from __future__ import annotations

from tests.conftest import load_module


def _signal(**overrides):
    payload = {
        "asset": "BTC-USD",
        "score": 0.67,
        "effective_min_signal_score": 0.74,
        "buy": False,
        "strategy_name": "trend_pullback_breakout",
        "context_status": "FAVORAVEL",
        "data_source": "twelvedata",
        "macro_alert_active": False,
        "rejection_reasons": ["score_below_minimum"],
    }
    payload.update(overrides)
    return payload


def test_rejection_reason_maps_to_score_below_min(isolated_storage):
    bottleneck = load_module("core.strategy_bottleneck")

    item = bottleneck.classify_strategy_bottleneck(_signal())

    assert item["bottleneck_category"] == "SCORE_BELOW_MIN"
    assert item["status"] == "DIAGNOSTIC_ONLY"


def test_rejection_reason_maps_to_rsi_out_of_range(isolated_storage):
    bottleneck = load_module("core.strategy_bottleneck")

    item = bottleneck.classify_strategy_bottleneck(_signal(rejection_reasons=["reversal_not_eligible"]))

    assert item["bottleneck_category"] == "RSI_OUT_OF_RANGE"


def test_rejection_reason_maps_to_trend_not_confirmed(isolated_storage):
    bottleneck = load_module("core.strategy_bottleneck")

    item = bottleneck.classify_strategy_bottleneck(_signal(rejection_reasons=["trend_not_confirmed"]))

    assert item["bottleneck_category"] == "TREND_NOT_CONFIRMED"


def test_rejection_reason_maps_to_momentum_or_secondary_confirmation(isolated_storage):
    bottleneck = load_module("core.strategy_bottleneck")

    momentum = bottleneck.classify_strategy_bottleneck(_signal(rejection_reasons=["confidence_too_low"]))
    secondary = bottleneck.classify_strategy_bottleneck(_signal(rejection_reasons=["breakout_not_confirmed"]))

    assert momentum["bottleneck_category"] == "MOMENTUM_WEAK"
    assert secondary["bottleneck_category"] == "SECONDARY_CONFIRMATION_WEAK"


def test_feed_block_is_classified_without_approval_side_effect(isolated_storage):
    bottleneck = load_module("core.strategy_bottleneck")

    summary = bottleneck.summarize_strategy_bottlenecks(
        signals=[_signal(rejection_reasons=["fallback_blocked"], data_source="fallback")]
    )

    assert summary["feed_block_count"] == 1
    assert summary["total_strategy_rejections"] == 0
    assert summary["mode"] == "DIAGNOSTIC_ONLY"
    assert all("trade_approved" not in item for item in summary["closest_candidates"])


def test_closest_candidates_are_capped(isolated_storage):
    bottleneck = load_module("core.strategy_bottleneck")
    signals = [
        _signal(asset="BTC-USD", score=0.60 + (idx / 100), rejection_reasons=["score_below_minimum"])
        for idx in range(12)
    ]

    summary = bottleneck.summarize_strategy_bottlenecks(signals=signals, max_candidates=10)

    assert summary["score_below_min_count"] == 12
    assert len(summary["closest_candidates"]) == 10


def test_portuguese_reason_maps_to_score_below_min(isolated_storage):
    bottleneck = load_module("core.strategy_bottleneck")

    item = bottleneck.classify_strategy_bottleneck(
        _signal(rejection_reasons=["Score ajustado ficou abaixo do minimo exigido."])
    )

    assert item["bottleneck_category"] == "SCORE_BELOW_MIN"
