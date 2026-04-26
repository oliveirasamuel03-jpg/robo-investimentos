from __future__ import annotations

from tests.conftest import load_module


def _state() -> dict:
    return {
        "trader": {"watchlist": ["BTC-USD", "ETH-USD"], "max_open_positions": 2},
        "broker": {"mode": "paper", "effective_mode": "paper", "provider": "paper"},
        "risk": {"daily_loss_block_active": False},
        "positions": [],
    }


def _signal(**overrides):
    payload = {
        "asset": "BTC-USD",
        "score": 0.72,
        "effective_min_signal_score": 0.74,
        "buy": False,
        "strategy_name": "trend_pullback_breakout",
        "context_status": "FAVORAVEL",
        "rejection_reasons": ["score_below_minimum"],
        "signal_timestamp": "2026-04-26T00:00:00+00:00",
    }
    payload.update(overrides)
    return payload


def _live_feed() -> dict:
    return {
        "feed_status": "LIVE",
        "provider_effective": "twelvedata",
        "fallback_symbols": [],
        "source_breakdown": {"market": 2, "cached": 0, "fallback": 0, "unknown": 0},
    }


def test_near_approved_detected_under_safe_conditions(isolated_storage):
    preview_module = load_module("core.calibration_preview")

    preview = preview_module.build_calibration_preview(
        signals=[_signal()],
        state=_state(),
        paper_state={"positions": {}},
        market_data_status=_live_feed(),
        macro_alert={"macro_alert_active": False},
        enabled=True,
        margin=0.04,
        max_examples=10,
    )

    assert preview["mode"] == "PREVIEW_ONLY"
    assert preview["near_approved_count"] == 1
    assert preview["near_approved_rate"] == 1.0
    assert preview["near_approved_examples"][0]["status"] == "PREVIEW_ONLY"
    assert "trade_approved" not in preview["near_approved_examples"][0]


def test_not_near_approved_when_feed_is_fallback(isolated_storage):
    preview_module = load_module("core.calibration_preview")

    preview = preview_module.build_calibration_preview(
        signals=[_signal()],
        state=_state(),
        paper_state={"positions": {}},
        market_data_status={"feed_status": "FALLBACK", "fallback_symbols": ["BTC-USD"]},
        macro_alert={"macro_alert_active": False},
        enabled=True,
        margin=0.04,
        max_examples=10,
    )

    assert preview["near_approved_count"] == 0
    assert preview["unsafe_conditions_count"] == 1


def test_not_near_approved_when_macro_alert_active(isolated_storage):
    preview_module = load_module("core.calibration_preview")

    preview = preview_module.build_calibration_preview(
        signals=[_signal()],
        state=_state(),
        paper_state={"positions": {}},
        market_data_status=_live_feed(),
        macro_alert={"macro_alert_active": True},
        enabled=True,
        margin=0.04,
        max_examples=10,
    )

    assert preview["near_approved_count"] == 0


def test_not_near_approved_when_context_is_critical(isolated_storage):
    preview_module = load_module("core.calibration_preview")

    preview = preview_module.build_calibration_preview(
        signals=[_signal(context_status="CRITICO")],
        state=_state(),
        paper_state={"positions": {}},
        market_data_status=_live_feed(),
        macro_alert={"macro_alert_active": False},
        enabled=True,
        margin=0.04,
        max_examples=10,
    )

    assert preview["near_approved_count"] == 0


def test_not_near_approved_when_broker_is_not_paper(isolated_storage):
    preview_module = load_module("core.calibration_preview")
    state = _state()
    state["broker"] = {"mode": "live", "effective_mode": "live", "provider": "alpaca"}

    preview = preview_module.build_calibration_preview(
        signals=[_signal()],
        state=state,
        paper_state={"positions": {}},
        market_data_status=_live_feed(),
        macro_alert={"macro_alert_active": False},
        enabled=True,
        margin=0.04,
        max_examples=10,
    )

    assert preview["near_approved_count"] == 0


def test_not_near_approved_for_risk_critical_rejection(isolated_storage):
    preview_module = load_module("core.calibration_preview")

    preview = preview_module.build_calibration_preview(
        signals=[_signal(rejection_reasons=["daily_loss_guard", "score_below_minimum"])],
        state=_state(),
        paper_state={"positions": {}},
        market_data_status=_live_feed(),
        macro_alert={"macro_alert_active": False},
        enabled=True,
        margin=0.04,
        max_examples=10,
    )

    assert preview["near_approved_count"] == 0


def test_examples_are_capped(isolated_storage):
    preview_module = load_module("core.calibration_preview")
    signals = [_signal(asset=f"BTC-USD", signal_timestamp=f"2026-04-26T00:0{i}:00+00:00") for i in range(6)]

    preview = preview_module.build_calibration_preview(
        signals=signals,
        state=_state(),
        paper_state={"positions": {}},
        market_data_status=_live_feed(),
        macro_alert={"macro_alert_active": False},
        enabled=True,
        margin=0.04,
        max_examples=3,
    )

    assert preview["near_approved_count"] == 6
    assert len(preview["near_approved_examples"]) == 3
