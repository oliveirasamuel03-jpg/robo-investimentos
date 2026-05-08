from __future__ import annotations

import json

import pandas as pd

from tests.conftest import load_module


def _trend_frame(periods: int, *, freq: str, direction: str = "up", final_breakout: bool = True, source: str = "market") -> pd.DataFrame:
    start = 100.0
    end = 140.0 if direction == "up" else 70.0
    values = [start + ((end - start) * idx / max(periods - 1, 1)) for idx in range(periods)]
    if final_breakout and periods > 4:
        if direction == "up":
            values[-1] = max(values[:-1]) * 1.08
        else:
            values[-1] = min(values[:-1]) * 0.92
    rows = []
    for idx, close in enumerate(values):
        if direction == "up":
            open_price = close * (0.995 if idx == periods - 1 else 0.998)
        else:
            open_price = close * (1.005 if idx == periods - 1 else 1.002)
        rows.append(
            {
                "datetime": pd.Timestamp("2026-01-01", tz="UTC") + pd.to_timedelta(idx, unit=freq),
                "open": open_price,
                "high": max(open_price, close) * 1.002,
                "low": min(open_price, close) * 0.998,
                "close": close,
                "volume": 1000 + idx,
                "data_source": source,
                "provider_name": "twelvedata",
            }
        )
    return pd.DataFrame(rows)


def _payload(*, direction_1d: str = "up", direction_4h: str = "up", direction_1h: str = "up", final_breakout: bool = True):
    return {
        "BTC-USD": {
            "1d": _trend_frame(80, freq="D", direction=direction_1d, final_breakout=final_breakout),
            "4h": _trend_frame(100, freq="h", direction=direction_4h, final_breakout=final_breakout),
            "1h": _trend_frame(120, freq="h", direction=direction_1h, final_breakout=final_breakout),
        }
    }


def test_short_data_returns_insufficient_data():
    mtf = load_module("core.multi_timeframe_swing_audit")
    result = mtf.build_multi_timeframe_swing_audit(
        market_data={"BTC-USD": {"1d": _trend_frame(5, freq="D"), "4h": _trend_frame(5, freq="h"), "1h": _trend_frame(5, freq="h")}},
        market_data_status={"feed_status": "LIVE", "provider_effective": "twelvedata"},
    )

    assert result["mode"] == "SHADOW_ONLY"
    assert result["top_alignment_status"] == "INSUFFICIENT_DATA"
    assert "insufficient_data" in result["recent_candidates"][0]["missing_for_setup"]
    assert result["estimated_provider_calls"] == 0


def test_aligned_1d_4h_1h_generates_strong_alignment_without_trade_authority():
    mtf = load_module("core.multi_timeframe_swing_audit")
    result = mtf.build_multi_timeframe_swing_audit(
        market_data=_payload(),
        market_data_status={"feed_status": "LIVE", "provider_effective": "twelvedata"},
    )
    candidate = result["recent_candidates"][0]

    assert result["mode"] == "SHADOW_ONLY"
    assert result["shadow_only"] is True
    assert candidate["alignment_status"] == "STRONG_ALIGNMENT"
    assert candidate["supports_trend_pullback_breakout"] is True
    assert candidate["should_keep_blocked"] is True
    assert "positions" not in result
    assert "wallet" not in result
    assert "pnl" not in result


def test_conflict_between_daily_and_h4_is_marked_conflict():
    mtf = load_module("core.multi_timeframe_swing_audit")
    result = mtf.build_multi_timeframe_swing_audit(
        market_data=_payload(direction_1d="up", direction_4h="down", direction_1h="up"),
        market_data_status={"feed_status": "LIVE", "provider_effective": "twelvedata"},
    )

    assert result["top_alignment_status"] == "CONFLICT"
    assert "timeframe_conflict" in result["recent_candidates"][0]["missing_for_setup"]


def test_missing_bos_keeps_setup_blocked():
    mtf = load_module("core.multi_timeframe_swing_audit")
    result = mtf.build_multi_timeframe_swing_audit(
        market_data=_payload(final_breakout=False),
        market_data_status={"feed_status": "LIVE", "provider_effective": "twelvedata"},
    )

    candidate = result["recent_candidates"][0]
    assert "h4_bos_missing" in candidate["missing_for_setup"]
    assert candidate["should_keep_blocked"] is True


def test_fallback_feed_blocks_or_degrades_diagnostic():
    mtf = load_module("core.multi_timeframe_swing_audit")
    result = mtf.build_multi_timeframe_swing_audit(
        market_data=_payload(),
        market_data_status={"feed_status": "FALLBACK", "provider_effective": "twelvedata"},
    )

    assert result["top_alignment_status"] == "INSUFFICIENT_DATA"
    assert result["provider_guard"] == "feed_status_fallback"
    assert "feed_not_live" in result["recent_candidates"][0]["missing_for_setup"]


def test_daily_base_interval_does_not_synthesize_intraday_live_frames():
    mtf = load_module("core.multi_timeframe_swing_audit")
    result = mtf.build_multi_timeframe_swing_audit(
        market_data={"BTC-USD": _trend_frame(80, freq="D")},
        market_data_status={"feed_status": "LIVE", "provider_effective": "twelvedata", "effective_interval": "1d"},
    )

    assert "4h=blocked_base_interval_1d" in result["timeframe_fallbacks"]
    assert "1h=blocked_base_interval_1d" in result["timeframe_fallbacks"]
    assert result["top_alignment_status"] == "INSUFFICIENT_DATA"


def test_old_state_loads_with_multi_timeframe_defaults(isolated_storage):
    config = load_module("core.config")
    config.BOT_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    config.BOT_STATE_FILE.write_text(
        json.dumps({"wallet_value": 1000.0, "cash": 1000.0, "positions": []}),
        encoding="utf-8",
    )
    state_store = load_module("core.state_store")
    state = state_store.load_bot_state()

    assert state["multi_timeframe_swing_audit"]["mode"] == "SHADOW_ONLY"
    assert state["multi_timeframe_swing_audit"]["shadow_only"] is True
    assert state["multi_timeframe_swing_audit"]["recent_candidates"] == []
