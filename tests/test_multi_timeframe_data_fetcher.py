from __future__ import annotations

import pandas as pd

from tests.conftest import load_module


def _frame(rows: int = 80, *, source: str = "market", final_close_breakout: bool = True, wick_only: bool = False) -> pd.DataFrame:
    values = [100.0 + idx * 0.4 for idx in range(rows)]
    if final_close_breakout:
        values[-1] = max(values[:-1]) * 1.05
    records = []
    for idx, close in enumerate(values):
        high = close * 1.002
        if wick_only and idx == rows - 1:
            high = max(values[:-1]) * 1.08
            close = max(values[:-1]) * 0.998
        records.append(
            {
                "datetime": pd.Timestamp("2026-01-01", tz="UTC") + pd.to_timedelta(idx, unit="h"),
                "open": close * 0.995,
                "high": high,
                "low": close * 0.99,
                "close": close,
                "volume": 1000,
                "data_source": source,
                "provider_name": "twelvedata",
            }
        )
    return pd.DataFrame(records)


def test_cache_hit_avoids_new_provider_call():
    fetcher = load_module("core.multi_timeframe_data_fetcher")
    fetcher.clear_multi_timeframe_intraday_cache()
    calls = []

    def fake_fetch(symbol, **kwargs):
        calls.append((symbol, kwargs["interval"]))
        return _frame(), {"feed_status": "LIVE"}

    first = fetcher.fetch_multi_timeframe_intraday_data(
        symbols=["BTC-USD"],
        market_data_status={"feed_status": "LIVE", "provider_effective": "twelvedata"},
        timeframes=["4h"],
        fetch_frame_func=fake_fetch,
    )
    second = fetcher.fetch_multi_timeframe_intraday_data(
        symbols=["BTC-USD"],
        market_data_status={"feed_status": "LIVE", "provider_effective": "twelvedata"},
        timeframes=["4h"],
        fetch_frame_func=fake_fetch,
    )

    assert first.summary["provider_calls_attempted"] == 1
    assert second.summary["cache_hits"] == 1
    assert second.summary["provider_calls_attempted"] == 0
    assert len(calls) == 1


def test_cache_miss_respects_max_calls_per_cycle():
    fetcher = load_module("core.multi_timeframe_data_fetcher")
    fetcher.clear_multi_timeframe_intraday_cache()

    def fake_fetch(symbol, **kwargs):
        return _frame(), {"feed_status": "LIVE"}

    result = fetcher.fetch_multi_timeframe_intraday_data(
        symbols=["BTC-USD"],
        market_data_status={"feed_status": "LIVE", "provider_effective": "twelvedata"},
        timeframes=["4h", "1h"],
        max_calls_per_cycle=1,
        fetch_frame_func=fake_fetch,
    )

    assert result.summary["provider_calls_attempted"] == 1
    assert result.summary["provider_calls_skipped"] == 1
    assert result.summary["provider_budget_guard_active"] is True
    assert result.summary["provider_guard_reason"] == "provider_budget_guard_blocked"


def test_feed_fallback_blocks_fetch_when_live_required():
    fetcher = load_module("core.multi_timeframe_data_fetcher")
    fetcher.clear_multi_timeframe_intraday_cache()
    calls = []

    def fake_fetch(symbol, **kwargs):
        calls.append(symbol)
        return _frame(), {"feed_status": "LIVE"}

    result = fetcher.fetch_multi_timeframe_intraday_data(
        symbols=["BTC-USD"],
        market_data_status={"feed_status": "FALLBACK", "provider_effective": "twelvedata"},
        fetch_frame_func=fake_fetch,
    )

    assert result.summary["provider_calls_attempted"] == 0
    assert result.summary["provider_guard_reason"] == "feed_not_live"
    assert calls == []


def test_provider_error_does_not_break_fetcher():
    fetcher = load_module("core.multi_timeframe_data_fetcher")
    fetcher.clear_multi_timeframe_intraday_cache()

    def fake_fetch(symbol, **kwargs):
        raise RuntimeError("provider down")

    result = fetcher.fetch_multi_timeframe_intraday_data(
        symbols=["BTC-USD"],
        market_data_status={"feed_status": "LIVE", "provider_effective": "twelvedata"},
        timeframes=["4h"],
        fetch_frame_func=fake_fetch,
    )

    assert result.summary["provider_calls_attempted"] == 1
    assert result.summary["intraday_data_quality"] == "INSUFFICIENT"
    assert "provider down" in result.summary["last_error"]


def test_real_intraday_frames_are_passed_to_shadow_audit_only():
    fetcher = load_module("core.multi_timeframe_data_fetcher")
    mtf = load_module("core.multi_timeframe_swing_audit")
    fetcher.clear_multi_timeframe_intraday_cache()

    def fake_fetch(symbol, **kwargs):
        return _frame(), {"feed_status": "LIVE"}

    intraday = fetcher.fetch_multi_timeframe_intraday_data(
        symbols=["BTC-USD"],
        market_data_status={"feed_status": "LIVE", "provider_effective": "twelvedata", "effective_interval": "1d"},
        timeframes=["4h", "1h"],
        fetch_frame_func=fake_fetch,
    )
    audit = mtf.build_multi_timeframe_swing_audit(
        market_data={"BTC-USD": _frame(rows=80)},
        market_data_status={"feed_status": "LIVE", "provider_effective": "twelvedata", "effective_interval": "1d"},
        intraday_market_data=intraday.frames,
        intraday_fetch_summary=intraday.summary,
    )

    assert audit["mode"] == "SHADOW_ONLY"
    assert audit["uses_real_intraday_data"] is True
    assert audit["estimated_provider_calls"] == 2
    assert audit["recent_candidates"][0]["should_keep_blocked"] is True
    assert "positions" not in audit
    assert "realized_pnl" not in audit


def test_bos_requires_close_not_wick_only():
    mtf = load_module("core.multi_timeframe_swing_audit")
    diagnostic = mtf.build_timeframe_diagnostic(_frame(wick_only=True), "4h")

    assert diagnostic["bos_confirmed"] is False
