from __future__ import annotations

import pandas as pd

from tests.conftest import load_module


def _up_structure_frame(mode: str = "confirmed", rows: int = 64) -> pd.DataFrame:
    records = []
    for idx in range(rows):
        close = 100.0 + idx * 0.16
        high = close * 1.003
        low = close * 0.997
        if idx == 15:
            close, high, low = 101.0, 102.0, 100.0
        if idx == 25:
            close, high, low = 111.5, 112.0, 110.0
        if idx == 34:
            close, high, low = 105.0, 106.0, 104.0
        if idx == 44:
            close, high, low = 113.0, 114.0, 111.8
        if idx == 52:
            close, high, low = 109.0, 110.0, 108.0
        if idx == rows - 2 and mode == "failed":
            close, high, low = 114.6, 115.0, 113.0
        if idx == rows - 1:
            if mode == "confirmed":
                close, high, low = 114.7, 115.2, 113.4
            elif mode == "weak":
                close, high, low = 114.05, 114.5, 113.2
            elif mode == "wick_only":
                close, high, low = 113.8, 115.2, 113.0
            elif mode == "forming":
                close, high, low = 112.0, 113.0, 111.0
            elif mode == "failed":
                close, high, low = 113.7, 114.2, 112.8
        records.append(
            {
                "datetime": pd.Timestamp("2026-01-01", tz="UTC") + pd.to_timedelta(idx, unit="h"),
                "open": close * 0.996,
                "high": high,
                "low": low,
                "close": close,
                "volume": 1000 + idx,
                "data_source": "market",
                "provider_name": "twelvedata",
            }
        )
    return pd.DataFrame(records)


def _down_structure_frame(rows: int = 64) -> pd.DataFrame:
    records = []
    for idx in range(rows):
        close = 140.0 - idx * 0.18
        high = close * 1.003
        low = close * 0.997
        if idx == 15:
            close, high, low = 137.0, 138.0, 136.0
        if idx == 25:
            close, high, low = 128.5, 130.0, 128.0
        if idx == 34:
            close, high, low = 135.0, 136.0, 134.0
        if idx == 44:
            close, high, low = 126.5, 127.5, 126.0
        if idx == 52:
            close, high, low = 131.0, 132.0, 130.0
        if idx == rows - 1:
            close, high, low = 125.6, 126.4, 125.2
        records.append(
            {
                "datetime": pd.Timestamp("2026-01-01", tz="UTC") + pd.to_timedelta(idx, unit="h"),
                "open": close * 1.004,
                "high": high,
                "low": low,
                "close": close,
                "volume": 1000 + idx,
                "data_source": "market",
                "provider_name": "twelvedata",
            }
        )
    return pd.DataFrame(records)


def _audit(h4: pd.DataFrame, h1: pd.DataFrame | None = None):
    module = load_module("core.bos_pivot_trace_audit")
    return module.build_bos_pivot_trace_audit(
        intraday_market_data={"BTC-USD": {"4h": h4, "1h": h1 if h1 is not None else h4.copy()}},
        market_data_status={"feed_status": "LIVE", "provider_effective": "twelvedata"},
        signals=[
            {
                "asset": "BTC-USD",
                "strategy_name": "trend_pullback_breakout",
                "rejection_reasons": ["score_below_minimum", "secondary_confirmation_weak"],
            }
        ],
    )


def _row(result: dict, timeframe: str) -> dict:
    rows = [row for row in result["recent_candidates"] if row.get("timeframe") == timeframe]
    assert rows
    return rows[0]


def test_short_intraday_data_returns_insufficient_data():
    result = _audit(_up_structure_frame(rows=10))
    row = _row(result, "4h")

    assert result["mode"] == "SHADOW_ONLY"
    assert row["bos_state"] == "INSUFFICIENT_DATA"
    assert row["pivot_state"] == "INSUFFICIENT_DATA"
    assert row["should_keep_blocked"] is True


def test_wick_above_structure_does_not_confirm_bos():
    result = _audit(_up_structure_frame("wick_only"))
    row = _row(result, "4h")

    assert row["bos_state"] == "BOS_BY_WICK_ONLY"
    assert row["wick_crossed_level"] is True
    assert row["close_confirmed_level"] is False
    assert row["should_keep_blocked"] is True


def test_close_above_structure_with_buffer_confirms_bos():
    result = _audit(_up_structure_frame("confirmed"))
    row = _row(result, "4h")

    assert row["bos_state"] in {"BOS_BY_CLOSE_CONFIRMED", "BOS_RETEST_CONFIRMED"}
    assert row["close_confirmed_level"] is True
    assert result["confirmed_bos_count"] >= 1
    assert "positions" not in result
    assert "pnl" not in result


def test_small_close_above_structure_is_weak_bos():
    result = _audit(_up_structure_frame("weak"))
    row = _row(result, "4h")

    assert row["bos_state"] == "BOS_BY_CLOSE_WEAK"
    assert row["close_above_or_below_level"] is True
    assert row["close_confirmed_level"] is False


def test_breakout_returning_inside_structure_marks_failed_or_high_risk():
    result = _audit(_up_structure_frame("failed"))
    row = _row(result, "4h")

    assert row["bos_state"] == "BOS_FAILED"
    assert row["false_breakout_risk"] == "HIGH"


def test_pivot_forming_is_not_triggered():
    result = _audit(_up_structure_frame("forming"))
    row = _row(result, "4h")

    assert row["pivot_state"] == "PIVOT_FORMING"
    assert row["pivot_state"] != "PIVOT_TRIGGERED"
    assert row["should_keep_blocked"] is True


def test_pivot_triggered_requires_minimum_structure():
    result = _audit(_up_structure_frame("confirmed"))
    row = _row(result, "4h")

    assert row["pivot_state"] in {"PIVOT_CONFIRMED", "PIVOT_TRIGGERED"}
    assert row["swing_high_reference"] is not None
    assert row["swing_low_reference"] is not None


def test_h1_confirmation_without_h4_marks_h1_leads_or_h4_missing():
    result = _audit(_up_structure_frame("forming"), _up_structure_frame("confirmed"))
    rows = result["recent_candidates"]
    relationship = {row["relationship_to_higher_tf"] for row in rows}

    assert relationship & {"H1_LEADS_H4", "H4_STRUCTURE_MISSING"}
    assert result["h1_bos_only_count"] >= 1


def test_h1_conflict_against_h4_is_marked_conflict():
    result = _audit(_up_structure_frame("confirmed"), _down_structure_frame())
    relationships = {row["relationship_to_higher_tf"] for row in result["recent_candidates"]}

    assert "H1_CONFLICTS_H4" in relationships
    assert all(row["should_keep_blocked"] is True for row in result["recent_candidates"])


def test_trace_audit_never_changes_real_trading_fields():
    result = _audit(_up_structure_frame("confirmed"))

    assert result["shadow_only"] is True
    assert "positions" not in result
    assert "wallet" not in result
    assert "realized_pnl" not in result
    assert "min_signal_score" not in result
