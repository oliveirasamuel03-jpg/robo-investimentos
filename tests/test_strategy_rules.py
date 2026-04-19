from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from tests.conftest import load_module


def test_should_buy_accepts_high_quality_trend_setup(isolated_storage):
    paper_engine = load_module("paper_trading_engine")
    config = paper_engine.PaperTradingConfig()

    latest = pd.Series(
        {
            "close": 105.0,
            "ma20": 103.0,
            "ma50": 100.0,
            "rsi": 54.0,
            "atr_pct": 0.018,
            "momentum": 0.009,
            "momentum_8": 0.022,
            "breakout_20": -0.006,
            "pullback_to_ma20": 0.004,
            "ma20_slope": 0.008,
            "ma50_slope": 0.006,
        }
    )

    should_buy, score = paper_engine._should_buy(latest, config)

    assert should_buy is True
    assert score >= config.min_signal_score


def test_should_buy_rejects_weak_or_overstretched_setup(isolated_storage):
    paper_engine = load_module("paper_trading_engine")
    config = paper_engine.PaperTradingConfig()

    latest = pd.Series(
        {
            "close": 99.0,
            "ma20": 100.5,
            "ma50": 101.0,
            "rsi": 72.0,
            "atr_pct": 0.11,
            "momentum": -0.012,
            "momentum_8": -0.026,
            "breakout_20": -0.08,
            "pullback_to_ma20": -0.04,
            "ma20_slope": -0.008,
            "ma50_slope": -0.006,
        }
    )

    should_buy, score = paper_engine._should_buy(latest, config)

    assert should_buy is False
    assert score < 0.7


def test_should_sell_uses_trailing_stop_and_position_timeout(isolated_storage):
    paper_engine = load_module("paper_trading_engine")
    config = paper_engine.PaperTradingConfig(holding_minutes=60)

    position = {
        "avg_price": 100.0,
        "peak_price": 110.0,
        "trailing_stop": 106.0,
        "atr_pct": 0.02,
        "opened_at": (datetime.now(timezone.utc) - timedelta(minutes=75)).isoformat(),
    }

    latest = pd.Series(
        {
            "close": 105.0,
            "ma20": 106.5,
            "ma50": 101.5,
            "rsi": 61.0,
            "momentum": 0.002,
            "atr_pct": 0.02,
        }
    )

    should_sell, reason = paper_engine._should_sell(position, latest, config.holding_minutes, config)

    assert should_sell is True
    assert reason in {"trailing stop", "holding expirado"}


def test_position_size_notional_scales_down_when_volatility_is_high(isolated_storage):
    paper_engine = load_module("paper_trading_engine")
    config = paper_engine.PaperTradingConfig(ticket_value=100.0)

    calm_candidate = {"score": 0.79, "atr_pct": 0.012}
    volatile_candidate = {"score": 0.79, "atr_pct": 0.05}

    calm_notional = paper_engine._position_size_notional(calm_candidate, config, cash=1000.0)
    volatile_notional = paper_engine._position_size_notional(volatile_candidate, config, cash=1000.0)

    assert calm_notional >= volatile_notional
    assert volatile_notional >= config.min_trade_notional * 0.65
