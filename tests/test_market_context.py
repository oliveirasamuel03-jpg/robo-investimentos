from __future__ import annotations

import pandas as pd

from tests.conftest import load_module


def _crypto_frame(trend: str = "up") -> pd.DataFrame:
    if trend == "up":
        close = [100, 102, 104, 106, 108]
        ma20 = [98, 99, 100, 102, 104]
        ma50 = [95, 96, 97, 98, 100]
        ma20_slope = [0.0, 0.005, 0.006, 0.007, 0.008]
        ma50_slope = [0.0, 0.003, 0.004, 0.0045, 0.005]
    elif trend == "down":
        close = [108, 106, 104, 100, 96]
        ma20 = [109, 108, 107, 105, 103]
        ma50 = [110, 109, 108, 107, 106]
        ma20_slope = [0.0, -0.004, -0.005, -0.006, -0.007]
        ma50_slope = [0.0, -0.003, -0.004, -0.005, -0.006]
    else:
        close = [100, 101, 100, 101, 100]
        ma20 = [100, 100, 100, 100, 100]
        ma50 = [99, 99, 99, 99, 99]
        ma20_slope = [0.0, 0.0, 0.0, 0.0, 0.0]
        ma50_slope = [0.0, 0.0, 0.0, 0.0, 0.0]

    return pd.DataFrame(
        {
            "datetime": pd.date_range("2026-04-01", periods=5, freq="D"),
            "close": close,
            "ma20": ma20,
            "ma50": ma50,
            "ma20_slope": ma20_slope,
            "ma50_slope": ma50_slope,
            "atr_pct": [0.03, 0.03, 0.03, 0.03, 0.03],
        }
    )


def test_get_market_context_returns_favorable_for_consistent_crypto_watchlist(isolated_storage):
    market_context = load_module("core.market_context")

    context = market_context.get_market_context(
        {
            "BTC-USD": _crypto_frame("up"),
            "ETH-USD": _crypto_frame("up"),
        }
    )

    assert context["market_context_status"] == "FAVORAVEL"
    assert context["market_context_score"] >= 75


def test_apply_context_filter_blocks_weak_crypto_signal_in_bad_context(isolated_storage):
    market_context = load_module("core.market_context")

    result = market_context.apply_context_filter(
        {"asset": "BTC-USD", "score": 0.72, "base_min_signal_score": 0.70},
        {
            "market_context_status": "DESFAVORAVEL",
            "market_context_impact": "Score minimo de cripto sobe.",
        },
    )

    assert result["blocked"] is True
    assert result["blocked_reason"] == "context_blocked"
    assert result["effective_min_signal_score"] > 0.70


def test_get_market_context_falls_back_to_neutral_on_invalid_input(isolated_storage):
    market_context = load_module("core.market_context")

    context = market_context.get_market_context({"BTC-USD": None})

    assert context["market_context_status"] == "NEUTRO"
