from __future__ import annotations

import pandas as pd

from tests.conftest import load_module


def _frame(closes: list[float], *, source: str = "market") -> pd.DataFrame:
    df = pd.DataFrame({"close": closes})
    df["open"] = df["close"].shift(1).fillna(df["close"])
    df["high"] = df[["open", "close"]].max(axis=1) * 1.01
    df["low"] = df[["open", "close"]].min(axis=1) * 0.99
    df["ma20"] = df["close"].rolling(20, min_periods=1).mean()
    df["ma50"] = df["close"].rolling(50, min_periods=1).mean()
    df["rsi"] = 50.0
    df["momentum"] = df["close"].pct_change(3).fillna(0.0)
    df["momentum_8"] = df["close"].pct_change(8).fillna(0.0)
    df["atr_pct"] = 0.02
    df["data_source"] = source
    df["provider_name"] = "twelvedata" if source == "market" else "fallback"
    return df


def test_fibonacci_levels_for_up_and_down_trends():
    module = load_module("core.market_structure_audit")

    up = module.calculate_fibonacci_levels(100.0, 200.0, "UP")
    down = module.calculate_fibonacci_levels(100.0, 200.0, "DOWN")

    assert up["fib_0"] == 200.0
    assert up["fib_382"] == 161.8
    assert up["fib_618"] == 138.2
    assert up["fib_100"] == 100.0
    assert down["fib_0"] == 100.0
    assert down["fib_382"] == 138.2
    assert down["fib_618"] == 161.8
    assert down["fib_100"] == 200.0


def test_insufficient_data_does_not_create_shadow_candidate():
    module = load_module("core.market_structure_audit")

    result = module.analyze_symbol_market_structure("BTC-USD", _frame([100.0, 101.0, 102.0]))

    assert result["market_structure_minimum_sample_met"] is False
    assert result["market_structure_shadow_candidate"] is False
    assert "dados insuficientes" in result["primary_blockers"]


def test_price_action_bos_uses_objective_close_break_rule():
    module = load_module("core.market_structure_audit")
    closes = [100.0 + (idx * 0.5) for idx in range(30)] + [112.0, 113.0, 130.0]
    frame = _frame(closes)

    action = module.detect_price_action(frame, "UP")

    assert action["bos_detected"] is True


def test_fallback_feed_does_not_generate_strong_structural_candidate():
    module = load_module("core.market_structure_audit")
    closes = [120, 115, 110, 105, 100, 105, 112, 120, 132, 146, 160, 178, 195]
    closes.extend([190, 185, 180, 175, 172, 174, 176, 179, 181] * 4)

    audit = module.build_market_structure_audit(market_data={"BTC-USD": _frame(closes, source="fallback")})

    assert audit["market_structure_candidates_count"] == 0
    assert audit["market_structure_best_candidates"][0]["market_structure_shadow_candidate"] is False
    assert "feed invalido" in audit["market_structure_best_candidates"][0]["primary_blockers"]


def test_critical_context_blocks_structural_candidate():
    module = load_module("core.market_structure_audit")
    closes = [120, 115, 110, 105, 100, 105, 112, 120, 132, 146, 160, 178, 195]
    closes.extend([190, 185, 180, 175, 172, 174, 176, 179, 181] * 4)

    audit = module.build_market_structure_audit(
        market_data={"BTC-USD": _frame(closes)},
        market_context={"market_context_status": "CRITICO"},
    )

    assert audit["market_structure_candidates_count"] == 0
    assert "contexto critico" in audit["market_structure_best_candidates"][0]["primary_blockers"]


def test_market_structure_audit_is_shadow_only_and_has_no_trade_authority():
    module = load_module("core.market_structure_audit")
    closes = [120, 115, 110, 105, 100, 105, 112, 120, 132, 146, 160, 178, 195]
    closes.extend([190, 185, 180, 175, 172, 174, 176, 179, 181] * 4)

    audit = module.build_market_structure_audit(market_data={"BTC-USD": _frame(closes)})

    assert audit["market_structure_audit_mode"] == "SHADOW_ONLY"
    assert "trade_approved" not in audit
    assert "open_position" not in audit
    assert "broker" not in audit
    assert "order" not in audit
