from __future__ import annotations

from tests.conftest import load_module


def test_equilibrado_profile_preserves_current_baseline(isolated_storage):
    profiles = load_module("core.trader_profiles")

    config = profiles.get_trader_profile_config(
        "Equilibrado",
        base_ticket_value=100.0,
        base_holding_minutes=60,
        base_max_open_positions=3,
    )

    assert config["name"] == "Equilibrado"
    assert config["ticket_value"] == 100.0
    assert config["holding_minutes"] == 60
    assert config["max_open_positions"] == 3
    assert config["min_signal_score"] == 0.62
    assert config["min_atr_pct"] == 0.003
    assert config["max_atr_pct"] == 0.08
    assert config["trailing_stop_atr_mult"] == 2.4
    assert config["reentry_cooldown_minutes"] == 20
    assert config["min_position_age_minutes"] == 5
    assert config["take_profit_rsi"] == 74.0
    assert config["weak_momentum_exit_threshold"] == -0.002
    assert config["trend_break_buffer_pct"] == 0.99
    assert config["hard_stop_loss_pct"] == 0.04


def test_profiles_change_risk_envelope_without_breaking_baseline(isolated_storage):
    profiles = load_module("core.trader_profiles")

    reserved = profiles.get_trader_profile_config(
        "Reservado",
        base_ticket_value=100.0,
        base_holding_minutes=60,
        base_max_open_positions=3,
    )
    aggressive = profiles.get_trader_profile_config(
        "Agressivo",
        base_ticket_value=100.0,
        base_holding_minutes=60,
        base_max_open_positions=3,
    )

    assert reserved["ticket_value"] < aggressive["ticket_value"]
    assert reserved["max_open_positions"] < aggressive["max_open_positions"]
    assert reserved["min_signal_score"] > aggressive["min_signal_score"]
    assert reserved["max_atr_pct"] < aggressive["max_atr_pct"]
    assert reserved["reentry_cooldown_minutes"] > aggressive["reentry_cooldown_minutes"]
    assert reserved["min_position_age_minutes"] > aggressive["min_position_age_minutes"]
    assert reserved["take_profit_rsi"] < aggressive["take_profit_rsi"]
    assert reserved["hard_stop_loss_pct"] < aggressive["hard_stop_loss_pct"]
