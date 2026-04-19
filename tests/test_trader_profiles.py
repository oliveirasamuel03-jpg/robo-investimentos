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
