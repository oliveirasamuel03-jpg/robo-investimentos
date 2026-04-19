from __future__ import annotations

from copy import deepcopy

from core.config import MAX_HOLDING_MINUTES, MAX_TICKET, MIN_HOLDING_MINUTES, MIN_TICKET


DEFAULT_TRADER_PROFILE = "Equilibrado"
TRADER_PROFILE_ORDER = ["Reservado", "Equilibrado", "Agressivo"]


PROFILE_DEFINITIONS: dict[str, dict] = {
    "Reservado": {
        "name": "Reservado",
        "description": "Menor risco, menos operacoes",
        "accent": "#38bdf8",
        "ticket_multiplier": 0.75,
        "holding_multiplier": 0.85,
        "max_open_positions_multiplier": 0.67,
        "max_open_positions_cap": 2,
        "min_signal_score": 0.69,
        "min_atr_pct": 0.0045,
        "max_atr_pct": 0.055,
        "trailing_stop_atr_mult": 2.1,
        "rsi_entry_min": 45.0,
        "rsi_entry_max": 60.0,
        "pullback_min": -0.015,
        "pullback_max": 0.025,
        "momentum_min": -0.0025,
        "momentum_8_min": -0.01,
        "breakout_min": -0.02,
        "ma20_slope_min": -0.0015,
        "ma50_slope_min": -0.0025,
    },
    "Equilibrado": {
        "name": "Equilibrado",
        "description": "Melhor equilibrio entre risco e retorno",
        "accent": "#22c55e",
        "ticket_multiplier": 1.0,
        "holding_multiplier": 1.0,
        "max_open_positions_multiplier": 1.0,
        "max_open_positions_cap": None,
        "min_signal_score": 0.62,
        "min_atr_pct": 0.003,
        "max_atr_pct": 0.08,
        "trailing_stop_atr_mult": 2.4,
        "rsi_entry_min": 42.0,
        "rsi_entry_max": 64.0,
        "pullback_min": -0.02,
        "pullback_max": 0.035,
        "momentum_min": -0.005,
        "momentum_8_min": -0.02,
        "breakout_min": -0.04,
        "ma20_slope_min": -0.003,
        "ma50_slope_min": -0.004,
    },
    "Agressivo": {
        "name": "Agressivo",
        "description": "Maior risco, busca mais retorno",
        "accent": "#f97316",
        "ticket_multiplier": 1.2,
        "holding_multiplier": 1.2,
        "max_open_positions_multiplier": 1.34,
        "max_open_positions_cap": 6,
        "min_signal_score": 0.58,
        "min_atr_pct": 0.0025,
        "max_atr_pct": 0.11,
        "trailing_stop_atr_mult": 2.9,
        "rsi_entry_min": 40.0,
        "rsi_entry_max": 67.0,
        "pullback_min": -0.03,
        "pullback_max": 0.045,
        "momentum_min": -0.008,
        "momentum_8_min": -0.03,
        "breakout_min": -0.06,
        "ma20_slope_min": -0.005,
        "ma50_slope_min": -0.006,
    },
}


def normalize_trader_profile(profile_name: str | None) -> str:
    if not profile_name:
        return DEFAULT_TRADER_PROFILE

    raw = str(profile_name).strip().lower()
    for name in TRADER_PROFILE_ORDER:
        if raw == name.lower():
            return name
    return DEFAULT_TRADER_PROFILE


def list_trader_profiles() -> list[dict]:
    return [deepcopy(PROFILE_DEFINITIONS[name]) for name in TRADER_PROFILE_ORDER]


def get_trader_profile_config(
    profile_name: str | None,
    *,
    base_ticket_value: float,
    base_holding_minutes: int,
    base_max_open_positions: int,
) -> dict:
    name = normalize_trader_profile(profile_name)
    profile = deepcopy(PROFILE_DEFINITIONS[name])

    resolved_ticket = min(MAX_TICKET, max(MIN_TICKET, round(float(base_ticket_value) * float(profile["ticket_multiplier"]), 2)))
    resolved_holding = min(
        MAX_HOLDING_MINUTES,
        max(MIN_HOLDING_MINUTES, int(round(int(base_holding_minutes) * float(profile["holding_multiplier"])))),
    )
    resolved_max_open = max(
        1,
        int(round(int(base_max_open_positions) * float(profile["max_open_positions_multiplier"]))),
    )
    cap = profile.get("max_open_positions_cap")
    if cap is not None:
        resolved_max_open = min(int(cap), resolved_max_open)

    profile["ticket_value"] = resolved_ticket
    profile["holding_minutes"] = resolved_holding
    profile["max_open_positions"] = max(1, resolved_max_open)
    return profile
