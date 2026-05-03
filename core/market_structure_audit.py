from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any

import pandas as pd

MODE = "SHADOW_ONLY"
MIN_BARS = 40
LOOKBACK_BARS = 90
SWING_WINDOW = 3
MIN_SWING_DISTANCE_PCT = 0.025
SHADOW_CANDIDATE_SCORE = 0.70
MAX_BEST_CANDIDATES = 10

FIB_RATIOS = {
    "fib_0": 0.0,
    "fib_236": 0.236,
    "fib_382": 0.382,
    "fib_500": 0.500,
    "fib_618": 0.618,
    "fib_764": 0.764,
    "fib_100": 1.0,
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_float(value: Any, default: float | None = 0.0) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _round(value: Any, digits: int = 6) -> float | None:
    numeric = _as_float(value, None)
    return None if numeric is None else round(float(numeric), digits)


def _last_text(frame: pd.DataFrame, column: str, fallback: str = "") -> str:
    if frame.empty or column not in frame.columns:
        return fallback
    values = [str(item).strip() for item in frame[column].dropna().tolist() if str(item).strip()]
    return values[-1] if values else fallback


def _is_live_frame(frame: pd.DataFrame) -> bool:
    return _last_text(frame, "data_source", "unknown").lower() == "market"


def _clean_frame(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    required = ["high", "low", "close"]
    if any(column not in frame.columns for column in required):
        return pd.DataFrame()
    data = frame.copy()
    if "open" not in data.columns:
        data["open"] = data["close"]
    for column in ("open", "high", "low", "close"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["open", "high", "low", "close"])
    return data.tail(LOOKBACK_BARS).reset_index(drop=True)


def _find_swings(frame: pd.DataFrame, *, window: int = SWING_WINDOW) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    highs: list[dict[str, Any]] = []
    lows: list[dict[str, Any]] = []
    if len(frame) < (window * 2) + 1:
        return highs, lows
    for idx in range(window, len(frame) - window):
        high = float(frame.loc[idx, "high"])
        low = float(frame.loc[idx, "low"])
        high_window = frame["high"].iloc[idx - window : idx + window + 1]
        low_window = frame["low"].iloc[idx - window : idx + window + 1]
        if high >= float(high_window.max()) and high > float(high_window.drop(high_window.index[window]).max()):
            highs.append({"idx": idx, "price": high})
        if low <= float(low_window.min()) and low < float(low_window.drop(low_window.index[window]).min()):
            lows.append({"idx": idx, "price": low})
    return highs, lows


def detect_swing_points(frame: pd.DataFrame | None, *, window: int = SWING_WINDOW) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return _find_swings(_clean_frame(frame), window=window)


def _select_structure(frame: pd.DataFrame, highs: list[dict[str, Any]], lows: list[dict[str, Any]]) -> dict[str, Any]:
    if not highs or not lows:
        return {"valid": False, "reason": "sem swing high/low valido", "direction": "INCONCLUSIVE"}

    latest_high = highs[-1]
    latest_low = lows[-1]
    ma20 = _as_float(frame["ma20"].iloc[-1], None) if "ma20" in frame.columns else None
    ma50 = _as_float(frame["ma50"].iloc[-1], None) if "ma50" in frame.columns else None

    if int(latest_low["idx"]) < int(latest_high["idx"]):
        direction = "UP"
        low = latest_low
        high = latest_high
    elif int(latest_high["idx"]) < int(latest_low["idx"]):
        direction = "DOWN"
        low = latest_low
        high = latest_high
    elif ma20 is not None and ma50 is not None and ma20 >= ma50:
        direction = "UP"
        low = latest_low
        high = latest_high
    else:
        direction = "DOWN"
        low = latest_low
        high = latest_high

    current_price = float(frame["close"].iloc[-1])
    swing_low = float(low["price"])
    swing_high = float(high["price"])
    distance_pct = abs(swing_high - swing_low) / max(abs(current_price), 1e-9)
    if swing_high <= swing_low or distance_pct < MIN_SWING_DISTANCE_PCT:
        return {
            "valid": False,
            "reason": "distancia minima entre topo/fundo nao atendida",
            "direction": "INCONCLUSIVE",
            "swing_low": swing_low,
            "swing_high": swing_high,
            "distance_pct": distance_pct,
        }

    return {
        "valid": True,
        "reason": "estrutura objetiva encontrada",
        "direction": direction,
        "swing_low": swing_low,
        "swing_high": swing_high,
        "distance_pct": distance_pct,
    }


def calculate_fibonacci_levels(swing_low: float, swing_high: float, direction: str) -> dict[str, float]:
    distance = abs(float(swing_high) - float(swing_low))
    if str(direction).upper() == "DOWN":
        return {key: round(float(swing_low) + (distance * ratio), 6) for key, ratio in FIB_RATIOS.items()}
    return {key: round(float(swing_high) - (distance * ratio), 6) for key, ratio in FIB_RATIOS.items()}


def _fib_zone(price: float, levels: dict[str, float], direction: str) -> str:
    if not levels:
        return "INCONCLUSIVE"
    fib0 = float(levels["fib_0"])
    fib382 = float(levels["fib_382"])
    fib618 = float(levels["fib_618"])
    fib764 = float(levels["fib_764"])
    fib100 = float(levels["fib_100"])
    price = float(price)
    if str(direction).upper() == "DOWN":
        if price < fib0:
            return "BREAKOUT_ZONE"
        if fib0 <= price <= fib382:
            return "SHALLOW_ZONE"
        if fib382 < price <= fib618:
            return "MEDIUM_ZONE"
        if fib618 < price <= fib764:
            return "DEEP_ZONE"
        if price > fib100:
            return "INVALIDATION_ZONE"
        return "NEUTRAL_ZONE"
    if price > fib0:
        return "BREAKOUT_ZONE"
    if fib382 <= price <= fib0:
        return "SHALLOW_ZONE"
    if fib618 <= price < fib382:
        return "MEDIUM_ZONE"
    if fib764 <= price < fib618:
        return "DEEP_ZONE"
    if price < fib100:
        return "INVALIDATION_ZONE"
    return "NEUTRAL_ZONE"


def _prior_range(frame: pd.DataFrame, bars: int = 20) -> tuple[float, float]:
    prior = frame.iloc[max(0, len(frame) - bars - 1) : max(1, len(frame) - 1)]
    if prior.empty:
        return float(frame["high"].max()), float(frame["low"].min())
    return float(prior["high"].max()), float(prior["low"].min())


def _detect_pivot_bos(frame: pd.DataFrame, direction: str) -> dict[str, bool]:
    if len(frame) < 3:
        return {"pivot_detected": False, "bos_detected": False, "false_breakout_risk": False}
    last = frame.iloc[-1]
    previous = frame.iloc[-2]
    close = float(last["close"])
    open_ = float(last.get("open", close))
    high = float(last["high"])
    low = float(last["low"])
    prior_high, prior_low = _prior_range(frame)
    buffer = 0.001

    if str(direction).upper() == "DOWN":
        bearish_pivot = close < float(previous["low"]) and close < open_
        bos = close < prior_low * (1.0 - buffer)
        false_breakout = low < prior_low * (1.0 - buffer) and close >= prior_low
        return {
            "pivot_detected": bool(bearish_pivot),
            "bos_detected": bool(bos),
            "false_breakout_risk": bool(false_breakout),
        }

    bullish_pivot = close > float(previous["high"]) and close > open_
    bos = close > prior_high * (1.0 + buffer)
    false_breakout = high > prior_high * (1.0 + buffer) and close <= prior_high
    return {
        "pivot_detected": bool(bullish_pivot),
        "bos_detected": bool(bos),
        "false_breakout_risk": bool(false_breakout),
    }


def detect_price_action(frame: pd.DataFrame | None, direction: str) -> dict[str, bool]:
    data = _clean_frame(frame)
    return _detect_pivot_bos(data, direction) if not data.empty else {
        "pivot_detected": False,
        "bos_detected": False,
        "false_breakout_risk": False,
    }


def _classify_regime(frame: pd.DataFrame, structure: dict[str, Any]) -> str:
    if len(frame) < MIN_BARS or not bool(structure.get("valid")):
        return "INCONCLUSIVE"
    close = float(frame["close"].iloc[-1])
    atr_avg = None
    if "atr_pct" in frame.columns:
        atr_avg = _as_float(pd.to_numeric(frame["atr_pct"], errors="coerce").tail(20).mean(), None)
    if atr_avg is None:
        candle_range = (pd.to_numeric(frame["high"], errors="coerce") - pd.to_numeric(frame["low"], errors="coerce")) / frame["close"].replace(0, pd.NA)
        atr_avg = _as_float(candle_range.tail(20).mean(), 0.0)
    ma20 = _as_float(frame["ma20"].iloc[-1], None) if "ma20" in frame.columns else None
    ma50 = _as_float(frame["ma50"].iloc[-1], None) if "ma50" in frame.columns else None
    trend_strength = abs(float(ma20 or close) - float(ma50 or close)) / max(abs(close), 1e-9)
    swing_distance = float(structure.get("distance_pct", 0.0) or 0.0)
    if float(atr_avg or 0.0) >= 0.065:
        return "HIGH_VOLATILITY"
    if float(atr_avg or 0.0) <= 0.008:
        return "LOW_VOLATILITY"
    if trend_strength >= 0.018 or swing_distance >= 0.08:
        return "TREND"
    if swing_distance <= 0.05:
        return "RANGE"
    return "INCONCLUSIVE"


def _signal_for_symbol(signals_by_symbol: dict[str, dict[str, Any]], symbol: str) -> dict[str, Any]:
    return dict(signals_by_symbol.get(str(symbol).upper(), {}) or {})


def _score_components(
    *,
    structure_valid: bool,
    fib_zone: str,
    pivot_detected: bool,
    bos_detected: bool,
    false_breakout_risk: bool,
    feed_live: bool,
    context_status: str,
    regime: str,
    frame: pd.DataFrame,
    direction: str,
) -> tuple[dict[str, float], float, list[str]]:
    components = {
        "structure_clear": 0.25 if structure_valid else 0.0,
        "fib_zone_valid": 0.20 if fib_zone in {"MEDIUM_ZONE", "DEEP_ZONE"} else (0.08 if fib_zone in {"SHALLOW_ZONE", "BREAKOUT_ZONE"} else 0.0),
        "pivot_confirmed": 0.15 if pivot_detected else 0.0,
        "bos_confirmed": 0.20 if bos_detected else 0.0,
        "regime_fit": 0.10 if regime == "TREND" else (0.04 if regime in {"RANGE", "LOW_VOLATILITY"} else 0.0),
        "feed_live": 0.10 if feed_live else 0.0,
        "context_safe": 0.10 if str(context_status).upper() != "CRITICO" else 0.0,
    }
    penalties: list[str] = []
    score = sum(components.values())
    if false_breakout_risk:
        score -= 0.20
        penalties.append("false_breakout_risk")
    rsi = _as_float(frame["rsi"].iloc[-1], None) if "rsi" in frame.columns else None
    momentum = _as_float(frame["momentum"].iloc[-1], None) if "momentum" in frame.columns else None
    if rsi is not None and (rsi < 28.0 or rsi > 76.0):
        score -= 0.07
        penalties.append("rsi_extreme")
    if momentum is not None:
        if str(direction).upper() == "UP" and momentum < -0.015:
            score -= 0.07
            penalties.append("momentum_against_structure")
        if str(direction).upper() == "DOWN" and momentum > 0.015:
            score -= 0.07
            penalties.append("momentum_against_structure")
    if regime == "HIGH_VOLATILITY":
        score -= 0.08
        penalties.append("high_volatility_regime")
    return components, round(max(0.0, min(1.0, score)), 4), penalties


def default_market_structure_audit_state(reason: str = "No market structure audit data yet.") -> dict[str, Any]:
    return {
        "market_structure_audit_enabled": True,
        "market_structure_audit_mode": MODE,
        "market_structure_audit_last_run_at": "",
        "market_structure_top_symbol": "",
        "market_structure_top_score": None,
        "market_structure_top_zone": "",
        "market_structure_top_recommendation": "sem dados suficientes",
        "market_structure_candidates_count": 0,
        "market_structure_best_candidates": [],
        "market_structure_setup_confluence": {},
        "market_structure_fib_summary": {},
        "market_structure_blockers_summary": {},
        "market_structure_regime_summary": {},
        "market_structure_data_sufficiency": "NO_DATA",
        "market_structure_minimum_sample_met": False,
        "market_structure_why_no_candidate": reason,
    }


def analyze_symbol_market_structure(
    symbol: str,
    frame: pd.DataFrame | None,
    *,
    signal: dict[str, Any] | None = None,
    context_status: str = "NEUTRO",
) -> dict[str, Any]:
    data = _clean_frame(frame)
    feed_live = _is_live_frame(frame if frame is not None else pd.DataFrame())
    current_price = _round(data["close"].iloc[-1]) if not data.empty else None
    minimum_sample_met = len(data) >= MIN_BARS
    primary_blockers: list[str] = []
    secondary_blockers: list[str] = []
    confluence_notes: list[str] = []

    if not minimum_sample_met:
        primary_blockers.append("dados insuficientes")
    if not feed_live:
        primary_blockers.append("feed invalido")
    if str(context_status).upper() == "CRITICO":
        primary_blockers.append("contexto critico")

    highs, lows = _find_swings(data) if minimum_sample_met else ([], [])
    structure = _select_structure(data, highs, lows) if minimum_sample_met else {"valid": False, "reason": "dados insuficientes", "direction": "INCONCLUSIVE"}
    if not bool(structure.get("valid")):
        primary_blockers.append(str(structure.get("reason") or "sem swing high/low valido"))

    direction = str(structure.get("direction") or "INCONCLUSIVE")
    swing_low = _as_float(structure.get("swing_low"), None)
    swing_high = _as_float(structure.get("swing_high"), None)
    levels = calculate_fibonacci_levels(float(swing_low), float(swing_high), direction) if swing_low is not None and swing_high is not None and direction in {"UP", "DOWN"} else {}
    fib_zone = _fib_zone(float(current_price or 0.0), levels, direction) if current_price is not None and levels else "INCONCLUSIVE"
    pivot_bos = _detect_pivot_bos(data, direction) if minimum_sample_met and direction in {"UP", "DOWN"} else {
        "pivot_detected": False,
        "bos_detected": False,
        "false_breakout_risk": False,
    }
    regime = _classify_regime(data, structure)
    is_pullback = fib_zone in {"MEDIUM_ZONE", "DEEP_ZONE"}
    is_reaction = is_pullback and bool(pivot_bos["pivot_detected"])
    is_breakout_zone = fib_zone == "BREAKOUT_ZONE" or bool(pivot_bos["bos_detected"])

    if fib_zone not in {"MEDIUM_ZONE", "DEEP_ZONE", "BREAKOUT_ZONE"}:
        secondary_blockers.append("preco fora de zona Fibonacci util")
    if not bool(pivot_bos["pivot_detected"]):
        secondary_blockers.append("sem pivo")
    if not bool(pivot_bos["bos_detected"]):
        secondary_blockers.append("sem BOS")
    if bool(pivot_bos["false_breakout_risk"]):
        secondary_blockers.append("risco de falso rompimento")
    if regime in {"RANGE", "HIGH_VOLATILITY", "INCONCLUSIVE"}:
        secondary_blockers.append("regime pouco favoravel")

    components, score, penalties = _score_components(
        structure_valid=bool(structure.get("valid")),
        fib_zone=fib_zone,
        pivot_detected=bool(pivot_bos["pivot_detected"]),
        bos_detected=bool(pivot_bos["bos_detected"]),
        false_breakout_risk=bool(pivot_bos["false_breakout_risk"]),
        feed_live=feed_live,
        context_status=context_status,
        regime=regime,
        frame=data if not data.empty else pd.DataFrame({"close": [0.0]}),
        direction=direction,
    )
    secondary_blockers.extend(penalties)

    signal_payload = dict(signal or {})
    reasons = set(str(item) for item in list(signal_payload.get("rejection_reasons", []) or []))
    confirms_trend_pullback = bool(direction == "UP" and is_pullback and not pivot_bos["false_breakout_risk"])
    confirms_breakout = bool(pivot_bos["bos_detected"] and fib_zone in {"BREAKOUT_ZONE", "SHALLOW_ZONE"})
    confirms_reversal = bool(fib_zone == "DEEP_ZONE" and pivot_bos["pivot_detected"])
    disagrees = bool(score < 0.45 or fib_zone == "INVALIDATION_ZONE")
    improves_quality = bool(score >= 0.65 and ("no_setup_eligible" in reasons or "breakout_not_confirmed" in reasons or "confidence_too_low" in reasons))
    weak_confluence = not any((confirms_trend_pullback, confirms_breakout, confirms_reversal))
    if weak_confluence:
        secondary_blockers.append("confluencia fraca")
    confluence_notes.append("estrutura confirma pullback" if confirms_trend_pullback else "estrutura nao confirma pullback")
    confluence_notes.append("BOS confirma breakout" if confirms_breakout else "BOS nao confirmou breakout")
    confluence_notes.append("reversal em zona tecnica" if confirms_reversal else "reversal sem confluencia suficiente")

    shadow_candidate = bool(
        score >= SHADOW_CANDIDATE_SCORE
        and not primary_blockers
        and not bool(pivot_bos["false_breakout_risk"])
        and (is_pullback or is_breakout_zone or is_reaction)
        and not weak_confluence
    )
    if shadow_candidate:
        status = "STRUCTURAL_CANDIDATE_SHADOW"
        recommendation = "estrutura boa, setup atual rigido"
        why_no_candidate = ""
    elif primary_blockers:
        status = "BLOCKED"
        recommendation = "sem dados suficientes" if not minimum_sample_met else "observar mais"
        why_no_candidate = primary_blockers[0]
    elif fib_zone not in {"MEDIUM_ZONE", "DEEP_ZONE", "BREAKOUT_ZONE"}:
        status = "NO_CANDIDATE"
        recommendation = "Fibonacci nao ajuda"
        why_no_candidate = "preco fora de zona Fibonacci util"
    elif weak_confluence:
        status = "NO_CANDIDATE"
        recommendation = "manter so como filtro"
        why_no_candidate = "confluencia fraca"
    elif bool(pivot_bos["false_breakout_risk"]):
        status = "NO_CANDIDATE"
        recommendation = "observar mais"
        why_no_candidate = "risco de falso rompimento"
    else:
        status = "NO_CANDIDATE"
        recommendation = "observar mais"
        why_no_candidate = "sem pivo ou BOS suficiente"

    return {
        "symbol": str(symbol),
        "timeframe_source": "worker_cycle",
        "market_regime": regime,
        "market_structure_data_sufficiency": "OK" if minimum_sample_met else "INSUFFICIENT_DATA",
        "market_structure_minimum_sample_met": bool(minimum_sample_met),
        "market_structure_why_no_candidate": why_no_candidate,
        "structure_direction": direction,
        "relevant_swing_low": _round(swing_low),
        "relevant_swing_high": _round(swing_high),
        "current_price": current_price,
        **{key: _round(value) for key, value in levels.items()},
        "current_fib_zone": fib_zone,
        "is_in_pullback_zone": bool(is_pullback),
        "is_in_reaction_zone": bool(is_reaction),
        "is_in_breakout_zone": bool(is_breakout_zone),
        "bos_detected": bool(pivot_bos["bos_detected"]),
        "pivot_detected": bool(pivot_bos["pivot_detected"]),
        "false_breakout_risk": bool(pivot_bos["false_breakout_risk"]),
        "structure_score": score,
        "market_structure_score": score,
        "market_structure_score_components": components,
        "market_structure_score_reason": "; ".join(primary_blockers + secondary_blockers) or "estrutura objetiva avaliada",
        "market_structure_shadow_candidate": bool(shadow_candidate),
        "market_structure_shadow_rank": None,
        "structure_status": status,
        "structure_recommendation": recommendation,
        "primary_blockers": list(dict.fromkeys(primary_blockers)),
        "secondary_blockers": list(dict.fromkeys(secondary_blockers)),
        "confluence_notes": list(dict.fromkeys(confluence_notes)),
        "structure_confirms_trend_pullback": bool(confirms_trend_pullback),
        "structure_confirms_breakout": bool(confirms_breakout),
        "structure_confirms_reversal": bool(confirms_reversal),
        "structure_disagrees_with_current_setup": bool(disagrees),
        "structure_would_improve_candidate_quality": bool(improves_quality),
        "structure_should_become_filter_future": bool(score >= 0.60 and not shadow_candidate),
        "structure_should_become_score_component_future": bool(shadow_candidate or score >= 0.70),
    }


def build_market_structure_audit(
    *,
    market_data: dict[str, pd.DataFrame] | None,
    signals: list[dict[str, Any]] | None = None,
    market_context: dict[str, Any] | None = None,
    enabled: bool = True,
) -> dict[str, Any]:
    if not enabled:
        return default_market_structure_audit_state("Market structure audit disabled.")
    context_status = str((market_context or {}).get("market_context_status") or "NEUTRO")
    signals_by_symbol = {
        str(signal.get("asset") or "").upper(): dict(signal)
        for signal in list(signals or [])
        if str(signal.get("asset") or "").strip()
    }
    rows = [
        analyze_symbol_market_structure(
            symbol,
            frame,
            signal=_signal_for_symbol(signals_by_symbol, symbol),
            context_status=context_status,
        )
        for symbol, frame in dict(market_data or {}).items()
    ]
    if not rows:
        state = default_market_structure_audit_state("dados insuficientes")
        state["market_structure_audit_last_run_at"] = _utc_now_iso()
        return state

    ranked = sorted(rows, key=lambda row: float(row.get("market_structure_score", 0.0) or 0.0), reverse=True)
    for idx, row in enumerate(ranked, start=1):
        row["market_structure_shadow_rank"] = idx
    candidates = [row for row in ranked if bool(row.get("market_structure_shadow_candidate", False))]
    top = candidates[0] if candidates else ranked[0]
    blockers = Counter()
    fib_zones = Counter()
    regimes = Counter()
    confluence = {
        "trend_pullback": 0,
        "breakout": 0,
        "reversal": 0,
        "disagrees": 0,
        "would_improve_quality": 0,
    }
    for row in rows:
        fib_zones[str(row.get("current_fib_zone") or "INCONCLUSIVE")] += 1
        regimes[str(row.get("market_regime") or "INCONCLUSIVE")] += 1
        for blocker in list(row.get("primary_blockers", []) or []) + list(row.get("secondary_blockers", []) or []):
            blockers[str(blocker)] += 1
        confluence["trend_pullback"] += int(bool(row.get("structure_confirms_trend_pullback")))
        confluence["breakout"] += int(bool(row.get("structure_confirms_breakout")))
        confluence["reversal"] += int(bool(row.get("structure_confirms_reversal")))
        confluence["disagrees"] += int(bool(row.get("structure_disagrees_with_current_setup")))
        confluence["would_improve_quality"] += int(bool(row.get("structure_would_improve_candidate_quality")))

    minimum_sample_met = any(bool(row.get("market_structure_minimum_sample_met")) for row in rows)
    why_no_candidate = "" if candidates else str(top.get("market_structure_why_no_candidate") or "sem candidato estrutural forte")
    return {
        "market_structure_audit_enabled": True,
        "market_structure_audit_mode": MODE,
        "market_structure_audit_last_run_at": _utc_now_iso(),
        "market_structure_top_symbol": str(top.get("symbol") or ""),
        "market_structure_top_score": top.get("market_structure_score"),
        "market_structure_top_zone": str(top.get("current_fib_zone") or ""),
        "market_structure_top_recommendation": str(top.get("structure_recommendation") or "sem dados suficientes"),
        "market_structure_candidates_count": len(candidates),
        "market_structure_best_candidates": ranked[:MAX_BEST_CANDIDATES],
        "market_structure_setup_confluence": confluence,
        "market_structure_fib_summary": dict(fib_zones.most_common()),
        "market_structure_blockers_summary": dict(blockers.most_common(8)),
        "market_structure_regime_summary": dict(regimes.most_common()),
        "market_structure_data_sufficiency": "OK" if minimum_sample_met else "INSUFFICIENT_DATA",
        "market_structure_minimum_sample_met": bool(minimum_sample_met),
        "market_structure_why_no_candidate": why_no_candidate,
    }
