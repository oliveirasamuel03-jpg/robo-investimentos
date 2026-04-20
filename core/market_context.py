from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def is_crypto_symbol(symbol: str | None) -> bool:
    normalized = str(symbol or "").strip().upper()
    return normalized.endswith("-USD") or normalized in {"BTCUSD", "ETHUSD", "SOLUSD"}


def _clip(value: float, floor: float, ceil: float) -> float:
    return max(floor, min(ceil, float(value)))


def _empty_context(reason: str = "Contexto indisponivel; robo segue em modo neutro.") -> dict[str, Any]:
    return {
        "market_context_status": "NEUTRO",
        "market_context_reason": reason,
        "market_context_updated_at": _utc_now_iso(),
        "market_context_score": 50.0,
        "market_context_impact": "Sem restricao adicional sobre sinais de cripto.",
        "market_context_regime": "indefinido",
        "watchlist_consistency": None,
        "btc_move_pct": None,
        "btc_volatility_pct": None,
    }


def _latest_row(frame: pd.DataFrame) -> pd.Series | None:
    if frame is None or frame.empty:
        return None
    try:
        return frame.iloc[-1]
    except Exception:
        return None


def _safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _trend_alignment(frame: pd.DataFrame) -> str:
    latest = _latest_row(frame)
    if latest is None:
        return "indefinido"

    close = _safe_float(latest.get("close"))
    ma20 = _safe_float(latest.get("ma20"))
    ma50 = _safe_float(latest.get("ma50"))
    ma20_slope = _safe_float(latest.get("ma20_slope")) or 0.0
    ma50_slope = _safe_float(latest.get("ma50_slope")) or 0.0

    if None in (close, ma20, ma50):
        return "indefinido"
    if close >= ma20 >= ma50 and ma20_slope >= 0 and ma50_slope >= 0:
        return "tendencia"
    if close < ma50:
        return "baixa"
    return "lateral"


def get_market_context(data: dict[str, pd.DataFrame] | None) -> dict[str, Any]:
    try:
        frames = {str(symbol).upper(): frame for symbol, frame in (data or {}).items() if isinstance(frame, pd.DataFrame)}
        crypto_frames = {symbol: frame for symbol, frame in frames.items() if is_crypto_symbol(symbol)}
        if not crypto_frames:
            return _empty_context("Sem ativos cripto na watchlist desta rodada; contexto fica neutro.")

        btc_frame = crypto_frames.get("BTC-USD") or next(iter(crypto_frames.values()))
        latest = _latest_row(btc_frame)
        if latest is None:
            return _empty_context()

        btc_close = _safe_float(latest.get("close"))
        btc_ma20 = _safe_float(latest.get("ma20"))
        btc_ma50 = _safe_float(latest.get("ma50"))
        atr_pct = _safe_float(latest.get("atr_pct")) or 0.0
        recent_close = _safe_float(btc_frame["close"].iloc[-21]) if len(btc_frame) >= 21 else btc_close
        btc_move_pct = None
        if btc_close not in (None, 0) and recent_close not in (None, 0):
            btc_move_pct = (float(btc_close) / float(recent_close)) - 1.0

        regime = _trend_alignment(btc_frame)
        consistency_hits = 0
        consistency_total = 0
        for frame in crypto_frames.values():
            alignment = _trend_alignment(frame)
            if alignment == "indefinido":
                continue
            consistency_total += 1
            if alignment == "tendencia":
                consistency_hits += 1
        watchlist_consistency = (consistency_hits / consistency_total) if consistency_total > 0 else None

        score = 50.0
        reasons: list[str] = []

        if regime == "tendencia":
            score += 18
            reasons.append("BTC em tendencia de alta.")
        elif regime == "baixa":
            score -= 24
            reasons.append("BTC abaixo da estrutura principal.")
        else:
            score -= 8
            reasons.append("BTC em regime mais lateral.")

        if btc_move_pct is not None:
            if btc_move_pct >= 0.08:
                score += 14
                reasons.append("Movimento recente do BTC positivo.")
            elif btc_move_pct <= -0.08:
                score -= 18
                reasons.append("Movimento recente do BTC negativo.")

        if atr_pct >= 0.09:
            score -= 18
            reasons.append("Volatilidade cripto muito alta.")
        elif atr_pct >= 0.06:
            score -= 8
            reasons.append("Volatilidade cripto elevada.")
        else:
            score += 6
            reasons.append("Volatilidade sob controle.")

        if watchlist_consistency is not None:
            if watchlist_consistency >= 0.67:
                score += 10
                reasons.append("Watchlist cripto com leitura consistente.")
            elif watchlist_consistency <= 0.34:
                score -= 15
                reasons.append("Watchlist cripto inconsistente.")

        score = round(_clip(score, 0.0, 100.0), 2)

        if score >= 75:
            status = "FAVORAVEL"
            impact = "Sinais de cripto seguem sem restricao extra alem do filtro normal."
        elif score >= 45:
            status = "NEUTRO"
            impact = "Sinais de cripto seguem com filtro base, sem penalidade adicional."
        elif score >= 25:
            status = "DESFAVORAVEL"
            impact = "Score minimo de cripto sobe e sinais fracos passam a ser bloqueados."
        else:
            status = "CRITICO"
            impact = "Novas entradas de cripto ficam bloqueadas ate o contexto voltar a estabilizar."

        return {
            "market_context_status": status,
            "market_context_reason": " ".join(reasons).strip() or "Contexto neutro.",
            "market_context_updated_at": _utc_now_iso(),
            "market_context_score": score,
            "market_context_impact": impact,
            "market_context_regime": regime,
            "watchlist_consistency": None if watchlist_consistency is None else round(float(watchlist_consistency), 4),
            "btc_move_pct": None if btc_move_pct is None else round(float(btc_move_pct), 6),
            "btc_volatility_pct": round(float(atr_pct), 6),
        }
    except Exception:
        return _empty_context("Falha ao calcular contexto; filtro volta ao modo neutro.")


def apply_context_filter(signal: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    base_score = float(signal.get("score", 0.0) or 0.0)
    base_min_signal_score = float(signal.get("base_min_signal_score", 0.0) or 0.0)
    asset = str(signal.get("asset") or "").upper()
    status = str(context.get("market_context_status") or "NEUTRO").upper()

    if not is_crypto_symbol(asset):
        return {
            "adjusted_score": round(base_score, 4),
            "effective_min_signal_score": round(base_min_signal_score, 4),
            "blocked": False,
            "blocked_reason": "",
            "impact_message": "Sem ajuste de contexto para este ativo.",
            "context_status": status,
        }

    adjusted_score = base_score
    effective_min_signal_score = base_min_signal_score
    blocked = False
    blocked_reason = ""

    if status == "DESFAVORAVEL":
        adjusted_score = base_score - 0.06
        effective_min_signal_score = base_min_signal_score + 0.06
        if adjusted_score < effective_min_signal_score:
            blocked = True
            blocked_reason = "context_blocked"
    elif status == "CRITICO":
        adjusted_score = base_score - 0.12
        effective_min_signal_score = base_min_signal_score + 0.12
        blocked = True
        blocked_reason = "context_blocked"
    elif status == "FAVORAVEL":
        adjusted_score = min(base_score + 0.02, 1.0)

    return {
        "adjusted_score": round(adjusted_score, 4),
        "effective_min_signal_score": round(effective_min_signal_score, 4),
        "blocked": blocked,
        "blocked_reason": blocked_reason,
        "impact_message": str(context.get("market_context_impact") or ""),
        "context_status": status,
    }
