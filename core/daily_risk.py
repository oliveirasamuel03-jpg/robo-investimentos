from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from core.config import DAILY_LOSS_LIMIT_BRL_DEFAULT


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso(now: datetime | None = None) -> str:
    return (now or _utc_now()).astimezone(timezone.utc).isoformat()


def utc_day_key(now: datetime | None = None) -> str:
    return (now or _utc_now()).astimezone(timezone.utc).strftime("%Y-%m-%d")


def parse_iso_datetime(value: object) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        normalized = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return default


def calculate_daily_realized_pnl(trades: list[dict[str, Any]] | None, day_key: str) -> float:
    total = 0.0
    target_day = str(day_key or "").strip()
    for trade in trades or []:
        side = str(trade.get("side") or "").strip().upper()
        if side != "SELL":
            continue
        trade_at = parse_iso_datetime(trade.get("timestamp"))
        if trade_at is None or trade_at.strftime("%Y-%m-%d") != target_day:
            continue
        total += _safe_float(trade.get("realized_pnl"), default=0.0)
    return round(total, 2)


def evaluate_daily_loss_guard(
    previous_state: dict[str, Any] | None,
    trades: list[dict[str, Any]] | None,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    current_time = (now or _utc_now()).astimezone(timezone.utc)
    now_iso = _utc_now_iso(current_time)
    day_key = utc_day_key(current_time)

    previous = dict(previous_state or {})
    previous_day_key = str(previous.get("daily_loss_day_key") or "").strip()
    previous_block_active = bool(previous.get("daily_loss_block_active", False))
    previous_blocked_at = str(previous.get("daily_loss_blocked_at") or "").strip()
    previous_reset_at = str(previous.get("daily_loss_reset_at") or "").strip()

    limit_brl = max(1.0, _safe_float(previous.get("daily_loss_limit_brl"), DAILY_LOSS_LIMIT_BRL_DEFAULT))
    realized_today = calculate_daily_realized_pnl(trades, day_key=day_key)
    consumed_brl = max(0.0, -float(realized_today))
    remaining_brl = max(0.0, float(limit_brl) - float(consumed_brl))
    block_active = bool(consumed_brl >= float(limit_brl))

    day_rolled = bool(previous_day_key and previous_day_key != day_key)
    blocked_at = previous_blocked_at
    reset_at = previous_reset_at
    reason = ""
    transition = "none"

    if day_rolled:
        reset_at = now_iso
        blocked_at = ""
        transition = "reset_day"

    if block_active:
        reason = (
            f"Limite de perda diaria atingido: R$ {consumed_brl:,.2f} consumidos "
            f"de R$ {limit_brl:,.2f} no dia UTC {day_key}. Novas entradas bloqueadas."
        )
        if not previous_block_active or day_rolled or not blocked_at:
            blocked_at = now_iso
            transition = "blocked"
    else:
        if previous_block_active and not day_rolled:
            transition = "released"
        blocked_at = ""

    return {
        "daily_loss_limit_brl": round(float(limit_brl), 2),
        "daily_loss_day_key": day_key,
        "daily_realized_pnl_brl": round(float(realized_today), 2),
        "daily_loss_consumed_brl": round(float(consumed_brl), 2),
        "daily_loss_remaining_brl": round(float(remaining_brl), 2),
        "daily_loss_block_active": bool(block_active),
        "daily_loss_block_reason": reason,
        "daily_loss_blocked_at": blocked_at,
        "daily_loss_blocked_value_brl": round(float(consumed_brl), 2) if block_active else 0.0,
        "daily_loss_reset_at": reset_at,
        "last_transition": transition,
        "last_updated_at": now_iso,
    }
