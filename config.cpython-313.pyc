from __future__ import annotations

from datetime import datetime, timedelta


def compute_expiry_timestamp(holding_minutes: int) -> str:
    return (datetime.utcnow() + timedelta(minutes=holding_minutes)).isoformat()


def is_position_expired(expiry_ts: str | None) -> bool:
    if not expiry_ts:
        return False
    return datetime.utcnow() >= datetime.fromisoformat(expiry_ts)
