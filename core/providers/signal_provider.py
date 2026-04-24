from __future__ import annotations

from dataclasses import dataclass

from core.config import (
    EXTERNAL_SIGNAL_ALLOWED_SOURCES,
    EXTERNAL_SIGNAL_ALLOWED_TIMEFRAMES,
    EXTERNAL_SIGNAL_DEDUPE_SECONDS,
    EXTERNAL_SIGNAL_MAX_AGE_SECONDS,
    EXTERNAL_SIGNAL_WEBHOOK_ENABLED,
)


def parse_csv_values(raw: str | None) -> tuple[str, ...]:
    values: list[str] = []
    for item in str(raw or "").split(","):
        normalized = item.strip()
        if normalized and normalized not in values:
            values.append(normalized)
    return tuple(values)


@dataclass(frozen=True)
class SignalProviderConfig:
    """Audit-only external signal provider configuration.

    External signals are complementary inputs for observation and audit. They
    never approve trades, open positions, change scores, or bypass guards.
    """

    enabled: bool
    allowed_sources: tuple[str, ...]
    allowed_timeframes: tuple[str, ...]
    max_age_seconds: int
    dedupe_seconds: int
    role: str = "external_signal_audit"


def current_signal_provider_config() -> SignalProviderConfig:
    return SignalProviderConfig(
        enabled=bool(EXTERNAL_SIGNAL_WEBHOOK_ENABLED),
        allowed_sources=parse_csv_values(EXTERNAL_SIGNAL_ALLOWED_SOURCES),
        allowed_timeframes=parse_csv_values(EXTERNAL_SIGNAL_ALLOWED_TIMEFRAMES),
        max_age_seconds=int(EXTERNAL_SIGNAL_MAX_AGE_SECONDS),
        dedupe_seconds=int(EXTERNAL_SIGNAL_DEDUPE_SECONDS),
    )
