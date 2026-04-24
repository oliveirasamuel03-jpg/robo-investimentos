from __future__ import annotations

from dataclasses import dataclass

from core.config import MARKET_DATA_FALLBACK_PROVIDER, MARKET_DATA_PROVIDER


@dataclass(frozen=True)
class MarketDataProviderConfig:
    """Current operational market provider configuration.

    This is a read-only foundation for future provider adapters. It does not
    replace core.market_data and does not alter feed selection.
    """

    primary_provider: str
    fallback_provider: str
    role: str = "operational_market_data"


def current_market_provider_config() -> MarketDataProviderConfig:
    return MarketDataProviderConfig(
        primary_provider=str(MARKET_DATA_PROVIDER or "twelvedata").strip().lower(),
        fallback_provider=str(MARKET_DATA_FALLBACK_PROVIDER or "yahoo").strip().lower(),
    )
