"""Provider foundation for future integrations.

The current operational market-data flow remains in core.market_data. These
lightweight definitions only document and prepare provider roles without
changing runtime behavior.
"""

from core.providers.base import ProviderIdentity
from core.providers.market_data_provider import MarketDataProviderConfig, current_market_provider_config
from core.providers.signal_provider import SignalProviderConfig, current_signal_provider_config

__all__ = [
    "MarketDataProviderConfig",
    "ProviderIdentity",
    "SignalProviderConfig",
    "current_market_provider_config",
    "current_signal_provider_config",
]
