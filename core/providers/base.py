from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ProviderIdentity:
    """Minimal identity shared by provider adapters.

    Provider roles are intentionally separate:
    - market providers are operational data sources for the robot.
    - signal providers are complementary audit inputs and never execution authority.
    """

    name: str
    role: str
    enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


class Provider(Protocol):
    identity: ProviderIdentity
