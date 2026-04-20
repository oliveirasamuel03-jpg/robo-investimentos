from __future__ import annotations

import importlib
import sys

import pytest


MODULES_TO_CLEAR = [
    "core.config",
    "core.persistence",
    "core.state_store",
    "core.market_data",
    "core.trader_profiles",
    "core.trader_reports",
    "core.auth.users_store",
    "paper_trading_engine",
    "engines.quant_bridge",
    "engines.trader_engine",
    "bot_engine",
]


def _clear_modules() -> None:
    for module_name in MODULES_TO_CLEAR:
        sys.modules.pop(module_name, None)


@pytest.fixture
def isolated_storage(tmp_path, monkeypatch):
    storage_dir = tmp_path / "storage"
    monkeypatch.setenv("ROBO_STORAGE_DIR", str(storage_dir))
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("AUTH_REQUIRED", "false")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("ADMIN_USERNAME", raising=False)
    monkeypatch.delenv("ADMIN_PASSWORD", raising=False)
    _clear_modules()
    yield storage_dir
    _clear_modules()


def load_module(name: str):
    if name in sys.modules:
        return importlib.reload(sys.modules[name])
    return importlib.import_module(name)
