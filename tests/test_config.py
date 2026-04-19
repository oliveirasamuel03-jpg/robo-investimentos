from __future__ import annotations

from tests.conftest import load_module


def test_config_uses_runtime_subdirectories(isolated_storage):
    config = load_module("core.config")

    assert config.STORAGE_DIR == isolated_storage
    assert config.RUNTIME_DIR == isolated_storage / "runtime"
    assert config.CACHE_DIR == isolated_storage / "cache"
    assert config.REPORTS_DIR == isolated_storage / "reports"
    assert config.BOT_STATE_FILE.parent == config.RUNTIME_DIR
    assert config.AUTH_USERS_FILE.parent == config.RUNTIME_DIR
    assert config.RUNTIME_DIR.exists()
