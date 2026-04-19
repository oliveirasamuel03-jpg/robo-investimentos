from __future__ import annotations

from tests.conftest import load_module


def test_legacy_bot_engine_uses_runtime_directory(isolated_storage):
    config = load_module("core.config")
    bot_engine = load_module("bot_engine")

    bot_engine.ensure_runtime_files()

    assert bot_engine.CONFIG_PATH == config.LEGACY_RUNTIME_CONFIG_FILE
    assert bot_engine.STATE_PATH == config.LEGACY_BOT_STATE_FILE
    assert bot_engine.LOG_PATH == config.LEGACY_BOT_LOG_FILE
    assert bot_engine.CONFIG_PATH.exists()
    assert bot_engine.STATE_PATH.exists()
    assert bot_engine.LOG_PATH.exists()
