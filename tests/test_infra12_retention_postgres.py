"""Retention races and interrupted archive writes in the disposable local DB."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import importlib
import threading

import pytest

from tests.test_infra12_postgres import pg, pytestmark


def test_retention_read_replace_serializes_with_concurrent_append(pg, monkeypatch):
    p, _, _, _ = pg
    config = importlib.import_module("core.config")
    retention = importlib.import_module("core.retention")
    runtime = config.BOT_LOG_FILE
    old = {"timestamp": "2026-01-01T00:00:00+00:00", "level": "INFO", "message": "old-synthetic"}
    new = {"timestamp": "2026-04-20T00:00:00+00:00", "level": "INFO", "message": "new-synthetic"}
    p.replace_table(runtime, [old], config.BOT_LOG_COLUMNS)
    read_started, release_read, append_started, append_done = (threading.Event() for _ in range(4))
    original_read = retention.read_table

    def paused_read(path, *args, **kwargs):
        frame = original_read(path, *args, **kwargs)
        if path == runtime:
            read_started.set()
            assert release_read.wait(timeout=10)
        return frame

    monkeypatch.setattr(retention, "read_table", paused_read)

    def append():
        append_started.set()
        p.append_table_row(runtime, new, config.BOT_LOG_COLUMNS)
        append_done.set()

    with ThreadPoolExecutor(max_workers=2) as pool:
        archive_future = pool.submit(retention.archive_old_bot_logs, 60,
                                     now=datetime(2026, 4, 20, tzinfo=timezone.utc))
        assert read_started.wait(timeout=10)
        append_future = pool.submit(append)
        try:
            assert append_started.wait(timeout=10)
            assert not append_done.wait(timeout=0.1), "append bypassed archive transaction"
        finally:
            release_read.set()
        summary = archive_future.result(timeout=10)
        append_future.result(timeout=10)
    assert p.read_table(runtime).to_dict(orient="records") == [new]
    assert p.read_table(summary["files_updated"][0]).to_dict(orient="records") == [old]


def test_retention_failure_rolls_back_archive_copy_and_runtime_removal(pg, monkeypatch):
    p, conn, _, _ = pg
    config = importlib.import_module("core.config")
    retention = importlib.import_module("core.retention")
    runtime = config.BOT_LOG_FILE
    old = {"timestamp": "2026-01-01T00:00:00+00:00", "level": "INFO", "message": "old-synthetic"}
    p.replace_table(runtime, [old], config.BOT_LOG_COLUMNS)
    original_replace = retention.replace_table

    def fail_runtime(path, *args, **kwargs):
        if path == runtime:
            raise OSError("synthetic interruption after archive copy")
        return original_replace(path, *args, **kwargs)

    monkeypatch.setattr(retention, "replace_table", fail_runtime)
    with pytest.raises(OSError, match="synthetic interruption"):
        retention.archive_old_bot_logs(60, now=datetime(2026, 4, 20, tzinfo=timezone.utc))
    assert p.read_table(runtime).to_dict(orient="records") == [old]
    assert conn.execute("SELECT DISTINCT namespace FROM app_rows").fetchall() == [(p._table_namespace(runtime),)]


def test_weekly_summary_can_be_recomputed_without_unversioned_overwrite(pg):
    _, _, _, _ = pg
    retention = importlib.import_module("core.retention")
    retention._save_weekly_summary({"synthetic": "first"}, week_key="2026_W16", archived=False)
    retention._save_weekly_summary({"synthetic": "second"}, week_key="2026_W16", archived=False)
    assert retention.load_weekly_summary({"week_key": "2026_W16", "location": "runtime"}) == {"synthetic": "second"}
