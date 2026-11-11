"""Control revisions fence drafts even across a later explicit restart."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event
import importlib

import pytest
from tests.test_infra121_idempotency import _paper_fixture


def test_draft_before_stop_ack_restart_cannot_commit(isolated_storage, monkeypatch):
    paper, cfg = _paper_fixture(monkeypatch)
    store = importlib.import_module('core.state_store')
    control = importlib.import_module('core.stop_control')
    p = importlib.import_module('core.persistence')
    control.start_robot(store.load_bot_state())
    entered, release = Event(), Event()
    feed = paper.load_market_data_result
    def slow_feed(*args, **kwargs):
        assert p._ACTIVE_CONNECTION.get() is None
        entered.set()
        assert release.wait(10)
        return feed(*args, **kwargs)
    monkeypatch.setattr(paper, 'load_market_data_result', slow_feed)
    before = dict(paper.load_paper_state(cfg))
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(paper.run_paper_cycle, cfg)
        try:
            assert entered.wait(5)
            assert control.request_stop().persisted
            assert control.acknowledge_stop('synthetic-worker').acknowledged
            control.start_robot(store.load_bot_state())
        finally:
            release.set()
        with pytest.raises(p.PersistenceConflictError):
            pending.result(15)
    assert paper.load_paper_state(cfg) == before
    assert paper.read_paper_trades() == []
    assert store.load_bot_state()['bot_status'] == 'RUNNING'


def test_nonpaper_commit_guard_preserves_entire_draft(isolated_storage, monkeypatch):
    paper, cfg = _paper_fixture(monkeypatch)
    before = paper.load_paper_state(cfg)
    draft = paper.load_paper_state(cfg)
    draft['run_count'] += 1
    monkeypatch.setattr(paper, 'build_broker_runtime_config', lambda _: {
        'provider': 'paper', 'configured_mode': 'live',
        'live_requested': True, 'real_mode_enabled': False,
    })
    with pytest.raises(RuntimeError, match='exclusively paper'):
        paper._commit_paper_draft(draft, [{'side': 'BUY'}], [], cfg)
    assert paper.load_paper_state(cfg) == before
    assert paper.read_paper_trades() == []
