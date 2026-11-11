"""Regression cases for state conflicts, local corruption and interrupted writes."""
from concurrent.futures import ThreadPoolExecutor
import json
import threading

import pytest
from tests.conftest import load_module


def test_stale_snapshot_cannot_undo_stopped(isolated_storage):
    p = load_module('core.persistence')
    path = isolated_storage / 'state.json'
    stale = p.load_json_state('state', lambda: {'bot_status': 'RUNNING'}, path)
    current = p.load_json_state('state', dict, path)
    current['bot_status'] = 'STOPPED'
    p.save_json_state('state', current, path)
    stale['cash'] = 100
    with pytest.raises(p.PersistenceConflictError):
        p.save_json_state('state', stale, path)
    assert p.load_json_state('state', dict, path) == {'bot_status': 'STOPPED'}


def test_snapshot_is_not_portable_to_other_store(isolated_storage):
    p = load_module('core.persistence')
    first = isolated_storage / 'a.json'
    second = isolated_storage / 'b.json'
    snapshot = p.load_json_state('a', dict, first)
    with pytest.raises(p.PersistenceConflictError):
        p.save_json_state('b', snapshot, second)
    assert not second.exists()


def test_concurrent_state_increments_are_not_lost(isolated_storage):
    p = load_module('core.persistence')
    path = isolated_storage / 'state.json'
    barrier = threading.Barrier(8)
    def run(_):
        barrier.wait(timeout=5)
        for _ in range(10):
            p.update_json_state('state', lambda: {'count': 0}, path,
                                lambda state: {'count': state['count'] + 1})
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(run, range(8)))
    assert p.load_json_state('state', dict, path)['count'] == 80


@pytest.mark.parametrize('kind', ['json', 'csv'])
def test_concurrent_appends_preserve_every_record(isolated_storage, kind):
    p = load_module('core.persistence')
    path = isolated_storage / f'rows.{kind}'
    barrier = threading.Barrier(8)
    def run(i):
        barrier.wait(timeout=5)
        if kind == 'json':
            p.append_json_row('rows', {'id': i}, path)
        else:
            p.append_table_row(path, {'id': i}, ['id'])
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(run, range(8)))
    rows = p.load_json_rows('rows', path) if kind == 'json' else p.read_table(path).to_dict('records')
    assert sorted(row['id'] for row in rows) == list(range(8))


@pytest.mark.parametrize('text', ['{"cash": NaN}', '{"cash": Infinity}', '{"cash": 1e999}', '{"cash": 1,"cash": 2}'])
def test_nonfinite_or_ambiguous_json_is_preserved(isolated_storage, text):
    p = load_module('core.persistence')
    path = isolated_storage / 'state.json'
    path.parent.mkdir(parents=True)
    path.write_text(text)
    with pytest.raises(p.PersistenceDataError):
        p.load_json_state('state', dict, path)
    with pytest.raises(p.PersistenceDataError):
        p.save_json_state('state', {}, path)
    assert path.read_text() == text


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_write_does_not_replace_history(isolated_storage, value):
    p = load_module('core.persistence')
    path = isolated_storage / 'rows.json'
    p.append_json_row('rows', {'cash': 10}, path)
    before = path.read_bytes()
    with pytest.raises(ValueError):
        p.append_json_row('rows', {'cash': value}, path)
    assert path.read_bytes() == before


@pytest.mark.parametrize('content', ['', 'id,value\n1,"unterminated', 'id\n\xff'])
def test_corrupt_csv_is_never_treated_as_empty(isolated_storage, content):
    p = load_module('core.persistence')
    path = isolated_storage / 'rows.csv'
    path.parent.mkdir(parents=True)
    path.write_bytes(content.encode('latin1'))
    before = path.read_bytes()
    with pytest.raises(p.PersistenceDataError):
        p.append_table_row(path, {'id': 2}, ['id'])
    assert path.read_bytes() == before


def test_interrupted_updater_preserves_prior_state(isolated_storage):
    p = load_module('core.persistence')
    path = isolated_storage / 'state.json'
    p.save_json_state('state', {'cash': 100}, path)
    def interrupted(state):
        state['cash'] = 0
        raise RuntimeError('synthetic interruption')
    with pytest.raises(RuntimeError):
        p.update_json_state('state', dict, path, interrupted)
    assert json.loads(path.read_text()) == {'cash': 100}


def test_stopped_trader_does_not_invoke_paper_cycle(isolated_storage, monkeypatch):
    """STOPPED remains a guard even with the persistence transaction wrapper."""
    store = load_module('core.state_store')
    trader = load_module('engines.trader_engine')
    state = store.load_bot_state()
    state['bot_status'] = 'STOPPED'
    store.save_bot_state(state)
    called = []
    def forbidden(*args, **kwargs):
        called.append(True)
        raise AssertionError('STOPPED must not invoke the paper cycle')
    monkeypatch.setattr(trader, 'run_paper_cycle', forbidden)
    assert trader.run_trader_cycle() == {'status': 'STOPPED'}
    assert not called
    assert store.load_bot_state()['bot_status'] == 'STOPPED'
    assert store.load_bot_state()['security']['real_mode_enabled'] is False
