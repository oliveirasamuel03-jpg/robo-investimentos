"""Offline audit-tool checks and legacy boundary counterexamples."""
import ast
import importlib
import json

import pytest
from scripts.studies.infra126_inventory import Scan, build


def test_inventory_resolves_initializing_readers_and_stale_legacy_sinks():
    result = build()
    names = {row['symbol']: row for row in result['potential_writers']}
    for symbol in ('core.persistence.load_json_state', 'core.state_store.load_bot_state',
                   'bot_engine.save_json', 'core.retention._archive_old_rows',
                   'core.commit_protocol.reconcile'):
        assert symbol in names
    assert names['bot_engine.save_json']['group'] == 'LEGACY_LOCAL'
    assert result['contracts_A_to_G']['LEGACY_LOCAL']['F_revision'] == 'none'
    assert all(len(result['contracts_A_to_G'][row['group']]) == 7 for row in names.values())
    assert any(row['kind'] == 'sql_read_with_row_lock' for row in result['explicit_sinks'])


def test_inventory_does_not_call_select_for_update_a_dml_writer():
    scan = Scan('synthetic.py', ast.parse('''
def lock_only(conn):
    conn.execute("SELECT payload_json FROM app_state WHERE namespace='x' FOR UPDATE")
def actual_writer(conn):
    conn.execute("UPDATE app_state SET payload_json='{}'")
'''))
    assert [sink['kind'] for sink in scan.sinks] == ['sql_read_with_row_lock', 'sql_write_or_ddl']


def test_in_memory_download_is_distinct_from_durable_csv_export():
    scan = Scan('synthetic.py', ast.parse('''
def export(frame):
    download = frame.to_csv(index=False)
    frame.to_csv("synthetic.csv", index=False)
'''))
    assert [sink['kind'] for sink in scan.sinks] == ['in_memory_csv_export_not_writer', 'csv_file_sink']


@pytest.mark.parametrize('payload', ['{', '{"status": NaN}', '{"cash":1,"cash":2}'])
def test_legacy_load_is_not_the_strict_financial_contract(isolated_storage, payload):
    legacy = importlib.import_module('bot_engine')
    p = importlib.import_module('core.persistence')
    path = isolated_storage / 'synthetic-legacy.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload)
    fallback = {'synthetic_default': True}
    returned = legacy.load_json(path, fallback)
    if payload == '{':
        assert returned == fallback and returned is not fallback
    else:
        assert isinstance(returned, dict)
    with pytest.raises(p.PersistenceDataError):
        p._decode_json(payload, dict, 'synthetic-only')
    assert legacy.STATE_PATH.name != 'bot_state.json'
    assert path.read_text() == payload
