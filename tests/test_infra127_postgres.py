"""Deterministic retention, containment and recovery on synthetic local data."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import importlib
import json
import os
import subprocess
import sys
import threading
import time

import psycopg
import pytest
from tests.test_infra12_postgres import pg, pytestmark, ROOT
from tests.test_infra123_postgres import _seed_full_synthetic_history


def evidence(conn, name, result):
    major = str(conn.execute('SHOW server_version_num').fetchone()[0])[:2]
    path = ROOT / f'docs/retomada/evidencias/infra127-proof-pg{major}.json'
    data = json.loads(path.read_text()) if path.exists() else {'scope': 'LOCAL_SYNTHETIC_PAPER_ONLY', 'cases': {}}
    data['cases'][name] = result
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')


def snapshot(conn):
    return (conn.execute('SELECT namespace,payload_json,updated_at FROM app_state ORDER BY namespace').fetchall(),
            conn.execute('SELECT seq,namespace,row_json,created_at FROM app_rows ORDER BY seq').fetchall())


def damage_report(conn):
    seq, raw = conn.execute("SELECT seq,row_json FROM app_rows WHERE namespace='table:trade_reports.csv' LIMIT 1").fetchone()
    damaged = json.loads(raw)
    damaged['entry_price'] += 1
    conn.execute('UPDATE app_rows SET row_json=%s WHERE seq=%s', (json.dumps(damaged), seq))
    return seq, raw


def test_noop_has_no_write_api_calls_and_preserves_full_evidence(pg, monkeypatch):
    p, conn, store, control, paper, trader, q, cycle = _seed_full_synthetic_history(pg, monkeypatch)
    retention = importlib.import_module('core.retention')
    before, manifest = snapshot(conn), q.manifest()
    def forbidden(*args, **kwargs):
        raise AssertionError('no-op attempted a persistence write')
    monkeypatch.setattr(p, 'save_json_state', forbidden)
    monkeypatch.setattr(p, 'replace_json_rows', forbidden)
    for _ in range(3):
        result = retention.archive_old_trade_reports(36500)
        assert not result.get('blocked') and result['files_updated'] == []
        assert snapshot(conn) == before and q.manifest() == manifest
    evidence(conn, 'noop', {'repetitions': 3, 'all_bytes_seq_timestamps_equal': True,
        'write_api_calls': 0, 'manifest_equal': True, 'cycle_id_preserved': True})


def test_identical_financial_duplicates_are_preserved_not_cleaned(pg, monkeypatch):
    p, conn, store, control, paper, trader, q, cycle = _seed_full_synthetic_history(pg, monkeypatch)
    retention = importlib.import_module('core.retention')
    original_report = json.loads(conn.execute("SELECT row_json FROM app_rows WHERE namespace='table:trade_reports.csv' LIMIT 1").fetchone()[0])
    reordered = dict(reversed(list(original_report.items())))
    conn.execute("INSERT INTO app_rows(namespace,row_json) VALUES ('table:trade_reports.csv',%s)",
                 (json.dumps(reordered),))
    before = snapshot(conn)[1]
    result = retention.archive_old_trade_reports(36500)
    assert result['identical_rows_preserved'] == 1 and result['retained_rows'] == 2
    assert snapshot(conn)[1] == before
    # Preservation is not approval of a duplicate ledger: new finance rechecks
    # the signed preimage and refuses the changed count/content.
    with pytest.raises(q.ReconciliationRequired):
        trader.run_trader_cycle()
    assert q.gate_status()['blocked'] and len(paper.read_paper_trades()) == 3
    evidence(conn, 'identical_duplicates', {'original_rows_preserved': 2,
        'new_finance_rejected_by_fresh_proof': True, 'no_cleanup': True})


@pytest.mark.parametrize('payload', ['{', '{"cash":NaN}', '{"trade_id":"one","trade_id":"two"}'])
def test_strict_corruption_is_preserved_and_quarantined(pg, monkeypatch, payload):
    p, conn, store, control, paper, trader, q, cycle = _seed_full_synthetic_history(pg, monkeypatch)
    retention = importlib.import_module('core.retention')
    conn.execute("UPDATE app_rows SET row_json=%s WHERE namespace='table:trade_reports.csv'", (payload,))
    before = snapshot(conn)[1]
    result = retention.archive_old_trade_reports(36500)
    assert result['blocked'] and result['reason'] == 'RETENTION_CORRUPTION'
    assert snapshot(conn)[1] == before and q.gate_status()['blocked']
    with psycopg.connect(pg[2], autocommit=True) as independent:
        state = json.loads(independent.execute('SELECT payload_json FROM app_state WHERE namespace=%s', (q.QUARANTINE,)).fetchone()[0])
        assert state['active']
    evidence(conn, 'corruption_' + str(len(payload)), {'source_not_overwritten': True, 'quarantine_independently_observed': True})


def restart_blocked():
    program = """
import json,sys
from scripts.test_infra12_postgres import _loopback_only
sys.addaudithook(_loopback_only)
from core.commit_protocol import assert_execution_allowed, ReconciliationRequired, gate_status
try: assert_execution_allowed()
except ReconciliationRequired: print(json.dumps(gate_status()))
else: raise AssertionError('unsafe persisted history admitted after restart')
"""
    process = subprocess.run([sys.executable, '-c', program], cwd=ROOT, env=dict(os.environ),
                             capture_output=True, text=True, check=True, timeout=30)
    return json.loads(process.stdout)


def legacy_report():
    return {'trade_id': 'synthetic-unreceipted', 'closed_at': '2026-01-01T00:00:00+00:00',
            'asset': 'FAKE', 'quantity': 1.25, 'entry_price': 10, 'exit_reason': 'synthetic',
            'realized_pnl': 0, 'extra_identity': {'source': 'synthetic', 'version': 1}}


def test_valid_unreceipted_archive_preserves_objects_and_catalog_atomically(pg, monkeypatch):
    p, conn, _, root = pg
    cfg = importlib.import_module('core.config')
    retention = importlib.import_module('core.retention')
    original = legacy_report()
    store = importlib.import_module('core.state_store')
    untouched_bot = deepcopy(store.DEFAULT_STATE)
    untouched_bot.update(cash=cfg.LEGACY_VALIDATION_INITIAL_CAPITAL_BRL,
                         wallet_value=cfg.LEGACY_VALIDATION_INITIAL_CAPITAL_BRL,
                         positions=[], realized_pnl=0, last_run_at='')
    untouched_bot['validation'].update(validation_started_at='', last_report={})
    untouched_bot['trader']['max_open_positions'] = 3
    p._ensure_db_schema()
    conn.execute("INSERT INTO app_state(namespace,payload_json) VALUES ('bot_state',%s) ON CONFLICT(namespace) DO UPDATE SET payload_json=EXCLUDED.payload_json",
                 (json.dumps(untouched_bot),))
    p.replace_table(cfg.TRADER_REPORTS_FILE, [original])
    result = retention.archive_old_trade_reports(60, now=datetime(2026, 4, 20, tzinfo=timezone.utc))
    assert result['archived_rows'] == 1 and result['retained_rows'] == 0
    archive = result['files_updated'][0]
    assert p.load_json_rows(p._table_namespace(archive), root / 'unused.json') == [original]
    assert p.load_json_rows(p._table_namespace(cfg.TRADER_REPORTS_FILE), root / 'unused.json') == []
    persisted_bot = json.loads(conn.execute("SELECT payload_json FROM app_state WHERE namespace='bot_state'").fetchone()[0])
    assert archive in persisted_bot['retention']['archive_catalog']['reports']
    assert {k:v for k,v in persisted_bot.items() if k != 'retention'} == {k:v for k,v in untouched_bot.items() if k != 'retention'}
    before = snapshot(conn)
    again = retention.archive_old_trade_reports(60, now=datetime(2026, 4, 20, tzinfo=timezone.utc))
    assert again['files_updated'] == [] and snapshot(conn) == before
    # Incompatible SQL writer: introduce an archived-source conflict, then
    # lose containment through both a failed write and an outer rollback.
    conflicting = {**original, 'entry_price': original['entry_price'] + 1}
    conn.execute('INSERT INTO app_rows(namespace,row_json) VALUES (%s,%s)',
                 (p._table_namespace(cfg.TRADER_REPORTS_FILE), json.dumps(conflicting)))
    before_conflict = snapshot(conn)[1]
    q = importlib.import_module('core.commit_protocol')
    save = p.save_json_state
    with monkeypatch.context() as patch:
        def unavailable(namespace, *args, **kwargs):
            if namespace == q.QUARANTINE:
                raise OSError('PYTHON_INJECTION missing quarantine acknowledgment')
            return save(namespace, *args, **kwargs)
        patch.setattr(p, 'save_json_state', unavailable)
        with pytest.raises(OSError, match='PYTHON_INJECTION'):
            retention.archive_old_trade_reports(60, now=datetime(2026, 4, 20, tzinfo=timezone.utc))
    with pytest.raises(ValueError, match='caller rollback'):
        with p.persistence_transaction():
            conflict = retention.archive_old_trade_reports(60, now=datetime(2026, 4, 20, tzinfo=timezone.utc))
            assert conflict['blocked'] and conflict['reason'] == 'RETENTION_CONFLICT'
            raise ValueError('caller rollback')
    assert snapshot(conn)[1] == before_conflict and not q._quarantine()['active']
    assert q.gate_status()['blocked'], 'Persisted archive evidence must block even after containment rollback'
    restarted = restart_blocked()
    assert restarted['blocked'] and q._quarantine()['active']
    evidence(conn, 'valid_unreceipted_archive', {'all_fields_preserved': True,
        'non_retention_bot_fields_equal': True, 'archive_conflict_preserved': True,
        'failed_quarantine_and_outer_rollback_cannot_reopen_finance': True,
        'new_process_reconstitutes_containment': True,
        'catalog_same_commit': True, 'second_run_no_writes': True, 'receipt_compatibility_claim': False})


def test_failure_after_archive_copy_rolls_back_runtime_archive_and_catalog(pg, monkeypatch):
    p, conn, _, _ = pg
    cfg = importlib.import_module('core.config')
    retention = importlib.import_module('core.retention')
    p.replace_table(cfg.TRADER_REPORTS_FILE, [legacy_report()])
    before = snapshot(conn)
    original = p.replace_json_rows
    def fail(namespace, *args, **kwargs):
        if namespace == p._table_namespace(cfg.TRADER_REPORTS_FILE):
            raise OSError('PYTHON_INJECTION after synthetic archive copy')
        return original(namespace, *args, **kwargs)
    monkeypatch.setattr(p, 'replace_json_rows', fail)
    with pytest.raises(OSError, match='PYTHON_INJECTION'):
        retention.archive_old_trade_reports(60, now=datetime(2026, 4, 20, tzinfo=timezone.utc))
    assert snapshot(conn) == before
    collision_source = pg[3] / 'bot_log_2026_01.csv'
    p.replace_table(collision_source, [{'timestamp': '2026-01-01T00:00:00+00:00', 'message': 'synthetic-evidence'}])
    before_collision = snapshot(conn)
    collision_error = None
    try:
        retention._archive_old_rows(runtime_path=collision_source, archive_dir=pg[3]/'other',
            prefix='bot_log', columns=['timestamp','message'], timestamp_column='timestamp',
            key_columns=['timestamp','message'], retention_days=60,
            now=datetime(2026, 4, 20, tzinfo=timezone.utc))
    except p.PersistenceDataError as exc:
        collision_error = str(exc)
    remaining = conn.execute('SELECT count(*) FROM app_rows WHERE namespace=%s', (p._table_namespace(collision_source),)).fetchone()[0]
    evidence(conn, 'namespace_collision', {'source_rows_before': 1, 'source_rows_after': remaining, 'refused': collision_error is not None})
    assert collision_error and 'namespace' in collision_error
    assert snapshot(conn) == before_collision
    evidence(conn, 'archive_rollback', {'fault': 'PYTHON_INJECTION', 'all_rows_catalog_rolled_back': True})


def test_receipted_archival_is_refused_without_touching_manifest_or_receipts(pg, monkeypatch):
    p, conn, store, control, paper, trader, q, cycle = _seed_full_synthetic_history(pg, monkeypatch)
    retention = importlib.import_module('core.retention')
    before, manifest = snapshot(conn), q.manifest()
    result = retention.run_retention_job(now=datetime(2027, 5, 1, tzinfo=timezone.utc))
    assert result['blocked'] and result['last_error'] == 'FINANCIAL_ARCHIVE_REQUIRES_VERSIONED_RECEIPTS'
    assert snapshot(conn) == before and q.manifest() == manifest
    assert q.reconcile(cycle)['outcome'] == 'COMMITTED'
    evidence(conn, 'receipted_archival_refused', {'manifest_receipts_rows_unchanged': True,
        'financial_archive_requires_future_protocol': True, 'maintenance_not_financial_quarantine': True})


@pytest.mark.parametrize('failure', ['quarantine_write', 'outer_rollback'])
def test_failed_containment_never_allows_finance_after_python_restart(pg, monkeypatch, failure):
    p, conn, store, control, paper, trader, q, cycle = _seed_full_synthetic_history(pg, monkeypatch)
    damage_report(conn)
    with monkeypatch.context() as patch:
        if failure == 'quarantine_write':
            original = p.save_json_state
            def reject(namespace, *args, **kwargs):
                if namespace == q.QUARANTINE:
                    raise OSError('PYTHON_INJECTION quarantine unavailable')
                return original(namespace, *args, **kwargs)
            patch.setattr(p, 'save_json_state', reject)
            result = q.reconcile(cycle)
            assert result['outcome'] == 'UNKNOWN_COMMIT' and result['quarantine_persistence'] == 'UNCONFIRMED'
        else:
            with pytest.raises(ValueError):
                with p.persistence_transaction():
                    result = q.reconcile(cycle)
                    assert result['quarantine_persistence'] == 'STAGED_IN_CALLER_TRANSACTION'
                    raise ValueError('synthetic caller rollback')
    assert not q._quarantine()['active']
    program = '''
import json
from core.commit_protocol import assert_execution_allowed, ReconciliationRequired, gate_status
try: assert_execution_allowed()
except ReconciliationRequired: print(json.dumps(gate_status()))
else: raise AssertionError('restart accepted unsafe preimage')
'''
    restarted = subprocess.run([sys.executable, '-c', program], cwd=ROOT, env=dict(os.environ),
                               check=True, capture_output=True, text=True, timeout=30)
    assert json.loads(restarted.stdout)['blocked'] and q._quarantine()['active']
    assert len(paper.read_paper_trades()) == 3
    evidence(conn, 'containment_failure_' + failure, {'failure_type': 'PYTHON_INJECTION_OR_CALLER_ROLLBACK',
        'no_false_persistence_claim': True, 'new_python_process_refused_finance': True,
        'fresh_preimage_recheck_recreated_durable_quarantine': True, 'trades_unchanged': 3})


def test_explicit_proof_and_quarantine_generation_are_required_to_release(pg, monkeypatch):
    p, conn, store, control, paper, trader, q, cycle = _seed_full_synthetic_history(pg, monkeypatch)
    seq, raw = damage_report(conn)
    assert q.reconcile(cycle)['quarantine_persistence'] == 'CONFIRMED'
    identity = q.gate_status()['quarantine_id']
    # Synthetic fixture repair only. Runtime provides no automatic repair API.
    conn.execute('UPDATE app_rows SET row_json=%s WHERE seq=%s', (raw, seq))
    assert q.reconcile(cycle)['blocked']
    assert q.reconcile(cycle, release_quarantine=True, expected_quarantine_id='stale')['blocked']
    result = q.reconcile(cycle, release_quarantine=True, expected_quarantine_id=identity)
    assert result['outcome'] == 'COMMITTED' and not result['blocked']
    assert not q._quarantine()['active'] and q._quarantine()['evidence']
    q.quarantine_integrity('RETENTION_CONFLICT', cycle_id=cycle, evidence={'source': 'synthetic-only'})
    next_id = q.gate_status()['quarantine_id']
    assert next_id != identity
    assert q.reconcile(cycle, release_quarantine=True, expected_quarantine_id=identity)['blocked']
    assert q.reconcile(cycle, release_quarantine=True, expected_quarantine_id=next_id)['blocked']
    assert q.gate_status()['quarantine_id'] == next_id
    evidence(conn, 'release_contract', {'full_proof_and_matching_generation_required': True,
        'retention_conflict_not_cleared_by_financial_receipt_alone': True, 'no_financial_retry': True})


def test_backup_restore_preserves_quarantine_stop_and_all_financial_rows(pg, monkeypatch, tmp_path):
    from scripts.local_synthetic_backup import dump_restore
    p, conn, store, control, paper, trader, q, cycle = _seed_full_synthetic_history(pg, monkeypatch)
    damage_report(conn)
    assert q.reconcile(cycle)['quarantine_persistence'] == 'CONFIRMED'
    identity = q.gate_status()['quarantine_id']
    assert control.request_stop('synthetic-quarantine-backup').persisted
    before = snapshot(conn)[1]
    restored_dsn, proof = dump_restore(pg[2], os.environ['ATLAS_INFRA12_CONTAINER'], tmp_path)
    with monkeypatch.context() as patch:
        patch.setenv('DATABASE_URL', restored_dsn)
        assert store.load_bot_state()['bot_status'] == 'STOPPED'
        assert q.gate_status()['quarantine_id'] == identity
        store.update_worker_heartbeat('synthetic-restarted-worker')
        assert q.gate_status()['quarantine_id'] == identity
        with pytest.raises(p.PersistenceConflictError):
            store.reset_state()
        with pytest.raises(p.PersistenceConflictError):
            control.start_robot(store.load_bot_state(), requested_by='synthetic-forbidden-start')
        with pytest.raises(p.PersistenceConflictError):
            p.update_json_state(q.QUARANTINE, dict, q._QUARANTINE_PATH, lambda _: {'active': False, 'evidence': []})
        assert trader.run_trader_cycle()['status'] == 'STOPPED'
    with psycopg.connect(restored_dsn, autocommit=True) as restored:
        assert snapshot(restored)[1] == before
    evidence(conn, 'backup_quarantine', {'backup': proof, 'quarantine_id_preserved': True,
        'stop_persisted': True, 'heartbeat_reset_start_cannot_clear': True, 'financial_rows_equal': True})

    # Actual crash/restart of THIS test's server only, with quarantine active.
    from scripts.local_synthetic_backup import owned_container, DOCKER
    container = os.environ['ATLAS_INFRA12_CONTAINER']
    owned_container(container)
    expected_quarantine = dict(q._quarantine())
    assert conn.execute('SHOW fsync').fetchone()[0] == 'on'
    assert conn.execute('SHOW synchronous_commit').fetchone()[0] == 'on'
    subprocess.run(DOCKER + ['kill', '--signal=KILL', container], check=True, capture_output=True, timeout=10)
    failed_stop = control.request_stop('synthetic-stop-during-quarantine-server-crash')
    assert failed_stop.status == 'UNCONFIRMED' and not failed_stop.persisted
    subprocess.run(DOCKER + ['start', container], check=True, capture_output=True, timeout=10)
    deadline = time.monotonic() + 30
    while True:
        try:
            with psycopg.connect(pg[2], autocommit=True) as recovered:
                recovered.execute('SELECT 1')
            break
        except psycopg.OperationalError:
            assert time.monotonic() < deadline
            threading.Event().wait(.05)  # readiness predicate, not fault timing evidence
    assert dict(q._quarantine()) == expected_quarantine
    assert q.gate_status()['blocked'] and store.load_bot_state()['bot_status'] == 'STOPPED'
    with pytest.raises(q.ReconciliationRequired):
        q.assert_execution_allowed()
    logs = subprocess.run(DOCKER + ['logs', '--tail', '100', container], check=True,
                          capture_output=True, text=True, timeout=10)
    recovery = [line for line in (logs.stdout + logs.stderr).splitlines()
                if any(word in line for word in ('was interrupted', 'not properly shut down',
                    'redo starts', 'redo done', 'automatic recovery', 'ready to accept connections'))]
    assert recovery
    with psycopg.connect(pg[2], autocommit=True) as recovered:
        assert snapshot(recovered)[1] == before
        evidence(recovered, 'quarantine_real_postgres_crash', {
            'fault': 'REAL_OWNED_POSTGRES_SIGKILL_AND_RESTART', 'quarantine_unchanged': True,
            'financial_rows_equal': True, 'stop_unconfirmed_while_down': True,
            'stop_still_persisted_after_recovery': True, 'no_financial_retry': True,
            'fsync': 'on', 'synchronous_commit': 'on', 'server_recovery_excerpt': recovery[-8:]})


def test_two_concurrent_writers_reject_divergence_without_replaying_trades(pg, monkeypatch):
    p, conn, store, control, paper, trader, q, cycle = _seed_full_synthetic_history(pg, monkeypatch)
    damage_report(conn)
    barrier = threading.Barrier(2)
    def attempt():
        barrier.wait(timeout=10)
        with pytest.raises(q.ReconciliationRequired):
            q.assert_execution_allowed()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(attempt) for _ in range(2)]
        for future in futures:
            future.result(30)
    assert q.gate_status()['blocked'] and len(paper.read_paper_trades()) == 3
    assert control.request_stop('synthetic-stop-after-race').persisted
    assert control.acknowledge_stop('synthetic-worker').acknowledged
    evidence(conn, 'concurrent_containment', {'barrier': True, 'writers_rejected': 2,
        'trades_not_replayed': True, 'stop_and_ack_confirmed': True})


def test_quarantine_during_feed_fences_pending_draft(pg, monkeypatch):
    p, conn, store, control, paper, trader, q, cycle = _seed_full_synthetic_history(pg, monkeypatch)
    entered, release = threading.Event(), threading.Event()
    original = paper.load_market_data_result
    def feed(*args, **kwargs):
        assert p._ACTIVE_CONNECTION.get() is None
        entered.set()
        assert release.wait(20)
        return original(*args, **kwargs)
    monkeypatch.setattr(paper, 'load_market_data_result', feed)
    before = snapshot(conn)[1]
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(trader.run_trader_cycle)
        try:
            assert entered.wait(20)
            result = q.quarantine_integrity('RETENTION_CONFLICT', cycle_id=cycle,
                                            evidence={'source': 'synthetic-concurrent-study'})
            assert result['quarantine_persistence'] == 'CONFIRMED'
            assert control.request_stop('synthetic-stop-during-quarantine-feed').persisted
        finally:
            release.set()
        try:
            answer = future.result(30)
            assert answer['status'] == 'STOPPED'
        except q.ReconciliationRequired:
            pass
    assert snapshot(conn)[1] == before and q.gate_status()['blocked']
    evidence(conn, 'feed_quarantine_stop', {'deterministic_events': True,
        'draft_not_committed': True, 'stop_available': True, 'financial_rows_equal': True})


def test_repeated_complete_retention_job_has_no_storage_changes(pg, monkeypatch):
    p, conn, store, control, paper, trader, q, cycle = _seed_full_synthetic_history(pg, monkeypatch)
    retention = importlib.import_module('core.retention')
    now = datetime.now(timezone.utc)
    first = retention.run_retention_job(now=now, retention_days=36500)
    assert not first.get('blocked')
    before = snapshot(conn)
    def forbidden(*args, **kwargs):
        raise AssertionError('unchanged complete job attempted a data write')
    monkeypatch.setattr(p, 'replace_json_rows', forbidden)
    monkeypatch.setattr(p, 'save_json_state', forbidden)
    second = retention.run_retention_job(now=now, retention_days=36500)
    assert not second.get('blocked') and snapshot(conn) == before
    evidence(conn, 'complete_job_noop', {'all_state_rows_timestamps_equal': True, 'write_api_calls': 0})


def test_current_receipt_tampering_cannot_be_laundered_into_a_new_cycle(pg, monkeypatch):
    p, conn, store, control, paper, trader, q, cycle = _seed_full_synthetic_history(pg, monkeypatch)
    receipt = dict(q._receipt(cycle))
    receipt['planned_trades'] = ['0' * 64]
    conn.execute('UPDATE app_state SET payload_json=%s WHERE namespace=%s',
                 (json.dumps(receipt), q.RECEIPT_PREFIX + cycle))
    before = snapshot(conn)[1]
    with pytest.raises(q.ReconciliationRequired):
        trader.run_trader_cycle()
    assert q.gate_status()['blocked'] and snapshot(conn)[1] == before
    evidence(conn, 'prior_receipt_tamper', {'full_receipt_validation_reused': True,
        'new_intent_rejected': True, 'financial_rows_equal': True})


def test_quarantine_after_durable_intent_rolls_back_entire_financial_unit(pg, monkeypatch):
    p, conn, store, control, paper, trader, q, cycle = _seed_full_synthetic_history(pg, monkeypatch)
    before = snapshot(conn)[1]
    original = q.begin_intent
    def intervene(*args, **kwargs):
        identity = original(*args, **kwargs)
        result = q.quarantine_integrity('RETENTION_CONFLICT', cycle_id=identity,
                                       evidence={'source': 'synthetic-boundary'})
        assert result['quarantine_persistence'] == 'CONFIRMED'
        return identity
    monkeypatch.setattr(q, 'begin_intent', intervene)
    with pytest.raises(q.ReconciliationRequired):
        trader.run_trader_cycle()
    assert snapshot(conn)[1] == before and q.gate_status()['blocked']
    assert len(paper.read_paper_trades()) == 3
    evidence(conn, 'intent_to_commit_quarantine', {'intent_committed_before_intervention': True,
        'entire_financial_unit_rolled_back': True, 'no_automatic_retry': True})


def test_late_cycle_a_diagnostic_never_clears_newer_quarantine_b(pg, monkeypatch):
    p, conn, store, control, paper, trader, q, current = _seed_full_synthetic_history(pg, monkeypatch)
    receipts = conn.execute('SELECT payload_json FROM app_state WHERE left(namespace,%s)=%s',
                            (len(q.RECEIPT_PREFIX), q.RECEIPT_PREFIX)).fetchall()
    old = next(json.loads(raw)['cycle_id'] for (raw,) in receipts if json.loads(raw)['cycle_id'] != current)
    q.quarantine_integrity('RETENTION_CONFLICT', cycle_id=current, evidence={'source': 'synthetic-B'})
    before_gate, before_quarantine, before_receipt = dict(q._gate()), dict(q._quarantine()), dict(q._receipt(current))
    answer = q.post_commit_failure(old, confirmed=True)
    assert answer['stale_diagnostic'] and answer['current_cycle_id'] == current
    historical = q.reconcile(old, release_quarantine=True,
                             expected_quarantine_id=before_quarantine['quarantine_id'])
    assert historical['historical'] and historical['blocked']
    assert dict(q._gate()) == before_gate and dict(q._quarantine()) == before_quarantine
    assert dict(q._receipt(current)) == before_receipt
    evidence(conn, 'late_a_vs_quarantine_b', {'gate_quarantine_new_receipt_equal': True,
        'historical_receipt_cannot_release_current_quarantine': True})


@pytest.mark.parametrize('payload', ['{}', '{', '{"active":false,"evidence":null}'])
def test_unreadable_quarantine_is_not_reset_or_ignored(pg, monkeypatch, payload):
    p, conn, store, control, paper, trader, q, cycle = _seed_full_synthetic_history(pg, monkeypatch)
    conn.execute('UPDATE app_state SET payload_json=%s WHERE namespace=%s', (payload, q.QUARANTINE))
    with pytest.raises((p.PersistenceDataError, ValueError)):
        q.gate_status()
    with pytest.raises((p.PersistenceDataError, ValueError)):
        trader.run_trader_cycle()
    assert conn.execute('SELECT payload_json FROM app_state WHERE namespace=%s', (q.QUARANTINE,)).fetchone()[0] == payload
    assert control.request_stop('synthetic-stop-despite-corrupt-quarantine').persisted
    evidence(conn, 'corrupt_quarantine_' + str(len(payload)), {'execution_refused': True,
        'corrupt_record_preserved': True, 'stop_still_persists': True})


def test_lost_quarantine_commit_reply_is_unconfirmed_until_independent_read(pg, monkeypatch):
    p, conn, store, control, paper, trader, q, cycle = _seed_full_synthetic_history(pg, monkeypatch)
    original_boundary, original_save = p._transaction_context, p.save_json_state
    armed = {'value': False}
    def save(namespace, *args, **kwargs):
        result = original_save(namespace, *args, **kwargs)
        if namespace == q.QUARANTINE:
            armed['value'] = True
        return result
    @contextmanager
    def lost_reply(connection, witness):
        with original_boundary(connection, witness):
            yield
        if armed['value']:
            armed['value'] = False
            raise psycopg.OperationalError('PYTHON_INJECTION reply lost AFTER actual quarantine COMMIT')
    with monkeypatch.context() as patch:
        patch.setattr(p, 'save_json_state', save)
        patch.setattr(p, '_transaction_context', lost_reply)
        result = q.quarantine_integrity('RETENTION_CONFLICT', cycle_id=cycle,
                                       evidence={'source': 'synthetic-lost-response'})
    assert result['quarantine_persistence'] == 'UNCONFIRMED'
    with psycopg.connect(pg[2], autocommit=True) as independent:
        observed = json.loads(independent.execute('SELECT payload_json FROM app_state WHERE namespace=%s', (q.QUARANTINE,)).fetchone()[0])
        assert observed['active']
    with pytest.raises(q.ReconciliationRequired):
        trader.run_trader_cycle()
    evidence(conn, 'quarantine_commit_lost_reply', {'fault': 'PYTHON_INJECTION_AFTER_REAL_SERVER_COMMIT',
        'independent_connection_observed_active_quarantine': True, 'false_success_not_reported': True,
        'not_a_real_tcp_failure': True, 'trades_not_replayed': True})


@pytest.mark.parametrize('failure', ['quarantine_write', 'outer_rollback'])
def test_unanchored_runtime_conflict_cannot_reopen_after_failed_quarantine(pg, monkeypatch, failure):
    p, conn, _, _ = pg
    cfg = importlib.import_module('core.config')
    retention = importlib.import_module('core.retention')
    q = importlib.import_module('core.commit_protocol')
    first = legacy_report()
    p.replace_table(cfg.TRADER_REPORTS_FILE, [first, {**first, 'entry_price': first['entry_price'] + 1}])
    before = snapshot(conn)[1]
    if failure == 'quarantine_write':
        save = p.save_json_state
        with monkeypatch.context() as patch:
            def unavailable(namespace, *args, **kwargs):
                if namespace == q.QUARANTINE:
                    raise OSError('PYTHON_INJECTION unavailable quarantine')
                return save(namespace, *args, **kwargs)
            patch.setattr(p, 'save_json_state', unavailable)
            with pytest.raises(OSError, match='PYTHON_INJECTION'):
                retention.archive_old_trade_reports(36500)
    else:
        with pytest.raises(ValueError, match='caller rollback'):
            with p.persistence_transaction():
                result = retention.archive_old_trade_reports(36500)
                assert result['blocked'] and result['reason'] == 'RETENTION_CONFLICT'
                raise ValueError('caller rollback')
    assert not q._quarantine()['active'] and snapshot(conn)[1] == before
    assert q.gate_status()['blocked'], 'Unanchored conflicting history must not be admitted'
    assert restart_blocked()['blocked'] and q._quarantine()['active']
    assert snapshot(conn)[1] == before
    evidence(conn, 'unanchored_conflict_' + failure, {'persisted_data_enforces_block': True,
        'new_python_process_refused_finance': True, 'original_rows_equal': True,
        'no_prior_receipt_assumed': True, 'financial_retry': False})
