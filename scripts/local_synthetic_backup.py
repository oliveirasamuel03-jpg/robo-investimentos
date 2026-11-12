"""Backup/restore ONLY a runner-owned loopback PostgreSQL with synthetic data.

No external DSN/container or existing restore target is accepted. Dumps remain
private temporary artifacts. Restore data is compared before an audited STOP
quarantine; this procedure never starts a worker or executes a financial cycle.
"""
from __future__ import annotations
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import time
from uuid import uuid4

DOCKER = ['docker', '--host=unix:///var/run/docker.sock']


def owned_container(container):
    if container != os.environ.get('ATLAS_INFRA12_CONTAINER') or not re.fullmatch(r'[a-f0-9]{64}',container):
        raise ValueError('Container does not belong to this runner invocation')
    result = subprocess.run(DOCKER + ['inspect', '--format', '{{index .Config.Labels "atlas.scope"}}', container],
                            check=True, capture_output=True, text=True, timeout=10)
    if result.stdout.strip() != 'infra12-local-synthetic':
        raise ValueError('Container is not owned by the isolated test runner')


def local_params(dsn):
    from psycopg.conninfo import conninfo_to_dict
    params = conninfo_to_dict(dsn)
    if (params.get('host') != '127.0.0.1' or params.get('user') != 'atlas_fixture'
            or not params.get('dbname', '').startswith(('atlas_infra12_', 'atlas_infra123_'))):
        raise ValueError('Only the synthetic local runner database is accepted')
    return params


def inventory(conn):
    from psycopg import sql
    sequence = conn.execute("SELECT n.nspname,c.relname FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace WHERE c.oid=pg_get_serial_sequence('public.app_rows','seq')::regclass").fetchone()
    if sequence is None:
        raise AssertionError('Missing sequence owned by app_rows.seq')
    def serial(rows):
        return [[v.isoformat() if hasattr(v, 'isoformat') else v for v in row] for row in rows]
    return {
        'app_state': serial(conn.execute('SELECT namespace,payload_json,updated_at FROM app_state ORDER BY namespace').fetchall()),
        'app_rows': serial(conn.execute('SELECT seq,namespace,row_json,created_at FROM app_rows ORDER BY seq').fetchall()),
        'sequence_name': list(sequence),
        'sequence': serial(conn.execute(sql.SQL('SELECT last_value,is_called FROM {}').format(sql.Identifier(*sequence))).fetchall()),
        'indexes': serial(conn.execute("SELECT tablename,indexname,indexdef FROM pg_indexes WHERE schemaname='public' ORDER BY tablename,indexname").fetchall()),
    }


def digest(snapshot):
    return hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode()).hexdigest()


def dump_restore(dsn, container, directory):
    started = time.perf_counter()
    import psycopg
    from psycopg import sql
    from psycopg.conninfo import make_conninfo
    import core.persistence as p
    params = local_params(dsn)
    owned_container(container)
    published = subprocess.run(DOCKER+['inspect','--format','{{json .NetworkSettings.Ports}}',container],
                               check=True,capture_output=True,text=True,timeout=10)
    bindings = json.loads(published.stdout)['5432/tcp']
    if len(bindings) != 1 or bindings[0]['HostIp'] != '127.0.0.1' or bindings[0]['HostPort'] != params.get('port'):
        raise ValueError('DSN does not match the owned loopback PostgreSQL port')
    # Quiesce the source BEFORE exporting a snapshot. Otherwise an intent
    # captured PREPARED could still commit on the source after the dump, while
    # an isolated restore would only observe its older preimage. This is a STOP
    # control action, never a financial retry or a RUNNING recovery action.
    previous_url = os.environ.get('DATABASE_URL')
    try:
        os.environ['DATABASE_URL'] = dsn
        from core.stop_control import request_stop
        from core.state_store import load_bot_state
        source_previous_status = load_bot_state().get('bot_status')
        source_stop = request_stop('synthetic_backup_quiesce')
        if not source_stop.persisted or load_bot_state().get('bot_status') != 'STOPPED':
            raise AssertionError('Source STOP unconfirmed; backup is not authorized')
    finally:
        if previous_url is None:
            os.environ.pop('DATABASE_URL', None)
        else:
            os.environ['DATABASE_URL'] = previous_url
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with psycopg.connect(dsn, autocommit=True) as source:
        source.execute("SET statement_timeout='5s'")
        # Session lock covers the exported read-only snapshot, sequences and dump.
        # It conflicts with the same xact lock used by all cooperating writers.
        source.execute('SELECT pg_advisory_lock(%s)', (int.from_bytes(hashlib.sha256(b'atlas:persistence-unit').digest()[:8], 'big', signed=True),))
        locked = time.perf_counter()
        with source.transaction():
            source.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY')
            snapshot = source.execute('SELECT pg_export_snapshot()').fetchone()[0]
            before = inventory(source)
            state_rows = {row[0]: row[1] for row in before['app_state']}
            if json.loads(state_rows['bot_state']).get('bot_status') != 'STOPPED':
                raise AssertionError('Source control changed before backup fence; abort')
            if 'paper_commit_control' not in state_rows:
                raise AssertionError('Source financial control is missing; refuse an operational restore')
            dump_started = time.perf_counter()
            dump = subprocess.run(DOCKER + ['exec', container, 'pg_dump', '-U', 'atlas_fixture',
                    '-d', params['dbname'], '-Fc', '--snapshot', snapshot],
                    check=True, capture_output=True, timeout=30).stdout
            dump_seconds = time.perf_counter() - dump_started
        source.execute('SELECT pg_advisory_unlock_all()')
        lock_seconds = time.perf_counter() - locked
        restore_name = 'atlas_infra123_restore_' + uuid4().hex[:16]
        source.execute(sql.SQL('CREATE DATABASE {}').format(sql.Identifier(restore_name)))
    dump_file = directory/'synthetic.dump'
    dump_file.write_bytes(dump)
    dump_file.chmod(0o600)
    restored_dsn = make_conninfo(dsn, dbname=restore_name)
    with psycopg.connect(restored_dsn, autocommit=True) as restored:
        assert restored.execute("SELECT to_regclass('public.app_state')").fetchone()[0] is None
    restore_started = time.perf_counter()
    subprocess.run(DOCKER + ['exec', '-i', container, 'pg_restore', '-U', 'atlas_fixture',
                   '-d', restore_name, '--single-transaction', '--exit-on-error'],
                   input=dump, check=True, capture_output=True, timeout=30)
    restore_seconds = time.perf_counter() - restore_started
    with psycopg.connect(restored_dsn, autocommit=True) as restored:
        after = inventory(restored)
    if before != after:
        raise AssertionError('Restored rows, STOP/UNKNOWN/receipts, indexes or sequence differ')
    # No app/worker may use the restore before this quarantine succeeds.
    previous = os.environ.get('DATABASE_URL')
    try:
        os.environ['DATABASE_URL'] = restored_dsn
        from core.stop_control import request_stop
        from core.state_store import load_bot_state
        from core.commit_protocol import gate_status
        gate_before = gate_status()
        stop = request_stop('synthetic_restore_quarantine')
        if not stop.persisted or load_bot_state().get('bot_status') != 'STOPPED':
            raise AssertionError('Restore remains quarantined: STOP was not verified')
        gate_after = gate_status()
        if gate_before['blocked'] and (not gate_after['blocked'] or gate_before['cycle_id'] != gate_after['cycle_id']):
            raise AssertionError('Restore quarantine changed an unresolved financial intent')
    finally:
        if previous is None:
            os.environ.pop('DATABASE_URL', None)
        else:
            os.environ['DATABASE_URL'] = previous
    evidence = {
        'scope': 'LOCAL_SYNTHETIC_ONLY', 'dump_sha256': hashlib.sha256(dump).hexdigest(),
        'dump_bytes': len(dump), 'source_inventory_sha256': digest(before),
        'restored_inventory_sha256': digest(after), 'all_rows_equal_before_quarantine': before == after,
        'state_namespaces': len(before['app_state']), 'rows': len(before['app_rows']),
        'indexes_equal': before['indexes'] == after['indexes'], 'sequence_equal': before['sequence'] == after['sequence'],
        'source_stop_verified_before_snapshot': True, 'source_previous_bot_status': source_previous_status,
        'stop_quarantine_verified': True, 'pending_intent_preserved': gate_before['blocked'] == gate_after['blocked'],
        'gate_blocked_after_quarantine': gate_after['blocked'],
        'dump_seconds': dump_seconds, 'restore_seconds': restore_seconds,
        'backup_writer_lock_seconds': lock_seconds, 'total_seconds': time.perf_counter()-started,
    }
    return restored_dsn, evidence
