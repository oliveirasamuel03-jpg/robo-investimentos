"""Writer/retention counterexamples and toy privileges, never runtime changes.

Every positive assertion about a defect describes the current limitation; it
does not approve that behavior. The gateway has generic markers, no ledger.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
import importlib
import json
import secrets
import threading

import psycopg
from psycopg import sql
from psycopg.conninfo import make_conninfo
import pytest
from tests.test_infra12_postgres import pg, pytestmark, ROOT
from tests.test_infra122_postgres import _setup
from tests.test_infra123_postgres import _seed_full_synthetic_history


def evidence(conn, name, values):
    major = str(conn.execute('SHOW server_version_num').fetchone()[0])[:2]
    path = ROOT / f'docs/retomada/evidencias/infra126-contract-pg{major}.json'
    data = json.loads(path.read_text()) if path.exists() else {
        'scope': 'LOCAL_SYNTHETIC_STUDY_ONLY', 'runtime_changed': False, 'cases': {}}
    data['cases'][name] = values
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')


def rows(conn):
    return conn.execute('SELECT seq,namespace,row_json FROM app_rows ORDER BY seq').fetchall()


def test_noop_retention_changes_current_receipt_without_persisting_a_gate(pg, monkeypatch):
    p, conn, store, control, paper, trader, q, cycle = _seed_full_synthetic_history(pg, monkeypatch)
    retention = importlib.import_module('core.retention')
    before = q.manifest()
    before_rows = rows(conn)
    result = retention.archive_old_trade_reports(retention_days=36500)
    assert result['archived_rows'] == 0 and result['retained_rows'] == 1
    assert q.manifest() != before and rows(conn) != before_rows
    reports = p.load_json_rows('table:trade_reports.csv', ROOT / 'unused.json')
    assert len(reports) == 1 and 'cycle_id' not in reports[0]
    answer = q.reconcile(cycle)
    assert answer['outcome'] == 'UNKNOWN_COMMIT' and answer['blocked']
    # Defect proof: returned blocking status is not persisted if gate was clear.
    assert not q.gate_status()['blocked']
    q.assert_execution_allowed()  # NO subsequent financial operation is executed.
    assert len(paper.read_paper_trades()) == 3
    evidence(conn, 'noop_retention_and_unpersisted_block', {
        'archived_rows': 0, 'manifest_changed': True, 'cycle_id_removed_from_report': True,
        'reconcile_result': answer['outcome'], 'returned_blocked': True,
        'persisted_gate_blocked': False, 'execution_guard_still_allows': True,
        'classification': 'CRITICAL_OPEN_DEFECT_NOT_FIXED_IN_STUDY'})


def test_retained_conflicting_reports_collapse_without_archive(pg, monkeypatch):
    p, conn, store, control, paper, trader, q, cycle = _seed_full_synthetic_history(pg, monkeypatch)
    retention = importlib.import_module('core.retention')
    report = p.load_json_rows('table:trade_reports.csv', ROOT / 'unused.json')[0]
    changed = {**report, 'entry_price': report['entry_price'] + 1}
    # Deliberately inconsistent SYNTHETIC records: maintenance must not be
    # mistaken for a lossless evidence archive or an integrity certification.
    conn.execute("INSERT INTO app_rows(namespace,row_json) VALUES ('table:trade_reports.csv',%s)",
                 (json.dumps(changed),))
    result = retention.archive_old_trade_reports(retention_days=36500)
    after = p.load_json_rows('table:trade_reports.csv', ROOT / 'unused.json')
    assert result['archived_rows'] == 0 and len(after) == 1
    assert after[0]['entry_price'] == changed['entry_price']
    assert conn.execute("SELECT count(*) FROM app_rows WHERE namespace LIKE 'table:trade_reports_%%' AND namespace <> 'table:trade_reports.csv'").fetchone()[0] == 0
    evidence(conn, 'retained_conflicting_reports', {'synthetic_reports_before': 2,
        'reports_after': 1, 'archived_rows': 0, 'different_payload_same_dedupe_key': True,
        'classification': 'HIGH_LOSS_OF_EVIDENCE_NOT_FIXED'})


def test_unknown_rolls_back_retention_and_catalog_unit(pg, monkeypatch):
    p, conn, store, control, paper, trader, q, cycle = _seed_full_synthetic_history(pg, monkeypatch)
    retention = importlib.import_module('core.retention')
    q.post_commit_failure(cycle, confirmed=True)
    before_rows = rows(conn)
    before_state = conn.execute('SELECT namespace,payload_json FROM app_state ORDER BY namespace').fetchall()
    with pytest.raises(q.ReconciliationRequired):
        with p.persistence_transaction():
            result = retention.run_retention_job(now=datetime(2027, 5, 1, tzinfo=timezone.utc))
            store.update_retention_status(result)
    assert rows(conn) == before_rows
    assert conn.execute('SELECT namespace,payload_json FROM app_state ORDER BY namespace').fetchall() == before_state
    assert q.gate_status()['blocked']
    assert control.request_stop('synthetic-retention-unknown').persisted
    assert store.load_bot_state()['bot_status'] == 'STOPPED'
    evidence(conn, 'unknown_retention_atomic_rollback', {'runtime_archive_catalog_unchanged': True,
        'unknown_preserved': True, 'stop_persisted': True, 'financial_retry': False})


def test_backup_of_retained_history_is_exact_but_does_not_repair_receipts(pg, monkeypatch, tmp_path):
    from scripts.local_synthetic_backup import dump_restore
    p, conn, store, control, paper, trader, q, cycle = _seed_full_synthetic_history(pg, monkeypatch)
    retention = importlib.import_module('core.retention')
    with p.persistence_transaction():
        result = retention.run_retention_job(now=datetime(2027, 5, 1, tzinfo=timezone.utc))
        store.update_retention_status(result)
    assert result['last_summary']['trade_reports_archived_rows'] == 1
    before_rows = rows(conn)
    restored_dsn, report = dump_restore(pg[2], __import__('os').environ['ATLAS_INFRA12_CONTAINER'], tmp_path)
    with psycopg.connect(restored_dsn, autocommit=True) as restored:
        assert rows(restored) == before_rows
    with monkeypatch.context() as patch:
        patch.setenv('DATABASE_URL', restored_dsn)
        assert store.load_bot_state()['bot_status'] == 'STOPPED'
        answer = q.reconcile(cycle)
        assert answer['outcome'] == 'UNKNOWN_COMMIT'
        assert len(paper.read_paper_trades()) == 3
    evidence(conn, 'retention_backup_restore', {'all_rows_equal': True,
        'archived_reports': 1, 'restored_stopped': True, 'receipt_still_unknown': True,
        'backup_is_not_integrity_repair': True, 'financial_retry': False})


def test_table_namespaces_alias_different_directories(pg):
    p, conn, _, root = pg
    first, second = root / 'one' / 'synthetic.csv', root / 'two' / 'SYNTHETIC.csv'
    p.replace_table(first, [{'marker': 'one'}], columns=['marker'])
    p.replace_table(second, [{'marker': 'two'}], columns=['marker'])
    assert p.read_table(first, columns=['marker']).iloc[0]['marker'] == 'two'
    assert conn.execute("SELECT count(DISTINCT namespace) FROM app_rows").fetchone()[0] == 1
    evidence(conn, 'basename_namespace_collision', {'different_paths_same_namespace': True,
        'contract': 'reserve canonical basenames; directory is not namespace isolation'})


@pytest.mark.parametrize('payload', ['{"value":NaN}', '{"value":1,"value":2}'])
def test_readonly_inventory_is_not_strict_integrity_evidence(pg, payload):
    from scripts.audit_database_readonly import inventory
    p, conn, _, root = pg
    p.load_json_state('synthetic-audit', dict, root / 'state.json')
    conn.execute("UPDATE app_state SET payload_json=%s WHERE namespace='synthetic-audit'", (payload,))
    report = inventory(conn)
    assert report['read_only']
    assert report['tables']['app_state']['invalid_payloads'] == 0
    with pytest.raises(p.PersistenceDataError):
        p.load_json_state('synthetic-audit', dict, root / 'state.json')
    evidence(conn, 'readonly_parser_' + ('nonfinite' if 'NaN' in payload else 'duplicate_keys'),
        {'readonly_tool_counts_as_valid': True, 'strict_runtime_rejects': True,
         'read_only_audit_is_not_financial_authority': True})


def test_nonadmin_raw_table_writer_bypasses_global_advisory_and_gate(pg, monkeypatch):
    p, conn, store, control, paper, trader, q = _setup(pg, monkeypatch)
    cycle = trader.run_trader_cycle()['cycle_id']
    q.post_commit_failure(cycle, confirmed=True)
    name, password = 'infra126_raw_' + secrets.token_hex(5), secrets.token_urlsafe(24)
    conn.execute(sql.SQL('CREATE ROLE {} LOGIN PASSWORD {} NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS').format(sql.Identifier(name), sql.Literal(password)))
    conn.execute(sql.SQL('GRANT USAGE ON SCHEMA public TO {}').format(sql.Identifier(name)))
    conn.execute(sql.SQL('GRANT SELECT,UPDATE ON app_state TO {}').format(sql.Identifier(name)))
    params = make_conninfo(pg[2], user=name, password=password)
    with p.persistence_transaction():
        with psycopg.connect(params, autocommit=True) as raw:
            assert raw.execute("SELECT rolsuper FROM pg_roles WHERE rolname=current_user").fetchone()[0] is False
            raw.execute("SET lock_timeout='500ms'")
            # Own synthetic gate only; no historical data or trade is removed.
            raw.execute("UPDATE app_state SET payload_json=%s WHERE namespace='paper_commit_control'", ('{}',))
    assert not q.gate_status()['blocked']
    evidence(conn, 'ordinary_dml_bypasses_guard', {'superuser': False,
        'advisory_lock_held': True, 'raw_update_succeeded': True,
        'pending_gate_cleared_outside_protocol': True, 'financial_cycle_executed_after_bypass': False})


@contextmanager
def gateway(pg):
    """Only synthetic roles in this invocation's private disposable server."""
    _, conn, dsn, _ = pg
    suffix = secrets.token_hex(5)
    owner, app, auditor = ['infra126_' + kind + '_' + suffix for kind in ('owner', 'app', 'audit')]
    password = secrets.token_urlsafe(24)
    for role in (owner, app, auditor):
        conn.execute(sql.SQL('CREATE ROLE {} {} NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS').format(
            sql.Identifier(role), sql.SQL('NOLOGIN') if role == owner else sql.SQL('LOGIN PASSWORD {}').format(sql.Literal(password))))
    # The owner gets CREATE only on this disposable database to install the toy.
    database = conn.execute('SELECT current_database()').fetchone()[0]
    conn.execute(sql.SQL('GRANT CREATE ON DATABASE {} TO {}').format(sql.Identifier(database), sql.Identifier(owner)))
    conn.execute(sql.SQL('SET ROLE {}').format(sql.Identifier(owner)))
    try:
        conn.execute((ROOT / 'scripts/studies/infra126_writer_contract_local.sql').read_text())
    finally:
        conn.execute('RESET ROLE')
    conn.execute(sql.SQL('GRANT USAGE ON SCHEMA infra126_contract TO {},{}').format(sql.Identifier(app), sql.Identifier(auditor)))
    conn.execute(sql.SQL('GRANT EXECUTE ON FUNCTION infra126_contract.write_marker(integer,bigint,text,text) TO {}').format(sql.Identifier(app)))
    conn.execute(sql.SQL('GRANT SELECT ON ALL TABLES IN SCHEMA infra126_contract TO {}').format(sql.Identifier(auditor)))
    with psycopg.connect(make_conninfo(dsn, user=app, password=password), autocommit=True) as writer, psycopg.connect(make_conninfo(dsn, user=auditor, password=password), autocommit=True) as reader:
        yield conn, writer, reader, owner, make_conninfo(dsn, user=app, password=password)


def marker(conn, version=1, revision=0, identity='synthetic', content='one'):
    return conn.execute('SELECT infra126_contract.write_marker(%s,%s,%s,%s)', (version, revision, identity, content)).fetchone()[0]


def test_gateway_actual_login_has_minimum_privileges_and_safe_search_path(pg):
    with gateway(pg) as (admin, app, reader, owner, dsn):
        for connection in (app, reader):
            assert connection.execute('SELECT rolsuper,rolcreatedb,rolcreaterole,rolreplication,rolbypassrls FROM pg_roles WHERE rolname=current_user').fetchone() == (False,) * 5
        assert marker(app) == 1
        assert reader.execute('SELECT count(*) FROM infra126_contract.markers').fetchone()[0] == 1
        commands = ["UPDATE infra126_contract.authority SET pending=false", "DELETE FROM infra126_contract.markers",
            "INSERT INTO infra126_contract.markers VALUES ('outside','bad')", 'TRUNCATE infra126_contract.markers',
            'ALTER TABLE infra126_contract.markers ADD COLUMN bypass int',
            'CREATE TABLE infra126_contract.bypass(id int)', "SET session_replication_role='replica'"]
        for command in commands:
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                app.execute(command)
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            app.execute(sql.SQL('SET ROLE {}').format(sql.Identifier(owner)))
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            marker(reader, revision=1, identity='reader-write')
        config = admin.execute("SELECT proconfig FROM pg_proc WHERE oid='infra126_contract.write_marker(integer,bigint,text,text)'::regprocedure").fetchone()[0]
        assert config == ['search_path=pg_catalog, infra126_contract']
        evidence(admin, 'least_privilege_marker_gateway', {'real_restricted_logins': True,
            'raw_dml_ddl_truncate_role_escalation_denied': True, 'reader_write_denied': True,
            'security_definer_search_path_fixed': True, 'financial_implementation': False})


def test_gateway_versions_rollback_and_pending_fence(pg):
    with gateway(pg) as (admin, app, reader, owner, dsn):
        with pytest.raises(psycopg.errors.RaiseException, match='INCOMPATIBLE_WRITER'):
            marker(app, version=0)
        with pytest.raises(ValueError):
            with app.transaction():
                marker(app)
                raise ValueError('synthetic rollback before COMMIT')
        assert reader.execute('SELECT revision FROM infra126_contract.authority').fetchone()[0] == 0
        assert reader.execute('SELECT count(*) FROM infra126_contract.markers').fetchone()[0] == 0
        with pytest.raises(ValueError):
            with admin.transaction():
                admin.execute('UPDATE infra126_contract.authority SET accepted_version=2')
                raise ValueError('synthetic version migration rollback')
        assert marker(app) == 1
        admin.execute('UPDATE infra126_contract.authority SET accepted_version=2,pending=true')
        with pytest.raises(psycopg.errors.RaiseException, match='INCOMPATIBLE_WRITER'):
            marker(app, revision=1, identity='old')
        with pytest.raises(psycopg.errors.RaiseException, match='PENDING_INTENT'):
            marker(app, version=2, revision=1, identity='new')
        assert reader.execute('SELECT count(*) FROM infra126_contract.markers').fetchone()[0] == 1
        evidence(admin, 'version_and_pending_marker_contract', {'old_version_rejected': True,
            'rollback_atomic': True, 'pending_not_cleared': True, 'migration_rollback_local_only': True})


def test_gateway_concurrent_writers_have_one_cas_winner_without_retry(pg):
    with gateway(pg) as (admin, app, reader, owner, dsn):
        barrier = threading.Barrier(2)
        def compete(identity):
            with psycopg.connect(dsn, autocommit=True) as writer:
                barrier.wait(timeout=5)
                try:
                    return marker(writer, identity=identity)
                except psycopg.errors.RaiseException as exc:
                    assert 'STALE_REVISION' in str(exc)
                    return 'STALE_REVISION'
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(compete, identity) for identity in ('A', 'B')]
            answers = [future.result(timeout=10) for future in futures]
        assert answers.count(1) == answers.count('STALE_REVISION') == 1
        assert reader.execute('SELECT count(*) FROM infra126_contract.markers').fetchone()[0] == 1
        evidence(admin, 'concurrent_marker_cas', {'barrier_synchronized': True,
            'successful_writers': 1, 'stale_writers': 1, 'automatic_retry': False})


def test_privileged_sql_can_change_content_without_revision_even_with_gateway(pg):
    with gateway(pg) as (admin, app, reader, owner, dsn):
        assert marker(app) == 1
        before = reader.execute('SELECT revision FROM infra126_contract.authority').fetchone()[0]
        admin.execute("UPDATE infra126_contract.markers SET content='ADMINISTRATIVELY_CHANGED'")
        assert reader.execute('SELECT revision FROM infra126_contract.authority').fetchone()[0] == before
        assert reader.execute('SELECT content FROM infra126_contract.markers').fetchone()[0] == 'ADMINISTRATIVELY_CHANGED'
        evidence(admin, 'privileged_sql_not_proven_by_revision', {'same_revision': True,
            'changed_content': True, 'requires_independent_content_audit': True,
            'universal_integrity_claim': False})
