"""ISOLATED LOCAL PROTOTYPE. Not imported by application/runtime code.

Full existing verification on REPEATABLE READ READ ONLY, then exact raw recheck
under the writer fence AND short SQL table locks. No counter/cache is proof.
Only reconciliation metadata is published; never runs/retries a financial unit.
Requires this invocation's disposable synthetic runner and PAPER environment.
"""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass, field
import importlib.util
import json
import os
from pathlib import Path
import secrets
import time

import psycopg
from psycopg.conninfo import conninfo_to_dict, make_conninfo


@dataclass(frozen=True)
class _Proof:
    owner: str = field(repr=False)
    identity: tuple = field(repr=False)
    capture: tuple = field(repr=False)
    selected_cycle: str | None
    result_json: str = field(repr=False)
    writes: tuple = field(repr=False)

    @property
    def result(self):
        return json.loads(self.result_json)


class _ReadOnlyAdapter:
    def __init__(self, persistence, connection, source_url):
        self.p, self.conn, self.source_url = persistence, connection, source_url
        self.writes = []

    def __getattr__(self, name):
        return getattr(self.p, name)

    @contextmanager
    def _reader(self):
        yield self.conn

    def _state_source(self, namespace, path):
        return (self.source_url, namespace)

    def load_json_state(self, namespace, default_factory, path):
        row = self.conn.execute('SELECT payload_json FROM app_state WHERE namespace=%s', (namespace,)).fetchone()
        raw = row[0] if row else None
        value = self.p._decode_json(raw, dict, 'prototype snapshot') if row else default_factory()
        return self.p.StateSnapshot(value, self._state_source(namespace, path), raw)

    def save_json_state(self, namespace, value, path):
        # No SQL write, lock, gate release or financial confirmation here.
        self.writes.append((namespace, self.p._encode(value), value.expected))


class SnapshotPrototype:
    def __init__(self, dsn, *, connect=None, publication_proxy=None):
        from scripts.local_synthetic_backup import local_params, owned_container
        if os.environ.get('ATLAS_INFRA12_LOCAL_RUN') != 'synthetic-disposable-only':
            raise ValueError('Prototype requires the disposable synthetic runner')
        if os.environ.get('BROKER_MODE') != 'paper' or os.environ.get('BROKER_PROVIDER') != 'paper':
            raise ValueError('Prototype is PAPER only')
        params = local_params(dsn)
        admin = conninfo_to_dict(os.environ['ATLAS_INFRA12_LOCAL_DSN'])
        if params.get('port') != admin.get('port'):
            raise ValueError('Prototype source must use the owned runner port')
        owned_container(os.environ['ATLAS_INFRA12_CONTAINER'])
        self.dsn = dsn
        self.publication_dsn = dsn
        if publication_proxy is not None:
            from tests.infra123_tcp_proxy import LocalPostgresProxy
            if not isinstance(publication_proxy, LocalPostgresProxy) or publication_proxy.target != ('127.0.0.1', int(params['port'])):
                raise ValueError('Only an owned loopback proxy to this test server is accepted')
            self.publication_dsn = make_conninfo(dsn, port=publication_proxy.port, sslmode='disable', gssencmode='disable')
        self.connect = connect or (lambda url: psycopg.connect(url, autocommit=True))
        self.owner = secrets.token_hex(16)  # ephemeral, never persisted/exported as authority
        self.metrics = []

    @staticmethod
    def _identity(conn):
        return tuple(str(v) for v in conn.execute("SELECT current_database(), (SELECT oid FROM pg_database WHERE datname=current_database()), (SELECT system_identifier FROM pg_control_system()), pg_postmaster_start_time()").fetchone())

    @staticmethod
    def _capture(conn, cycle_id):
        import core.commit_protocol as q
        namespaces = ['paper_state', 'bot_state', q.GATE]
        if cycle_id:
            namespaces.append(q.RECEIPT_PREFIX+cycle_id)
        states = tuple(conn.execute('SELECT namespace,payload_json FROM app_state WHERE namespace=ANY(%s) ORDER BY namespace', (namespaces,)).fetchall())
        rows = tuple(conn.execute('SELECT seq,namespace,row_json FROM app_rows WHERE namespace=ANY(%s) ORDER BY seq', (sorted(q.FINANCIAL_ROWS),)).fetchall())
        return (states, rows)

    def _after_snapshot(self, connection):
        """Test seam; no work in the prototype's default path."""

    def _instrument_verifier(self, verifier):
        """Test profiling seam; default does not change verification."""

    def _before_publish_commit(self, connection):
        """Test seam, not a financial retry/cancellation facility."""

    def prepare(self, cycle_id=None, *, release=True):
        import core.persistence as p
        import core.commit_protocol as q
        if p._ACTIVE_CONNECTION.get() is not None:
            raise RuntimeError('Prototype verification must run outside a writer transaction')
        started = time.perf_counter()
        with self.connect(self.dsn) as conn, conn.transaction():
            conn.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY')
            conn.execute("SET LOCAL statement_timeout='15s'")
            identity = self._identity(conn)
            adapter = _ReadOnlyAdapter(p, conn, p.get_database_url())
            gate = adapter.load_json_state(q.GATE, lambda: {'active_cycle_id':None}, q._GATE_PATH)
            selected = cycle_id or gate.get('active_cycle_id')
            capture = self._capture(conn, selected)
            self._after_snapshot(conn)
            # Run the UNCHANGED verifier source in private globals. Only its reads
            # and bookkeeping sink are adapted; no monkeypatch of runtime modules.
            spec = importlib.util.spec_from_file_location('_infra125_private_verifier', Path(q.__file__))
            verifier = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(verifier)
            verifier.p = adapter
            @contextmanager
            def readonly_scope():
                yield
            verifier._scope = readonly_scope
            self._instrument_verifier(verifier)
            result = verifier.reconcile(cycle_id, release=release)
            if result.get('outcome') == 'UNKNOWN_COMMIT' or not result.get('resolution_recorded'):
                adapter.writes.clear()
            proof = _Proof(self.owner, identity, capture, selected, p._encode(result), tuple(adapter.writes))
        self.metrics.append({'phase':'snapshot_full_verification','ms':(time.perf_counter()-started)*1000,
                             'writer_lock_held':False})
        return proof

    def publish(self, proof):
        import core.persistence as p
        import core.commit_protocol as q
        failure = {'outcome':'UNKNOWN_COMMIT','blocked':True,'resolution_recorded':False,
                   'cycle_id':getattr(proof,'selected_cycle',None),'prototype':True}
        if not isinstance(proof,_Proof) or proof.owner != self.owner:
            return {**failure,'reason':'FOREIGN_OR_RESTARTED_PROOF'}
        result = proof.result
        if result.get('outcome') == 'UNKNOWN_COMMIT' or not result.get('resolution_recorded'):
            return {**result,'prototype':True}  # never writes/releases on missing evidence
        entered = time.perf_counter()
        verified_current = False
        acquired = None
        wait_started = None
        try:
            with self.connect(self.publication_dsn) as conn:
                with p._transaction_context(conn, None):
                    conn.execute("SET LOCAL lock_timeout='5s'")
                    wait_started = time.perf_counter()
                    p._lock_namespace(conn,'atlas:persistence-unit')
                    acquired = time.perf_counter()
                    p._lock_namespace(conn,q.GATE)
                    # Brief table fence covers uncooperative INSERT/UPDATE/DELETE,
                    # namespace moves and TRUNCATE, not only versioned app writers.
                    conn.execute('LOCK TABLE app_state,app_rows IN SHARE ROW EXCLUSIVE MODE')
                    if self._identity(conn) != proof.identity or self._capture(conn, proof.selected_cycle) != proof.capture:
                        return {**failure,'reason':'SNAPSHOT_CHANGED_OR_SERVER_RESTARTED'}
                    verified_current = True
                    token = p._ACTIVE_CONNECTION.set(conn)
                    try:
                        with q.protocol_write():
                            for namespace, payload, expected in proof.writes:
                                if namespace != q.GATE and namespace != q.RECEIPT_PREFIX+proof.selected_cycle:
                                    raise ValueError('Prototype cannot write a financial namespace')
                                state = p.StateSnapshot(p._decode_json(payload,dict,'verified bookkeeping'),
                                                        (p.get_database_url(),namespace), expected)
                                path = q._GATE_PATH if namespace==q.GATE else q._receipt_path(proof.selected_cycle)
                                p.save_json_state(namespace,state,path)
                        self._before_publish_commit(conn)
                    finally:
                        p._ACTIVE_CONNECTION.reset(token)
            return {**result,'prototype':True,'evidence':'FULL_READ_ONLY_SNAPSHOT_AND_EXACT_FENCED_RAW_MATCH'}
        except Exception as exc:
            # Bookkeeping failure cannot become rollback/retry of a financial unit.
            return {**(result if verified_current else failure),'blocked':True,'resolution_recorded':False,
                    'prototype':True,'reason':'RESOLUTION_PUBLICATION_UNCONFIRMED','error_code':type(exc).__name__}
        finally:
            finished=time.perf_counter()
            self.metrics.append({'phase':'publication','wait_global_ms':((acquired or finished)-wait_started)*1000 if wait_started else None,
                                 'hold_until_client_return_ms':(finished-acquired)*1000 if acquired else 0})

    def reconcile(self, cycle_id=None, *, release=True):
        try:
            return self.publish(self.prepare(cycle_id,release=release))
        except Exception as exc:
            return {'outcome':'UNKNOWN_COMMIT','blocked':True,'resolution_recorded':False,
                    'cycle_id':cycle_id,'prototype':True,'reason':'SNAPSHOT_READ_FAILED','error_code':type(exc).__name__}
