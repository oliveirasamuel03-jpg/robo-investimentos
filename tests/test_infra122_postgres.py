"""COMMIT boundary injection against a real disposable PostgreSQL server.

The lost-response exception is simulated in Python AFTER an actual server
COMMIT. A separate connection observes the receipt/ledger before reconciliation.
It is not a claim to have dropped real network packets or failed infrastructure.
"""
from contextlib import contextmanager
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest
from tests.test_infra12_postgres import pg, pytestmark, ROOT
from tests.test_infra121_postgres import _running, _synthetic_market


def _setup(pg, monkeypatch):
    p, conn, dsn, root = pg
    store, control, paper, trader = _running()
    _synthetic_market(monkeypatch, paper)
    paper.load_paper_state()
    q = importlib.import_module('core.commit_protocol')
    return p, conn, store, control, paper, trader, q


def _response_fault(monkeypatch, p, *, committed):
    import psycopg
    original = p._transaction_context
    @contextmanager
    def boundary(conn, witness):
        with original(conn, witness):
            yield
            if witness is not None and not committed:
                raise psycopg.OperationalError('SIMULATED communication failure during COMMIT attempt')
        if witness is not None and committed:
            raise psycopg.OperationalError('SIMULATED lost reply AFTER actual server COMMIT')
    monkeypatch.setattr(p, '_transaction_context', boundary)
    return original


def test_failure_before_commit_has_positive_not_committed_proof(pg, monkeypatch):
    p, conn, _, _, paper, trader, q = _setup(pg, monkeypatch)
    before = dict(paper.load_paper_state())
    def before_send(conn, witness):
        raise ValueError('synthetic failure BEFORE sending COMMIT')
    monkeypatch.setattr(p, '_before_financial_commit', before_send)
    with pytest.raises(ValueError) as caught:
        trader.run_trader_cycle()
    assert caught.value.commit_outcome == 'NOT_COMMITTED'
    assert q.gate_status()['outcome'] == 'NOT_COMMITTED' and not q.gate_status()['blocked']
    assert paper.load_paper_state() == before and paper.read_paper_trades() == []
    assert conn.execute("SELECT count(*) FROM app_rows WHERE namespace='paper_trades'").fetchone()[0] == 0


@pytest.mark.parametrize('server_committed', [False, True])
def test_unknown_boundary_is_fenced_until_explicit_evidence_reconciliation(pg, monkeypatch, server_committed):
    p, conn, _, _, paper, trader, q = _setup(pg, monkeypatch)
    original = _response_fault(monkeypatch, p, committed=server_committed)
    with pytest.raises(q.UnknownCommit) as caught:
        trader.run_trader_cycle()
    cycle_id = caught.value.cycle_id
    # This is an independent connection, observing real server persistence.
    payload = json.loads(conn.execute('SELECT payload_json FROM app_state WHERE namespace=%s',
                                     (q.RECEIPT_PREFIX + cycle_id,)).fetchone()[0])
    assert payload['status'] == ('COMMITTED' if server_committed else 'PREPARED')
    assert len(paper.read_paper_trades()) == (1 if server_committed else 0)
    assert q.gate_status()['blocked']
    monkeypatch.setattr(p, '_transaction_context', original)
    with pytest.raises(q.ReconciliationRequired):
        trader.run_trader_cycle()
    resolved = q.reconcile(cycle_id)
    assert resolved['outcome'] == ('COMMITTED' if server_committed else 'NOT_COMMITTED')
    assert not resolved['blocked'] and not q.gate_status()['blocked']
    # Reconciliation itself did not append or replay any financial event.
    assert len(paper.read_paper_trades()) == (1 if server_committed else 0)


@pytest.mark.parametrize('where', ['paper_report', 'trader_diagnostic'])
def test_post_commit_failure_is_not_reclassified_as_rollback(pg, monkeypatch, where):
    p, conn, _, _, paper, trader, q = _setup(pg, monkeypatch)
    def failed(*args, **kwargs):
        raise RuntimeError('synthetic diagnostic failed after commit')
    if where == 'paper_report':
        monkeypatch.setattr(paper, 'build_paper_report', failed)
    else:
        monkeypatch.setattr(trader, 'update_market_data_status', failed)
    result = trader.run_trader_cycle()
    assert result['committed'] is True
    assert result['commit_outcome'] == 'POST_COMMIT_DIAGNOSTIC_FAILURE'
    assert len(paper.read_paper_trades()) == 1
    assert q.gate_status()['blocked']
    assert q.gate_status()['outcome'] == 'POST_COMMIT_DIAGNOSTIC_FAILURE'
    with pytest.raises(q.ReconciliationRequired):
        trader.run_trader_cycle()
    assert q.reconcile()['outcome'] == 'COMMITTED'
    assert len(paper.read_paper_trades()) == 1


def test_worker_restart_unknown_preserves_fence_and_stop(pg, monkeypatch):
    p, conn, store, control, paper, trader, q = _setup(pg, monkeypatch)
    original = _response_fault(monkeypatch, p, committed=True)
    with pytest.raises(q.UnknownCommit):
        trader.run_trader_cycle()
    monkeypatch.setattr(p, '_transaction_context', original)
    # A fresh interpreter has no memory of the lost response.
    program = """
import json
from core.commit_protocol import gate_status, ReconciliationRequired
from engines.trader_engine import run_trader_cycle
try:
 run_trader_cycle()
except ReconciliationRequired:
 print(json.dumps({'blocked':gate_status()['blocked']}))
else:
 raise AssertionError('restart must not execute another cycle')
"""
    completed = subprocess.run([sys.executable, '-c', program], cwd=ROOT,
                               capture_output=True, text=True, check=True, timeout=25)
    assert json.loads(completed.stdout.strip()) == {'blocked': True}
    assert control.request_stop('synthetic-operator').persisted
    assert control.acknowledge_stop('synthetic-worker').acknowledged
    assert trader.run_trader_cycle()['status'] == 'STOPPED'
    assert q.gate_status()['blocked'] and len(paper.read_paper_trades()) == 1
    with pytest.raises(q.ReconciliationRequired):
        control.start_robot(store.load_bot_state())
    with pytest.raises(q.ReconciliationRequired):
        store.reset_state()


@pytest.mark.parametrize('damage', ['receipt', 'balance', 'position', 'trade', 'report'])
def test_insufficient_or_divergent_evidence_never_releases_fence(pg, monkeypatch, damage):
    p, conn, _, _, paper, trader, q = _setup(pg, monkeypatch)
    original = _response_fault(monkeypatch, p, committed=True)
    with pytest.raises(q.UnknownCommit) as caught:
        trader.run_trader_cycle()
    cycle_id = caught.value.cycle_id
    monkeypatch.setattr(p, '_transaction_context', original)
    if damage == 'receipt':
        conn.execute('UPDATE app_state SET payload_json=%s WHERE namespace=%s',
                     ('{}', q.RECEIPT_PREFIX + cycle_id))
    elif damage in {'balance', 'position'}:
        state = json.loads(conn.execute("SELECT payload_json FROM app_state WHERE namespace='paper_state'").fetchone()[0])
        if damage == 'balance':
            state['cash'] += 1
        else:
            state['positions'] = {}
        conn.execute("UPDATE app_state SET payload_json=%s WHERE namespace='paper_state'", (json.dumps(state),))
    else:
        namespace = 'paper_trades' if damage == 'trade' else 'table:trade_reports.csv'
        conn.execute('INSERT INTO app_rows(namespace,row_json) VALUES (%s,%s)',
                     (namespace, json.dumps({'synthetic_divergence': True, 'cycle_id': cycle_id})))
    result = q.reconcile(cycle_id)
    assert result['outcome'] == 'UNKNOWN_COMMIT' and result['blocked']
    assert q.gate_status()['blocked']
    with pytest.raises(q.ReconciliationRequired):
        trader.run_trader_cycle()


def test_read_outage_then_reconnection_requires_explicit_reconciliation(pg, monkeypatch):
    p, _, _, _, paper, trader, q = _setup(pg, monkeypatch)
    original_tx = _response_fault(monkeypatch, p, committed=True)
    with pytest.raises(q.UnknownCommit):
        trader.run_trader_cycle()
    monkeypatch.setattr(p, '_transaction_context', original_tx)
    import psycopg
    connect = p._connect
    def unavailable():
        raise psycopg.OperationalError('synthetic unavailable')
    monkeypatch.setattr(p, '_connect', unavailable)
    assert q.reconcile()['outcome'] == 'UNKNOWN_COMMIT'
    with pytest.raises(psycopg.OperationalError):
        trader.run_trader_cycle()
    monkeypatch.setattr(p, '_connect', connect)
    assert q.gate_status()['blocked']
    assert q.reconcile()['outcome'] == 'COMMITTED'
    assert len(paper.read_paper_trades()) == 1


@pytest.mark.parametrize('side', ['BUY', 'SELL'])
def test_duplicate_identity_cannot_publish_again(pg, monkeypatch, side):
    p, _, _, _, paper, trader, q = _setup(pg, monkeypatch)
    trader.run_trader_cycle()
    if side == 'SELL':
        monkeypatch.setattr(paper, '_should_sell', lambda *_: (True, 'synthetic-close'))
        trader.run_trader_cycle()
    trades = paper.read_paper_trades()
    event = next(row for row in trades if row['side'] == side)
    draft = paper.load_paper_state()
    with pytest.raises(p.PersistenceDataError, match='already persisted'):
        paper._commit_paper_draft(draft, [event], [], paper.PaperTradingConfig())
    assert paper.read_paper_trades() == trades
    assert q.gate_status()['outcome'] == 'NOT_COMMITTED'


def test_legacy_fc593_response_loss_reproduction(pg, tmp_path):
    """Base version really commits on PG, raises generic error, then admits cycle2."""
    p, conn, dsn, _ = pg
    archive, legacy = tmp_path/'legacy.tar', tmp_path/'legacy'
    legacy.mkdir()
    subprocess.run(['git','archive','fc5931254c85d1b36591818fcafa0e204e37c5b3','-o',str(archive)], cwd=ROOT, check=True)
    with tarfile.open(archive) as source:
        source.extractall(legacy, filter='data')
    program = """
import json
from contextlib import contextmanager
import psycopg
from tests.test_infra121_postgres import _running, _synthetic_market
from pytest import MonkeyPatch
store, control, paper, trader = _running()
m=MonkeyPatch();_synthetic_market(m,paper)
original=paper._paper_commit_scope
@contextmanager
def lost_reply():
 with original():
  yield
 raise psycopg.OperationalError('SIMULATED lost response AFTER server COMMIT')
paper._paper_commit_scope=lost_reply
try:trader.run_trader_cycle()
except psycopg.OperationalError:pass
else:raise AssertionError('expected simulated lost response')
first=len(paper.read_paper_trades())
paper._paper_commit_scope=original
paper._should_sell=lambda *_:(True,'synthetic-close')
second=trader.run_trader_cycle()
print(json.dumps({'base':'fc5931254c85d1b36591818fcafa0e204e37c5b3',
 'fault':'PYTHON_SIMULATED_REPLY_LOSS_AFTER_ACTUAL_POSTGRES_COMMIT',
 'server_committed_trades_after_exception':first,'second_cycle_was_allowed':second['committed'],
 'server_trades_after_second_cycle':len(paper.read_paper_trades())}))
"""
    env=dict(os.environ, DATABASE_URL=dsn, ROBO_STORAGE_DIR=str(tmp_path/'legacy-storage'))
    result=subprocess.run([sys.executable,'-c',program],cwd=legacy,env=env,
                          capture_output=True,text=True,check=True,timeout=30)
    proof=json.loads(result.stdout.strip().splitlines()[-1])
    assert proof['server_committed_trades_after_exception']==1
    assert proof['second_cycle_was_allowed'] and proof['server_trades_after_second_cycle']==2
    # Independent server connection verifies the two actual records.
    assert conn.execute("SELECT count(*) FROM app_rows WHERE namespace='paper_trades'").fetchone()[0]==2
    (ROOT/'docs/retomada/evidencias/infra122-legacy-response-loss.json').write_text(json.dumps(proof,indent=2)+'\n')


def test_failed_verification_after_positive_commit_reply_is_post_commit_failure(pg, monkeypatch):
    p, conn, _, _, paper, trader, q = _setup(pg, monkeypatch)
    verify = q.reconcile
    def unavailable(*args, **kwargs):
        return {'outcome': 'UNKNOWN_COMMIT', 'blocked': True, 'reason': 'SIMULATED_READ_FAILURE'}
    monkeypatch.setattr(q, 'reconcile', unavailable)
    result = trader.run_trader_cycle()
    assert result['commit_outcome'] == 'POST_COMMIT_DIAGNOSTIC_FAILURE'
    assert result['committed'] is True and len(paper.read_paper_trades()) == 1
    assert q.gate_status()['blocked']
    monkeypatch.setattr(q, 'reconcile', verify)
    assert q.reconcile()['outcome'] == 'COMMITTED'


def test_confirmed_commit_then_scope_cleanup_failure_is_post_commit(pg, monkeypatch):
    p, conn, _, _, paper, trader, q = _setup(pg, monkeypatch)
    original = paper._paper_commit_scope
    @contextmanager
    def failed_cleanup(witness=None):
        with original(witness):
            yield
        if witness is not None:
            assert witness.phase == 'COMMITTED'
            raise OSError('SIMULATED cleanup failure AFTER positive COMMIT reply')
    monkeypatch.setattr(paper, '_paper_commit_scope', failed_cleanup)
    result = trader.run_trader_cycle()
    assert result['committed'] and result['commit_outcome'] == 'POST_COMMIT_DIAGNOSTIC_FAILURE'
    assert len(paper.read_paper_trades()) == 1 and q.gate_status()['blocked']
    monkeypatch.setattr(paper, '_paper_commit_scope', original)
    assert q.reconcile()['outcome'] == 'COMMITTED'


def test_worker_post_return_failure_fences_verified_cycle_without_financial_retry(pg, monkeypatch):
    p, conn, _, _, paper, trader, q = _setup(pg, monkeypatch)
    result = trader.run_trader_cycle()
    assert result['commit_outcome'] == 'COMMITTED' and not q.gate_status()['blocked']
    # Same bounded failure handler used by the worker; no continuous loop starts.
    failure = q.post_commit_failure(result['cycle_id'])
    assert failure['outcome'] == 'POST_COMMIT_DIAGNOSTIC_FAILURE'
    assert q.gate_status()['blocked']
    with pytest.raises(q.ReconciliationRequired):
        trader.run_trader_cycle()
    assert q.reconcile()['outcome'] == 'COMMITTED' and len(paper.read_paper_trades()) == 1


def test_committed_but_inconsistent_platform_is_detected_not_silently_reconciled(pg, monkeypatch):
    p, conn, _, _, paper, trader, q = _setup(pg, monkeypatch)
    # Simulate a defective finalizer while the server really commits. The receipt
    # hashes alone must not bless a logically inconsistent platform mirror.
    monkeypatch.setattr(trader, 'sync_platform_positions_from_paper', lambda: None)
    result = trader.run_trader_cycle()
    assert result['commit_outcome'] == 'POST_COMMIT_DIAGNOSTIC_FAILURE'
    assert q.reconcile()['outcome'] == 'UNKNOWN_COMMIT'
    assert q.gate_status()['blocked'] and len(paper.read_paper_trades()) == 1


def test_reconciliation_bookkeeping_failure_does_not_erase_observed_commit_evidence(pg, monkeypatch):
    p, conn, _, _, paper, trader, q = _setup(pg, monkeypatch)
    original = _response_fault(monkeypatch, p, committed=True)
    with pytest.raises(q.UnknownCommit):
        trader.run_trader_cycle()
    monkeypatch.setattr(p, '_transaction_context', original)
    save = p.save_json_state
    def failure(namespace, *args, **kwargs):
        if namespace.startswith(q.RECEIPT_PREFIX):
            raise OSError('SIMULATED reconciliation bookkeeping failure')
        return save(namespace, *args, **kwargs)
    monkeypatch.setattr(p, 'save_json_state', failure)
    result = q.reconcile()
    assert result['outcome'] == 'COMMITTED' and result['resolution_recorded'] is False
    assert result['blocked'] and q.gate_status()['blocked']
    monkeypatch.setattr(p, 'save_json_state', save)
    assert q.reconcile()['outcome'] == 'COMMITTED'
    assert len(paper.read_paper_trades()) == 1
