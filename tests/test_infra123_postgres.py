"""Real local TCP, crash recovery, synthetic restores and bounded measurements."""
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import importlib
import json
import os
import selectors
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest
from psycopg.conninfo import conninfo_to_dict, make_conninfo
from tests.test_infra12_postgres import pg, pytestmark, ROOT
from tests.test_infra122_postgres import _setup
from tests.infra123_tcp_proxy import LocalPostgresProxy


def evidence(conn, category, name, value):
    major = str(conn.execute('SHOW server_version_num').fetchone()[0])[:2]
    path = ROOT/f'docs/retomada/evidencias/infra123-{category}-pg{major}.json'
    data = json.loads(path.read_text()) if path.exists() else {'scope': 'LOCAL_SYNTHETIC_ONLY', 'cases': {}}
    data['cases'][name] = value
    path.write_text(json.dumps(data, indent=2, sort_keys=True)+'\n')


@contextmanager
def proxy_database(monkeypatch, dsn):
    params = conninfo_to_dict(dsn)
    assert params['host'] == '127.0.0.1'
    with LocalPostgresProxy('127.0.0.1', int(params['port'])) as proxy:
        proxied = make_conninfo(dsn, port=proxy.port, sslmode='disable', gssencmode='disable')
        with monkeypatch.context() as patch:
            patch.setenv('DATABASE_URL', proxied)
            yield proxy, proxied, patch


@pytest.mark.parametrize('mode', ['before_commit', 'partial_commit', 'drop_commit_reply'])
def test_real_socket_cut_at_financial_commit_boundary(pg, monkeypatch, mode):
    p, conn, store, control, paper, trader, q = _setup(pg, monkeypatch)
    dsn = pg[2]
    before = dict(paper.load_paper_state())
    with proxy_database(monkeypatch, dsn) as (proxy, proxied, patch):
        # A bounded financial process dies without cleanup after UNKNOWN; the
        # proxy supervisor survives, then an entirely new interpreter restarts.
        # IPC gates the exact wire fault, with no arbitrary sleep or real feed.
        worker_program = """
import json,os,sys
import pytest
import core.persistence as p
import core.commit_protocol as q
import paper_trading_engine as paper
from engines.trader_engine import run_trader_cycle
from tests.test_infra121_postgres import _synthetic_market
from scripts.test_infra12_postgres import _loopback_only
sys.addaudithook(_loopback_only)
_synthetic_market(pytest.MonkeyPatch(),paper)
def boundary(connection,witness):
 print(json.dumps({'backend_pid':connection.info.backend_pid}),flush=True)
 assert sys.stdin.readline().strip()=='continue'
p._before_financial_commit=boundary
try:run_trader_cycle()
except q.UnknownCommit as error:
 print(json.dumps({'cycle_id':error.cycle_id}),flush=True)
 os._exit(79)
else:raise AssertionError('actual TCP loss must produce UNKNOWN_COMMIT')
"""
        worker = subprocess.Popen([sys.executable,'-c',worker_program],cwd=ROOT,env=dict(os.environ),
                                  stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        try:
            with selectors.DefaultSelector() as ready:
                ready.register(worker.stdout,selectors.EVENT_READ)
                assert ready.select(20), 'Financial child did not reach the COMMIT boundary'
                pid = json.loads(worker.stdout.readline())['backend_pid']
            if mode == 'before_commit':
                proxy.cut(pid,'before_commit')
            else:
                proxy.arm_commit(pid,mode)
            output, errors = worker.communicate('continue\n',timeout=25)
            assert worker.returncode == 79, 'Financial child did not exit after UNKNOWN_COMMIT'
            cycle_id = json.loads(output.strip())['cycle_id']
        finally:
            if worker.poll() is None:
                worker.kill()
                worker.communicate(timeout=5)
        # Direct independent connection observes server truth, not proxy counters.
        receipt = json.loads(conn.execute('SELECT payload_json FROM app_state WHERE namespace=%s',
                                         (q.RECEIPT_PREFIX+cycle_id,)).fetchone()[0])
        committed = mode == 'drop_commit_reply'
        assert receipt['status'] == ('COMMITTED' if committed else 'PREPARED')
        count = conn.execute("SELECT count(*) FROM app_rows WHERE namespace='paper_trades'").fetchone()[0]
        assert count == int(committed) and q.gate_status()['blocked']
        with pytest.raises(q.ReconciliationRequired):
            trader.run_trader_cycle()
        # New interpreter has no witness/RAM from the failed process.
        program = """
import json
from core.commit_protocol import gate_status,ReconciliationRequired
from engines.trader_engine import run_trader_cycle
try:run_trader_cycle()
except ReconciliationRequired:print(json.dumps({'blocked':gate_status()['blocked']}))
else:raise AssertionError('restart may not run a financial cycle')
"""
        restarted = subprocess.run([sys.executable,'-c',program],cwd=ROOT,env=dict(os.environ),
                                   capture_output=True,text=True,check=True,timeout=20)
        assert json.loads(restarted.stdout.strip()) == {'blocked': True}
        resolved = q.reconcile(cycle_id)
        assert resolved['outcome'] == ('COMMITTED' if committed else 'NOT_COMMITTED')
        assert not resolved['blocked']
        assert len(paper.read_paper_trades()) == count
        if not committed:
            assert paper.load_paper_state() == before
        events = list(proxy.events)
        assert any(event.get('real_socket_close') for event in events)
        if committed:
            assert any(event.get('server_command_complete_commit') for event in events)
        evidence(conn, 'tcp', mode, {'fault_type': 'REAL_LOOPBACK_TCP_SOCKET_CLOSE',
            'python_exception_injected': False, 'server_receipt_observed': receipt['status'],
            'server_trades_observed': count, 'restart_blocked': True, 'python_process_abrupt_exit_code':79,
            'reconciled_outcome': resolved['outcome'], 'trades_not_reexecuted': True, 'proxy_events': events})


def test_real_disconnect_after_driver_has_confirmed_commit(pg, monkeypatch):
    p, conn, _, _, paper, trader, q = _setup(pg, monkeypatch)
    with proxy_database(monkeypatch, pg[2]) as (proxy, proxied, patch):
        original = p._transaction_context
        @contextmanager
        def disconnect_after_ack(connection, witness):
            with original(connection, witness):
                yield
            if witness is not None:
                # transaction.__exit__ already returned: libpq received COMMIT ACK.
                proxy.cut(connection.info.backend_pid, 'after_driver_confirmed_commit')
        patch.setattr(p, '_transaction_context', disconnect_after_ack)
        result = trader.run_trader_cycle()
        assert result['commit_outcome'] == 'COMMITTED' and not q.gate_status()['blocked']
        assert len(paper.read_paper_trades()) == 1
        evidence(conn,'tcp','after_known_confirmation', {'fault_type':'REAL_LOOPBACK_TCP_SOCKET_CLOSE',
            'driver_ack_before_cut':True,'outcome':result['commit_outcome'],'server_trades':1,
            'new_connection_recovered':True,'proxy_events':list(proxy.events)})


def test_real_socket_failure_in_post_commit_diagnostic(pg, monkeypatch):
    p, conn, _, _, paper, trader, q = _setup(pg, monkeypatch)
    with proxy_database(monkeypatch, pg[2]) as (proxy, proxied, patch):
        def diagnostic(payload):
            with p._connect() as connection:
                proxy.cut(connection.info.backend_pid, 'post_commit_diagnostic_connection')
                # Actual libpq socket error, not a manufactured Python exception.
                connection.execute('SELECT 1')
        patch.setattr(trader,'update_market_data_status',diagnostic)
        result = trader.run_trader_cycle()
        assert result['commit_outcome'] == 'POST_COMMIT_DIAGNOSTIC_FAILURE'
        assert result['committed'] and len(paper.read_paper_trades()) == 1
        assert q.gate_status()['blocked']
        assert q.reconcile()['outcome'] == 'COMMITTED'
        assert len(paper.read_paper_trades()) == 1
        evidence(conn,'tcp','post_commit_diagnostic', {'fault_type':'REAL_LOOPBACK_TCP_SOCKET_CLOSE',
            'python_exception_injected':False,'outcome':result['commit_outcome'],
            'trades_not_reexecuted':True,'proxy_events':list(proxy.events)})


def _seed_full_synthetic_history(pg, monkeypatch, *, unknown=False):
    p, conn, store, control, paper, trader, q = _setup(pg, monkeypatch)
    trader.run_trader_cycle()  # BUY AAPL
    monkeypatch.setattr(paper,'_should_sell',lambda *_:(True,'synthetic-close'))
    trader.run_trader_cycle()  # SELL AAPL + closed report
    monkeypatch.setattr(paper,'_should_sell',lambda *_:(False,''))
    state = store.load_bot_state()
    state['trader']['watchlist'] = ['MSFT']  # fictitious fixture universe, no strategy/limits changed
    store.save_bot_state(state)
    if unknown:
        with proxy_database(monkeypatch,pg[2]) as (proxy,proxied,patch):
            patch.setattr(p,'_before_financial_commit',lambda c,w:proxy.arm_commit(c.info.backend_pid,'drop_commit_reply'))
            with pytest.raises(q.UnknownCommit) as caught:
                trader.run_trader_cycle()  # actual BUY MSFT committed, response lost by real socket
            cycle_id = caught.value.cycle_id
    else:
        cycle_id = trader.run_trader_cycle()['cycle_id']
    return p,conn,store,control,paper,trader,q,cycle_id


def test_real_postgres_sigkill_restart_preserves_unknown_stop_and_financial_data(pg, monkeypatch):
    import psycopg
    from scripts.local_synthetic_backup import owned_container, DOCKER
    p,conn,store,control,paper,trader,q,cycle_id = _seed_full_synthetic_history(pg,monkeypatch,unknown=True)
    container = os.environ['ATLAS_INFRA12_CONTAINER']
    owned_container(container)
    assert conn.execute('SHOW fsync').fetchone()[0] == 'on'
    assert conn.execute('SHOW synchronous_commit').fetchone()[0] == 'on'
    stop = control.request_stop('synthetic-before-crash')
    assert stop.persisted and control.acknowledge_stop('synthetic-worker').acknowledged
    before = q.manifest()
    before_trades = paper.read_paper_trades()
    subprocess.run(DOCKER+['kill','--signal=KILL',container],check=True,capture_output=True,timeout=10)
    unavailable_stop = control.request_stop('synthetic-during-real-outage')
    assert not unavailable_stop.persisted and unavailable_stop.status == 'UNCONFIRMED'
    assert q.reconcile()['outcome'] == 'UNKNOWN_COMMIT'
    subprocess.run(DOCKER+['start',container],check=True,capture_output=True,timeout=10)
    deadline = time.monotonic()+30
    while True:
        try:
            with psycopg.connect(pg[2],autocommit=True) as recovered:
                recovered.execute('SELECT 1')
            break
        except psycopg.OperationalError:
            if time.monotonic()>deadline:
                raise
            threading.Event().wait(.05)  # readiness predicate, not timing evidence for the fault
    assert q.gate_status()['blocked'] and q.gate_status()['cycle_id'] == cycle_id
    assert q.manifest() == before and paper.read_paper_trades() == before_trades
    assert store.load_bot_state()['bot_status'] == 'STOPPED'
    assert store.load_bot_state()['stop_control']['request_id'] == stop.request_id
    with pytest.raises(q.ReconciliationRequired):
        p.append_json_row('paper_trades',{'synthetic':'forbidden'},paper.PAPER_TRADES_FILE)
    assert trader.run_trader_cycle()['status'] == 'STOPPED'
    assert q.reconcile()['outcome'] == 'COMMITTED'
    assert trader.run_trader_cycle()['status'] == 'STOPPED'
    assert paper.read_paper_trades() == before_trades
    logs = subprocess.run(DOCKER+['logs','--tail','100',container],capture_output=True,text=True,check=True,timeout=10)
    recovery = [line for line in (logs.stdout+logs.stderr).splitlines()
                if any(word in line for word in ('was interrupted','not properly shut down','redo starts','redo done','automatic recovery','ready to accept connections'))]
    assert recovery, 'No server recovery/readiness evidence'
    with psycopg.connect(pg[2],autocommit=True) as recovered:
        evidence(recovered,'recovery','sigkill',{'fault_type':'REAL_POSTGRES_CONTAINER_SIGKILL_AND_RESTART',
            'owned_anonymous_volume_retained':True,'fsync':'on','synchronous_commit':'on',
            'financial_manifest_equal':True,'unknown_preserved':True,'stop_preserved':True,
            'stop_failed_while_server_down':True,'trades_not_reexecuted':True,
            'server_recovery_excerpt':recovery[-8:]})


@pytest.mark.parametrize('scenario',['clean','unknown','stopped','missing_receipt','corrupt_receipt'])
def test_synthetic_backup_restore_preserves_all_state_and_quarantines(pg, monkeypatch, tmp_path, scenario):
    import psycopg
    from scripts.local_synthetic_backup import dump_restore
    unknown = scenario in {'unknown','missing_receipt','corrupt_receipt'}
    p,conn,store,control,paper,trader,q,cycle_id = _seed_full_synthetic_history(pg,monkeypatch,unknown=unknown)
    from core.auth.users_store import create_user
    create_user('fixture-backup-admin','synthetic-backup-password','admin')
    if scenario == 'stopped':
        assert control.request_stop('synthetic-before-backup').persisted
        assert control.acknowledge_stop('synthetic-worker').acknowledged
    if scenario == 'missing_receipt':
        # Deliberate loss of ONE disposable synthetic receipt, never financial history.
        conn.execute('DELETE FROM app_state WHERE namespace=%s',(q.RECEIPT_PREFIX+cycle_id,))
    elif scenario == 'corrupt_receipt':
        conn.execute('UPDATE app_state SET payload_json=%s WHERE namespace=%s',('{',q.RECEIPT_PREFIX+cycle_id))
    original_stop = store.load_bot_state().get('stop_control',{}).get('request_id')
    restored_dsn, proof = dump_restore(pg[2],os.environ['ATLAS_INFRA12_CONTAINER'],tmp_path/'backup')
    with monkeypatch.context() as patch:
        patch.setenv('DATABASE_URL',restored_dsn)
        assert store.load_bot_state()['bot_status'] == 'STOPPED'
        assert q.gate_status()['blocked'] == unknown
        if scenario == 'stopped':
            assert store.load_bot_state()['stop_control']['request_id'] == original_stop
        from core.auth.users_store import get_user
        from core.auth.security import verify_password
        assert verify_password('synthetic-backup-password',get_user('fixture-backup-admin')['password_hash'])
        assert len(paper.read_paper_trades()) == 3 and len(paper.load_paper_state()['positions']) == 1
        reports = p.load_json_rows('table:trade_reports.csv',Path('unused.json'))
        assert len(reports) == 1
        before_trades = paper.read_paper_trades()
        if unknown:
            with pytest.raises(q.ReconciliationRequired):
                p.append_json_row('paper_trades',{'synthetic':'blocked'},paper.PAPER_TRADES_FILE)
            outcome = q.reconcile(cycle_id)['outcome']
            assert outcome == ('COMMITTED' if scenario == 'unknown' else 'UNKNOWN_COMMIT')
            assert q.gate_status()['blocked'] == (scenario != 'unknown')
        assert trader.run_trader_cycle()['status'] == 'STOPPED'
        assert paper.read_paper_trades() == before_trades
        with psycopg.connect(restored_dsn,autocommit=True) as restored:
            maximum = restored.execute('SELECT max(seq) FROM app_rows').fetchone()[0]
            next_seq = restored.execute("INSERT INTO app_rows(namespace,row_json) VALUES ('synthetic_sequence_probe','{}') RETURNING seq").fetchone()[0]
            assert next_seq > maximum
        proof.update(scenario=scenario,synthetic_user_verified=True,trades=3,closed_reports=1,
                     open_positions=1,sequence_advanced=True,no_financial_retry=True)
    evidence(conn,'backup',scenario,proof)


@pytest.mark.parametrize('pending_newer',[False,True])
def test_late_diagnostic_ordering_under_postgres_fence(pg, monkeypatch, pending_newer):
    p,conn,_,_,paper,trader,q = _setup(pg,monkeypatch)
    first = trader.run_trader_cycle()['cycle_id']
    newer_finished = threading.Event()
    def late_diagnostic():
        assert newer_finished.wait(15)
        return q.post_commit_failure(first,confirmed=True)
    with ThreadPoolExecutor(max_workers=1) as pool:
        delayed = pool.submit(late_diagnostic)
        if pending_newer:
            with proxy_database(monkeypatch,pg[2]) as (proxy,proxied,patch):
                patch.setattr(p,'_before_financial_commit',lambda c,w:proxy.arm_commit(c.info.backend_pid,'drop_commit_reply'))
                with pytest.raises(q.UnknownCommit) as caught:
                    trader.run_trader_cycle()
                second = caught.value.cycle_id
        else:
            second = trader.run_trader_cycle()['cycle_id']
        gate_before = dict(q._gate())
        receipt_before = dict(q._receipt(second))
        current_manifest = q.manifest()
        newer_finished.set()
        late = delayed.result(15)
    assert late['stale_diagnostic'] and late['current_cycle_id'] == second
    assert dict(q._gate()) == gate_before and dict(q._receipt(second)) == receipt_before
    assert q.manifest() == current_manifest and len(paper.read_paper_trades()) == 1
    old = q.reconcile(first)
    if pending_newer:
        assert old['outcome'] == 'UNKNOWN_COMMIT' and q.gate_status()['cycle_id'] == second
        assert q.reconcile(second)['outcome'] == 'COMMITTED'
    else:
        assert old['historical'] and old['current_cycle_id'] == second
        assert not q.gate_status()['blocked']
    evidence(conn,'ordering','pending_newer' if pending_newer else 'newer_confirmed',
             {'late_diagnostic_did_not_change_gate':True,'newer_receipt_unchanged':True,
              'financial_manifest_unchanged':True,'reconciliation_did_not_replay_trades':True})


@pytest.mark.parametrize('historical_events',[0,1000,5000])
def test_bounded_protocol_measurements_and_stop_contention(pg, monkeypatch, tmp_path, historical_events):
    import resource
    import shutil
    import tracemalloc
    from scripts.local_synthetic_backup import dump_restore
    p,conn,store,control,paper,trader,q = _setup(pg,monkeypatch)
    assert historical_events <= 5000 and shutil.disk_usage(ROOT).free > 256*1024*1024
    initial_cash = paper.load_paper_state()['cash']
    # Valid closed synthetic pairs: zero PnL, no open historical positions, no
    # changes to capital, ticket, score, risk thresholds or strategy configuration.
    with conn.cursor().copy('COPY app_rows(namespace,row_json) FROM STDIN') as copy:
        for number in range(historical_events//2):
            trade_id = f'synthetic-history-{number}'
            timestamp = f'2026-01-01T00:{number//60%60:02d}:{number%60:02d}+00:00'
            for side in ('BUY','SELL'):
                row = {'trade_id':trade_id,'timestamp':timestamp,'asset':'AAPL','side':side,
                       'quantity':1.0,'price':10.0,'gross_value':10.0,'cost':0.0,
                       'cash_after':initial_cash-(10 if side=='BUY' else 0),'realized_pnl':0.0}
                copy.write_row(('paper_trades',json.dumps(row)))
                copy.write_row(('table:trader_orders.csv',json.dumps({
                    k:v for k,v in {**row,'source':'synthetic-history'}.items()
                    if k in {'timestamp','asset','side','quantity','price','gross_value','cost','cash_after','source'}})))
            report = {'trade_id':trade_id,'asset':'AAPL','quantity':1.0,'entry_price':10.0,
                      'exit_price':10.0,'gross_entry_value':10.0,'gross_exit_value':10.0,
                      'realized_pnl':0.0,'opened_at':timestamp,'closed_at':timestamp}
            copy.write_row(('table:trade_reports.csv',json.dumps(report)))
    timings = []
    growth = []
    financial_pids = set()
    acquired = {}
    financial_locked, release_financial, stop_at_lock = (threading.Event() for _ in range(3))
    stop_pid = []
    stop_commit_at = []
    race = {'enabled':False}
    lock = p._lock_namespace
    transaction = p._transaction_context

    def timed_lock(connection, namespace):
        pid = connection.info.backend_pid
        if namespace == 'atlas:persistence-unit' and threading.current_thread().name.startswith('perf-stop'):
            stop_pid.append(pid)
            stop_at_lock.set()
        started = time.perf_counter()
        result = lock(connection,namespace)
        if namespace == 'atlas:persistence-unit' and pid in financial_pids:
            acquired[pid] = time.perf_counter()
            if race['enabled']:
                race['acquired_at'] = acquired[pid]
                financial_locked.set()
                assert release_financial.wait(10)
        return result

    @contextmanager
    def timed_transaction(connection, witness):
        pid = connection.info.backend_pid
        if witness is not None:
            financial_pids.add(pid)
        with transaction(connection,witness):
            yield
            commit_started = time.perf_counter()
        finished = time.perf_counter()
        if witness is not None:
            timings.append({'commit_ack_ms':(finished-commit_started)*1000,
                            'writer_lock_ms':(finished-acquired[pid])*1000})
            financial_pids.discard(pid)
        elif threading.current_thread().name.startswith('perf-stop'):
            stop_commit_at.append(finished)

    monkeypatch.setattr(p,'_lock_namespace',timed_lock)
    monkeypatch.setattr(p,'_transaction_context',timed_transaction)
    tracemalloc.start()
    started = time.perf_counter()
    for _ in range(2):
        result = trader.run_trader_cycle()
        assert result['committed']
        values = conn.execute("SELECT count(*),sum(octet_length(payload_json)) FROM app_state WHERE namespace LIKE 'paper_cycle_receipt:%%'").fetchone()
        growth.append({'receipts':values[0],'payload_bytes':values[1]})
    race['enabled'] = True
    with ThreadPoolExecutor(max_workers=1,thread_name_prefix='perf-finance') as finance_pool, ThreadPoolExecutor(
            max_workers=1,thread_name_prefix='perf-stop') as stop_pool:
        financial = finance_pool.submit(trader.run_trader_cycle)
        assert financial_locked.wait(15)
        requested = time.perf_counter()
        stop = stop_pool.submit(control.request_stop,'synthetic-performance-stop')
        try:
            assert stop_at_lock.wait(5)
            deadline = time.monotonic()+3
            while True:
                waiting = conn.execute("SELECT EXISTS(SELECT 1 FROM pg_locks WHERE pid=%s AND locktype='advisory' AND NOT granted)",
                                       (stop_pid[-1],)).fetchone()[0]
                if waiting:
                    break
                assert time.monotonic() < deadline, 'STOP did not reach the actual advisory-lock wait'
                threading.Event().wait(.005)
        finally:
            setup_released = time.perf_counter()
            release_financial.set()
        receipt = stop.result(15)
        stop_returned = time.perf_counter()
        result = financial.result(30)
    race['enabled'] = False
    assert result['committed']
    if receipt.persisted:
        ack_started = time.perf_counter()
        ack = control.acknowledge_stop('synthetic-performance-worker')
        ack_ms = (time.perf_counter()-ack_started)*1000
        assert ack.acknowledged
    else:
        assert receipt.status == 'UNCONFIRMED' and not receipt.acknowledged
        ack_ms = None  # A timeout is a measured blocking condition, never a pass target.
    current_cycle = result['cycle_id']
    recon_started = time.perf_counter()
    assert q.reconcile(current_cycle)['outcome'] == 'COMMITTED'
    reconciliation_ms = (time.perf_counter()-recon_started)*1000
    current_memory, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak_memory < 128*1024*1024 and time.perf_counter()-started < 90
    values = conn.execute("SELECT count(*),sum(octet_length(payload_json)) FROM app_state WHERE namespace LIKE 'paper_cycle_receipt:%%'").fetchone()
    growth.append({'receipts':values[0],'payload_bytes':values[1]})
    assert len(paper.read_paper_trades()) == historical_events+1
    manifest_bytes = len(json.dumps(q._receipt(current_cycle)['after']).encode())
    _, backup = dump_restore(pg[2],os.environ['ATLAS_INFRA12_CONTAINER'],tmp_path/'perf-backup')
    assert time.perf_counter()-started < 90
    stop_ms = (stop_returned-requested)*1000
    evidence(conn,'performance',str(historical_events),{
        'historical_financial_events':historical_events,'historical_reports':historical_events//2,
        'actual_cycles':3,'new_financial_trades':1,'production_frequency_changed':False,
        'financial_transactions':timings,'stop_request_to_verified_return_ms':stop_ms,
        'stop_request_to_commit_ack_ms':(stop_commit_at[0]-requested)*1000 if stop_commit_at else None,
        'stop_persisted':receipt.persisted,'stop_ack_ms':ack_ms,
        'controlled_lock_setup_ms':(setup_released-race['acquired_at'])*1000,
        'reconciliation_ms':reconciliation_ms,'python_tracemalloc_peak_bytes':peak_memory,
        'process_rss_high_water_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'receipts_growth':growth,'current_manifest_bytes':manifest_bytes,
        'backup_dump_seconds':backup['dump_seconds'],'backup_restore_seconds':backup['restore_seconds'],
        'backup_writer_lock_seconds':backup['backup_writer_lock_seconds'],'dump_bytes':backup['dump_bytes'],
        'observed_status':'MEASURED' if receipt.persisted else 'BLOCKED_STOP_TIMEOUT',
        'limits':{'historical_events_max':5000,'python_peak_max_bytes':128*1024*1024,'case_seconds_max':90,
                  'dump_restore_each_seconds_max':30,'server_memory_bytes':512*1024*1024}})
