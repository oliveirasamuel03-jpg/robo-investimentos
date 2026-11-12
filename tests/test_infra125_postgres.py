"""Isolated snapshot candidate: equivalence, adversarial SQL and bounded A/B."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import wraps
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest
import psycopg
from psycopg.conninfo import make_conninfo,conninfo_to_dict
from tests.test_infra12_postgres import pg,pytestmark,ROOT
from tests.test_infra122_postgres import _setup
from tests.test_infra123_postgres import _seed_full_synthetic_history,proxy_database
from tests.test_infra124_postgres import seed_history,statistics
from scripts.prototypes.infra125_snapshot import SnapshotPrototype

DDL=(ROOT/'scripts/prototypes/infra125_revision_local.sql').read_text()


def evidence(conn,category,name,value):
    major=str(conn.execute('SHOW server_version_num').fetchone()[0])[:2]
    path=ROOT/f'docs/retomada/evidencias/infra125-{category}-pg{major}.json'
    data=json.loads(path.read_text()) if path.exists() else {'scope':'LOCAL_SYNTHETIC_ONLY','cases':{}}
    data['cases'][name]=value
    path.write_text(json.dumps(data,indent=2,sort_keys=True)+'\n')


def pending(pg,monkeypatch):
    values=_seed_full_synthetic_history(pg,monkeypatch)
    *_,q,cycle=values
    q.post_commit_failure(cycle,confirmed=True)
    assert q.gate_status()['blocked']
    return values


def financial_rows(conn):
    return conn.execute("SELECT seq,namespace,row_json FROM app_rows WHERE namespace IN ('paper_trades','table:trader_orders.csv','table:trade_reports.csv') ORDER BY seq").fetchall()


@pytest.mark.parametrize('damage',[
 'intact','json','same_count_value','delete_old','insert_unwanted','event_order',
 'receipt','cash','position','duplicate_trade','duplicate_report','missing_receipt','corrupt_receipt'])
def test_static_full_verifier_equivalence_and_corruption(pg,monkeypatch,damage):
    p,conn,store,control,paper,trader,q,cycle=pending(pg,monkeypatch)
    row=conn.execute("SELECT seq,row_json FROM app_rows WHERE namespace='paper_trades' ORDER BY seq LIMIT 1").fetchone()
    if damage=='json':conn.execute('UPDATE app_rows SET row_json=%s WHERE seq=%s',('{',row[0]))
    elif damage=='same_count_value':
        changed=json.loads(row[1]);changed['price']=999.0
        conn.execute('UPDATE app_rows SET row_json=%s WHERE seq=%s',(json.dumps(changed),row[0]))
    elif damage=='delete_old':conn.execute('DELETE FROM app_rows WHERE seq=%s',(row[0],))
    elif damage in {'insert_unwanted','duplicate_trade'}:
        payload=json.loads(row[1])
        if damage=='insert_unwanted':payload['trade_id']='unwanted-synthetic'
        conn.execute("INSERT INTO app_rows(namespace,row_json) VALUES ('paper_trades',%s)",(json.dumps(payload),))
    elif damage=='duplicate_report':
        conn.execute("INSERT INTO app_rows(namespace,row_json) SELECT namespace,row_json FROM app_rows WHERE namespace='table:trade_reports.csv' LIMIT 1")
    elif damage=='event_order':
        seqs=[r[0] for r in conn.execute("SELECT seq FROM app_rows WHERE namespace='paper_trades' ORDER BY seq LIMIT 2")]
        with conn.transaction():
            conn.execute('UPDATE app_rows SET seq=-1 WHERE seq=%s',(seqs[0],))
            conn.execute('UPDATE app_rows SET seq=%s WHERE seq=%s',(seqs[0],seqs[1]))
            conn.execute('UPDATE app_rows SET seq=%s WHERE seq=-1',(seqs[1],))
    elif damage in {'receipt','missing_receipt','corrupt_receipt'}:
        namespace=q.RECEIPT_PREFIX+cycle
        if damage=='missing_receipt':conn.execute('DELETE FROM app_state WHERE namespace=%s',(namespace,))
        elif damage=='corrupt_receipt':conn.execute('UPDATE app_state SET payload_json=%s WHERE namespace=%s',('{',namespace))
        else:
            receipt=dict(q._receipt(cycle));receipt['expected_paper']='0'*64
            conn.execute('UPDATE app_state SET payload_json=%s WHERE namespace=%s',(json.dumps(receipt),namespace))
    elif damage in {'cash','position'}:
        state=dict(paper.load_paper_state())
        if damage=='cash':state['cash']+=1
        else:next(iter(state['positions'].values()))['quantity']+=1
        conn.execute("UPDATE app_state SET payload_json=%s WHERE namespace='paper_state'",(json.dumps(state),))
    rows=financial_rows(conn)
    proto=SnapshotPrototype(pg[2]);proof=proto.prepare(cycle,release=False)
    baseline=q.reconcile(cycle,release=False)
    assert proof.result['outcome']==baseline['outcome']==('COMMITTED' if damage=='intact' else 'UNKNOWN_COMMIT')
    # Baseline may update verification metadata: get a NEW snapshot, never reuse a stale one.
    candidate=proto.reconcile(cycle,release=False)
    assert candidate['outcome']==baseline['outcome'] and candidate['blocked']
    assert financial_rows(conn)==rows and q.gate_status()['blocked']
    evidence(conn,'equivalence',damage,{'baseline':baseline['outcome'],'candidate':candidate['outcome'],
        'same_outcome':True,'financial_rows_unchanged':True,'gate_kept_closed':True,'prototype_integrated':False})


@pytest.mark.parametrize('change',['value','receipt','heartbeat','retention'])
def test_snapshot_recheck_refuses_any_intervening_relevant_write(pg,monkeypatch,change):
    p,conn,store,control,paper,trader,q,cycle=pending(pg,monkeypatch)
    proto=SnapshotPrototype(pg[2]);proof=proto.prepare(cycle)
    if change=='value':
        conn.execute("UPDATE app_rows SET row_json=replace(row_json,'AAPL','FAKE') WHERE seq=(SELECT min(seq) FROM app_rows WHERE namespace='paper_trades')")
    elif change=='receipt':
        conn.execute('UPDATE app_state SET payload_json=payload_json || %s WHERE namespace=%s',(' ',q.RECEIPT_PREFIX+cycle))
    elif change=='heartbeat':store.update_worker_heartbeat('synthetic-intervening-heartbeat')
    else:
        # Same external namespace move used by retention, without deleting history.
        conn.execute("UPDATE app_rows SET namespace='table:synthetic_archive.csv' WHERE namespace='table:trade_reports.csv'")
    before=financial_rows(conn)
    result=proto.publish(proof)
    assert result['outcome']=='UNKNOWN_COMMIT' and result['blocked'] and not result['resolution_recorded']
    assert q.gate_status()['blocked'] and financial_rows(conn)==before
    evidence(conn,'races',change,{'stale_proof_rejected':True,'no_automatic_reverification':True,
        'heartbeat_false_conflict':change=='heartbeat','financial_retry':False})


def test_repeatable_read_is_consistent_while_stop_persists(pg,monkeypatch):
    p,conn,store,control,paper,trader,q,cycle=pending(pg,monkeypatch)
    proto=SnapshotPrototype(pg[2]);captured=threading.Event();release=threading.Event()
    def pause(connection):
        assert connection.execute('SHOW transaction_isolation').fetchone()[0]=='repeatable read'
        assert connection.execute('SHOW transaction_read_only').fetchone()[0]=='on'
        captured.set();assert release.wait(10)
    monkeypatch.setattr(proto,'_after_snapshot',pause)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future=pool.submit(proto.prepare,cycle)
        assert captured.wait(10)
        stop=control.request_stop('synthetic-stop-during-snapshot')
        assert stop.persisted and not future.done()
        release.set();proof=future.result(15)
    assert proof.result['outcome']=='COMMITTED'  # old consistent financial evidence
    result=proto.publish(proof)
    assert result['outcome']=='UNKNOWN_COMMIT' and q.gate_status()['blocked']
    assert store.load_bot_state()['bot_status']=='STOPPED'
    monkeypatch.setattr(proto,'_after_snapshot',lambda *_:None)
    assert proto.reconcile(cycle)['outcome']=='COMMITTED'
    assert trader.run_trader_cycle()['status']=='STOPPED' and len(paper.read_paper_trades())==3
    evidence(conn,'races','stop_during_snapshot',{'stop_persisted_without_writer_fence':True,
        'old_snapshot_consistent':True,'old_proof_could_not_clear_gate':True,'fresh_explicit_reconciliation_no_trade':True})


def test_publication_and_stop_ordering_is_deterministic(pg,monkeypatch):
    p,conn,store,control,paper,trader,q,cycle=pending(pg,monkeypatch)
    proto=SnapshotPrototype(pg[2]);proof=proto.prepare(cycle)
    ready=threading.Event();release=threading.Event()
    monkeypatch.setattr(proto,'_before_publish_commit',lambda *_:(ready.set(),release.wait(10)))
    with ThreadPoolExecutor(max_workers=2) as pool:
        publishing=pool.submit(proto.publish,proof);assert ready.wait(10)
        stopping=pool.submit(control.request_stop,'synthetic-stop-during-metadata-commit')
        deadline=time.monotonic()+3
        while not conn.execute("SELECT EXISTS(SELECT 1 FROM pg_locks WHERE locktype='advisory' AND NOT granted)").fetchone()[0]:
            assert time.monotonic()<deadline;threading.Event().wait(.005)
        assert not stopping.done()
        release.set();result=publishing.result(15);stop=stopping.result(15)
    assert result['outcome']=='COMMITTED' and result['resolution_recorded'] and stop.persisted
    assert control.acknowledge_stop('synthetic-worker').acknowledged
    assert trader.run_trader_cycle()['status']=='STOPPED' and len(paper.read_paper_trades())==3
    evidence(conn,'races','stop_during_publication',{'actual_advisory_wait_observed':True,'stop_verified':True,'no_new_financial_unit':True})


def test_external_writer_cannot_change_rows_inside_publication_fence(pg,monkeypatch):
    p,conn,store,control,paper,trader,q,cycle=pending(pg,monkeypatch)
    proto=SnapshotPrototype(pg[2]);proof=proto.prepare(cycle,release=False)
    ready=threading.Event();release=threading.Event();writer_pid=[];writer_started=threading.Event()
    monkeypatch.setattr(proto,'_before_publish_commit',lambda *_:(ready.set(),release.wait(10)))
    def external():
        with psycopg.connect(pg[2],autocommit=True) as writer:
            writer_pid.append(writer.info.backend_pid);writer_started.set()
            writer.execute("UPDATE app_rows SET row_json=replace(row_json,'AAPL','FAKE') WHERE namespace='paper_trades'")
    with ThreadPoolExecutor(max_workers=2) as pool:
        publishing=pool.submit(proto.publish,proof);assert ready.wait(10)
        changing=pool.submit(external);assert writer_started.wait(5)
        deadline=time.monotonic()+3
        while not conn.execute("SELECT EXISTS(SELECT 1 FROM pg_locks WHERE pid=%s AND locktype='relation' AND NOT granted)",(writer_pid[0],)).fetchone()[0]:
            assert time.monotonic()<deadline;threading.Event().wait(.005)
        assert not changing.done();release.set()
        assert publishing.result(15)['outcome']=='COMMITTED';changing.result(15)
    assert proto.reconcile(cycle)['outcome']=='UNKNOWN_COMMIT' and q.gate_status()['blocked']
    evidence(conn,'races','external_sql_table_fence',{'actual_relation_wait_observed':True,'later_tampering_detected':True})


def test_bookkeeping_rollback_never_reexecutes_financial_unit(pg,monkeypatch):
    p,conn,store,control,paper,trader,q,cycle=pending(pg,monkeypatch)
    proto=SnapshotPrototype(pg[2]);proof=proto.prepare(cycle)
    before=proof.capture;rows=financial_rows(conn)
    def fail(*_):raise ValueError('synthetic bookkeeping fault before COMMIT')
    monkeypatch.setattr(proto,'_before_publish_commit',fail)
    result=proto.publish(proof)
    assert result['outcome']=='COMMITTED' and not result['resolution_recorded'] and result['blocked']
    assert proto._capture(conn,cycle)==before and financial_rows(conn)==rows and q.gate_status()['blocked']
    evidence(conn,'recovery','bookkeeping_rollback',{'fault_type':'PYTHON_INJECTION_BEFORE_METADATA_COMMIT','financial_retry':False,'all_metadata_rolled_back':True})


def test_real_lost_financial_reply_restart_and_explicit_candidate_reconciliation(pg,monkeypatch):
    p,conn,store,control,paper,trader,q,cycle=_seed_full_synthetic_history(pg,monkeypatch,unknown=True)
    program="""
from core.commit_protocol import assert_execution_allowed,ReconciliationRequired
try:assert_execution_allowed()
except ReconciliationRequired:print('BLOCKED')
else:raise AssertionError('UNKNOWN lost after new interpreter')
"""
    process=subprocess.run([sys.executable,'-c',program],cwd=ROOT,env=dict(os.environ),check=True,capture_output=True,text=True,timeout=20)
    assert process.stdout.strip()=='BLOCKED'
    with pytest.raises(q.ReconciliationRequired):trader.run_trader_cycle()
    assert control.request_stop('synthetic-unknown-stop').persisted
    result=SnapshotPrototype(pg[2]).reconcile(cycle)
    assert result['outcome']=='COMMITTED' and result['resolution_recorded']
    assert len(paper.read_paper_trades())==3 and trader.run_trader_cycle()['status']=='STOPPED'
    evidence(conn,'recovery','financial_reply_lost',{'fault_type':'REAL_LOOPBACK_TCP_SERVER_REPLY_WITHHELD',
        'fresh_interpreter_blocked':True,'stop_preserved':True,'explicit_candidate_outcome':result['outcome'],'trades':3,'financial_retry':False})


def test_real_lost_candidate_publication_reply_is_not_a_financial_retry(pg,monkeypatch):
    from tests.infra123_tcp_proxy import LocalPostgresProxy
    p,conn,store,control,paper,trader,q,cycle=pending(pg,monkeypatch)
    port=int(conninfo_to_dict(pg[2])['port'])
    with LocalPostgresProxy('127.0.0.1',port) as proxy:
        proto=SnapshotPrototype(pg[2],publication_proxy=proxy)
        monkeypatch.setattr(proto,'_before_publish_commit',lambda connection:proxy.arm_commit(connection.info.backend_pid,'drop_commit_reply'))
        result=proto.reconcile(cycle)
        assert result['outcome']=='COMMITTED' and not result['resolution_recorded'] and result['blocked']
        assert not q.gate_status()['blocked']  # independent server observation, not application ACK
        assert len(paper.read_paper_trades())==3
        assert any(v.get('server_command_complete_commit') for v in proxy.events)
        events=list(proxy.events)
    assert SnapshotPrototype(pg[2]).reconcile(cycle)['outcome']=='COMMITTED'
    assert len(paper.read_paper_trades())==3
    evidence(conn,'recovery','metadata_reply_lost',{'fault_type':'REAL_LOOPBACK_TCP_SERVER_REPLY_WITHHELD',
        'server_gate_release_observed_independently':True,'application_did_not_claim_resolution_recorded':True,
        'financial_retry':False,'proxy_events':events})


def test_snapshot_read_real_tcp_failure_keeps_gate_closed(pg,monkeypatch):
    from tests.infra123_tcp_proxy import LocalPostgresProxy
    p,conn,store,control,paper,trader,q,cycle=pending(pg,monkeypatch)
    with LocalPostgresProxy('127.0.0.1',int(conninfo_to_dict(pg[2])['port'])) as proxy:
        url=make_conninfo(pg[2],port=proxy.port,sslmode='disable',gssencmode='disable')
        proto=SnapshotPrototype(pg[2],connect=lambda _:psycopg.connect(url,autocommit=True))
        monkeypatch.setattr(proto,'_after_snapshot',lambda connection:proxy.cut(connection.info.backend_pid,'snapshot_read_failure'))
        result=proto.reconcile(cycle)
        assert result['outcome']=='UNKNOWN_COMMIT' and q.gate_status()['blocked']
        assert len(paper.read_paper_trades())==3
        evidence(conn,'recovery','snapshot_tcp_loss',{'fault_type':'REAL_LOOPBACK_TCP_SOCKET_CLOSE','gate_kept_closed':True,'proxy_events':list(proxy.events)})


def test_namespace_revision_is_not_an_integrity_proof(pg,monkeypatch):
    p,conn,store,control,paper,trader,q,cycle=pending(pg,monkeypatch)
    conn.execute(DDL)
    # Old uninstrumented SQL writer participates through the local trigger.
    conn.execute("UPDATE app_rows SET row_json=row_json WHERE namespace='paper_trades'")
    before=conn.execute('SELECT namespace,revision FROM infra125_namespace_revision ORDER BY namespace').fetchall()
    try:
        conn.execute("SET session_replication_role='replica'")  # own synthetic superuser only
        conn.execute("UPDATE app_rows SET row_json=replace(row_json,'AAPL','FAKE') WHERE namespace='paper_trades'")
    finally:conn.execute("SET session_replication_role='origin'")
    after=conn.execute('SELECT namespace,revision FROM infra125_namespace_revision ORDER BY namespace').fetchall()
    assert before==after  # A revision-only reuse would incorrectly accept the old proof.
    assert q.reconcile(cycle)['outcome']=='UNKNOWN_COMMIT'
    assert SnapshotPrototype(pg[2]).reconcile(cycle)['outcome']=='UNKNOWN_COMMIT'
    assert q.gate_status()['blocked']
    evidence(conn,'revisions','privileged_bypass',{'revision_unchanged_despite_tampering':True,
        'revision_only_optimization':'REJECTED_LOSS_OF_COVERAGE','full_and_snapshot_detected':True,
        'ddl_scope':'LOCAL_DISPOSABLE_ONLY'})


def test_revision_rollback_namespace_move_truncate_and_second_install_refusal(pg,monkeypatch):
    p,conn,store,control,paper,trader,q,cycle=pending(pg,monkeypatch)
    conn.execute(DDL)
    with pytest.raises(ValueError):
        with conn.transaction():
            conn.execute("UPDATE app_rows SET row_json=row_json WHERE namespace='paper_trades'")
            raise ValueError('synthetic transaction rollback')
    assert conn.execute('SELECT count(*) FROM infra125_namespace_revision').fetchone()[0]==0
    conn.execute("UPDATE app_rows SET namespace='synthetic_archive' WHERE namespace='table:trade_reports.csv'")
    assert {r[0] for r in conn.execute('SELECT namespace FROM infra125_namespace_revision')}=={'table:trade_reports.csv','synthetic_archive'}
    # Roll back TRUNCATE: no financial history is deleted even in this fixture.
    with conn.transaction():
        conn.execute('SAVEPOINT synthetic_truncate')
        conn.execute('TRUNCATE app_rows')
        assert conn.execute("SELECT revision FROM infra125_namespace_revision WHERE namespace='*:truncate'").fetchone()[0]==1
        conn.execute('ROLLBACK TO SAVEPOINT synthetic_truncate')
    assert len(financial_rows(conn))>0
    with pytest.raises(psycopg.errors.DuplicateTable):conn.execute(DDL)
    conn.execute('ROLLBACK')
    assert q.reconcile(cycle)['outcome']=='UNKNOWN_COMMIT'
    assert SnapshotPrototype(pg[2]).reconcile(cycle)['outcome']=='UNKNOWN_COMMIT'
    evidence(conn,'revisions','transactional_behavior',{'rollback_restored_revisions':True,'namespace_move_tracked':True,
        'truncate_epoch_observed_only_in_rolled_back_savepoint':True,'second_install_refused':True,'remote_migration':False})


@pytest.mark.parametrize('missing',[False,True])
def test_snapshot_backup_restore_remains_stopped_and_requires_fresh_proof(pg,monkeypatch,tmp_path,missing):
    from scripts.local_synthetic_backup import dump_restore
    p,conn,store,control,paper,trader,q,cycle=_seed_full_synthetic_history(pg,monkeypatch,unknown=True)
    conn.execute(DDL)
    if missing:conn.execute('DELETE FROM app_state WHERE namespace=%s',(q.RECEIPT_PREFIX+cycle,))
    proto=SnapshotPrototype(pg[2]);proof=proto.prepare(cycle)
    restored,backup=dump_restore(pg[2],os.environ['ATLAS_INFRA12_CONTAINER'],tmp_path/'snapshot-backup')
    # A proof cannot be transported to another instance or another database OID.
    assert SnapshotPrototype(restored).publish(proof)['reason']=='FOREIGN_OR_RESTARTED_PROOF'
    if not missing:
        proto.connect=lambda _:psycopg.connect(restored,autocommit=True)
        assert proto.publish(proof)['outcome']=='UNKNOWN_COMMIT'
    with monkeypatch.context() as patch:
        patch.setenv('DATABASE_URL',restored)
        assert store.load_bot_state()['bot_status']=='STOPPED' and q.gate_status()['blocked']
        result=SnapshotPrototype(restored).reconcile(cycle)
        assert result['outcome']==('UNKNOWN_COMMIT' if missing else 'COMMITTED')
        assert q.gate_status()['blocked']==missing and len(paper.read_paper_trades())==3
        assert trader.run_trader_cycle()['status']=='STOPPED'
        with psycopg.connect(restored,autocommit=True) as target:
            assert target.execute("SELECT count(*) FROM pg_trigger WHERE tgname LIKE 'infra125_%%'").fetchone()[0]==4
            count=target.execute('SELECT count(*) FROM infra125_namespace_revision').fetchone()[0]
            assert count>0
    evidence(conn,'backup','missing_receipt' if missing else 'valid_receipt',{**backup,
        'candidate_outcome':result['outcome'],'old_proof_not_portable':True,'local_revision_triggers_restored':True,
        'stop_preserved':True,'financial_retry':False})


def test_old_proof_and_diagnostic_cannot_release_new_cycle(pg,monkeypatch):
    p,conn,store,control,paper,trader,q,first=pending(pg,monkeypatch)
    proto=SnapshotPrototype(pg[2]);old=proto.prepare(first)
    assert q.reconcile(first)['outcome']=='COMMITTED'
    second=trader.run_trader_cycle()['cycle_id'];q.post_commit_failure(second,confirmed=True)
    gate=dict(q._gate());receipt=dict(q._receipt(second))
    assert proto.publish(old)['outcome']=='UNKNOWN_COMMIT'
    assert q.post_commit_failure(first,confirmed=True)['stale_diagnostic']
    assert dict(q._gate())==gate and dict(q._receipt(second))==receipt
    assert proto.reconcile(first)['outcome']=='UNKNOWN_COMMIT'
    assert proto.reconcile(second)['outcome']=='COMMITTED' and len(paper.read_paper_trades())==3
    evidence(conn,'races','old_cycle_identity',{'newer_gate_preserved':True,'newer_receipt_preserved':True,'financial_retry':False})


def test_real_postgres_restart_invalidates_snapshot_and_preserves_unknown_stop(pg,monkeypatch):
    from scripts.local_synthetic_backup import owned_container,DOCKER
    p,conn,store,control,paper,trader,q,cycle=_seed_full_synthetic_history(pg,monkeypatch,unknown=True)
    assert control.request_stop('synthetic-pre-snapshot-crash').persisted
    proto=SnapshotPrototype(pg[2]);proof=proto.prepare(cycle);before=proof.capture
    container=os.environ['ATLAS_INFRA12_CONTAINER'];owned_container(container)
    subprocess.run(DOCKER+['kill','--signal=KILL',container],check=True,capture_output=True,timeout=10)
    failed_stop=control.request_stop('synthetic-db-down')
    assert not failed_stop.persisted and failed_stop.status=='UNCONFIRMED'
    assert proto.reconcile(cycle)['outcome']=='UNKNOWN_COMMIT'
    subprocess.run(DOCKER+['start',container],check=True,capture_output=True,timeout=10)
    deadline=time.monotonic()+30
    while True:
        try:
            with psycopg.connect(pg[2],autocommit=True) as recovered:recovered.execute('SELECT 1')
            break
        except psycopg.OperationalError:
            assert time.monotonic()<deadline;threading.Event().wait(.05)
    with psycopg.connect(pg[2],autocommit=True) as recovered:
        assert proto._capture(recovered,cycle)==before
        assert proto.publish(proof)['outcome']=='UNKNOWN_COMMIT' and q.gate_status()['blocked']
        assert proto.reconcile(cycle)['outcome']=='COMMITTED'
        assert store.load_bot_state()['bot_status']=='STOPPED' and len(paper.read_paper_trades())==3
        assert trader.run_trader_cycle()['status']=='STOPPED'
        evidence(recovered,'recovery','postgres_restart',{'fault_type':'REAL_OWNED_POSTGRES_SIGKILL_AND_RESTART',
            'canonical_capture_equal':True,'old_snapshot_rejected_by_server_epoch':True,
            'stop_while_down_unconfirmed':True,'fresh_explicit_reconciliation':'COMMITTED','financial_retry':False})


def test_broad_table_fence_is_an_integration_blocker_with_external_writer(pg,monkeypatch):
    p,conn,store,control,paper,trader,q,cycle=pending(pg,monkeypatch)
    with psycopg.connect(pg[2],autocommit=True) as external,external.transaction():
        external.execute("INSERT INTO app_state(namespace,payload_json) VALUES ('synthetic_unrelated_external','{}')")
        start=time.perf_counter();baseline=q.reconcile(cycle,release=False)
        baseline_ms=(time.perf_counter()-start)*1000
        assert baseline['outcome']=='COMMITTED'
        proto=SnapshotPrototype(pg[2]);proof=proto.prepare(cycle,release=False)
        with ThreadPoolExecutor(max_workers=2) as pool:
            publication=pool.submit(proto.publish,proof)
            deadline=time.monotonic()+3
            while not conn.execute("SELECT EXISTS(SELECT 1 FROM pg_locks WHERE locktype='relation' AND NOT granted)").fetchone()[0]:
                assert time.monotonic()<deadline;threading.Event().wait(.005)
            stop_started=time.perf_counter()
            stopping=pool.submit(control.request_stop,'synthetic-broad-lock-stop')
            # The unrelated external transaction remains open. Its ROW EXCLUSIVE
            # lock is compatible with current bookkeeping, but blocks the new fence.
            result=publication.result(8)
            stop=stopping.result(8);stop_ms=(time.perf_counter()-stop_started)*1000
        assert result['outcome']=='UNKNOWN_COMMIT' and not result['resolution_recorded']
        assert q.gate_status()['blocked'] and len(paper.read_paper_trades())==3
        assert stop.persisted or stop.status=='UNCONFIRMED'
    evidence(conn,'blockers','external_unrelated_table_lock',{'baseline_reconciliation_ms':baseline_ms,
        'candidate_error_code':result.get('error_code'),'candidate_outcome':result['outcome'],
        'stop_verified_ms':stop_ms,'stop_persisted':stop.persisted,'stop_status':stop.status,
        'relation_lock_wait_observed':True,'lock_timeout_seconds':5,'financial_retry':False,
        'integration':'REJECTED_BROAD_LOCK_CONTENTION'})


@pytest.mark.parametrize('implementation',['current','snapshot'])
@pytest.mark.parametrize('events',[0,1000,5000])
def test_snapshot_benchmark_same_conditions(pg,monkeypatch,events,implementation):
    import resource,shutil,tracemalloc
    from tests.infra125_metrics import Measurements,CountedConnection
    p,conn,store,control,paper,trader,q=_setup(pg,monkeypatch)
    assert events<=5000 and shutil.disk_usage(ROOT).free>256*1024**2
    seed_history(conn,paper,events)
    measured=Measurements(p,monkeypatch)
    original_reconcile=q.reconcile
    proto=SnapshotPrototype(pg[2],connect=lambda url:CountedConnection(psycopg.connect(url,autocommit=True),measured))
    def instrument_private(verifier):
        for name in ('manifest','_hash','_receipt','reconcile'):
            measured.measure(verifier,name,'snapshot_verifier.'+name)
    monkeypatch.setattr(proto,'_instrument_verifier',instrument_private)
    stop_verified=threading.Event()
    def reconcile(*args,**kwargs):
        if threading.current_thread().name.startswith('infra125-finance'):
            # Both A/B implementations use the same deterministic schedule:
            # STOP completes before the post-financial verification snapshot.
            # Unordered STOP/snapshot conflicts are tested separately, not hidden.
            assert stop_verified.wait(15)
        return (original_reconcile if implementation=='current' else proto.reconcile)(*args,**kwargs)
    monkeypatch.setattr(q,'reconcile',reconcile)
    for module,names in [(q,('manifest','begin_intent','record_unit_result','reconcile','reject_duplicate_events','_hash','_receipt')),
                         (p,('_decode_json',)),(trader,('_sync_trader_orders_into_storage','sync_platform_positions_from_paper'))]:
        for name in names:measured.measure(module,name)
    locked=threading.Event();release=threading.Event();stop_attempt=threading.Event();stop_pid=[];lock_at=[]
    def on_financial(pid):
        lock_at.append(time.perf_counter());locked.set();assert release.wait(10)
    measured.on_financial_lock=on_financial
    measured.on_stop_attempt=lambda pid:(stop_pid.append(pid),stop_attempt.set())
    samples=[];growth=[];started=time.perf_counter();timeouts=0
    for trial in range(10):
        assert time.perf_counter()-started<300
        assert shutil.disk_usage(ROOT).free>256*1024**2
        if trial:control.start_robot(store.load_bot_state(),requested_by='synthetic-snapshot-benchmark')
        locked.clear();release.clear();stop_attempt.clear();stop_verified.clear()
        unit_start=len(measured.units);sql_start=measured.statements;ctrl_start=measured.controls
        cycle_time=[];tracemalloc.start()
        with ThreadPoolExecutor(max_workers=1,thread_name_prefix='infra125-finance') as finances,ThreadPoolExecutor(
                max_workers=1,thread_name_prefix='infra125-stop') as stops:
            def run():
                at=time.perf_counter();result=trader.run_trader_cycle()
                cycle_time.append((time.perf_counter()-at)*1000);return result
            finance=finances.submit(run);assert locked.wait(30)
            requested=time.perf_counter();stopping=stops.submit(control.request_stop,'synthetic-snapshot-study')
            try:
                assert stop_attempt.wait(5)
                deadline=time.monotonic()+3
                while not conn.execute("SELECT EXISTS(SELECT 1 FROM pg_locks WHERE pid=%s AND locktype='advisory' AND NOT granted)",(stop_pid[-1],)).fetchone()[0]:
                    assert time.monotonic()<deadline;threading.Event().wait(.005)
            finally:
                setup_ms=(time.perf_counter()-lock_at[-1])*1000;release.set()
            receipt=stopping.result(15);stop_return=time.perf_counter();stop_verified.set()
            result=finance.result(60)
        assert result['committed'] and result['commit_outcome']=='COMMITTED'
        ack_ms=None
        if receipt.persisted:
            at=time.perf_counter();ack=control.acknowledge_stop('synthetic-snapshot-worker')
            ack_ms=(time.perf_counter()-at)*1000;assert ack.acknowledged
        else:
            assert receipt.status=='UNCONFIRMED' and not receipt.acknowledged;timeouts+=1
        at=time.perf_counter();resolution=q.reconcile(result['cycle_id'])
        reconciliation_ms=(time.perf_counter()-at)*1000
        assert resolution['outcome']=='COMMITTED' and resolution['resolution_recorded']
        _,peak=tracemalloc.get_traced_memory();tracemalloc.stop();assert peak<128*1024**2
        units=measured.units[unit_start:]
        financial=[u for u in units if u['kind']=='financial'];assert len(financial)==1
        stop_units=[u for u in units if u['thread'].startswith('infra125-stop')];assert stop_units or not receipt.persisted
        worker=[u for u in units if u['thread'].startswith('infra125-finance')]
        stop_committed=(stop_units[-1]['end']-requested)*1000 if stop_units else None
        samples.append({'iteration':trial+1,'financial_lock_ms':financial[0]['hold_ms'],
            'financial_lock_wait_ms':financial[0]['wait_ms'],'commit_ack_ms':financial[0]['commit_ack_ms'],
            'stop_commit_ack_ms':stop_committed,'stop_verified_ms':(stop_return-requested)*1000,
            'stop_global_wait_ms':stop_units[-1]['wait_ms'] if stop_units else None,
            'ack_ms':ack_ms,'reconciliation_ms':reconciliation_ms,'cycle_total_ms':cycle_time[0],
            'worker_all_global_hold_ms':sum(u['hold_ms'] for u in worker),
            'worker_all_global_wait_ms':sum(u['wait_ms'] for u in worker),
            'sql_execute_count':measured.statements-sql_start,'sql_transaction_control_count':measured.controls-ctrl_start,
            'python_peak_bytes':peak,'process_rss_high_water_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            'controlled_setup_ms':setup_ms,'stop_persisted':receipt.persisted})
        count,total=conn.execute("SELECT count(*),sum(octet_length(payload_json)) FROM app_state WHERE namespace LIKE 'paper_cycle_receipt:%%'").fetchone()
        growth.append({'receipts':count,'payload_bytes':total})
        assert len(paper.read_paper_trades())==events+1 and not q.gate_status()['blocked']
        if not receipt.persisted:break  # no next financial trial after unconfirmed STOP
    numeric=[k for k,v in samples[0].items() if isinstance(v,(int,float)) and not isinstance(v,bool)]
    stats={k:statistics([s[k] for s in samples]) for k in numeric}
    budgets={'stop_verified_ms':2000,'ack_ms':1000,'reconciliation_ms':2000,'financial_lock_ms':1500}
    meets=len(samples)==10 and timeouts==0 and all(stats[k]['max']<=limit for k,limit in budgets.items())
    evidence(conn,'benchmark',f'{events}:{implementation}',{'implementation':implementation,'historical_events':events,
        'samples':samples,'statistics':stats,'stage_profile_inclusive':measured.stages,'sql_categories':measured.sql_categories,
        'query_count_scope':'application execute/cursor.execute plus transaction contexts; no SQL parameters logged',
        'transaction_rollback_count':measured.rollbacks,'receipts_growth':growth,'new_trades':1,'financial_retry':False,
        'snapshot_phase_metrics':proto.metrics if implementation=='snapshot' else [],'within_candidate_budgets':meets,
        'budgets_ms':budgets,'stop_timeouts':timeouts,'unexecuted_repetitions':10-len(samples),
        'scheduling':'SAME_BOTH: financial lock -> STOP queued -> COMMIT -> STOP verified -> post-commit verifier',
        'elapsed_seconds':time.perf_counter()-started,'limits':{'repetitions':10,'case_seconds':300,'python_peak_bytes':128*1024**2,
          'server_memory_bytes':512*1024**2,'server_cpus':1},'operational_integration':False,'hours72_authorized':False})


def test_local_revision_ddl_refuses_non_fixture_database(pg):
    with psycopg.connect(os.environ['ATLAS_INFRA12_LOCAL_DSN'],autocommit=True) as admin:
        with pytest.raises(psycopg.errors.RaiseException,match='prototype DDL refused'):
            admin.execute(DDL)
        admin.execute('ROLLBACK')
        assert admin.execute("SELECT to_regclass('public.infra125_namespace_revision')").fetchone()[0] is None


def test_two_snapshot_publishers_do_not_overwrite_each_other(pg,monkeypatch):
    p,conn,store,control,paper,trader,q,cycle=pending(pg,monkeypatch)
    first=SnapshotPrototype(pg[2]);second=SnapshotPrototype(pg[2])
    proofs=[first.prepare(cycle),second.prepare(cycle)]
    barrier=threading.Barrier(2)
    def publish(proto,proof):barrier.wait(5);return proto.publish(proof)
    with ThreadPoolExecutor(max_workers=2) as pool:
        a=pool.submit(publish,first,proofs[0]);b=pool.submit(publish,second,proofs[1])
        results=[a.result(15),b.result(15)]
    assert sorted(r['outcome'] for r in results)==['COMMITTED','UNKNOWN_COMMIT']
    assert sum(bool(r.get('resolution_recorded')) for r in results)==1
    assert not q.gate_status()['blocked'] and len(paper.read_paper_trades())==3
    evidence(conn,'races','two_publishers',{'only_one_metadata_commit':True,'stale_publisher_rejected':True,'financial_retry':False})


def test_actual_retention_rollback_and_archive_preserve_or_invalidate_proof(pg,monkeypatch):
    from datetime import datetime,timezone
    p,conn,store,control,paper,trader,q,cycle=pending(pg,monkeypatch)
    retention=importlib.import_module('core.retention')
    proto=SnapshotPrototype(pg[2]);proof=proto.prepare(cycle,release=False)
    before=proto._capture(conn,cycle)
    with pytest.raises(q.ReconciliationRequired):
        retention.run_retention_job(now=datetime(2027,5,1,tzinfo=timezone.utc))
    assert proto._capture(conn,cycle)==before
    assert proto.publish(proof)['outcome']=='COMMITTED' and q.gate_status()['blocked']
    assert q.reconcile(cycle)['outcome']=='COMMITTED'
    proof=proto.prepare(cycle)
    maintenance=retention.run_retention_job(now=datetime(2027,5,1,tzinfo=timezone.utc))
    assert maintenance['last_summary']['trade_reports_archived_rows']==1
    assert proto.publish(proof)['outcome']=='UNKNOWN_COMMIT'
    assert q.reconcile(cycle)['outcome']=='UNKNOWN_COMMIT'
    assert proto.reconcile(cycle)['outcome']=='UNKNOWN_COMMIT'
    assert len(paper.read_paper_trades())==3
    assert conn.execute("SELECT count(*) FROM app_rows WHERE namespace LIKE 'table:trade_reports_%%'").fetchone()[0]==1
    evidence(conn,'blockers','retention',{'active_gate_rolled_back_maintenance':True,'history_preserved_in_archive':True,
        'committed_archive_invalidated_old_proof':True,'current_verifier_also_unknown_after_archive':True,
        'archive_compatibility':'NOT_PROVEN_FOR_OPERATIONAL_INTEGRATION'})
