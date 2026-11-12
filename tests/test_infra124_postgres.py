"""Bounded synthetic contention measurements; no external feed or financial retry."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import wraps
import importlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import threading
import time
import tracemalloc

import pytest
from tests.test_infra12_postgres import pg, pytestmark, ROOT
from tests.test_infra122_postgres import _setup


def write_evidence(conn, category, name, value):
    major = str(conn.execute('SHOW server_version_num').fetchone()[0])[:2]
    label = os.environ.get('ATLAS_INFRA124_MEASUREMENT', 'after')
    path = ROOT/f'docs/retomada/evidencias/infra124-{category}-{label}-pg{major}.json'
    data = json.loads(path.read_text()) if path.exists() else {'scope':'LOCAL_SYNTHETIC_ONLY','cases':{}}
    data['cases'][name] = value
    path.write_text(json.dumps(data,indent=2,sort_keys=True)+'\n')


def seed_history(conn, paper, count):
    """Same zero-PnL pairs as INFRA-1.2.3; preserves all configured money/limits."""
    initial = paper.load_paper_state()['cash']
    with conn.cursor().copy('COPY app_rows(namespace,row_json) FROM STDIN') as copy:
        for number in range(count//2):
            identity=f'synthetic-history-{number}'
            stamp=f'2026-01-01T00:{number//60%60:02d}:{number%60:02d}+00:00'
            for side in ('BUY','SELL'):
                row={'trade_id':identity,'timestamp':stamp,'asset':'AAPL','side':side,
                     'quantity':1.0,'price':10.0,'gross_value':10.0,'cost':0.0,
                     'cash_after':initial-(10 if side=='BUY' else 0),'realized_pnl':0.0}
                copy.write_row(('paper_trades',json.dumps(row)))
                copy.write_row(('table:trader_orders.csv',json.dumps({k:v for k,v in
                    {**row,'source':'synthetic-history'}.items() if k in
                    {'timestamp','asset','side','quantity','price','gross_value','cost','cash_after','source'}})))
            copy.write_row(('table:trade_reports.csv',json.dumps({'trade_id':identity,'asset':'AAPL',
                'quantity':1.0,'entry_price':10.0,'exit_price':10.0,'gross_entry_value':10.0,
                'gross_exit_value':10.0,'realized_pnl':0.0,'opened_at':stamp,'closed_at':stamp})))


def statistics(values):
    values=sorted(v for v in values if v is not None)
    size=len(values)
    if not size:return {'n':0,'median':None,'p95':None,'max':None}
    median=(values[(size-1)//2]+values[size//2])/2
    return {'n':size,'median':median,'p95':values[math.ceil(.95*size)-1] if size>=10 else None,
            'p95_method':'nearest_rank; n=10 gives observed maximum' if size>=10 else 'insufficient_samples',
            'max':values[-1]}


@pytest.mark.parametrize('events',[0,1000,5000])
def test_protocol_load_with_repeated_stop(pg,monkeypatch,events):
    p,conn,store,control,paper,trader,q=_setup(pg,monkeypatch)
    assert shutil.disk_usage(ROOT).free>256*1024**2 and events<=5000
    seed_history(conn,paper,events)
    stages={}
    def measured(module,name):
        original=getattr(module,name)
        @wraps(original)
        def wrapped(*args,**kwargs):
            location='inside' if p._ACTIVE_CONNECTION.get() is not None else 'outside'
            start=time.perf_counter()
            try:return original(*args,**kwargs)
            finally:
                key=f'{module.__name__}.{name}:{location}'
                item=stages.setdefault(key,{'calls':0,'ms':0.0})
                item['calls']+=1;item['ms']+=(time.perf_counter()-start)*1000
        monkeypatch.setattr(module,name,wrapped)
    for module,names in [(q,['manifest','begin_intent','record_unit_result','reconcile','reject_duplicate_events','_hash']),
                         (p,['_decode_json']), (trader,['_sync_trader_orders_into_storage'])]:
        for name in names:measured(module,name)
    original_lock=p._lock_namespace;original_transaction=p._transaction_context
    financial_pids=set();acquired={};lock_times=[];commits=[];stop_commits=[];worker_locks=[]
    locked=threading.Event();release=threading.Event();stop_at_lock=threading.Event();stop_pid=[]
    race={}
    def timed_lock(connection,namespace):
        pid=connection.info.backend_pid
        if namespace=='atlas:persistence-unit' and threading.current_thread().name.startswith('infra124-stop'):
            stop_pid.append(pid);stop_at_lock.set()
        value=original_lock(connection,namespace)
        if namespace=='atlas:persistence-unit':
            acquired[pid]=time.perf_counter()
            if pid in financial_pids:
                race['acquired_at']=acquired[pid];locked.set()
                assert release.wait(10)
        return value
    @contextmanager
    def timed_transaction(connection,witness):
        pid=connection.info.backend_pid
        if witness is not None:financial_pids.add(pid)
        with original_transaction(connection,witness):
            yield
            commit_start=time.perf_counter()
        finished=time.perf_counter()
        if threading.current_thread().name.startswith('infra124-finance'):
            worker_locks.append((finished-acquired[pid])*1000)
        if witness is not None:
            lock_times.append((finished-acquired[pid])*1000);commits.append((finished-commit_start)*1000)
            financial_pids.remove(pid)
        elif threading.current_thread().name.startswith('infra124-stop'):
            stop_commits.append(finished)
    monkeypatch.setattr(p,'_lock_namespace',timed_lock)
    monkeypatch.setattr(p,'_transaction_context',timed_transaction)
    samples=[];growth=[];started=time.perf_counter();timed_out=False
    for iteration in range(10):
        assert time.perf_counter()-started<240, 'Bounded benchmark exhausted; do not infer unmeasured samples'
        if iteration:
            control.start_robot(store.load_bot_state(),requested_by='synthetic-benchmark-iteration')
        locked.clear();release.clear();stop_at_lock.clear();stop_commits.clear();worker_locks.clear()
        tracemalloc.start()
        with ThreadPoolExecutor(max_workers=1,thread_name_prefix='infra124-finance') as finance_pool, ThreadPoolExecutor(
                max_workers=1,thread_name_prefix='infra124-stop') as stop_pool:
            def run_measured_cycle():
                at=time.perf_counter()
                result=trader.run_trader_cycle()
                race['cycle_elapsed_ms']=(time.perf_counter()-at)*1000
                return result
            finance=finance_pool.submit(run_measured_cycle)
            assert locked.wait(30)
            request=time.perf_counter()
            stop=stop_pool.submit(control.request_stop,'synthetic-benchmark-stop')
            try:
                assert stop_at_lock.wait(5)
                deadline=time.monotonic()+3
                while not conn.execute("SELECT EXISTS(SELECT 1 FROM pg_locks WHERE pid=%s AND locktype='advisory' AND NOT granted)",(stop_pid[-1],)).fetchone()[0]:
                    assert time.monotonic()<deadline
                    threading.Event().wait(.005)
            finally:
                setup=time.perf_counter()-race['acquired_at'];release.set()
            receipt=stop.result(15);returned=time.perf_counter()
            result=finance.result(60)
        assert result['committed'] and result['commit_outcome']=='COMMITTED'
        ack_ms=None
        if receipt.persisted:
            at=time.perf_counter();ack=control.acknowledge_stop('synthetic-benchmark-worker')
            ack_ms=(time.perf_counter()-at)*1000;assert ack.acknowledged
        at=time.perf_counter();proof=q.reconcile(result['cycle_id'])
        recon_ms=(time.perf_counter()-at)*1000
        assert proof['outcome']=='COMMITTED' and proof['resolution_recorded']
        _,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
        assert peak<128*1024**2
        samples.append({'iteration':iteration+1,'financial_lock_ms':lock_times[-1],
            'commit_ack_ms':commits[-1],'stop_commit_ack_ms':(stop_commits[0]-request)*1000 if stop_commits else None,
            'stop_verified_ms':(returned-request)*1000,'stop_persisted':receipt.persisted,
            'stop_ack_ms':ack_ms,'reconciliation_ms':recon_ms,'python_peak_bytes':peak,
            'process_rss_high_water_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            'controlled_lock_setup_ms':setup*1000, 'cycle_elapsed_ms':race['cycle_elapsed_ms'],
            'worker_global_lock_ms':sum(worker_locks),
            'worker_outside_global_lock_ms_estimate':max(0,race['cycle_elapsed_ms']-sum(worker_locks))})
        count,total=conn.execute("SELECT count(*),sum(octet_length(payload_json)) FROM app_state WHERE namespace LIKE 'paper_cycle_receipt:%%'").fetchone()
        growth.append({'receipts':count,'payload_bytes':total})
        assert len(paper.read_paper_trades())==events+1
        if not receipt.persisted:
            assert receipt.status=='UNCONFIRMED' and not receipt.acknowledged
            timed_out=True
            # End this scenario; no next financial cycle/retry after failed STOP.
            break
    assert not q.gate_status()['blocked']
    keys=['financial_lock_ms','commit_ack_ms','stop_commit_ack_ms','stop_verified_ms','stop_ack_ms',
          'reconciliation_ms','python_peak_bytes','process_rss_high_water_bytes',
          'cycle_elapsed_ms','worker_global_lock_ms','worker_outside_global_lock_ms_estimate']
    summary={key:statistics([s[key] for s in samples]) for key in keys}
    budgets={'stop_verified_ms':2000,'stop_ack_ms':1000,'reconciliation_ms':2000,'financial_lock_ms':1500}
    within=not timed_out and len(samples)==10 and all(summary[k]['max']<=limit for k,limit in budgets.items())
    write_evidence(conn,'performance',str(events),{'historical_events':events,'samples':samples,'statistics':summary,
        'inclusive_stage_profile':stages,'profile_note':'Inclusive times overlap; do not sum parents/children. Entry location for wrappers; manifest/decode execute inside lock.',
        'receipts_growth':growth,'new_trades':1,'no_financial_retry':True,'lock_timeout_unchanged_seconds':5,
        'timeout_observed':timed_out,'unexecuted_repetitions':10-len(samples),'within_candidate_budgets':within,
        'budgets_ms':budgets,'elapsed_seconds':time.perf_counter()-started,'hours72_authorized':False,
        'limits':{'max_historical_events':5000,'repetitions':10,'measured_window_seconds':240,'python_peak_bytes':128*1024**2,
                  'server_memory_bytes':512*1024**2,'server_cpus':1}})


def test_order_sync_preserves_existing_rows_and_noop_does_not_rewrite(pg,monkeypatch):
    p,conn,_,_,paper,trader,q=_setup(pg,monkeypatch)
    seed_history(conn,paper,2)
    def orders():
        return conn.execute("SELECT seq,row_json,created_at FROM app_rows WHERE namespace='table:trader_orders.csv' ORDER BY seq").fetchall()
    before=orders()
    trader.run_trader_cycle()
    first=orders()
    assert first[:len(before)]==before, 'Existing canonical rows must retain seq, JSON and timestamp'
    assert len(first)==len(before)+1
    trader._sync_trader_orders_into_storage()
    assert orders()==first, 'A no-op sync must perform no historical replacement'
    assert len(paper.read_paper_trades())==3


def test_sync_failure_rolls_back_full_financial_unit(pg,monkeypatch):
    p,conn,_,_,paper,trader,q=_setup(pg,monkeypatch)
    seed_history(conn,paper,2)
    before=q.manifest()
    orders=conn.execute("SELECT seq,row_json,created_at FROM app_rows WHERE namespace='table:trader_orders.csv' ORDER BY seq").fetchall()
    original=q.record_unit_result
    def fail_after_receipt(*args,**kwargs):
        original(*args,**kwargs)
        raise ValueError('synthetic failure after financial rows and receipt before COMMIT')
    monkeypatch.setattr(q,'record_unit_result',fail_after_receipt)
    with pytest.raises(ValueError) as caught:trader.run_trader_cycle()
    assert caught.value.commit_outcome=='NOT_COMMITTED'
    assert q.manifest()==before and len(paper.read_paper_trades())==2
    assert conn.execute("SELECT seq,row_json,created_at FROM app_rows WHERE namespace='table:trader_orders.csv' ORDER BY seq").fetchall()==orders


@pytest.mark.parametrize('namespace',['paper_trades','table:trader_orders.csv','table:trade_reports.csv'])
def test_history_tampered_after_receipt_stays_fenced(pg,monkeypatch,namespace):
    p,conn,_,_,paper,trader,q=_setup(pg,monkeypatch)
    seed_history(conn,paper,2)
    def alter(connection,witness):
        row=connection.execute('SELECT seq,row_json FROM app_rows WHERE namespace=%s ORDER BY seq LIMIT 1',(namespace,)).fetchone()
        payload=json.loads(row[1]);payload['price' if namespace!='table:trade_reports.csv' else 'exit_price']=999.0
        connection.execute('UPDATE app_rows SET row_json=%s WHERE seq=%s',(json.dumps(payload),row[0]))
    monkeypatch.setattr(p,'_before_financial_commit',alter)
    result=trader.run_trader_cycle()
    assert result['commit_outcome']=='POST_COMMIT_DIAGNOSTIC_FAILURE' and result['committed']
    assert q.gate_status()['blocked'] and q.reconcile()['outcome']=='UNKNOWN_COMMIT'
    count=len(paper.read_paper_trades())
    with pytest.raises(q.ReconciliationRequired):trader.run_trader_cycle()
    assert len(paper.read_paper_trades())==count


def test_duplicate_buy_sell_and_report_still_rejected(pg,monkeypatch):
    p,conn,_,_,paper,trader,q=_setup(pg,monkeypatch)
    seed_history(conn,paper,2)
    rows=paper.read_paper_trades()
    report=p.load_json_rows('table:trade_reports.csv',Path('unused.json'))[0]
    for event in rows:
        with p.persistence_transaction(),pytest.raises(p.PersistenceDataError):
            q.reject_duplicate_events([event],[])
    with p.persistence_transaction(),pytest.raises(p.PersistenceDataError):
        q.reject_duplicate_events([],[report])
    assert len(paper.read_paper_trades())==2


def test_receipt_growth_uses_real_aborted_intents_and_maintenance_preserves_them(pg,monkeypatch):
    p,conn,store,control,paper,trader,q=_setup(pg,monkeypatch)
    samples=[];created=[];started=time.perf_counter()
    for target in (1,32,128):
        while len(created)<target:
            assert time.perf_counter()-started<120
            identity=q.begin_intent(paper.load_paper_state(),[],[])
            proof=q.reconcile(identity)
            assert proof['outcome']=='NOT_COMMITTED' and proof['resolution_recorded']
            created.append(identity)
        snapshot=conn.execute("SELECT namespace,payload_json FROM app_state WHERE namespace LIKE 'paper_cycle_receipt:%%' ORDER BY namespace").fetchall()
        assert len(snapshot)==target
        reads=[];reconciles=[]
        for _ in range(10):
            at=time.perf_counter();receipt=q._receipt(created[-1]);reads.append((time.perf_counter()-at)*1000)
            assert receipt['cycle_id']==created[-1]
            at=time.perf_counter();assert q.reconcile(created[-1])['outcome']=='NOT_COMMITTED'
            reconciles.append((time.perf_counter()-at)*1000)
        # Read-only maintenance inventory and existing retention must not purge receipts.
        from datetime import datetime,timezone
        retention=importlib.import_module('core.retention')
        before_maintenance=conn.execute("SELECT namespace,payload_json FROM app_state WHERE namespace LIKE 'paper_cycle_receipt:%%' ORDER BY namespace").fetchall()
        at=time.perf_counter();retention.run_retention_job(now=datetime(2026,11,12,tzinfo=timezone.utc))
        maintenance_ms=(time.perf_counter()-at)*1000
        assert conn.execute("SELECT namespace,payload_json FROM app_state WHERE namespace LIKE 'paper_cycle_receipt:%%' ORDER BY namespace").fetchall()==before_maintenance
        samples.append({'receipts':target,'payload_bytes':sum(len(text.encode()) for _,text in snapshot),
            'lookup_ms':statistics(reads),'reconciliation_ms':statistics(reconciles),
            'maintenance_ms':maintenance_ms,'all_receipts_preserved':True})
    assert paper.read_paper_trades()==[] and not q.gate_status()['blocked']
    write_evidence(conn,'receipts','growth',{'samples':samples,'real_prepared_intents_proven_not_committed':128,
        'financial_units_attempted':0,'financial_trades':0,'receipt_cleanup_implemented':False,
        'elapsed_seconds':time.perf_counter()-started,'max_creation_window_seconds':120,
        'note':'Real protocol bookkeeping with no financial transaction; not 128 trading cycles nor proof of 72h growth.'})


def test_reconciliation_reuses_only_current_fenced_observations(pg,monkeypatch):
    p,conn,_,_,paper,trader,q=_setup(pg,monkeypatch)
    first=trader.run_trader_cycle()['cycle_id']
    calls=[];original=p.load_json_rows
    def count(namespace,*args,**kwargs):
        calls.append(namespace)
        return original(namespace,*args,**kwargs)
    monkeypatch.setattr(p,'load_json_rows',count)
    expected=q.manifest()
    assert q.reconcile(first)['outcome']=='COMMITTED'
    assert not set(calls)&{'paper_trades','table:trade_reports.csv'}, 'Do not decode the same fenced rows twice'
    assert q.manifest()==expected
    # A new reconciliation MUST read the database afresh, even with unchanged counts.
    pending=q.begin_intent(paper.load_paper_state(),[],[])
    row=conn.execute("SELECT seq,row_json FROM app_rows WHERE namespace='paper_trades' ORDER BY seq LIMIT 1").fetchone()
    changed=json.loads(row[1]);changed['price']=999.0
    conn.execute('UPDATE app_rows SET row_json=%s WHERE seq=%s',(json.dumps(changed),row[0]))
    assert q.reconcile(pending)['outcome']=='UNKNOWN_COMMIT'
    assert q.gate_status()['blocked'] and q.gate_status()['cycle_id']==pending


@pytest.mark.parametrize('namespace',['paper_trades','table:trader_orders.csv','table:trade_reports.csv'])
def test_empty_plan_does_not_bypass_strict_historical_json_validation(pg,monkeypatch,namespace):
    p,conn,_,_,paper,trader,q=_setup(pg,monkeypatch)
    seed_history(conn,paper,2)
    conn.execute("UPDATE app_rows SET row_json=%s WHERE seq=(SELECT min(seq) FROM app_rows WHERE namespace=%s)",
                 ('{"duplicate":1,"duplicate":2}',namespace))
    count=conn.execute("SELECT count(*) FROM app_rows WHERE namespace='paper_trades'").fetchone()[0]
    with pytest.raises(p.PersistenceDataError):
        q.begin_intent(paper.load_paper_state(),[],[])
    assert conn.execute("SELECT count(*) FROM app_rows WHERE namespace='paper_trades'").fetchone()[0]==count
    assert conn.execute("SELECT count(*) FROM app_state WHERE namespace LIKE 'paper_cycle_receipt:%%'").fetchone()[0]==0


def test_save_still_refuses_corrupt_current_json_even_with_matching_cas(pg):
    p,conn,_,root=pg
    path=root/'synthetic.json'
    snapshot=p.load_json_state('synthetic_integrity',lambda:{'value':1},path)
    for invalid in ('{','{"k":1,"k":2}','[1]','{"x":NaN}'):
        conn.execute('UPDATE app_state SET payload_json=%s WHERE namespace=%s',(invalid,'synthetic_integrity'))
        candidate=p.StateSnapshot({'value':2},snapshot.source,invalid)
        with pytest.raises(p.PersistenceDataError):p.save_json_state('synthetic_integrity',candidate,path)
        assert conn.execute('SELECT payload_json FROM app_state WHERE namespace=%s',('synthetic_integrity',)).fetchone()[0]==invalid
