"""SQL/stage/lock measurements for the isolated snapshot study; no parameters logged."""
from contextlib import contextmanager
from functools import wraps
import threading
import time


class CountedConnection:
    def __init__(self, connection, metrics):self.connection,self.metrics=connection,metrics
    def __getattr__(self,name):return getattr(self.connection,name)
    def __enter__(self):self.connection.__enter__();return self
    def __exit__(self,*args):return self.connection.__exit__(*args)
    def execute(self,query,*args,**kwargs):
        started=time.perf_counter()
        try:return self.connection.execute(query,*args,**kwargs)
        finally:self.metrics.sql(query,(time.perf_counter()-started)*1000)
    def cursor(self,*args,**kwargs):return CountedCursor(self.connection.cursor(*args,**kwargs),self.metrics)
    @contextmanager
    def transaction(self,*args,**kwargs):
        # Normal benchmark paths count the BEGIN/SAVEPOINT and COMMIT/RELEASE
        # emitted by psycopg. Errors get a separately labelled rollback count.
        self.metrics.controls+=1
        try:
            with self.connection.transaction(*args,**kwargs):yield
        except BaseException:
            self.metrics.controls+=1;self.metrics.rollbacks+=1;raise
        else:self.metrics.controls+=1


class CountedCursor:
    def __init__(self,cursor,metrics):self.cursor,self.metrics=cursor,metrics
    def __getattr__(self,name):return getattr(self.cursor,name)
    def __enter__(self):self.cursor.__enter__();return self
    def __exit__(self,*args):return self.cursor.__exit__(*args)
    def execute(self,query,*args,**kwargs):
        start=time.perf_counter()
        try:return self.cursor.execute(query,*args,**kwargs)
        finally:self.metrics.sql(query,(time.perf_counter()-start)*1000)


class Measurements:
    def __init__(self,p,monkeypatch):
        self.p=p;self.patch=monkeypatch;self.statements=0;self.controls=0;self.rollbacks=0
        self.sql_categories={};self.stages={};self.units=[];self.pending={};self.financial=set()
        self.on_financial_lock=lambda *_:None
        self.on_stop_attempt=lambda *_:None
        original_lock=p._lock_namespace;original_transaction=p._transaction_context
        def lock(conn,namespace):
            pid=conn.info.backend_pid;thread=threading.current_thread().name
            if namespace=='atlas:persistence-unit' and thread.startswith('infra125-stop'):self.on_stop_attempt(pid)
            start=time.perf_counter();value=original_lock(conn,namespace)
            if namespace=='atlas:persistence-unit':
                self.pending[pid]={'acquired':time.perf_counter(),'wait_ms':(time.perf_counter()-start)*1000,
                    'thread':thread,'kind':'financial' if pid in self.financial else 'metadata'}
                if pid in self.financial:self.on_financial_lock(pid)
            return value
        @contextmanager
        def transaction(conn,witness):
            pid=conn.info.backend_pid
            if witness is not None:self.financial.add(pid)
            with original_transaction(conn,witness):
                yield
                commit_start=time.perf_counter()
            ended=time.perf_counter();item=self.pending.pop(pid,None)
            if item:
                self.units.append({**item,'end':ended,'hold_ms':(ended-item['acquired'])*1000,
                                   'commit_ack_ms':(ended-commit_start)*1000})
            self.financial.discard(pid)
        monkeypatch.setattr(p,'_lock_namespace',lock)
        monkeypatch.setattr(p,'_transaction_context',transaction)
        connect=p._connect
        monkeypatch.setattr(p,'_connect',lambda:CountedConnection(connect(),self))

    def sql(self,query,ms):
        text=str(query).lstrip().upper()
        kind='lock' if 'ADVISORY' in text or text.startswith('LOCK ') else 'read' if text.startswith(('SELECT','SHOW')) else 'other_or_write'
        item=self.sql_categories.setdefault(kind,{'calls':0,'ms':0.0})
        item['calls']+=1;item['ms']+=ms;self.statements+=1

    def measure(self,module,name,label=None):
        original=getattr(module,name);stage=label or f'{module.__name__}.{name}'
        @wraps(original)
        def wrapped(*args,**kwargs):
            where='inside' if self.p._ACTIVE_CONNECTION.get() is not None else 'outside'
            start=time.perf_counter()
            try:return original(*args,**kwargs)
            finally:
                item=self.stages.setdefault(stage+':'+where,{'calls':0,'ms':0.0})
                item['calls']+=1;item['ms']+=(time.perf_counter()-start)*1000
        self.patch.setattr(module,name,wrapped)
