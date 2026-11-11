"""Deterministic stop/commit regressions; disposable local PostgreSQL runner only."""
from concurrent.futures import ThreadPoolExecutor
import importlib
import json
import os
import subprocess
import sys
import threading

import pytest

from tests.test_infra12_postgres import ROOT, pg, pytestmark


def _modules():
    store = importlib.import_module("core.state_store")
    control = importlib.import_module("core.stop_control")
    paper = importlib.import_module("paper_trading_engine")
    trader = importlib.import_module("engines.trader_engine")
    return store, control, paper, trader


def _running():
    store, control, paper, trader = _modules()
    state = store.load_bot_state()
    state["trader"]["watchlist"] = ["AAPL"]
    store.save_bot_state(state)
    control.start_robot(store.load_bot_state(), requested_by="synthetic-fixture")
    assert store.load_bot_state()["bot_status"] == "RUNNING"
    return store, control, paper, trader


def _synthetic_market(monkeypatch, paper, *, on_feed=None):
    from tests.conftest import fake_market_result
    from tests.test_trader_smoke import _fake_prices

    def feed(symbols, config, requested_by="worker_cycle"):
        if on_feed:
            on_feed()
        return fake_market_result({str(symbol).upper(): _fake_prices() for symbol in symbols}, requested_by)

    monkeypatch.setattr(paper, "load_market_data_result", feed)
    monkeypatch.setattr(paper, "_evaluate_buy_signal", lambda latest, config: {
        "buy": True, "score": 0.95, "trend_ok": True, "score_ok": True,
        "rsi_ok": True, "atr_ok": True, "structure_ok": True, "momentum_ok": True,
        "trend_alignment": "aligned", "rejection_reasons": [],
    })
    monkeypatch.setattr(paper, "_should_sell", lambda *args: (False, ""))


def test_stop_persists_while_worker_awaits_feed_and_draft_never_commits(pg, monkeypatch):
    """A/B/D: acquisition owns no writer transaction; STOP fences its draft."""
    p, _, _, _ = pg
    store, control, paper, trader = _running()
    entered, release = threading.Event(), threading.Event()
    append_calls, shadow_calls = [], []
    append = paper._append_paper_trade
    shadow_fetch = paper.fetch_multi_timeframe_intraday_data

    def on_feed():
        assert p._ACTIVE_CONNECTION.get() is None
        entered.set()
        assert release.wait(15)

    def tracked_shadow_fetch(*args, **kwargs):
        assert p._ACTIVE_CONNECTION.get() is None
        shadow_calls.append(True)
        return shadow_fetch(*args, **kwargs)

    def tracked_append(trade):
        append_calls.append(trade)
        append(trade)

    _synthetic_market(monkeypatch, paper, on_feed=on_feed)
    monkeypatch.setattr(paper, "_append_paper_trade", tracked_append)
    monkeypatch.setattr(paper, "fetch_multi_timeframe_intraday_data", tracked_shadow_fetch)
    paper_before = dict(paper.load_paper_state())
    with ThreadPoolExecutor(max_workers=1) as pool:
        worker = pool.submit(trader.run_trader_cycle)
        try:
            assert entered.wait(10)
            receipt = control.request_stop(requested_by="synthetic-operator")
            assert receipt.persisted and not receipt.acknowledged
            assert receipt.status == "PERSISTED"
            assert not worker.done() and not release.is_set()
            observed = store.load_bot_state()
            assert observed["bot_status"] == "STOPPED"
            assert control.stop_control_status(observed)["request_id"] == receipt.request_id
        finally:
            release.set()
        result = worker.result(timeout=15)
    assert result["status"] == "STOPPED"
    assert not append_calls
    assert shadow_calls  # Existing shadow path was observed, never enabled here.
    assert paper.read_paper_trades() == []
    assert paper.load_paper_state() == paper_before
    final = control.stop_control_status(store.load_bot_state())
    assert final["status"] == "ACKNOWLEDGED"
    assert final["request_id"] == receipt.request_id


def test_stop_requested_during_commit_waits_then_persists_in_order(pg, monkeypatch):
    """C: no retroactive cancellation; a prior transaction may finish first."""
    p, _, _, root = pg
    store, control, _, _ = _running()
    transaction_held, release, stop_at_lock = (threading.Event() for _ in range(3))
    lock = p._lock_namespace

    def tracked_lock(conn, namespace):
        if threading.current_thread().name.startswith("stop-request") and namespace == "atlas:persistence-unit":
            stop_at_lock.set()
        return lock(conn, namespace)

    monkeypatch.setattr(p, "_lock_namespace", tracked_lock)

    def writer():
        with p.persistence_transaction("synthetic_ledger"):
            p.append_json_row("synthetic_ledger", {"id": "prior-commit"}, root / "unused.json")
            transaction_held.set()
            assert release.wait(10)

    with ThreadPoolExecutor(max_workers=1) as writer_pool, ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="stop-request") as stop_pool:
        prior = writer_pool.submit(writer)
        assert transaction_held.wait(10)
        stop = stop_pool.submit(control.request_stop, requested_by="synthetic-operator")
        try:
            assert stop_at_lock.wait(5)
            assert not stop.done()
        finally:
            release.set()
        prior.result(timeout=10)
        receipt = stop.result(timeout=10)
    assert receipt.persisted and receipt.status == "PERSISTED"
    assert p.load_json_rows("synthetic_ledger", root / "unused.json") == [{"id": "prior-commit"}]
    assert store.load_bot_state()["bot_status"] == "STOPPED"


def test_lock_timeout_reports_stop_unconfirmed_without_blind_retry(pg, monkeypatch):
    """H: real PG lock timeout, never display a failed write as a confirmed STOP."""
    p, _, _, _ = pg
    store, control, _, _ = _running()
    held, release = threading.Event(), threading.Event()

    def hold():
        with p.persistence_transaction():
            held.set()
            assert release.wait(12)

    with ThreadPoolExecutor(max_workers=1) as pool:
        writer = pool.submit(hold)
        assert held.wait(10)
        try:
            receipt = control.request_stop(requested_by="synthetic-operator")
            assert not receipt.persisted and not receipt.acknowledged
            assert receipt.status == "UNCONFIRMED" and receipt.error_code
        finally:
            release.set()
        writer.result(timeout=10)
    assert store.load_bot_state()["bot_status"] == "RUNNING"


def test_postgres_failure_during_stop_cannot_claim_persistence(pg, monkeypatch):
    """G: application gets no false receipt when the connection cannot open."""
    p, conn, _, _ = pg
    store, control, _, _ = _running()
    import psycopg
    attempts = []

    def unavailable():
        attempts.append(True)
        raise psycopg.OperationalError("synthetic PostgreSQL unavailable")

    monkeypatch.setattr(p, "_connect", unavailable)
    receipt = control.request_stop(requested_by="synthetic-operator")
    assert receipt.status == "UNCONFIRMED" and not receipt.persisted
    assert not receipt.acknowledged and receipt.error_code
    assert len(attempts) == 1
    payload = json.loads(conn.execute("SELECT payload_json FROM app_state WHERE namespace='bot_state'").fetchone()[0])
    assert payload["bot_status"] == "RUNNING"


def test_stop_and_heartbeat_preserve_each_other(pg):
    """E: heartbeat must never replace the stop intent with a stale status."""
    store, control, _, _ = _running()
    barrier = threading.Barrier(2)

    def stop():
        barrier.wait(5)
        return control.request_stop(requested_by="synthetic-operator")

    def heartbeat():
        barrier.wait(5)
        store.update_worker_heartbeat("online")

    with ThreadPoolExecutor(max_workers=2) as pool:
        stop_job, beat_job = pool.submit(stop), pool.submit(heartbeat)
        receipt = stop_job.result(timeout=10)
        beat_job.result(timeout=10)
    assert receipt.persisted
    state = store.load_bot_state()
    assert state["bot_status"] == "STOPPED" and state["worker_heartbeat"]
    assert control.stop_control_status(state)["request_id"] == receipt.request_id


def test_stale_configuration_snapshot_cannot_undo_confirmed_stop(pg):
    """F: a stale UI save is rejected; no retry changes the operator's intent."""
    p, _, _, _ = pg
    store, control, _, _ = _running()
    stale = store.load_bot_state()
    receipt = control.request_stop(requested_by="synthetic-operator")
    assert receipt.persisted
    stale["fixture_ui_setting"] = "stale-change"
    with pytest.raises(p.PersistenceConflictError):
        store.save_bot_state(stale)
    state = store.load_bot_state()
    assert state["bot_status"] == "STOPPED"
    assert "fixture_ui_setting" not in state
    assert control.stop_control_status(state)["request_id"] == receipt.request_id


def test_atomic_configuration_change_concurrent_with_stop_preserves_both(pg):
    store, control, _, _ = _running()
    barrier = threading.Barrier(2)

    def stop():
        barrier.wait(5)
        return control.request_stop(requested_by="synthetic-operator")

    def settings():
        barrier.wait(5)
        return store.update_bot_state(lambda state: {**state, "fixture_ui_setting": "synthetic-only"})

    with ThreadPoolExecutor(max_workers=2) as pool:
        stop_job, settings_job = pool.submit(stop), pool.submit(settings)
        receipt = stop_job.result(timeout=10)
        settings_job.result(timeout=10)
    state = store.load_bot_state()
    assert receipt.persisted and state["bot_status"] == "STOPPED"
    assert state["fixture_ui_setting"] == "synthetic-only"


def test_repeated_cycles_after_stop_do_not_fetch_or_create_entries(pg, monkeypatch):
    """L: STOP is observed on every fresh cycle without touching the provider."""
    store, control, paper, trader = _running()
    receipt = control.request_stop(requested_by="synthetic-operator")
    assert receipt.persisted

    def forbidden(*args, **kwargs):
        raise AssertionError("STOPPED cycle must not start another market request")

    monkeypatch.setattr(paper, "load_market_data_result", forbidden)
    for _ in range(4):
        assert trader.run_trader_cycle()["status"] == "STOPPED"
    assert paper.read_paper_trades() == []
    state = store.load_bot_state()
    assert state["bot_status"] == "STOPPED"
    assert control.stop_control_status(state)["status"] == "ACKNOWLEDGED"


def test_restart_in_new_process_keeps_stopped_without_starting_worker_loop(pg):
    """I: simulate process restart by importing and running one bounded cycle."""
    _, _, _, _ = pg
    store, control, paper, _ = _running()
    receipt = control.request_stop(requested_by="synthetic-operator")
    assert receipt.persisted
    program = """
import json
from core.state_store import load_bot_state
from engines.trader_engine import run_trader_cycle
from paper_trading_engine import read_paper_trades
before=load_bot_state()['bot_status']
result=run_trader_cycle()
print(json.dumps({'before':before,'after':load_bot_state()['bot_status'],
                  'status':result['status'],'trades':len(read_paper_trades())}))
"""
    result = subprocess.run([sys.executable, "-c", program], cwd=ROOT,
                            check=True, capture_output=True, text=True, timeout=25)
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload == {"before": "STOPPED", "after": "STOPPED", "status": "STOPPED", "trades": 0}
    assert store.load_bot_state()["bot_status"] == "STOPPED"
    assert paper.read_paper_trades() == []


def test_failure_after_trade_then_explicit_retry_does_not_duplicate(pg, monkeypatch):
    """J/idempotence: failed transaction is absent, explicit retry creates one BUY."""
    store, control, paper, trader = _running()
    _synthetic_market(monkeypatch, paper)
    before = dict(paper.load_paper_state())
    append, attempted = paper._append_paper_trade, []

    def failed_append(trade):
        append(trade)
        attempted.append(trade)
        raise ValueError("synthetic failure after trade before commit")

    monkeypatch.setattr(paper, "_append_paper_trade", failed_append)
    with pytest.raises(ValueError, match="after trade before commit"):
        trader.run_trader_cycle()
    assert len(attempted) == 1
    assert paper.read_paper_trades() == []
    assert paper.load_paper_state() == before
    monkeypatch.setattr(paper, "_append_paper_trade", append)
    trader.run_trader_cycle()
    trades = paper.read_paper_trades()
    assert len(trades) == 1 and trades[0]["side"] == "BUY"
    assert len({(trade["trade_id"], trade["side"]) for trade in trades}) == len(trades)
    assert len(paper.load_paper_state()["positions"]) == 1


def test_repeated_successful_cycles_do_not_duplicate_open_trade_or_report(pg, monkeypatch):
    _, _, paper, trader = _running()
    _synthetic_market(monkeypatch, paper)
    trader.run_trader_cycle()
    trader.run_trader_cycle()
    trades = paper.read_paper_trades()
    assert len(trades) == 1
    assert len({(trade["trade_id"], trade["side"]) for trade in trades}) == 1
    from core.trader_reports import read_trade_reports
    assert read_trade_reports().empty  # No SELL, so no closed-trade report.


def test_two_processes_serialize_stop_and_heartbeat_without_losing_intent(pg):
    """K: separate interpreters wait for a barrier before mutating the same state."""
    p, conn, _, _ = pg
    store, control, _, _ = _running()
    # A test-only advisory gate synchronizes the two child processes. It is not
    # part of the application protocol and cannot touch non-synthetic databases.
    gate = 12120261111
    conn.execute("SELECT pg_advisory_lock(%s)", (gate,))
    program = """
import os,sys
import psycopg
if sys.argv[1]=='stop':
    from core.stop_control import request_stop
else:
    from core.state_store import update_worker_heartbeat
with psycopg.connect(os.environ['DATABASE_URL'],autocommit=True) as barrier:
    barrier.execute('SELECT pg_advisory_lock(%s)',(12120261111,))
    barrier.execute('SELECT pg_advisory_unlock(%s)',(12120261111,))
if sys.argv[1]=='stop':
    receipt=request_stop(requested_by='synthetic-child')
    assert receipt.persisted,receipt.error_code
else:
    update_worker_heartbeat('synthetic-child')
"""
    children = [subprocess.Popen([sys.executable, "-c", program, action], cwd=ROOT,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                for action in ("stop", "heartbeat")]
    try:
        # Server-side condition establishes both children reached the gate. The
        # bounded Event waits only pace polling, not decide the concurrency order.
        waiting = threading.Event()
        for _ in range(500):
            count = conn.execute("SELECT count(*) FROM pg_locks WHERE locktype='advisory' AND NOT granted").fetchone()[0]
            if count >= 2:
                break
            waiting.wait(0.01)
        else:
            raise AssertionError("Both child processes did not reach PostgreSQL barrier")
        conn.execute("SELECT pg_advisory_unlock(%s)", (gate,))
        for child in children:
            stdout, stderr = child.communicate(timeout=20)
            assert child.returncode == 0, (stdout, stderr)
    finally:
        conn.execute("SELECT pg_advisory_unlock(%s)", (gate,))
        for child in children:
            if child.poll() is None:
                child.kill()
                child.communicate()
    state = store.load_bot_state()
    assert state["bot_status"] == "STOPPED" and state["worker_status"] == "synthetic-child"
    assert control.stop_control_status(state)["status"] == "PERSISTED"


@pytest.mark.parametrize("changed_guard", ["trader_disabled", "risk_blocked"])
def test_operational_guard_change_during_feed_discards_stale_draft(pg, monkeypatch, changed_guard):
    """B/F validation stage must re-read current guards, not the feed's snapshot."""
    p, _, _, _ = pg
    store, _, paper, trader = _running()
    entered, release = threading.Event(), threading.Event()

    def on_feed():
        entered.set()
        assert release.wait(15)

    _synthetic_market(monkeypatch, paper, on_feed=on_feed)
    before = dict(paper.load_paper_state())
    with ThreadPoolExecutor(max_workers=1) as pool:
        worker = pool.submit(trader.run_trader_cycle)
        try:
            assert entered.wait(10)
            def change(state):
                if changed_guard == "trader_disabled":
                    state["trader"]["enabled"] = False
                else:
                    state["risk"]["daily_loss_block_active"] = True
                    state["risk"]["daily_loss_block_reason"] = "synthetic guard changed during feed"
                return state
            store.update_bot_state(change)
        finally:
            release.set()
        with pytest.raises(p.PersistenceConflictError):
            worker.result(timeout=15)
    assert paper.read_paper_trades() == []
    assert paper.load_paper_state() == before
    state = store.load_bot_state()
    assert not state["trader"]["enabled"] if changed_guard == "trader_disabled" else state["risk"]["daily_loss_block_active"]


def test_failure_after_closed_report_rolls_back_sell_and_retry_is_not_duplicated(pg, monkeypatch):
    """A SELL, its closed report, cash and position share exactly one DB commit."""
    _, _, paper, trader = _running()
    _synthetic_market(monkeypatch, paper)
    trader.run_trader_cycle()
    before = dict(paper.load_paper_state())
    buy = paper.read_paper_trades()[0]
    monkeypatch.setattr(paper, "_should_sell", lambda *args: (True, "synthetic-close"))
    append_report = paper.append_trade_report
    attempted = []

    def failed_report(report):
        append_report(report)
        attempted.append(report)
        raise OSError("synthetic interruption after report before commit")

    monkeypatch.setattr(paper, "append_trade_report", failed_report)
    with pytest.raises(OSError, match="after report before commit"):
        trader.run_trader_cycle()
    from core.trader_reports import read_trade_reports
    assert len(attempted) == 1
    assert paper.load_paper_state() == before
    assert paper.read_paper_trades() == [buy]
    assert read_trade_reports().empty
    monkeypatch.setattr(paper, "append_trade_report", append_report)
    trader.run_trader_cycle()
    trader.run_trader_cycle()
    trades = paper.read_paper_trades()
    assert [trade["side"] for trade in trades] == ["BUY", "SELL"]
    assert trades[0]["trade_id"] == trades[1]["trade_id"] == buy["trade_id"]
    reports = read_trade_reports()
    assert len(reports) == 1 and reports.iloc[0]["trade_id"] == buy["trade_id"]
    assert paper.load_paper_state()["positions"] == {}
