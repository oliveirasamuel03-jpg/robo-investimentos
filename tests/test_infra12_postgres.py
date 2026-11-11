"""Real PostgreSQL integration; run only via scripts/test_infra12_postgres.py.

Every test owns a new synthetic database inside the runner's disposable server.
The ordinary offline suite skips this module before making any connection.
"""
from concurrent.futures import ThreadPoolExecutor
import importlib
import os
from pathlib import Path
import secrets
import subprocess
import sys
import threading

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("ATLAS_INFRA12_LOCAL_RUN") != "synthetic-disposable-only",
    reason="requires explicit disposable local PostgreSQL runner",
)

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = (ROOT / "scripts/sql/new_history_schema.sql").read_text()


@pytest.fixture
def pg(isolated_storage, monkeypatch):
    import psycopg
    from psycopg import sql
    from psycopg.conninfo import conninfo_to_dict, make_conninfo
    base = os.environ["ATLAS_INFRA12_LOCAL_DSN"]
    params = conninfo_to_dict(base)
    assert params["host"] == "127.0.0.1" and params["dbname"] == "postgres"
    assert os.environ["BROKER_MODE"] == os.environ["BROKER_PROVIDER"] == "paper"
    database = f"atlas_infra12_{secrets.token_hex(8)}"
    with psycopg.connect(base, autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database)))
    dsn = make_conninfo(base, dbname=database)
    monkeypatch.setenv("DATABASE_URL", dsn)
    persistence = importlib.import_module("core.persistence")
    isolated_storage.mkdir(parents=True, exist_ok=True)
    with psycopg.connect(dsn, autocommit=True) as conn:
        yield persistence, conn, dsn, isolated_storage
    # No need to drop: the entire disposable server is removed by the runner.


def test_new_schema_empty_stays_empty_and_refuses_second_application(pg):
    p, conn, _, _ = pg
    import psycopg
    conn.execute(SCHEMA)
    assert conn.execute("SELECT count(*) FROM app_state").fetchone()[0] == 0
    assert conn.execute("SELECT count(*) FROM app_rows").fetchone()[0] == 0
    with pytest.raises(psycopg.errors.RaiseException, match="Refusing initialization"):
        conn.execute(SCHEMA)
    conn.execute("ROLLBACK")
    assert conn.execute("SELECT count(*) FROM app_state").fetchone()[0] == 0


def test_new_schema_refuses_data_in_another_schema(pg):
    _, conn, _, _ = pg
    import psycopg
    conn.execute("CREATE SCHEMA archive_fixture")
    conn.execute("CREATE TABLE archive_fixture.history(id int)")
    conn.execute("INSERT INTO archive_fixture.history VALUES (1)")
    with pytest.raises(psycopg.errors.RaiseException, match="Refusing initialization"):
        conn.execute(SCHEMA)
    conn.execute("ROLLBACK")
    assert conn.execute("SELECT * FROM archive_fixture.history").fetchall() == [(1,)]
    assert conn.execute("SELECT to_regclass('public.app_state')").fetchone()[0] is None


def test_database_never_imports_old_local_json_or_csv(pg):
    p, _, _, root = pg
    state = root / "old.json"
    state.write_text('{"legacy":true}')
    rows = root / "old_rows.json"
    rows.write_text('[{"legacy":true}]')
    csv = root / "old.csv"
    csv.write_text("id\nlegacy\n")
    csv.with_suffix(".csv.json").write_text('[{"id":"legacy"}]')
    assert p.load_json_state("fresh", lambda: {"fresh": True}, state) == {"fresh": True}
    assert p.load_json_rows("fresh_rows", rows) == []
    assert p.read_table(csv).empty
    assert state.read_text() == '{"legacy":true}'
    assert rows.read_text() == '[{"legacy":true}]'


def test_concurrent_state_updates_do_not_lose_increments(pg):
    p, _, _, root = pg
    path = root / "state.json"
    p.load_json_state("counter", lambda: {"value": 0}, path)
    barrier = threading.Barrier(8)
    def increment(_):
        barrier.wait(timeout=10)
        for _ in range(12):
            p.update_json_state("counter", lambda: {"value": 0}, path,
                                lambda state: {"value": state["value"] + 1})
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(increment, range(8)))
    assert p.load_json_state("counter", dict, path)["value"] == 96


def test_simultaneous_append_keeps_every_synthetic_record_once(pg):
    p, _, _, root = pg
    path = root / "rows.json"
    p.replace_json_rows("events", [], path)
    barrier = threading.Barrier(8)
    def append(worker):
        barrier.wait(timeout=10)
        for index in range(10):
            p.append_json_row("events", {"id": f"{worker}-{index}"}, path)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(append, range(8)))
    rows = p.load_json_rows("events", path)
    assert len(rows) == len({row["id"] for row in rows}) == 80


def test_separate_python_processes_update_one_namespace_without_loss(pg):
    p, _, _, root = pg
    path = root / "state.json"
    p.load_json_state("process_counter", lambda: {"value": 0}, path)
    program = """
from pathlib import Path
from core import persistence as p
for _ in range(8):
    p.update_json_state('process_counter', lambda: {'value':0}, Path('unused.json'),
                        lambda state: {'value': state['value'] + 1})
"""
    def run(_):
        subprocess.run([sys.executable, "-c", program], cwd=ROOT,
                       check=True, capture_output=True, timeout=45)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(run, range(4)))
    assert p.load_json_state("process_counter", dict, path)["value"] == 32


def test_simultaneous_replacements_never_mix_two_histories(pg):
    p, _, _, root = pg
    path = root / "rows.json"
    p.replace_json_rows("replace_race", [{"writer": -1, "id": 0}], path)
    barrier = threading.Barrier(8)
    def replace(writer):
        barrier.wait(timeout=10)
        for _ in range(3):
            p.replace_json_rows("replace_race", [{"writer": writer, "id": i} for i in range(25)], path)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(replace, range(8)))
    rows = p.load_json_rows("replace_race", path)
    assert len(rows) == 25
    assert len({row["writer"] for row in rows}) == 1
    assert [row["id"] for row in rows] == list(range(25))


def test_grouped_state_and_history_rollback_together(pg):
    p, _, _, root = pg
    path = root / "state.json"
    p.save_json_state("account", {"cash": 10}, path)
    p.replace_json_rows("ledger", [{"id": 1}], path)
    with pytest.raises(ValueError, match="synthetic interruption"):
        with p.persistence_transaction("account", "ledger"):
            state = p.load_json_state("account", dict, path)
            state["cash"] = 0
            p.save_json_state("account", state, path)
            p.append_json_row("ledger", {"id": 2}, path)
            raise ValueError("synthetic interruption")
    assert p.load_json_state("account", dict, path) == {"cash": 10}
    assert p.load_json_rows("ledger", path) == [{"id": 1}]


def test_caught_nested_failure_rolls_back_its_savepoint_only(pg):
    p, _, _, root = pg
    path = root / "rows.json"
    with p.persistence_transaction("events"):
        p.append_json_row("events", {"id": "before"}, path)
        try:
            with p.persistence_transaction("events"):
                p.append_json_row("events", {"id": "failed"}, path)
                raise ValueError("synthetic nested failure")
        except ValueError:
            pass
        p.append_json_row("events", {"id": "after"}, path)
    assert p.load_json_rows("events", path) == [{"id": "before"}, {"id": "after"}]


def test_contended_lock_times_out_without_blind_retry_or_state_change(pg):
    p, conn, _, root = pg
    import psycopg
    path = root / "state.json"
    p.save_json_state("timeout", {"value": 0}, path)
    with conn.transaction():
        p._lock_namespace(conn, "atlas:persistence-unit")
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(p.update_json_state, "timeout", dict, path,
                                 lambda state: {"value": state["value"] + 1})
            with pytest.raises(psycopg.errors.LockNotAvailable):
                future.result(timeout=8)
    assert p.load_json_state("timeout", dict, path) == {"value": 0}


def test_readers_see_only_committed_grouped_writes(pg):
    p, _, _, root = pg
    path = root / "state.json"
    p.save_json_state("visibility", {"value": "before"}, path)
    p.replace_json_rows("visibility_rows", [{"value": "before"}], path)
    ready, release = threading.Event(), threading.Event()
    def writer():
        with p.persistence_transaction("visibility", "visibility_rows"):
            state = p.load_json_state("visibility", dict, path)
            state["value"] = "after"
            p.save_json_state("visibility", state, path)
            p.append_json_row("visibility_rows", {"value": "after"}, path)
            ready.set()
            assert release.wait(5)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(writer)
        try:
            assert ready.wait(5)
            assert p.load_json_state("visibility", dict, path) == {"value": "before"}
            assert p.load_json_rows("visibility_rows", path) == [{"value": "before"}]
        finally:
            release.set()
        future.result(timeout=10)
    assert p.load_json_state("visibility", dict, path) == {"value": "after"}
    assert p.load_json_rows("visibility_rows", path) == [{"value": "before"}, {"value": "after"}]


def test_replacement_refuses_to_erase_corrupt_history(pg):
    p, conn, _, root = pg
    path = root / "rows.json"
    p.replace_json_rows("corrupt_rows", [{"id": 1}], path)
    conn.execute("INSERT INTO app_rows(namespace,row_json) VALUES ('corrupt_rows','{')")
    with pytest.raises(p.PersistenceDataError):
        p.replace_json_rows("corrupt_rows", [], path)
    assert conn.execute("SELECT row_json FROM app_rows WHERE namespace='corrupt_rows' ORDER BY seq").fetchall() == [('{"id": 1}',), ('{',)]


@pytest.mark.parametrize("operation", ["disable", "demote"])
def test_last_admin_race_cannot_remove_both_admins(pg, operation):
    from core.auth import users_store as users
    for username in ("admin_a", "admin_b"):
        users.create_user(username, "synthetic-password-only", "admin")
    barrier = threading.Barrier(2)
    def remove(username):
        barrier.wait(5)
        try:
            if operation == "disable":
                users.set_user_disabled(username, True)
            else:
                users.update_user_role(username, "user")
            return "ok"
        except ValueError as exc:
            assert "At least one enabled admin" in str(exc)
            return "blocked"
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(remove, ("admin_a", "admin_b"))) == ["blocked", "ok"]
    assert sum(user["role"] == "admin" and not user["disabled"] for user in users.list_users()) == 1


def test_concurrent_duplicate_username_has_single_winner(pg):
    from core.auth import users_store as users
    barrier = threading.Barrier(4)
    def create(_):
        barrier.wait(5)
        try:
            users.create_user("same_fixture", "synthetic-password-only", "user")
            return "ok"
        except ValueError as exc:
            assert "already exists" in str(exc)
            return "blocked"
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert sorted(pool.map(create, range(4))) == ["blocked", "blocked", "blocked", "ok"]
    assert [user["username"] for user in users.list_users()] == ["same_fixture"]


def test_actual_paper_cycle_failure_after_trade_rolls_back_balance_and_history(pg, monkeypatch):
    import paper_trading_engine as paper
    from tests.conftest import fake_market_result
    from tests.test_trader_smoke import _fake_prices
    config = paper.PaperTradingConfig(custom_tickers=["AAPL"], allow_new_entries=True, history_limit=50)
    before = dict(paper.load_paper_state(config))
    before_trades = paper.read_paper_trades()
    monkeypatch.setattr(paper, "load_market_data_result",
                        lambda symbols, config, requested_by="worker_cycle": fake_market_result(
                            {str(symbol).upper(): _fake_prices() for symbol in symbols}, requested_by))
    monkeypatch.setattr(paper, "_evaluate_buy_signal", lambda latest, config: {
        "buy": True, "score": 0.95, "trend_ok": True, "score_ok": True,
        "rsi_ok": True, "atr_ok": True, "structure_ok": True, "momentum_ok": True,
        "trend_alignment": "aligned", "rejection_reasons": [],
    })
    monkeypatch.setattr(paper, "_should_sell", lambda *args: (False, ""))
    append = paper._append_paper_trade
    recorded = []
    def fail_after_append(trade):
        append(trade)
        recorded.append(trade)
        raise ValueError("synthetic failure after actual paper trade write")
    monkeypatch.setattr(paper, "_append_paper_trade", fail_after_append)
    with pytest.raises(ValueError, match="after actual paper trade write"):
        paper.run_paper_cycle(config)
    assert len(recorded) == 1  # The injected interruption really reached a write.
    assert paper.load_paper_state(config) == before
    assert paper.read_paper_trades() == before_trades


def test_stale_snapshot_refuses_overwrite(pg):
    p, _, _, root = pg
    path = root / "state.json"
    p.save_json_state("cas", {"value": 0}, path)
    first = p.load_json_state("cas", dict, path)
    stale = p.load_json_state("cas", dict, path)
    first["value"] = 1
    p.save_json_state("cas", first, path)
    stale["value"] = 2
    with pytest.raises(p.PersistenceConflictError):
        p.save_json_state("cas", stale, path)
    assert p.load_json_state("cas", dict, path)["value"] == 1


def test_replace_rollback_preserves_history_and_other_namespaces(pg):
    p, conn, _, root = pg
    import psycopg
    path = root / "rows.json"
    old = [{"id": 1}, {"id": 2}]
    p.replace_json_rows("history", old, path)
    p.append_json_row("other", {"id": 9}, path)
    conn.execute("ALTER TABLE app_rows ADD CONSTRAINT fixture_reject CHECK (row_json NOT LIKE '%%FAIL%%')")
    with pytest.raises(psycopg.errors.CheckViolation):
        p.replace_json_rows("history", [{"id": 3}, {"id": "FAIL"}], path)
    assert p.load_json_rows("history", path) == old
    assert p.load_json_rows("other", path) == [{"id": 9}]
    p.replace_json_rows("history", [{"id": 4}, {"id": 5}], path)
    assert p.load_json_rows("history", path) == [{"id": 4}, {"id": 5}]
    assert p.load_json_rows("other", path) == [{"id": 9}]


def test_updater_exception_rolls_back_state(pg):
    p, _, _, root = pg
    path = root / "state.json"
    p.save_json_state("state", {"cash": 10}, path)
    def fail(state):
        state["cash"] = 0
        raise ValueError("synthetic interruption")
    with pytest.raises(ValueError, match="synthetic interruption"):
        p.update_json_state("state", dict, path, fail)
    assert p.load_json_state("state", dict, path) == {"cash": 10}


@pytest.mark.parametrize("invalid", ['{', '[]', '{"cash":NaN}', '{"cash":1e999}', '{"positions":{"AAPL":{},"AAPL":{}}}'])
def test_corrupt_database_state_is_preserved_for_recovery(pg, invalid):
    p, conn, _, root = pg
    p.load_json_state("init", dict, root / "state.json")
    conn.execute("INSERT INTO app_state(namespace,payload_json) VALUES ('corrupt',%s)", (invalid,))
    with pytest.raises(p.PersistenceDataError):
        p.load_json_state("corrupt", dict, root / "state.json")
    assert conn.execute("SELECT payload_json FROM app_state WHERE namespace='corrupt'").fetchone()[0] == invalid


def test_json_and_csv_roundtrip_unicode_quoting_and_order(pg):
    p, _, _, root = pg
    rows = [{"id": 1, "text": 'ação, "PAPER"\nsegunda linha'}, {"id": 2, "text": "fim"}]
    path = root / "events.csv"
    p.replace_table(path, rows, ["id", "text"])
    assert p.read_table(path, ["id", "text"]).to_dict("records") == rows
    p.append_table_row(path, {"id": 3, "text": "último"}, ["id", "text"])
    assert p.read_table(path)["id"].tolist() == [1, 2, 3]
    assert not path.exists()  # PostgreSQL remains the sole source of truth.


def test_backend_interruption_rolls_back_and_next_connection_recovers(pg):
    p, conn, dsn, root = pg
    import psycopg
    path = root / "state.json"
    p.save_json_state("durable", {"cash": 10}, path)
    interrupted = psycopg.connect(dsn)
    try:
        interrupted.execute("UPDATE app_state SET payload_json='{\"cash\":0}' WHERE namespace='durable'")
        pid = interrupted.info.backend_pid
        assert conn.execute("SELECT pg_terminate_backend(%s)", (pid,)).fetchone()[0]
        with pytest.raises(psycopg.Error):
            interrupted.commit()
    finally:
        interrupted.close()
    assert p.load_json_state("durable", dict, path) == {"cash": 10}
    p.update_json_state("durable", dict, path, lambda state: {"cash": state["cash"] + 1})
    assert p.load_json_state("durable", dict, path) == {"cash": 11}


def test_stopped_and_paper_guards_in_actual_application(pg):
    _, _, _, _ = pg
    from core.state_store import load_bot_state, save_bot_state
    from core.broker import probe_broker_status
    from paper_trading_engine import load_paper_state
    state = load_bot_state()
    assert state["bot_status"] in {"PAUSED", "STOPPED"}  # Existing startup contract.
    state["bot_status"] = "STOPPED"
    save_bot_state(state)
    assert load_bot_state()["bot_status"] == "STOPPED"
    assert not state["security"]["real_mode_enabled"]
    broker = probe_broker_status(state["security"])
    assert broker["provider"] == broker["effective_mode"] == "paper"
    assert not broker["can_submit_orders"] and not broker["execution_enabled"]
    assert load_paper_state()["positions"] == {}


def test_synthetic_dump_restore_preserves_exact_payloads_and_sequence(pg):
    p, conn, dsn, root = pg
    import psycopg
    from psycopg import sql
    from psycopg.conninfo import conninfo_to_dict, make_conninfo
    p.save_json_state("fixture", {"mode": "paper", "cash": 10}, root / "state.json")
    p.replace_json_rows("history", [{"id": 1}, {"id": 2}], root / "rows.json")
    database = conninfo_to_dict(dsn)["dbname"]
    restored_name = f"atlas_infra12_restore_{secrets.token_hex(8)}"
    container = os.environ["ATLAS_INFRA12_CONTAINER"]
    command = ["docker", "--host=unix:///var/run/docker.sock", "exec", "-i", container]
    dumped = subprocess.run(command + ["pg_dump", "-U", "atlas_fixture", "-d", database, "-Fc"],
                            capture_output=True, check=True, timeout=30).stdout
    conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(restored_name)))
    subprocess.run(command + ["pg_restore", "-U", "atlas_fixture", "-d", restored_name,
                              "--single-transaction", "--exit-on-error"],
                   input=dumped, capture_output=True, check=True, timeout=30)
    with psycopg.connect(make_conninfo(dsn, dbname=restored_name), autocommit=True) as restored:
        for table, ordering in (("app_state", "namespace"), ("app_rows", "seq")):
            query = sql.SQL("SELECT * FROM {} ORDER BY {}").format(sql.Identifier(table), sql.Identifier(ordering))
            assert conn.execute(query).fetchall() == restored.execute(query).fetchall()
        maximum = restored.execute("SELECT max(seq) FROM app_rows").fetchone()[0]
        assert restored.execute("INSERT INTO app_rows(namespace,row_json) VALUES ('next','{}') RETURNING seq").fetchone()[0] > maximum
