"""Create, test, dump/restore and destroy a private disposable LOCAL cluster.

Requires preinstalled PostgreSQL binaries. No remote DSN is accepted. No .env.
Never starts web/worker. Unix socket or authenticated loopback only; synthetic data.
"""
import argparse
import hashlib
import json
import os
import secrets
import socket
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pg-bin", type=Path, required=True)
    parser.add_argument("--transport", choices=("unix", "loopback"), default="unix")
    args = parser.parse_args()
    pg_bin = args.pg_bin.resolve()
    for name in ("initdb", "pg_ctl", "pg_dump", "pg_restore", "postgres"):
        if not (pg_bin / name).is_file():
            raise SystemExit(f"Missing PostgreSQL binary: {name}")
    if (ROOT / ".env").exists():
        raise SystemExit("Move .env before local integration tests.")
    keep = {key: os.environ[key] for key in ("PATH", "HOME", "LANG", "TMPDIR", "SYSTEMROOT") if key in os.environ}
    os.environ.clear()
    os.environ.update(keep)
    os.environ.update(APP_ENV="development", AUTH_REQUIRED="true", BROKER_MODE="paper", BROKER_PROVIDER="paper")
    sys.path.insert(0, str(ROOT))
    results = []

    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        results.append(name)

    with tempfile.TemporaryDirectory(prefix="atlas-pg-") as directory:
        root = Path(directory)
        data, sock = root / "data", root / "socket"
        sock.mkdir(mode=0o700)
        os.environ["ROBO_STORAGE_DIR"] = str(root / "storage")
        log = root / "server.log"
        started = False
        password_file = root / "password"
        password_file.write_text(secrets.token_urlsafe(32))
        password_file.chmod(0o600)
        os.environ["PGPASSWORD"] = password_file.read_text()
        port = 55433
        host = str(sock)
        options = f"-k {sock} -p {port} -c listen_addresses='' -c unix_socket_permissions=0700"
        if args.transport == "loopback":
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", 0))
                port = probe.getsockname()[1]
            host = "127.0.0.1"
            options = f"-p {port} -c listen_addresses='127.0.0.1' -c unix_socket_directories=''"

        def run(name, *arguments):
            return subprocess.run([str(pg_bin / name), *map(str, arguments)],
                                  check=True, capture_output=True, text=True, timeout=60)

        try:
            run("initdb", "-D", data, "--no-locale", "--encoding=UTF8",
                "--auth-local=scram-sha-256", "--auth-host=scram-sha-256",
                "--pwfile", password_file, "--username=atlas_fixture")
            run("pg_ctl", "-D", data, "-l", log, "-o",
                options, "-w", "start")
            started = True
            dsn = f"host={host} port={port} dbname=postgres user=atlas_fixture"
            os.environ["DATABASE_URL"] = dsn
            import psycopg
            from core import persistence as p
            from scripts.audit_database_readonly import inventory
            missing_local = root / "absent.json"

            with psycopg.connect(dsn, autocommit=True) as conn:
                check("private_listener_only", conn.execute("SHOW listen_addresses").fetchone()[0]
                      == ("127.0.0.1" if args.transport == "loopback" else ""))
                empty_audit = inventory(conn)
                check("read_only_audit_does_not_create_schema", all(not t["present"] for t in empty_audit["tables"].values()))
                schema = (ROOT / "scripts/sql/new_history_schema.sql").read_text()
                conn.execute(schema)
                check("new_history_schema_starts_empty", all(
                    conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0
                    for table in ("app_state", "app_rows")))
                conn.execute("INSERT INTO app_state(namespace,payload_json) VALUES ('preserve_fixture','{}')")
                try:
                    conn.execute(schema)
                except psycopg.errors.RaiseException:
                    conn.execute("ROLLBACK")
                else:
                    raise AssertionError("New history initialization must refuse existing schema")
                check("new_history_refuses_existing_schema_without_data_loss",
                      conn.execute("SELECT payload_json FROM app_state WHERE namespace='preserve_fixture'").fetchone()[0] == "{}")
                p.save_json_state("fixture", {"mode": "paper"}, missing_local)
                check("json_state_roundtrip", p.load_json_state("fixture", dict, missing_local) == {"mode": "paper"})
                p.replace_json_rows("history", [{"id": 1}, {"id": 2}], missing_local)
                p.append_json_row("other", {"id": 9}, missing_local)
                try:
                    p.replace_json_rows("history", [{"id": 3}, {"bad": object()}], missing_local)
                except TypeError:
                    pass
                else:
                    raise AssertionError("Expected serialization failure")
                check("serialization_failure_rolls_back_delete_and_insert",
                      p.load_json_rows("history", missing_local) == [{"id": 1}, {"id": 2}])
                conn.execute("ALTER TABLE app_rows ADD CONSTRAINT fixture_reject CHECK (row_json NOT LIKE '%%FAIL%%')")
                try:
                    p.replace_json_rows("history", [{"id": 3}, {"id": "FAIL"}], missing_local)
                except psycopg.errors.CheckViolation:
                    pass
                else:
                    raise AssertionError("Expected SQL constraint failure")
                check("sql_failure_rolls_back_delete_and_insert",
                      p.load_json_rows("history", missing_local) == [{"id": 1}, {"id": 2}])
                check("other_namespace_preserved", p.load_json_rows("other", missing_local) == [{"id": 9}])
                p.replace_json_rows("history", [{"id": 4}, {"id": 5}], missing_local)
                check("successful_replace_commits", p.load_json_rows("history", missing_local) == [{"id": 4}, {"id": 5}])
                check("postgres_limit_one", p.load_json_rows("history", missing_local, 1) == [{"id": 5}])
                check("postgres_limit_zero", p.load_json_rows("history", missing_local, 0) == [])

                conn.execute("INSERT INTO app_state(namespace,payload_json) VALUES ('corrupt','{')")
                try:
                    p.load_json_state("corrupt", dict, missing_local)
                except p.PersistenceDataError:
                    pass
                else:
                    raise AssertionError("Corrupt DB state accepted")
                check("corrupt_state_preserved", conn.execute("SELECT payload_json FROM app_state WHERE namespace='corrupt'").fetchone()[0] == "{")
                audit = inventory(conn)
                check("read_only_inventory_detects_corruption", audit["read_only"] and audit["tables"]["app_state"]["invalid_payloads"] == 1)
                conn.execute("INSERT INTO app_rows(namespace,row_json) VALUES ('corrupt_rows','[]')")
                try:
                    p.load_json_rows("corrupt_rows", missing_local)
                except p.PersistenceDataError:
                    pass
                else:
                    raise AssertionError("Corrupt DB row silently ignored")
                check("corrupt_row_preserved", conn.execute("SELECT row_json FROM app_rows WHERE namespace='corrupt_rows'").fetchone()[0] == "[]")
                conn.execute("DELETE FROM app_rows WHERE namespace='corrupt_rows'")
                # Exercise real application loaders against this disposable cluster only.
                from core.auth.users_store import create_user, get_user
                from core.auth.security import verify_password
                from core.state_store import load_bot_state, save_bot_state
                from paper_trading_engine import load_paper_state
                create_user("fixture-admin", "synthetic-test-password", "admin")
                check("postgres_auth_roundtrip", verify_password("synthetic-test-password", get_user("fixture-admin")["password_hash"]))
                state = load_bot_state()
                state["bot_status"] = "STOPPED"
                save_bot_state(state)
                check("postgres_bot_state_roundtrip", load_bot_state()["bot_status"] == "STOPPED")
                check("postgres_paper_guards", not state["security"]["real_mode_enabled"]
                      and not state["broker"]["can_submit_orders"])
                check("postgres_paper_state_initialized", load_paper_state()["positions"] == {})
                # Delete only our known synthetic invalid fixture before backup validation.
                conn.execute("DELETE FROM app_state WHERE namespace='corrupt'")
                dump = root / "fixture.dump"
                run("pg_dump", "--dbname", dsn, "--format=custom", "--file", dump)
                digest = hashlib.sha256(dump.read_bytes()).hexdigest()
                conn.execute("CREATE DATABASE restored")
                restored_dsn = f"host={host} port={port} dbname=restored user=atlas_fixture"
                run("pg_restore", "--dbname", restored_dsn, "--single-transaction", "--exit-on-error", dump)
                with psycopg.connect(restored_dsn, autocommit=True) as restored:
                    for table, fields in (("app_state", "namespace,payload_json,updated_at"),
                                          ("app_rows", "seq,namespace,row_json,created_at")):
                        order = "namespace" if table == "app_state" else "seq"
                        query = f"SELECT {fields} FROM {table} ORDER BY {order}"
                        check(f"dump_restore_exact_{table}", conn.execute(query).fetchall() == restored.execute(query).fetchall())
                    maximum = restored.execute("SELECT max(seq) FROM app_rows").fetchone()[0]
                    seq = restored.execute("INSERT INTO app_rows(namespace,row_json) VALUES ('after_restore','{}') RETURNING seq").fetchone()[0]
                    check("restored_sequence_advances", seq > maximum)
                version = conn.execute("SHOW server_version").fetchone()[0]
        except Exception:
            if log.exists():
                print(log.read_text(), file=sys.stderr)
            raise
        finally:
            if started:
                run("pg_ctl", "-D", data, "-m", "fast", "-w", "stop")
        print(json.dumps({"passed": len(results), "checks": results, "server_version": version,
                          "fixture_dump_sha256": digest, "cluster_stopped": True,
                          "scope": "LOCAL_SYNTHETIC_ONLY_NOT_OLD_DATABASE_RECOVERY"}, indent=2))


if __name__ == "__main__":
    main()
