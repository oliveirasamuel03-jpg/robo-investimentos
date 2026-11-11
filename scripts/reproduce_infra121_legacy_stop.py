"""Reproduce the INFRA-1.2 STOP contention using only an archived local baseline.

Starts and removes its own local PostgreSQL 18 Docker container. No external DSN
or inherited application credential is accepted. Uses an Event-held fictitious
market request and a real advisory-lock timeout, without sleep-based race timing.
"""
from concurrent.futures import ThreadPoolExecutor
import importlib
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import tarfile
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = "5d7165217e42ff5121b5444f956d45a877118734"
DOCKER = ["docker", "--host=unix:///var/run/docker.sock"]


def main():
    if (ROOT / ".env").exists():
        raise SystemExit("Move .env before isolated reproduction")
    keep = {key: os.environ[key] for key in (
        "PATH", "HOME", "LANG", "TMPDIR", "SYSTEMROOT", "DOCKER_CONFIG",
        "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy",
        "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE") if key in os.environ}
    os.environ.clear()
    os.environ.update(keep)
    os.environ.update(APP_ENV="development", AUTH_REQUIRED="true",
                      BROKER_MODE="paper", BROKER_PROVIDER="paper")
    password = secrets.token_urlsafe(32)
    name = f"atlas-infra121-repro-{secrets.token_hex(8)}"
    container = None

    def docker(*args, **kwargs):
        return subprocess.run(DOCKER + list(args), check=True, capture_output=True,
                              timeout=60, **kwargs)

    with tempfile.TemporaryDirectory(prefix="atlas-infra121-legacy-") as directory:
        temporary = Path(directory)
        archive, legacy = temporary / "source.tar", temporary / "source"
        legacy.mkdir()
        subprocess.run(["git", "archive", BASE, "-o", str(archive)], cwd=ROOT, check=True)
        with tarfile.open(archive) as source:
            source.extractall(legacy, filter="data")
        try:
            container = docker("create", "--name", name,
                               "--label", "atlas.scope=infra121-legacy-local",
                               "--memory", "512m", "--cpus", "1",
                               "--publish", "127.0.0.1::5432", "--tmpfs", "/var/lib/postgresql",
                               "--env", "POSTGRES_PASSWORD", "--env", "POSTGRES_USER=atlas_fixture",
                               "postgres:18", env={**os.environ, "POSTGRES_PASSWORD": password}).stdout.decode().strip()
            docker("start", container)
            port_info = json.loads(docker("inspect", container).stdout)[0]["NetworkSettings"]["Ports"]["5432/tcp"]
            assert len(port_info) == 1 and port_info[0]["HostIp"] == "127.0.0.1"
            os.environ["DATABASE_URL"] = (f"host=127.0.0.1 port={port_info[0]['HostPort']} "
                                          "user=atlas_fixture dbname=postgres connect_timeout=3")
            os.environ["PGPASSWORD"] = password
            os.environ["ROBO_STORAGE_DIR"] = str(temporary / "storage")
            import psycopg
            deadline = time.monotonic() + 45
            while True:
                try:
                    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as conn:
                        version = conn.execute("SHOW server_version").fetchone()[0]
                    break
                except psycopg.OperationalError:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(0.2)  # Server readiness only, never the race synchronization.
            sys.path.insert(0, str(legacy))
            state_store = importlib.import_module("core.state_store")
            paper = importlib.import_module("paper_trading_engine")
            trader = importlib.import_module("engines.trader_engine")
            assert Path(state_store.__file__).resolve().is_relative_to(legacy)
            state = state_store.load_bot_state()
            state["bot_status"] = "RUNNING"
            state_store.save_bot_state(state)
            feed_entered, release_feed = threading.Event(), threading.Event()

            class SyntheticAbort(Exception):
                pass

            def held_feed(*args, **kwargs):
                feed_entered.set()
                assert release_feed.wait(15)
                raise SyntheticAbort("Fictitious feed released; roll back the legacy cycle")

            paper.load_market_data_result = held_feed
            trader.refresh_macro_alert_state = lambda: {}

            def legacy_stop():
                current = state_store.load_bot_state()
                current["bot_status"] = "STOPPED"
                state_store.save_bot_state(current)

            with ThreadPoolExecutor(max_workers=2) as pool:
                worker = pool.submit(trader.run_trader_cycle)
                assert feed_entered.wait(10), "Legacy worker did not reach fictitious feed"
                stop = pool.submit(legacy_stop)
                try:
                    try:
                        stop.result(timeout=8)
                    except psycopg.errors.LockNotAvailable:
                        stop_timed_out = True
                    else:
                        raise AssertionError("Legacy STOP unexpectedly succeeded under held feed")
                    assert not release_feed.is_set() and not worker.done()
                finally:
                    release_feed.set()
                try:
                    worker.result(timeout=10)
                except SyntheticAbort:
                    pass
                else:
                    raise AssertionError("Synthetic worker must roll back")
            final = state_store.load_bot_state()
            assert final["bot_status"] == "RUNNING"
            assert paper.read_paper_trades() == []
            result = {
                "base_commit": BASE, "scope": "LOCAL_SYNTHETIC_ONLY",
                "server_version": version, "listener": "127.0.0.1",
                "broker_mode": "paper", "broker_provider": "paper",
                "legacy_modules_loaded_from_git_archive": True,
                "worker_held_at_market_feed_by_event": True,
                "stop_failed_with_real_postgres_lock_timeout": stop_timed_out,
                "stop_was_not_persisted": final["bot_status"] == "RUNNING",
                "worker_rolled_back_after_synthetic_feed_failure": True,
                "synthetic_trades_committed": 0,
                "synchronization": "threading.Event; STOP result awaited while feed release Event remained unset",
            }
            target = ROOT / "docs/retomada/evidencias/infra121-legacy-stop.json"
            target.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result, indent=2), flush=True)
        finally:
            if container:
                docker("rm", "--force", "--volumes", container)
                print("Own disposable legacy reproduction container removed.", flush=True)


if __name__ == "__main__":
    main()
