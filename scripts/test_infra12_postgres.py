"""Run INFRA-1.2 integration tests in a disposable, local Docker PostgreSQL.

No DSN, credentials, remote Docker endpoint or existing database is accepted.
Only this invocation's uniquely named container (including its anonymous volumes)
is removed. The server binds exclusively to loopback; all records are synthetic.
Requires Docker and the project's development dependencies already installed.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
DOCKER = ["docker", "--host=unix:///var/run/docker.sock"]


def _loopback_only(event, arguments):
    """PG tests may open Python TCP sockets only on 127.0.0.1.

    libpq DSNs and Docker endpoints are separately constrained by the runner.
    This hook is installed only after the owned local server is ready.
    """
    if event == 'socket.getaddrinfo':
        if arguments[0] not in {'127.0.0.1', None}:
            raise RuntimeError('Local PostgreSQL tests: non-loopback resolution blocked')
    elif event in {'socket.connect', 'socket.bind', 'socket.sendto'}:
        address = arguments[-1]
        if isinstance(address, tuple) and address[0] != '127.0.0.1':
            raise RuntimeError('Local PostgreSQL tests: non-loopback TCP blocked')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", choices=("18", "16"), default="18")
    parser.add_argument("--alpine", action="store_true", help="Use the smaller official Alpine image")
    parser.add_argument("--junitxml", type=Path)
    parser.add_argument('--infra124-benchmark-only', action='store_true')
    parser.add_argument('--infra124-safety-only', action='store_true')
    parser.add_argument('--measurement-label', choices=('before','after'), default='after')
    args = parser.parse_args()
    image = f"postgres:{args.version}{'-alpine' if args.alpine else ''}"
    if (ROOT / ".env").exists():
        raise SystemExit("Move .env before isolated integration tests.")
    # Never inherit DATABASE_URL, broker, mail, market or Docker endpoint settings.
    keep = {key: os.environ[key] for key in (
        "PATH", "HOME", "LANG", "TMPDIR", "SYSTEMROOT", "DOCKER_CONFIG",
        "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy",
        "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE",
    ) if key in os.environ}
    os.environ.clear()
    os.environ.update(keep)
    os.environ.update(APP_ENV="development", AUTH_REQUIRED="true",
                      BROKER_MODE="paper", BROKER_PROVIDER="paper",
                      PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
                      ATLAS_INFRA124_MEASUREMENT=args.measurement_label)
    name = f"atlas-infra12-{secrets.token_hex(8)}"
    password = secrets.token_urlsafe(32)
    container = None

    def docker(*arguments, **kwargs):
        try:
            return subprocess.run(DOCKER + list(arguments), check=True,
                                  capture_output=True, timeout=60, **kwargs)
        except subprocess.CalledProcessError as exc:
            print(exc.stderr.decode(errors="replace"), file=sys.stderr)
            raise

    try:
        docker("info", "--format", "{{.ServerVersion}}")
        # Pulling an image is a local download, never provisioning infrastructure.
        image_info = json.loads(docker("image", "inspect", image).stdout)[0]
        mount = "/var/lib/postgresql" if args.version == "18" else "/var/lib/postgresql/data"
        with socket.socket() as port_probe:
            port_probe.bind(('127.0.0.1', 0))
            published_port = port_probe.getsockname()[1]
        container = docker("create", "--name", name,
                           "--label", "atlas.scope=infra12-local-synthetic",
                           "--memory", "512m", "--cpus", "1",
                           "--publish", f"127.0.0.1:{published_port}:5432", "--volume", mount,
                           "--env", "POSTGRES_PASSWORD", "--env", "POSTGRES_USER=atlas_fixture",
                           image,
                           env={**os.environ, "POSTGRES_PASSWORD": password}).stdout.decode().strip()
        # Record the container before start, so even a startup failure is cleaned up.
        docker("start", container)
        port_info = json.loads(docker("inspect", container).stdout)[0]["NetworkSettings"]["Ports"]["5432/tcp"]
        if len(port_info) != 1 or port_info[0]["HostIp"] != "127.0.0.1":
            raise RuntimeError("Disposable PostgreSQL must bind loopback exclusively")
        port = port_info[0]["HostPort"]
        import psycopg
        dsn = f"host=127.0.0.1 port={port} user=atlas_fixture dbname=postgres connect_timeout=3"
        os.environ["PGPASSWORD"] = password
        deadline = time.monotonic() + 45
        while True:
            try:
                with psycopg.connect(dsn, autocommit=True) as conn:
                    version = conn.execute("SHOW server_version").fetchone()[0]
                break
            except psycopg.OperationalError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.2)
        os.environ.update(ATLAS_INFRA12_LOCAL_DSN=dsn,
                          ATLAS_INFRA12_CONTAINER=container,
                          ATLAS_INFRA12_LOCAL_RUN="synthetic-disposable-only")
        print(json.dumps({"server_version": version, "image": image_info["RepoDigests"],
                          "scope": "LOCAL_SYNTHETIC_ONLY", "broker_mode": "paper",
                          "broker_provider": "paper", "listener": "127.0.0.1"}), flush=True)
        sys.path.insert(0, str(ROOT))
        os.chdir(ROOT)
        with tempfile.TemporaryDirectory(prefix="atlas-infra12-storage-") as directory:
            os.environ["ROBO_STORAGE_DIR"] = directory
            sys.addaudithook(_loopback_only)
            import pytest
            options = ["-q", "tests/test_infra12_postgres.py", "tests/test_infra12_retention_postgres.py", "tests/test_infra121_postgres.py", "tests/test_infra122_postgres.py", "tests/test_infra123_postgres.py", "tests/test_infra124_postgres.py"]
            if args.infra124_benchmark_only:
                options = ['-q', 'tests/test_infra124_postgres.py', '-k', 'test_protocol_load']
            if args.infra124_safety_only:
                options = ['-q', 'tests/test_infra124_postgres.py', '-k', 'not test_protocol_load']
            if args.junitxml:
                options += [f"--junitxml={args.junitxml.resolve()}"]
            return pytest.main(options)
    finally:
        if container:
            docker("rm", "--force", "--volumes", container)
            print("Disposable local PostgreSQL container and own volumes removed.", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
