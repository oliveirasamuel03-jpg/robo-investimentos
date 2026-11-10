"""Temporary authenticated web demo: loopback only, no DB/provider credentials.

All data is discarded on exit. This is not a migration or production launcher.
"""
from pathlib import Path
import argparse
import getpass
import os
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--no-bootstrap", action="store_true", help="Only show unconfigured login (smoke test)")
    args = parser.parse_args()
    if (ROOT / ".env").exists():
        raise SystemExit("Move the .env before running this isolated demo.")
    password = ""
    if not args.no_bootstrap:
        password = getpass.getpass("Senha temporaria para o usuario local-demo (minimo 12 caracteres): ")
        if len(password) < 12:
            raise SystemExit("Senha insuficiente.")
    keep = {key: os.environ[key] for key in ("PATH", "HOME", "LANG", "TMPDIR", "SYSTEMROOT") if key in os.environ}
    os.environ.clear()
    os.environ.update(keep)
    os.environ.update(APP_ENV="development", AUTH_REQUIRED="true", BROKER_MODE="paper", BROKER_PROVIDER="paper")
    if password:
        os.environ.update(ADMIN_USERNAME="local-demo", ADMIN_PASSWORD=password)

    def offline(event, values):
        if event == "socket.getaddrinfo" and values[0] not in {"localhost", "127.0.0.1", "::1"}:
            raise RuntimeError("Demo offline: DNS externo bloqueado")
        if event == "socket.connect":
            address = values[1]
            if not isinstance(address, tuple) or address[0] not in {"127.0.0.1", "::1"}:
                raise RuntimeError("Demo offline: conexao externa bloqueada")
        if event in {"socket.sendto", "subprocess.Popen", "os.system", "os.exec", "os.posix_spawn"}:
            raise RuntimeError("Demo offline: rede/processo externo bloqueado")

    sys.addaudithook(offline)
    sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)
    with tempfile.TemporaryDirectory(prefix="atlas-demo-") as directory:
        os.environ["ROBO_STORAGE_DIR"] = directory
        print("Demo PAPER offline; dados temporarios, descartados ao encerrar.")
        from streamlit.web import cli
        sys.argv = ["streamlit", "run", "app.py", "--server.address=127.0.0.1",
                    f"--server.port={args.port}", "--server.headless=true",
                    "--browser.gatherUsageStats=false", "--server.fileWatcherType=none"]
        return cli.main()


if __name__ == "__main__":
    main()
