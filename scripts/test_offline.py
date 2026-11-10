"""Run pytest with temporary storage, no inherited secrets and no network.

This runner is for local tests only; it never starts the worker loop.
"""
from pathlib import Path
import os
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def main():
    if (ROOT / ".env").exists():
        raise SystemExit("Remove/move the local .env before isolated tests.")
    # Preserve only runtime necessities. Never inherit DB/provider/mail credentials.
    keep = {key: os.environ[key] for key in ("PATH", "HOME", "LANG", "TMPDIR", "SYSTEMROOT") if key in os.environ}
    os.environ.clear()
    os.environ.update(keep)
    os.environ.update(APP_ENV="development", AUTH_REQUIRED="true",
                      BROKER_MODE="paper", BROKER_PROVIDER="paper",
                      PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    blocked = []

    def deny_external(event, args):
        if event in {"socket.connect", "socket.getaddrinfo", "socket.sendto",
                     "subprocess.Popen", "os.system", "os.exec", "os.posix_spawn"}:
            blocked.append(event)
            raise RuntimeError("Offline tests: network/process execution blocked")

    sys.addaudithook(deny_external)
    sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)
    with tempfile.TemporaryDirectory(prefix="atlas-offline-") as directory:
        os.environ["ROBO_STORAGE_DIR"] = directory
        import pytest
        result = pytest.main(sys.argv[1:] or ["-q"])
    print(f"Offline guard: {len(blocked)} network/process attempts blocked.")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
