"""INFRA-1.2.7 on the existing owned, loopback-only PostgreSQL runner."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    from scripts import test_infra12_postgres as owned
    import pytest
    focused = '--infra127-only' in sys.argv
    if focused:
        sys.argv.remove('--infra127-only')
    original = pytest.main
    def run(options):
        if focused:
            options = [v for v in options if not v.startswith('tests/')]
            options += ['tests/test_infra12_retention_postgres.py']
        return original([*options, 'tests/test_infra126_postgres.py', 'tests/test_infra127_postgres.py'])
    pytest.main = run
    try:
        return owned.main()
    finally:
        pytest.main = original


if __name__ == '__main__':
    raise SystemExit(main())
