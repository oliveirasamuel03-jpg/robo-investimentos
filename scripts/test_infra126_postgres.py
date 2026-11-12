"""Extend the existing owned-loopback runner without changing prior tests.

--contract-only selects the new study proofs. Otherwise all preceding 139
PostgreSQL cases run as well. No external DSN or container is accepted.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    from scripts import test_infra12_postgres as owned
    import pytest

    contract_only = '--contract-only' in sys.argv
    if contract_only:
        sys.argv.remove('--contract-only')
    previous = pytest.main

    def run(options):
        if contract_only:
            options = [value for value in options if not value.startswith('tests/')]
        return previous([*options, 'tests/test_infra126_postgres.py'])

    pytest.main = run
    try:
        return owned.main()
    finally:
        pytest.main = previous


if __name__ == '__main__':
    raise SystemExit(main())
