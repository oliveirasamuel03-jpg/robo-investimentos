import json

import pytest

from tests.conftest import load_module


@pytest.mark.parametrize("content", ['{', '[]', 'null', '"not a state"', ''])
def test_existing_invalid_state_is_not_reinitialized(isolated_storage, content):
    persistence = load_module("core.persistence")
    path = isolated_storage / "invalid.json"
    path.parent.mkdir(parents=True)
    path.write_text(content)
    with pytest.raises(persistence.PersistenceDataError):
        persistence.load_json_state("test", lambda: {"new": True}, path)
    assert path.read_text() == content


@pytest.mark.parametrize("content", ['{', '{}', '[{}, 1]', 'null', ''])
def test_append_refuses_corrupted_history(isolated_storage, content):
    persistence = load_module("core.persistence")
    path = isolated_storage / "rows.json"
    path.parent.mkdir(parents=True)
    path.write_text(content)
    with pytest.raises(persistence.PersistenceDataError):
        persistence.append_json_row("test", {"new": True}, path)
    assert path.read_text() == content


def test_absent_files_still_have_defaults(isolated_storage):
    persistence = load_module("core.persistence")
    path = isolated_storage / "absent.json"
    assert persistence.load_json_state("test", lambda: {"new": True}, path) == {"new": True}
    assert persistence.load_json_rows("test", path) == []
    assert not path.exists()


@pytest.mark.parametrize("failure", ["fsync", "replace"])
def test_atomic_write_failure_preserves_previous_file(isolated_storage, monkeypatch, failure):
    persistence = load_module("core.persistence")
    path = isolated_storage / "state.json"
    persistence.save_json_state("test", {"old": 1}, path)
    original = path.read_bytes()
    def fail(*args):
        raise OSError("injected disk failure")
    monkeypatch.setattr(persistence.os, failure, fail)
    with pytest.raises(OSError):
        persistence.save_json_state("test", {"new": 2}, path)
    assert path.read_bytes() == original
    assert list(path.parent.glob(".*.tmp")) == []


def test_serialization_failure_preserves_previous_file(isolated_storage):
    persistence = load_module("core.persistence")
    path = isolated_storage / "state.json"
    persistence.save_json_state("test", {"old": 1}, path)
    with pytest.raises(TypeError):
        persistence.save_json_state("test", {"bad": object()}, path)
    assert json.loads(path.read_text()) == {"old": 1}


def test_atomic_csv_write_failure_preserves_previous_file(isolated_storage, monkeypatch):
    persistence = load_module("core.persistence")
    path = isolated_storage / "rows.csv"
    persistence.replace_table(path, [{"id": 1}], ["id"])
    original = path.read_bytes()
    def fail(*args):
        raise OSError("injected replace failure")
    monkeypatch.setattr(persistence.os, "replace", fail)
    with pytest.raises(OSError):
        persistence.append_table_row(path, {"id": 2}, ["id"])
    assert path.read_bytes() == original


def test_zero_limit_does_not_initialize_database(isolated_storage, monkeypatch):
    persistence = load_module("core.persistence")
    monkeypatch.setenv("DATABASE_URL", "unused-fixture")
    def fail():
        raise AssertionError("Database must not be touched for limit=0")
    monkeypatch.setattr(persistence, "_connect", fail)
    assert persistence.load_json_rows("test", isolated_storage / "rows.json", limit=0) == []


def test_local_limits_and_order(isolated_storage):
    persistence = load_module("core.persistence")
    path = isolated_storage / "rows.json"
    persistence.replace_json_rows("test", [{"id": 1}, {"id": 2}], path)
    assert persistence.load_json_rows("test", path, 0) == []
    assert persistence.load_json_rows("test", path, 1) == [{"id": 2}]
    with pytest.raises(ValueError):
        persistence.load_json_rows("test", path, -1)


def test_corrupt_auth_store_blocks_bootstrap(isolated_storage, monkeypatch):
    store = load_module("core.auth.users_store")
    persistence = load_module("core.persistence")
    store.AUTH_USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    store.AUTH_USERS_FILE.write_text('{"users":')
    monkeypatch.setenv("ADMIN_USERNAME", "fixture-admin")
    monkeypatch.setenv("ADMIN_PASSWORD", "fixture-password")
    # Use RuntimeError because module reloads intentionally isolate persistence classes.
    with pytest.raises(RuntimeError, match="Invalid JSON"):
        store.ensure_bootstrap_admin()
    assert store.AUTH_USERS_FILE.read_text() == '{"users":'
