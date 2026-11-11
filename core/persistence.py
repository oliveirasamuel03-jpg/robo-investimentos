from __future__ import annotations

import json
import math
import hashlib
import threading
import fcntl
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import os
import tempfile
from pathlib import Path
from typing import Any, Callable

import pandas as pd

try:
    import psycopg
except Exception:
    psycopg = None


_DB_SCHEMA_READY = False
_DB_SCHEMA_URL = None
_ACTIVE_CONNECTION = ContextVar("atlas_persistence_connection", default=None)
_LOCAL_LOCKS = {}
_LOCAL_LOCKS_GUARD = threading.Lock()
_LOCAL_HELD = threading.local()


class PersistenceDataError(RuntimeError):
    """Existing data is invalid; do not replace it with an empty/default state."""


class PersistenceConflictError(RuntimeError):
    """The persisted state changed since it was read. Reload; never blindly retry."""


class StateSnapshot(dict):
    """A dict with an out-of-band content token (never serialized into user data)."""

    def __init__(self, payload, source, expected):
        super().__init__(payload)
        self.source = source
        self.expected = expected


def inherit_state_snapshot(payload, original):
    if isinstance(original, StateSnapshot):
        return StateSnapshot(payload, original.source, original.expected)
    return payload


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON object key")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"Non-finite JSON constant: {value}")


def _assert_finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Non-finite JSON number")
    if isinstance(value, dict):
        for item in value.values():
            _assert_finite(item)
    elif isinstance(value, list):
        for item in value:
            _assert_finite(item)


def _encode(payload):
    return json.dumps(payload, ensure_ascii=False, allow_nan=False)


@contextmanager
def _local_lock(path):
    # Stable sidecar inode: locking the data file would break after os.replace.
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LOCAL_LOCKS_GUARD:
        lock = _LOCAL_LOCKS.setdefault(str(path), threading.RLock())
    with lock:
        held = getattr(_LOCAL_HELD, "paths", set())
        if str(path) in held:
            yield
            return
        with path.with_name(f".{path.name}.lock").open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            _LOCAL_HELD.paths = held | {str(path)}
            try:
                yield
            finally:
                _LOCAL_HELD.paths = held
                fcntl.flock(handle, fcntl.LOCK_UN)


def _lock_namespace(conn, namespace):
    key = int.from_bytes(hashlib.sha256(namespace.encode()).digest()[:8], "big", signed=True)
    conn.execute("SELECT pg_advisory_xact_lock(%s)", (key,))


@contextmanager
def persistence_transaction(*namespaces):
    """One PostgreSQL unit of work. Local files retain per-file atomicity only.

    A single transaction lock precedes namespace locks, so nested writes to logs,
    auth, state and history cannot invert lock order. This conservatively serializes
    writers until a narrower aggregate contract can be introduced and homologated.
    """
    if not database_enabled():
        yield
        return
    active = _ACTIVE_CONNECTION.get()
    if active is not None:
        with active.transaction():  # savepoint: caught subunit failures cannot publish partial writes
            for namespace in sorted(set(namespaces)):
                _lock_namespace(active, namespace)
            yield
        return
    _ensure_db_schema()
    with _connect() as conn, conn.transaction():
        conn.execute("SET LOCAL lock_timeout = '5s'")
        _lock_namespace(conn, "atlas:persistence-unit")
        for namespace in sorted(set(namespaces)):
            _lock_namespace(conn, namespace)
        token = _ACTIVE_CONNECTION.set(conn)
        try:
            yield
        finally:
            _ACTIVE_CONNECTION.reset(token)


def transactional_persistence(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with persistence_transaction():
            return function(*args, **kwargs)
    return wrapped


@contextmanager
def _reader():
    active = _ACTIVE_CONNECTION.get()
    if active is not None:
        yield active
    else:
        with _connect() as conn:
            yield conn


def _decode_json(text: str, expected_type: type, source: str):
    try:
        data = json.loads(text, parse_constant=_reject_constant, object_pairs_hook=_unique_object)
        _assert_finite(data)
    except (ValueError, TypeError) as exc:
        raise PersistenceDataError(f"Invalid JSON in {source}; recovery required.") from exc
    if not isinstance(data, expected_type):
        raise PersistenceDataError(f"Invalid data structure in {source}; recovery required.")
    if expected_type is list and any(not isinstance(row, dict) for row in data):
        raise PersistenceDataError(f"Invalid row in {source}; recovery required.")
    return data


def _atomic_write_text(path: Path, text: str) -> None:
    """Publish a complete local file; pre-rename failure preserves the old file.

    The directory fsync makes the rename durable. If that fsync fails, the new
    file may already be visible: callers must reload before retrying. Per-path
    locks at callers serialize writes; this primitive only prevents torn files.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="",
                                         dir=path.parent, prefix=f".{path.name}.",
                                         suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def get_database_url() -> str:
    return str(os.getenv("DATABASE_URL", "") or "").strip()


def database_enabled() -> bool:
    return bool(get_database_url())


def _require_psycopg() -> None:
    if psycopg is None:
        raise RuntimeError("DATABASE_URL is set, but the 'psycopg[binary]' dependency is not installed.")


def _connect():
    _require_psycopg()
    return psycopg.connect(get_database_url(), autocommit=True)


def _ensure_db_schema() -> None:
    global _DB_SCHEMA_READY, _DB_SCHEMA_URL

    if not database_enabled() or (_DB_SCHEMA_READY and _DB_SCHEMA_URL == get_database_url()):
        return

    with _connect() as conn, conn.transaction():
        _lock_namespace(conn, "atlas:schema-init")
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS app_state (
                    namespace TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS app_rows (
                    seq BIGSERIAL PRIMARY KEY,
                    namespace TEXT NOT NULL,
                    row_json TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_app_rows_namespace_seq
                ON app_rows(namespace, seq)
                """
            )

    _DB_SCHEMA_READY = True
    _DB_SCHEMA_URL = get_database_url()


def _load_local_json_state(path: Path, default_factory: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return default_factory()
    except UnicodeError as exc:
        raise PersistenceDataError("Invalid state file encoding; recovery required.") from exc
    return _decode_json(text, dict, "local state")


def _load_local_json_rows(path: Path) -> list[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    except UnicodeError as exc:
        raise PersistenceDataError("Invalid rows file encoding; recovery required.") from exc
    return _decode_json(text, list, "local rows")


def _state_source(namespace, local_path):
    return (get_database_url(), namespace) if database_enabled() else ("local", str(Path(local_path).resolve()))


def load_json_state(namespace, default_factory, local_path):
    source = _state_source(namespace, local_path)
    if not database_enabled():
        with _local_lock(local_path):
            payload = _load_local_json_state(local_path, default_factory)
            expected = local_path.read_text(encoding="utf-8") if local_path.exists() else None
        return StateSnapshot(payload, source, expected)
    _ensure_db_schema()
    with _reader() as conn:
        row = conn.execute("SELECT payload_json FROM app_state WHERE namespace = %s", (namespace,)).fetchone()
    if row:
        return StateSnapshot(_decode_json(row[0], dict, "database state"), source, row[0])
    # A new DB is a new history. Never import an old local file implicitly.
    with persistence_transaction(namespace):
        conn = _ACTIVE_CONNECTION.get()
        text = _encode(default_factory())
        conn.execute("INSERT INTO app_state(namespace,payload_json) VALUES (%s,%s) ON CONFLICT DO NOTHING",
                     (namespace, text))
        row = conn.execute("SELECT payload_json FROM app_state WHERE namespace = %s", (namespace,)).fetchone()
        return StateSnapshot(_decode_json(row[0], dict, "database state"), source, row[0])


def _check_snapshot(payload, source, current, new, strict):
    if isinstance(payload, StateSnapshot):
        if payload.source != source or payload.expected != current:
            raise PersistenceConflictError("State changed since load; reload and review before retrying.")
    elif strict and current is not None and current != new:
        raise PersistenceConflictError("Refusing unversioned overwrite; load a state snapshot first.")


def save_json_state(namespace, payload, local_path):
    if not isinstance(payload, dict):
        raise PersistenceDataError("State must be a JSON object")
    text = _encode(payload)
    source = _state_source(namespace, local_path)
    if not database_enabled():
        with _local_lock(local_path):
            current = local_path.read_text(encoding="utf-8") if local_path.exists() else None
            if current is not None:
                _decode_json(current, dict, "local state")
            _check_snapshot(payload, source, current, text, strict=False)
            _atomic_write_text(local_path, text)
    else:
        with persistence_transaction(namespace):
            conn = _ACTIVE_CONNECTION.get()
            row = conn.execute("SELECT payload_json FROM app_state WHERE namespace = %s FOR UPDATE", (namespace,)).fetchone()
            current = row[0] if row else None
            if current is not None:
                _decode_json(current, dict, "database state")
            _check_snapshot(payload, source, current, text, strict=True)
            if current is None:
                changed = conn.execute("""INSERT INTO app_state(namespace,payload_json) VALUES (%s,%s)
                                        ON CONFLICT DO NOTHING""", (namespace, text)).rowcount
            else:
                changed = conn.execute("""UPDATE app_state SET payload_json=%s,updated_at=NOW()
                                        WHERE namespace=%s AND payload_json=%s""", (text,namespace,current)).rowcount
            if changed != 1:
                raise PersistenceConflictError("Concurrent state write; reload before retrying.")
    if isinstance(payload, StateSnapshot):
        payload.expected = text


def update_json_state(namespace, default_factory, local_path, updater):
    if not database_enabled():
        with _local_lock(local_path):
            # Do not nest flock through load/save: it would lock a second descriptor.
            current = local_path.read_text(encoding="utf-8") if local_path.exists() else None
            state = _decode_json(current, dict, "local state") if current is not None else default_factory()
            result = updater(state)
            if not isinstance(result, dict):
                raise TypeError("State updater must return a dict")
            text = _encode(result)
            _atomic_write_text(local_path, text)
            return StateSnapshot(result, _state_source(namespace, local_path), text)
    with persistence_transaction(namespace):
        state = load_json_state(namespace, default_factory, local_path)
        result = updater(state)
        if not isinstance(result, dict):
            raise TypeError("State updater must return a dict")
        result = inherit_state_snapshot(result, state)
        save_json_state(namespace, result, local_path)
        return result


def load_json_rows(namespace: str, local_path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    if limit is not None:
        limit = int(limit)
        if limit < 0:
            raise ValueError("limit must be non-negative")
        if limit == 0:
            return []
    if not database_enabled():
        rows = _load_local_json_rows(local_path)
        if limit is None:
            return rows
        return rows[-int(limit) :]

    _ensure_db_schema()
    query = "SELECT row_json FROM app_rows WHERE namespace = %s ORDER BY seq ASC"
    params: tuple[Any, ...] = (namespace,)

    if limit is not None:
        query = """
            SELECT row_json
            FROM (
                SELECT row_json, seq
                FROM app_rows
                WHERE namespace = %s
                ORDER BY seq DESC
                LIMIT %s
            ) recent
            ORDER BY seq ASC
        """
        params = (namespace, int(limit))

    with _reader() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            results = cur.fetchall()

    if results:
        rows: list[dict[str, Any]] = []
        for result in results:
            rows.append(_decode_json(str(result[0]), dict, "database row"))
        return rows

    return []


def append_json_row(namespace, row, local_path):
    if not isinstance(row, dict):
        raise PersistenceDataError("History row must be an object")
    text = _encode(row)
    if not database_enabled():
        with _local_lock(local_path):
            rows = _load_local_json_rows(local_path)
            rows.append(row)
            _atomic_write_text(local_path, _encode(rows))
        return
    with persistence_transaction(namespace):
        _ACTIVE_CONNECTION.get().execute("INSERT INTO app_rows(namespace,row_json) VALUES (%s,%s)", (namespace,text))


def replace_json_rows(namespace, rows, local_path):
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise PersistenceDataError("History must be a list of objects")
    texts = [_encode(row) for row in rows]
    if not database_enabled():
        with _local_lock(local_path):
            if local_path.exists():
                _load_local_json_rows(local_path)
            _atomic_write_text(local_path, _encode(rows))
        return
    with persistence_transaction(namespace):
        conn = _ACTIVE_CONNECTION.get()
        for row in conn.execute("SELECT row_json FROM app_rows WHERE namespace=%s", (namespace,)):
            _decode_json(row[0], dict, "database history before replacement")
        conn.execute("DELETE FROM app_rows WHERE namespace = %s", (namespace,))
        for text in texts:
            conn.execute("INSERT INTO app_rows(namespace,row_json) VALUES (%s,%s)", (namespace,text))


def _table_namespace(file_path: Path | str) -> str:
    return f"table:{Path(file_path).name.lower()}"


def read_table(file_path: Path | str, columns: list[str] | None = None) -> pd.DataFrame:
    path = Path(file_path)

    if not database_enabled():
        try:
            df = pd.read_csv(path)
        except FileNotFoundError:
            df = pd.DataFrame(columns=columns or [])
        except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeError) as exc:
            raise PersistenceDataError("Invalid CSV; recovery required.") from exc
    else:
        rows = load_json_rows(_table_namespace(path), path.with_suffix(path.suffix + ".json"))
        df = pd.DataFrame(rows)

    if columns:
        for column in columns:
            if column not in df.columns:
                df[column] = pd.Series(dtype="object")
        df = df[columns]

    return df


def append_table_row(file_path: Path | str, row: dict[str, Any], columns: list[str] | None = None) -> None:
    path = Path(file_path)

    if not database_enabled():
        with _local_lock(path):
            df = read_table(path, columns=columns)
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            _atomic_write_text(path, df.to_csv(index=False))
        return

    append_json_row(_table_namespace(path), row, path.with_suffix(path.suffix + ".json"))


def replace_table(file_path: Path | str, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    path = Path(file_path)

    if not database_enabled():
        path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows)
        if columns:
            for column in columns:
                if column not in df.columns:
                    df[column] = pd.Series(dtype="object")
            df = df[columns]
        with _local_lock(path):
            if path.exists():
                read_table(path)
            _atomic_write_text(path, df.to_csv(index=False))
        return

    replace_json_rows(_table_namespace(path), rows, path.with_suffix(path.suffix + ".json"))


def ensure_table_file(file_path, columns):
    """Create only a missing local CSV while holding the same lock as append."""
    if database_enabled():
        return
    path = Path(file_path)
    with _local_lock(path):
        if not path.exists():
            _atomic_write_text(path, pd.DataFrame(columns=columns).to_csv(index=False))


def ensure_json_rows_file(local_path):
    if database_enabled():
        return
    with _local_lock(local_path):
        if not local_path.exists():
            _atomic_write_text(local_path, "[]")
