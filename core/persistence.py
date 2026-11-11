from __future__ import annotations

import json
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


class PersistenceDataError(RuntimeError):
    """Existing data is invalid; do not replace it with an empty/default state."""


def _decode_json(text: str, expected_type: type, source: str):
    try:
        data = json.loads(text)
    except (ValueError, TypeError) as exc:
        raise PersistenceDataError(f"Invalid JSON in {source}; recovery required.") from exc
    if not isinstance(data, expected_type):
        raise PersistenceDataError(f"Invalid data structure in {source}; recovery required.")
    if expected_type is list and any(not isinstance(row, dict) for row in data):
        raise PersistenceDataError(f"Invalid row in {source}; recovery required.")
    return data


def _atomic_write_text(path: Path, text: str) -> None:
    """Publish a complete local file, preserving the old file on write failure.

    This prevents torn writes; it does not serialize concurrent read-modify-write.
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
    global _DB_SCHEMA_READY

    if not database_enabled() or _DB_SCHEMA_READY:
        return

    with _connect() as conn:
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


def load_json_state(
    namespace: str,
    default_factory: Callable[[], dict[str, Any]],
    local_path: Path,
) -> dict[str, Any]:
    if not database_enabled():
        return _load_local_json_state(local_path, default_factory)

    _ensure_db_schema()

    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT payload_json FROM app_state WHERE namespace = %s", (namespace,))
            row = cur.fetchone()

    if row:
        return _decode_json(str(row[0]), dict, "database state")

    payload = _load_local_json_state(local_path, default_factory)
    save_json_state(namespace, payload, local_path)
    return payload


def save_json_state(namespace: str, payload: dict[str, Any], local_path: Path) -> None:
    if not database_enabled():
        _atomic_write_text(local_path, json.dumps(payload, indent=2, ensure_ascii=False))
        return

    _ensure_db_schema()

    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO app_state(namespace, payload_json, updated_at)
                VALUES (%s, %s, NOW())
                ON CONFLICT (namespace)
                DO UPDATE SET payload_json = EXCLUDED.payload_json, updated_at = NOW()
                """,
                (namespace, json.dumps(payload, ensure_ascii=False)),
            )


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

    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            results = cur.fetchall()

    if results:
        rows: list[dict[str, Any]] = []
        for result in results:
            rows.append(_decode_json(str(result[0]), dict, "database row"))
        return rows

    rows = _load_local_json_rows(local_path)
    if rows:
        replace_json_rows(namespace, rows, local_path)
    if limit is None:
        return rows
    return rows[-int(limit) :]


def append_json_row(namespace: str, row: dict[str, Any], local_path: Path) -> None:
    if not database_enabled():
        local_path.parent.mkdir(parents=True, exist_ok=True)
        rows = _load_local_json_rows(local_path)
        rows.append(row)
        _atomic_write_text(local_path, json.dumps(rows, indent=2, ensure_ascii=False))
        return

    _ensure_db_schema()

    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO app_rows(namespace, row_json) VALUES (%s, %s)",
                (namespace, json.dumps(row, ensure_ascii=False)),
            )


def replace_json_rows(namespace: str, rows: list[dict[str, Any]], local_path: Path) -> None:
    if not database_enabled():
        _atomic_write_text(local_path, json.dumps(rows, indent=2, ensure_ascii=False))
        return

    _ensure_db_schema()

    with _connect() as conn:
        # Autocommit remains enabled for other operations; replacement is all-or-nothing.
        with conn.transaction(), conn.cursor() as cur:
            cur.execute("DELETE FROM app_rows WHERE namespace = %s", (namespace,))
            for row in rows:
                cur.execute(
                    "INSERT INTO app_rows(namespace, row_json) VALUES (%s, %s)",
                    (namespace, json.dumps(row, ensure_ascii=False)),
                )


def _table_namespace(file_path: Path | str) -> str:
    return f"table:{Path(file_path).name.lower()}"


def read_table(file_path: Path | str, columns: list[str] | None = None) -> pd.DataFrame:
    path = Path(file_path)

    if not database_enabled():
        try:
            df = pd.read_csv(path)
        except Exception:
            df = pd.DataFrame(columns=columns or [])
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
        path.parent.mkdir(parents=True, exist_ok=True)
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
        _atomic_write_text(path, df.to_csv(index=False))
        return

    replace_json_rows(_table_namespace(path), rows, path.with_suffix(path.suffix + ".json"))
