"""Explicit, read-only inventory of the robot's two PostgreSQL tables.

Never import app loaders: they can initialize schema or seed local data.
Only reads ATLAS_AUDIT_DATABASE_URL; never falls back to DATABASE_URL or .env.
Does not print credentials, user records, financial state or row contents.
"""
import json
import os

import psycopg


def inventory(conn):
    result = {"tables": {}}
    with conn.transaction():
        conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
        conn.execute("SET LOCAL statement_timeout = '30s'")
        result["server_version"] = conn.execute("SHOW server_version").fetchone()[0]
        result["read_only"] = conn.execute("SHOW transaction_read_only").fetchone()[0] == "on"
        for table, column in (("app_state", "payload_json"), ("app_rows", "row_json")):
            columns = conn.execute(
                "SELECT column_name, data_type FROM information_schema.columns "
                "WHERE table_schema = 'public' AND table_name = %s ORDER BY ordinal_position", (table,)
            ).fetchall()
            if not columns:
                result["tables"][table] = {"present": False}
                continue
            table_result = {"present": True, "columns": [list(item) for item in columns]}
            if column not in {item[0] for item in columns}:
                table_result["error"] = "expected_payload_column_missing"
                result["tables"][table] = table_result
                continue
            # Fixed identifiers above; never use untrusted SQL names.
            with conn.cursor(name=f"audit_{table}") as cur:
                cur.execute(f"SELECT {column} FROM public.{table}")
                checked = invalid = 0
                while batch := cur.fetchmany(256):
                    for (value,) in batch:
                        checked += 1
                        try:
                            if not isinstance(json.loads(value), dict):
                                invalid += 1
                        except (ValueError, TypeError):
                            invalid += 1
            table_result.update(rows_checked=checked, invalid_payloads=invalid)
            result["tables"][table] = table_result
    return result


def main():
    dsn = os.environ.get("ATLAS_AUDIT_DATABASE_URL", "").strip()
    if not dsn:
        raise SystemExit("Set ATLAS_AUDIT_DATABASE_URL securely; no connection attempted.")
    try:
        with psycopg.connect(dsn, connect_timeout=10, autocommit=True,
                             options="-c default_transaction_read_only=on") as conn:
            result = inventory(conn)
        print(json.dumps(result, indent=2))
    except Exception as exc:
        raise SystemExit(f"Read-only audit failed ({type(exc).__name__}); no credentials printed.") from None


if __name__ == "__main__":
    main()
