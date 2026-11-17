from __future__ import annotations

from datetime import datetime, timedelta, timezone
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from core.config import (
    ARCHIVE_LOGS_DIR,
    ARCHIVE_REPORTS_DIR,
    ARCHIVE_WEEKLY_REPORTS_DIR,
    BOT_LOG_COLUMNS,
    BOT_LOG_FILE,
    BOT_STATE_FILE,
    RETENTION_ARCHIVE_TRADER_ORDERS,
    RETENTION_DAYS,
    RETENTION_ENABLED,
    RETENTION_RUN_INTERVAL_HOURS,
    TRADER_ORDERS_COLUMNS,
    TRADER_ORDERS_FILE,
    TRADER_REPORTS_COLUMNS,
    TRADER_REPORTS_FILE,
    WEEKLY_REPORTS_DIR,
    WEEKLY_REPORT_RUNTIME_WEEKS,
)
from core.persistence import (
    load_json_state, read_table, replace_table, transactional_persistence, update_json_state,
)
import core.persistence as persistence
from core.state_store import load_bot_state
from core.trader_reports import (
    calculate_trade_report_metrics,
    generate_trade_suggestions,
    normalize_trade_reports_frame,
)


def _utc_now(now: datetime | None = None) -> datetime:
    if now is None:
        return datetime.now(timezone.utc)
    if now.tzinfo is None:
        return now.replace(tzinfo=timezone.utc)
    return now.astimezone(timezone.utc)


def _normalize_frame(df: pd.DataFrame | None, columns: list[str]) -> pd.DataFrame:
    frame = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
    for column in columns:
        if column not in frame.columns:
            frame[column] = pd.Series(dtype="object")
    return frame[columns].copy()


def _safe_dt_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(dtype="datetime64[ns, UTC]")
    return pd.to_datetime(df[column], errors="coerce", utc=True)


def _row_key_series(df: pd.DataFrame, key_columns: list[str]) -> pd.Series:
    working = _normalize_frame(df, list(dict.fromkeys(key_columns + list(df.columns))))
    key_frame = pd.DataFrame(index=working.index)
    for column in key_columns:
        key_frame[column] = (
            working[column]
            .astype("string")
            .fillna("")
            .str.strip()
            .replace({"<NA>": "", "nan": ""})
        )
    return key_frame.agg("|".join, axis=1)


def _dedupe_rows(df: pd.DataFrame, *, key_columns: list[str], sort_column: str | None = None) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    working = df.copy()
    if sort_column and sort_column in working.columns:
        sort_values = pd.to_datetime(working[sort_column], errors="coerce", utc=True)
        working = working.assign(_sort_value=sort_values).sort_values("_sort_value", kind="stable")

    # A business key is not a content identity. Conflicting payloads must never
    # disappear from a projection merely because one happened to be last.
    fingerprints = [json.dumps(row, sort_keys=True, default=str, ensure_ascii=False)
                    for row in _records(working, list(working.columns))]
    working = working.assign(_row_key=fingerprints)
    working = working.drop_duplicates("_row_key", keep="first")
    working = working.drop(columns=[c for c in ["_row_key", "_sort_value"] if c in working.columns])
    return working.reset_index(drop=True)


def _records(df: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    frame = _normalize_frame(df, columns)
    records: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        clean_row: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, pd.Timestamp):
                clean_row[key] = value.isoformat()
            elif pd.isna(value):
                clean_row[key] = None
            else:
                clean_row[key] = value
        records.append(clean_row)
    return records


def _publish_archive_paths(prefix, paths):
    """Same PostgreSQL unit as copy/removal, including direct API callers."""
    if not paths:
        return
    from core.state_store import DEFAULT_STATE
    kind = {'trade_reports': 'reports', 'trader_orders': 'orders', 'bot_log': 'logs'}[prefix]
    def publish(state):
        catalog = state.setdefault('retention', {}).setdefault('archive_catalog', {})
        catalog[kind] = sorted(set(catalog.get(kind, [])) | set(paths))
        return state
    # Patch the persisted document without invoking configuration migrations.
    # Standard persistence still enforces CAS and STOP/financial guards.
    persistence.update_json_state('bot_state', lambda: deepcopy(DEFAULT_STATE), BOT_STATE_FILE, publish)


def _check_archive_namespace(runtime_path, archive_path):
    if persistence._table_namespace(runtime_path) == persistence._table_namespace(archive_path):
        raise persistence.PersistenceDataError('Retention namespace collision; source and archive must be distinct.')


def _catalog_paths(kind: str) -> list[str]:
    state = load_bot_state()
    retention_state = state.get("retention", {}) or {}
    archive_catalog = retention_state.get("archive_catalog", {}) or {}
    values = archive_catalog.get(kind, []) or []
    return [str(value) for value in values if value]


def _known_archive_paths(kind: str, directory: Path, pattern: str) -> list[Path]:
    state_paths = [Path(value) for value in _catalog_paths(kind)]
    local_paths = list(directory.glob(pattern))
    combined: dict[str, Path] = {}
    for path in [*state_paths, *local_paths]:
        combined[str(path)] = path
    return [combined[key] for key in sorted(combined.keys())]


def _archive_file_name(prefix: str, stamp: pd.Period) -> str:
    return f"{prefix}_{stamp.year}_{int(stamp.month):02d}.csv"


def _weekly_csv_path(week_key: str, *, archived: bool) -> Path:
    filename = f"weekly_validation_archive_{week_key}.csv" if archived else f"weekly_validation_{week_key}.csv"
    return (ARCHIVE_WEEKLY_REPORTS_DIR if archived else WEEKLY_REPORTS_DIR) / filename


def _weekly_summary_path(week_key: str, *, archived: bool) -> Path:
    filename = f"weekly_validation_archive_{week_key}.json" if archived else f"weekly_validation_{week_key}.json"
    return (ARCHIVE_WEEKLY_REPORTS_DIR if archived else WEEKLY_REPORTS_DIR) / filename


def _weekly_summary_namespace(week_key: str, *, archived: bool) -> str:
    location = "archive" if archived else "runtime"
    return f"weekly_validation_summary:{location}:{week_key}"


@transactional_persistence
def _save_weekly_summary(summary: dict[str, Any], *, week_key: str, archived: bool) -> dict[str, Any]:
    path = _weekly_summary_path(week_key, archived=archived)
    namespace = _weekly_summary_namespace(week_key, archived=archived)
    if persistence.database_enabled():
        with persistence._reader() as conn:
            row = conn.execute('SELECT payload_json FROM app_state WHERE namespace=%s', (namespace,)).fetchone()
        previous = persistence._decode_json(row[0], dict, 'weekly summary') if row else None
    else:
        previous = persistence._decode_json(path.read_text(), dict, 'weekly summary') if path.exists() else None
    if previous is not None:
        meaningful = lambda value: {key: item for key, item in value.items() if key != 'generated_at'}
        if meaningful(previous) == meaningful(summary):
            return previous
    update_json_state(
        namespace, dict, path,
        lambda _current: summary,
    )
    return summary


def load_weekly_summary(report_entry: dict[str, Any]) -> dict[str, Any]:
    week_key = str(report_entry.get("week_key") or "")
    archived = str(report_entry.get("location") or "runtime").lower() == "archive"
    return load_json_state(
        _weekly_summary_namespace(week_key, archived=archived),
        lambda: {},
        _weekly_summary_path(week_key, archived=archived),
    )


def read_weekly_report_rows(report_entry: dict[str, Any]) -> pd.DataFrame:
    csv_path = report_entry.get("csv_path")
    if not csv_path:
        return pd.DataFrame(columns=TRADER_REPORTS_COLUMNS)
    return normalize_trade_reports_frame(read_table(csv_path, columns=TRADER_REPORTS_COLUMNS))


@transactional_persistence
def _archive_old_rows(
    *,
    runtime_path: Path,
    archive_dir: Path,
    prefix: str,
    columns: list[str],
    timestamp_column: str,
    key_columns: list[str],
    retention_days: int,
    now: datetime,
    preflight_only: bool = False,
) -> dict[str, Any]:
    namespace = persistence._table_namespace(runtime_path)
    from core import commit_protocol as protocol
    financial = namespace in protocol.FINANCIAL_ROWS
    source = []
    if financial:
        # Read controls without using initializing loaders in a no-op path.
        if persistence.database_enabled():
            with persistence._reader() as conn:
                controls = conn.execute('SELECT namespace FROM app_state WHERE namespace IN (%s,%s)',
                                        (protocol.GATE, protocol.QUARANTINE)).fetchall()
            if controls and protocol.gate_status()['blocked']:
                raise protocol.ReconciliationRequired(reason='RETENTION_BLOCKED_BY_FINANCIAL_CONTROL')
        elif protocol._GATE_PATH.exists() or protocol._QUARANTINE_PATH.exists():
            if protocol.gate_status()['blocked']:
                raise protocol.ReconciliationRequired(reason='RETENTION_BLOCKED_BY_FINANCIAL_CONTROL')
        try:
            if persistence.database_enabled():
                with persistence._reader() as conn:
                    raw = conn.execute('SELECT seq,row_json FROM app_rows WHERE namespace=%s ORDER BY seq',
                                       (namespace,)).fetchall()
                source = []
                for seq, text in raw:
                    source.append((seq, persistence._decode_json(text, dict, f'retention financial source seq={seq}')))
                records = [row for _, row in source]
                current_df = pd.DataFrame(records)
            else:
                current_df = read_table(runtime_path)
                records = _records(current_df, list(current_df.columns))
                source = list(enumerate(records))
        except persistence.PersistenceDataError:
            marked = protocol._quarantine_locked('RETENTION_CORRUPTION',
                cycle_id=protocol._gate().get('active_cycle_id') or protocol._gate().get('last_cycle_id'),
                evidence={'namespace': namespace, 'reason': 'STRICT_SOURCE_DECODING_FAILED',
                          **({'origin_id': seq, 'raw_payload_sha256': hashlib.sha256(text.encode()).hexdigest()}
                             if persistence.database_enabled() else {})})
            return {'blocked': True, 'reason': 'RETENTION_CORRUPTION', 'archived_rows': 0,
                    'retained_rows': len(raw) if persistence.database_enabled() else None,
                    'files_updated': [], 'oldest_runtime_at': '', **marked}
        seen, conflicts, identical = {}, [], 0
        for identity, record in source:
            # Hash the complete original object, including cycle_id/extra fields.
            content_hash = hashlib.sha256(persistence._encode(record).encode()).hexdigest()
            key = tuple(str(record.get(field, '')).strip() for field in key_columns)
            if key in seen:
                previous_id, previous_hash = seen[key]
                if previous_hash != content_hash:
                    conflicts.append({'namespace': namespace, 'key': list(key),
                        'origin_ids': [previous_id, identity], 'payload_sha256': [previous_hash, content_hash],
                        'cycle_id': record.get('cycle_id'), 'reason': 'SAME_KEY_DIFFERENT_CONTENT'})
                else:
                    identical += 1
            else:
                seen[key] = (identity, content_hash)
        if conflicts:
            marked = protocol._quarantine_locked('RETENTION_CONFLICT',
                cycle_id=protocol._gate().get('active_cycle_id') or protocol._gate().get('last_cycle_id'),
                evidence={'namespace': namespace, 'conflicts': conflicts})
            return {'blocked': True, 'reason': 'RETENTION_CONFLICT', 'archived_rows': 0,
                    'retained_rows': len(source), 'files_updated': [], 'oldest_runtime_at': '', **marked}
    else:
        current_df = read_table(runtime_path)
    timestamp_series = _safe_dt_series(current_df, timestamp_column)
    cutoff = now - timedelta(days=retention_days)

    archive_mask = timestamp_series.notna() & (timestamp_series < cutoff)
    archive_df = current_df.loc[archive_mask].copy()
    retained_df = current_df.loc[~archive_mask].copy()

    files_updated: list[str] = []
    archived_rows = 0

    oldest = timestamp_series.dropna().min().isoformat() if timestamp_series.notna().any() else ''
    if archive_df.empty:
        return {'archived_rows': 0, 'retained_rows': len(current_df), 'files_updated': [],
                'oldest_runtime_at': oldest, 'identical_rows_preserved': identical if financial else 0}

    if financial:
        if persistence.database_enabled():
            with persistence._reader() as conn:
                receipted = conn.execute('SELECT EXISTS(SELECT 1 FROM app_state WHERE left(namespace,%s)=%s)',
                                        (len(protocol.RECEIPT_PREFIX), protocol.RECEIPT_PREFIX)).fetchone()[0]
        else:
            receipted = any(protocol.RUNTIME_DIR.glob('paper_cycle_receipt_*.json'))
        if receipted:
            # Runtime namespace/seq/content are signed by v1 receipts. No safe
            # relocation contract exists: refuse, do not rewrite or re-sign.
            return {'blocked': True, 'reason': 'FINANCIAL_ARCHIVE_REQUIRES_VERSIONED_RECEIPTS',
                    'archived_rows': 0, 'retained_rows': len(current_df), 'files_updated': [],
                    'oldest_runtime_at': oldest}
        if not persistence.database_enabled():
            return {'blocked': True, 'reason': 'LOCAL_FINANCIAL_ARCHIVE_NOT_ATOMIC',
                    'archived_rows': 0, 'retained_rows': len(current_df), 'files_updated': [],
                    'oldest_runtime_at': oldest}
        if any(row.get('cycle_id') for _, row in source):
            marked = protocol._quarantine_locked('MISSING_FINANCIAL_RECEIPTS',
                evidence={'namespace': namespace, 'reason': 'SOURCE_REFERENCES_ABSENT_CYCLE_RECEIPTS'})
            return {'blocked': True, 'reason': 'MISSING_FINANCIAL_RECEIPTS', 'archived_rows': 0,
                    'retained_rows': len(source), 'files_updated': [], 'oldest_runtime_at': oldest, **marked}
        # Unreceipted synthetic/legacy rows can be moved losslessly in PG.
        # Plan every destination before writing anything; preserve raw objects.
        plans = []
        for period_value in sorted(timestamp_series[archive_mask].dt.tz_localize(None).dt.to_period('M').unique()):
            path = archive_dir / _archive_file_name(prefix, period_value)
            _check_archive_namespace(runtime_path, path)
            existing = persistence.load_json_rows(persistence._table_namespace(path), path.with_suffix('.csv.json'))
            selected = [row for (_, row), stamp in zip(source, timestamp_series)
                        if pd.notna(stamp) and stamp < cutoff and stamp.tz_localize(None).to_period('M') == period_value]
            by_key = {}
            for record in existing + selected:
                key = tuple(str(record.get(field, '')).strip() for field in key_columns)
                fingerprint = hashlib.sha256(persistence._encode(record).encode()).hexdigest()
                if key in by_key and by_key[key] != fingerprint:
                    marked = protocol._quarantine_locked('RETENTION_CONFLICT',
                        cycle_id=protocol._gate().get('active_cycle_id') or protocol._gate().get('last_cycle_id'),
                        evidence={'namespace': namespace, 'archive_namespace': persistence._table_namespace(path),
                                  'key': list(key), 'payload_sha256': [by_key[key], fingerprint],
                                  'reason': 'ARCHIVE_KEY_DIFFERENT_CONTENT'})
                    return {'blocked': True, 'reason': 'RETENTION_CONFLICT', 'archived_rows': 0,
                            'retained_rows': len(source), 'files_updated': [], 'oldest_runtime_at': oldest, **marked}
                by_key[key] = fingerprint
            plans.append((path, existing + selected, len(selected)))
        if preflight_only:
            return {'blocked': False, 'archived_rows': 0, 'retained_rows': len(source),
                    'files_updated': [], 'oldest_runtime_at': oldest}
        for path, merged, count in plans:
            persistence.replace_json_rows(persistence._table_namespace(path), merged, path.with_suffix('.csv.json'))
            files_updated.append(str(path))
            archived_rows += count
        retained = [row for (_, row), keep in zip(source, ~archive_mask) if keep]
        persistence.replace_json_rows(namespace, retained, runtime_path.with_suffix('.csv.json'))
        _publish_archive_paths(prefix, files_updated)
        return {'archived_rows': archived_rows, 'retained_rows': len(retained),
                'files_updated': files_updated, 'oldest_runtime_at': oldest}

    if not archive_df.empty:
        month_series = _safe_dt_series(archive_df, timestamp_column).dt.tz_localize(None).dt.to_period("M")
        for period_value in sorted(month_series.dropna().unique()):
            month_chunk = archive_df.loc[month_series == period_value].copy()
            archive_path = archive_dir / _archive_file_name(prefix, period_value)
            _check_archive_namespace(runtime_path, archive_path)
            existing_df = read_table(archive_path)
            merged_df = pd.concat([existing_df, month_chunk], ignore_index=True)
            merged_df = _dedupe_rows(merged_df, key_columns=key_columns, sort_column=timestamp_column)
            all_columns = list(merged_df.columns)
            replace_table(archive_path, _records(merged_df, all_columns), columns=all_columns)
            files_updated.append(str(archive_path))
            archived_rows += int(len(month_chunk))

    # Retained source rows keep their order/content. No deduplication of originals.
    all_columns = list(retained_df.columns)
    replace_table(runtime_path, _records(retained_df, all_columns), columns=all_columns)
    _publish_archive_paths(prefix, files_updated)

    oldest_runtime_at = None
    if not retained_df.empty:
        retained_ts = _safe_dt_series(retained_df, timestamp_column).dropna()
        if not retained_ts.empty:
            oldest_runtime_at = retained_ts.min().isoformat()

    return {
        "archived_rows": archived_rows,
        "retained_rows": int(len(retained_df)),
        "files_updated": files_updated,
        "oldest_runtime_at": oldest_runtime_at or "",
    }


def archive_old_trade_reports(retention_days: int = RETENTION_DAYS, *, now: datetime | None = None,
                              _preflight_only: bool = False) -> dict[str, Any]:
    return _archive_old_rows(
        runtime_path=TRADER_REPORTS_FILE,
        archive_dir=ARCHIVE_REPORTS_DIR,
        prefix="trade_reports",
        columns=TRADER_REPORTS_COLUMNS,
        timestamp_column="closed_at",
        key_columns=["trade_id", "closed_at", "asset", "exit_reason", "realized_pnl", "quantity"],
        retention_days=retention_days,
        now=_utc_now(now),
        preflight_only=_preflight_only,
    )


def archive_old_bot_logs(retention_days: int = RETENTION_DAYS, *, now: datetime | None = None) -> dict[str, Any]:
    return _archive_old_rows(
        runtime_path=BOT_LOG_FILE,
        archive_dir=ARCHIVE_LOGS_DIR,
        prefix="bot_log",
        columns=BOT_LOG_COLUMNS,
        timestamp_column="timestamp",
        key_columns=["timestamp", "level", "message"],
        retention_days=retention_days,
        now=_utc_now(now),
    )


def archive_old_trader_orders(retention_days: int = RETENTION_DAYS, *, now: datetime | None = None,
                             _preflight_only: bool = False) -> dict[str, Any]:
    return _archive_old_rows(
        runtime_path=TRADER_ORDERS_FILE,
        archive_dir=ARCHIVE_REPORTS_DIR,
        prefix="trader_orders",
        columns=TRADER_ORDERS_COLUMNS,
        timestamp_column="timestamp",
        key_columns=["timestamp", "asset", "side", "quantity", "price", "gross_value", "source"],
        retention_days=retention_days,
        now=_utc_now(now),
        preflight_only=_preflight_only,
    )


def _read_all_trade_reports() -> pd.DataFrame:
    frames = [normalize_trade_reports_frame(read_table(TRADER_REPORTS_FILE, columns=TRADER_REPORTS_COLUMNS))]
    for archive_path in _known_archive_paths("reports", ARCHIVE_REPORTS_DIR, "trade_reports_*.csv"):
        frames.append(normalize_trade_reports_frame(read_table(archive_path, columns=TRADER_REPORTS_COLUMNS)))
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=TRADER_REPORTS_COLUMNS)
    return normalize_trade_reports_frame(
        _dedupe_rows(combined, key_columns=["trade_id", "closed_at", "asset", "exit_reason", "realized_pnl", "quantity"], sort_column="closed_at")
    )


def _read_all_bot_logs() -> pd.DataFrame:
    frames = [_normalize_frame(read_table(BOT_LOG_FILE, columns=BOT_LOG_COLUMNS), BOT_LOG_COLUMNS)]
    for archive_path in _known_archive_paths("logs", ARCHIVE_LOGS_DIR, "bot_log_*.csv"):
        frames.append(_normalize_frame(read_table(archive_path, columns=BOT_LOG_COLUMNS), BOT_LOG_COLUMNS))
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=BOT_LOG_COLUMNS)
    return _dedupe_rows(combined, key_columns=["timestamp", "level", "message"], sort_column="timestamp")


def _parse_cycle_health_logs(logs_df: pd.DataFrame) -> pd.DataFrame:
    if logs_df.empty or "message" not in logs_df.columns:
        return pd.DataFrame(columns=["timestamp", "fallback", "feed_status", "health_level", "worker_status"])

    rows: list[dict[str, Any]] = []
    for row in logs_df.to_dict(orient="records"):
        message = str(row.get("message") or "")
        if not message.startswith("[cycle_health] "):
            continue
        payload: dict[str, Any] = {"timestamp": row.get("timestamp")}
        for chunk in message[len("[cycle_health] ") :].split(";"):
            if "=" not in chunk:
                continue
            key, value = chunk.split("=", 1)
            payload[key.strip()] = value.strip()
        rows.append(payload)

    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["timestamp", "fallback", "feed_status", "health_level", "worker_status"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    return frame


def _build_weekly_suggestions(
    reports_df: pd.DataFrame,
    *,
    early_closed_count: int,
    late_held_count: int,
    operational_errors: int,
    fallback_cycle_pct: float | None,
) -> list[str]:
    suggestions = [str(item.get("message")) for item in generate_trade_suggestions(reports_df) if item.get("message")]
    total_trades = int(len(reports_df))
    if total_trades > 0 and early_closed_count >= max(2, int(round(total_trades * 0.4))):
        suggestions.append("Muitos trades foram encerrados cedo demais; revisar exits suaves e filtros de entrada.")
    if total_trades > 0 and late_held_count >= max(2, int(round(total_trades * 0.3))):
        suggestions.append("Ha trades mantidos tempo demais; revisar holding e trailing stop para destravar capital.")
    if operational_errors > 0:
        suggestions.append("Ocorreram erros operacionais na semana; revisar logs antes de ampliar exposicao.")
    if fallback_cycle_pct is not None and fallback_cycle_pct >= 25:
        suggestions.append("O feed passou tempo relevante em fallback; manter foco em paper trading ate estabilizar dados.")

    unique: list[str] = []
    for text in suggestions:
        normalized = str(text).strip()
        if normalized and normalized not in unique:
            unique.append(normalized)
    return unique


def _final_weekly_observation(
    *,
    trades_count: int,
    operational_errors: int,
    fallback_cycle_pct: float | None,
    win_rate: float,
    early_closed_count: int,
) -> str:
    if operational_errors >= 5:
        return "instavel"
    if fallback_cycle_pct is not None and fallback_cycle_pct >= 50:
        return "instavel"
    if trades_count == 0 and (fallback_cycle_pct is not None and fallback_cycle_pct >= 25):
        return "instavel"
    if trades_count > 0 and (win_rate < 0.35 or early_closed_count >= max(2, int(round(trades_count * 0.4)))):
        return "precisa ajuste"
    return "apto para continuar em simulacao"


def _week_key(dt_value: pd.Timestamp) -> str:
    iso_year, iso_week, _ = dt_value.isocalendar()
    return f"{int(iso_year)}_W{int(iso_week):02d}"


def _week_bounds(week_key: str) -> tuple[datetime, datetime]:
    year_text, week_text = week_key.split("_W", 1)
    week_start = datetime.fromisocalendar(int(year_text), int(week_text), 1).replace(tzinfo=timezone.utc)
    return week_start, week_start + timedelta(days=7)


def _build_weekly_summary(
    *,
    week_key: str,
    reports_df: pd.DataFrame,
    logs_df: pd.DataFrame,
    now: datetime,
    archived: bool,
) -> dict[str, Any]:
    week_start, week_end = _week_bounds(week_key)
    report_metrics = calculate_trade_report_metrics(reports_df)

    best_assets: list[dict[str, Any]] = []
    worst_assets: list[dict[str, Any]] = []
    if not reports_df.empty:
        asset_group = (
            reports_df.assign(realized_pnl=pd.to_numeric(reports_df["realized_pnl"], errors="coerce").fillna(0.0))
            .groupby("asset", dropna=False)
            .agg(
                trades=("asset", "size"),
                pnl=("realized_pnl", "sum"),
                win_rate=("realized_pnl", lambda s: float((pd.to_numeric(s, errors="coerce") > 0).mean())),
            )
            .reset_index()
        )
        best_assets = asset_group.sort_values(["pnl", "win_rate"], ascending=[False, False]).head(3).to_dict(orient="records")
        worst_assets = asset_group.sort_values(["pnl", "win_rate"], ascending=[True, True]).head(3).to_dict(orient="records")

    durations = pd.to_numeric(reports_df.get("duration_minutes"), errors="coerce") if not reports_df.empty else pd.Series(dtype="float64")
    holding_limits = pd.to_numeric(reports_df.get("holding_minutes_limit"), errors="coerce") if not reports_df.empty else pd.Series(dtype="float64")
    early_threshold = holding_limits.fillna(20).apply(lambda value: max(5, float(value) * 0.25))
    late_threshold = holding_limits.fillna(60).apply(lambda value: max(30, float(value) * 1.1))
    early_closed_count = int(((durations.notna()) & (durations <= early_threshold)).sum()) if not reports_df.empty else 0
    late_held_count = int(((durations.notna()) & (durations >= late_threshold)).sum()) if not reports_df.empty else 0

    logs_working = logs_df.copy()
    logs_working["timestamp"] = pd.to_datetime(logs_working["timestamp"], errors="coerce", utc=True)
    operational_errors = int((logs_working["level"].astype("string").str.upper() == "ERROR").sum()) if not logs_working.empty else 0

    cycle_health = _parse_cycle_health_logs(logs_working)
    fallback_cycle_pct = None
    fallback_cycles = 0
    total_cycles = 0
    if not cycle_health.empty:
        total_cycles = int(len(cycle_health))
        fallback_cycles = int((cycle_health["fallback"].astype("string").fillna("0").isin(["1", "true", "True"])).sum())
        if total_cycles > 0:
            fallback_cycle_pct = round((fallback_cycles / total_cycles) * 100.0, 2)

    suggestions = _build_weekly_suggestions(
        reports_df,
        early_closed_count=early_closed_count,
        late_held_count=late_held_count,
        operational_errors=operational_errors,
        fallback_cycle_pct=fallback_cycle_pct,
    )

    observation = _final_weekly_observation(
        trades_count=int(report_metrics["total_trades"] or 0),
        operational_errors=operational_errors,
        fallback_cycle_pct=fallback_cycle_pct,
        win_rate=float(report_metrics["win_rate"] or 0.0),
        early_closed_count=early_closed_count,
    )

    summary = {
        "week_key": week_key,
        "week_label": week_key.replace("_", "-"),
        "week_start_at": week_start.isoformat(),
        "week_end_at": week_end.isoformat(),
        "generated_at": now.isoformat(),
        "location": "archive" if archived else "runtime",
        "trades_count": int(report_metrics["total_trades"] or 0),
        "win_rate": float(report_metrics["win_rate"] or 0.0),
        "payoff": None if report_metrics["payoff"] is None else float(report_metrics["payoff"]),
        "pnl": float(report_metrics["total_pnl"] or 0.0),
        "avg_duration_minutes": None if report_metrics["avg_duration"] is None else float(report_metrics["avg_duration"]),
        "best_trade": None if report_metrics["best_trade"] is None else float(report_metrics["best_trade"]),
        "worst_trade": None if report_metrics["worst_trade"] is None else float(report_metrics["worst_trade"]),
        "best_assets": best_assets,
        "worst_assets": worst_assets,
        "early_closed_trades": early_closed_count,
        "late_held_trades": late_held_count,
        "operational_errors": operational_errors,
        "fallback_cycle_pct": fallback_cycle_pct,
        "fallback_cycles": fallback_cycles,
        "total_cycles": total_cycles,
        "suggestions": suggestions,
        "observation_final": observation,
        "status_final_breakdown": {
            "gain": int((reports_df.get("status_final") == "gain").sum()) if not reports_df.empty else 0,
            "loss": int((reports_df.get("status_final") == "loss").sum()) if not reports_df.empty else 0,
            "flat": int((reports_df.get("status_final") == "flat").sum()) if not reports_df.empty else 0,
        },
    }
    return summary


def _build_weekly_reports(now: datetime, retention_days: int) -> dict[str, Any]:
    trade_reports = _read_all_trade_reports()
    bot_logs = _read_all_bot_logs()

    lookback_weeks = max(WEEKLY_REPORT_RUNTIME_WEEKS + 4, max(4, retention_days // 7))
    window_start = now - timedelta(weeks=lookback_weeks)

    working_reports = trade_reports.copy()
    if not working_reports.empty:
        working_reports["closed_at"] = pd.to_datetime(working_reports["closed_at"], errors="coerce", utc=True)
        working_reports = working_reports[working_reports["closed_at"] >= window_start]

    working_logs = bot_logs.copy()
    if not working_logs.empty:
        working_logs["timestamp"] = pd.to_datetime(working_logs["timestamp"], errors="coerce", utc=True)
        working_logs = working_logs[working_logs["timestamp"] >= window_start]

    week_keys: set[str] = set()
    if not working_reports.empty:
        week_keys.update(_week_key(value) for value in working_reports["closed_at"].dropna())
    if not working_logs.empty:
        week_keys.update(_week_key(value) for value in working_logs["timestamp"].dropna())

    index: list[dict[str, Any]] = []
    runtime_paths: list[str] = []
    archive_paths: list[str] = []

    for week_key in sorted(week_keys, reverse=True):
        week_start, week_end = _week_bounds(week_key)
        week_reports = (
            working_reports[(working_reports["closed_at"] >= week_start) & (working_reports["closed_at"] < week_end)].copy()
            if not working_reports.empty
            else pd.DataFrame(columns=TRADER_REPORTS_COLUMNS)
        )
        week_logs = (
            working_logs[(working_logs["timestamp"] >= week_start) & (working_logs["timestamp"] < week_end)].copy()
            if not working_logs.empty
            else pd.DataFrame(columns=BOT_LOG_COLUMNS)
        )
        weeks_old = max(0, int((now - week_start).days // 7))
        archived = weeks_old >= WEEKLY_REPORT_RUNTIME_WEEKS
        csv_path = _weekly_csv_path(week_key, archived=archived)
        summary_path = _weekly_summary_path(week_key, archived=archived)
        desired_rows = _records(week_reports, TRADER_REPORTS_COLUMNS)
        if persistence.database_enabled():
            existing_rows = persistence.load_json_rows(persistence._table_namespace(csv_path), csv_path.with_suffix('.csv.json'))
        else:
            existing_rows = _records(read_table(csv_path), TRADER_REPORTS_COLUMNS) if csv_path.exists() else None
        if existing_rows != desired_rows:
            replace_table(csv_path, desired_rows, columns=TRADER_REPORTS_COLUMNS)
        summary = _build_weekly_summary(
            week_key=week_key,
            reports_df=normalize_trade_reports_frame(week_reports),
            logs_df=_normalize_frame(week_logs, BOT_LOG_COLUMNS),
            now=now,
            archived=archived,
        )
        summary = _save_weekly_summary(summary, week_key=week_key, archived=archived)

        entry = {
            "week_key": week_key,
            "week_label": summary["week_label"],
            "week_start_at": summary["week_start_at"],
            "week_end_at": summary["week_end_at"],
            "generated_at": summary["generated_at"],
            "location": summary["location"],
            "csv_path": str(csv_path),
            "summary_path": str(summary_path),
            "summary_namespace": _weekly_summary_namespace(week_key, archived=archived),
            "trades_count": summary["trades_count"],
            "pnl": summary["pnl"],
            "win_rate": summary["win_rate"],
            "observation_final": summary["observation_final"],
        }
        index.append(entry)

        if archived:
            archive_paths.extend([str(csv_path), str(summary_path)])
        else:
            runtime_paths.extend([str(csv_path), str(summary_path)])

    return {
        "weekly_reports_index": index,
        "weekly_reports_generated": len(index),
        "runtime_paths": runtime_paths,
        "archive_paths": archive_paths,
    }


def should_run_retention_job(state: dict | None = None, *, now: datetime | None = None) -> bool:
    payload = state or load_bot_state()
    retention_state = payload.get("retention", {}) or {}
    if not bool(retention_state.get("enabled", RETENTION_ENABLED)):
        return False

    run_interval_hours = max(
        1,
        int(retention_state.get("run_interval_hours", RETENTION_RUN_INTERVAL_HOURS) or RETENTION_RUN_INTERVAL_HOURS),
    )
    last_run_at = pd.to_datetime(retention_state.get("last_run_at"), errors="coerce", utc=True)
    current_time = _utc_now(now)
    if pd.isna(last_run_at):
        return True
    return (current_time - last_run_at.to_pydatetime()) >= timedelta(hours=run_interval_hours)


@transactional_persistence
def run_retention_job(
    *,
    now: datetime | None = None,
    retention_days: int = RETENTION_DAYS,
    archive_trader_orders: bool = RETENTION_ARCHIVE_TRADER_ORDERS,
) -> dict[str, Any]:
    current_time = _utc_now(now)

    trade_reports_preflight = archive_old_trade_reports(retention_days=retention_days, now=current_time,
                                                       _preflight_only=True)
    if trade_reports_preflight.get('blocked'):
        return {'blocked': True, 'last_run_at': current_time.isoformat(),
                'last_error': trade_reports_preflight['reason'], 'last_error_at': current_time.isoformat(),
                'retention_result': trade_reports_preflight}
    # Validate orders before moving logs or writing any derived weekly reports.
    if archive_trader_orders:
        orders_preflight = archive_old_trader_orders(retention_days=retention_days, now=current_time,
                                                    _preflight_only=True)
        if orders_preflight.get('blocked'):
            return {'blocked': True, 'last_run_at': current_time.isoformat(),
                    'last_error': orders_preflight['reason'], 'last_error_at': current_time.isoformat(),
                    'retention_result': orders_preflight}
    trade_reports_summary = archive_old_trade_reports(retention_days=retention_days, now=current_time)
    bot_logs_summary = archive_old_bot_logs(retention_days=retention_days, now=current_time)

    orders_summary = {
        "archived_rows": 0,
        "retained_rows": int(len(read_table(TRADER_ORDERS_FILE, columns=TRADER_ORDERS_COLUMNS))),
        "files_updated": [],
        "oldest_runtime_at": "",
        "enabled": bool(archive_trader_orders),
    }
    if archive_trader_orders:
        archived_orders_summary = archive_old_trader_orders(retention_days=retention_days, now=current_time)
        archived_orders_summary["enabled"] = True
        orders_summary = archived_orders_summary

    weekly_payload = _build_weekly_reports(current_time, retention_days)

    archive_catalog = {
        "reports": [],
        "logs": [],
        "orders": [],
        "weekly_reports": sorted(set(_catalog_paths('weekly_reports') + weekly_payload["runtime_paths"] + weekly_payload["archive_paths"])),
    }
    archive_catalog["reports"] = sorted(
        {*(str(path) for path in _known_archive_paths("reports", ARCHIVE_REPORTS_DIR, "trade_reports_*.csv")), *trade_reports_summary["files_updated"]}
    )
    archive_catalog["logs"] = sorted(
        {*(str(path) for path in _known_archive_paths("logs", ARCHIVE_LOGS_DIR, "bot_log_*.csv")), *bot_logs_summary["files_updated"]}
    )
    if archive_trader_orders:
        archive_catalog["orders"] = sorted(set(_catalog_paths('orders') + orders_summary["files_updated"]))
    else:
        archive_catalog['orders'] = _catalog_paths('orders')

    summary = {
        "enabled": True,
        "retention_days": retention_days,
        "archive_trader_orders": bool(archive_trader_orders),
        "last_run_at": current_time.isoformat(),
        "last_success_at": current_time.isoformat(),
        "last_error": "",
        "last_error_at": "",
        "last_summary": {
            "trade_reports_archived_rows": trade_reports_summary["archived_rows"],
            "bot_logs_archived_rows": bot_logs_summary["archived_rows"],
            "trader_orders_archived_rows": orders_summary["archived_rows"],
            "runtime_trade_reports_rows": trade_reports_summary["retained_rows"],
            "runtime_bot_log_rows": bot_logs_summary["retained_rows"],
            "runtime_trader_orders_rows": orders_summary["retained_rows"],
            "archived_files_updated": sorted(
                set(
                    trade_reports_summary["files_updated"]
                    + bot_logs_summary["files_updated"]
                    + orders_summary["files_updated"]
                    + weekly_payload["archive_paths"]
                )
            ),
            "weekly_reports_generated": weekly_payload["weekly_reports_generated"],
            "oldest_runtime_report_at": trade_reports_summary["oldest_runtime_at"],
            "oldest_runtime_log_at": bot_logs_summary["oldest_runtime_at"],
            "oldest_runtime_order_at": orders_summary.get("oldest_runtime_at", ""),
        },
        "archive_catalog": archive_catalog,
        "weekly_reports_index": weekly_payload["weekly_reports_index"],
    }
    return summary


def get_retention_summary(state: dict | None = None) -> dict[str, Any]:
    payload = state or load_bot_state()
    retention_state = payload.get("retention", {}) or {}
    summary = retention_state.get("last_summary", {}) or {}
    return {
        "enabled": bool(retention_state.get("enabled", RETENTION_ENABLED)),
        "retention_days": int(retention_state.get("retention_days", RETENTION_DAYS) or RETENTION_DAYS),
        "run_interval_hours": int(
            retention_state.get("run_interval_hours", RETENTION_RUN_INTERVAL_HOURS) or RETENTION_RUN_INTERVAL_HOURS
        ),
        "archive_trader_orders": bool(retention_state.get("archive_trader_orders", RETENTION_ARCHIVE_TRADER_ORDERS)),
        "last_run_at": retention_state.get("last_run_at") or "",
        "last_success_at": retention_state.get("last_success_at") or "",
        "last_error": retention_state.get("last_error") or "",
        "last_error_at": retention_state.get("last_error_at") or "",
        "trade_reports_archived_rows": int(summary.get("trade_reports_archived_rows", 0) or 0),
        "bot_logs_archived_rows": int(summary.get("bot_logs_archived_rows", 0) or 0),
        "trader_orders_archived_rows": int(summary.get("trader_orders_archived_rows", 0) or 0),
        "runtime_trade_reports_rows": int(summary.get("runtime_trade_reports_rows", 0) or 0),
        "runtime_bot_log_rows": int(summary.get("runtime_bot_log_rows", 0) or 0),
        "runtime_trader_orders_rows": int(summary.get("runtime_trader_orders_rows", 0) or 0),
        "archived_files_updated": summary.get("archived_files_updated", []) or [],
        "weekly_reports_generated": int(summary.get("weekly_reports_generated", 0) or 0),
        "oldest_runtime_report_at": summary.get("oldest_runtime_report_at") or "",
        "oldest_runtime_log_at": summary.get("oldest_runtime_log_at") or "",
        "oldest_runtime_order_at": summary.get("oldest_runtime_order_at") or "",
        "weekly_reports_index": retention_state.get("weekly_reports_index", []) or [],
    }
