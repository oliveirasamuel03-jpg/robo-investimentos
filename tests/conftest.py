from __future__ import annotations

import importlib
import sys

import pytest


MODULES_TO_CLEAR = [
    "core.commit_protocol",
    "core.config",
    "core.persistence",
    "core.state_store",
    "core.stop_control",
    "core.ui_mutations",
    "core.market_data",
    "core.market_context",
    "core.macro_alerts",
    "core.external_signals",
    "core.providers.base",
    "core.providers.market_data_provider",
    "core.providers.signal_provider",
    "core.broker",
    "core.production_monitor",
    "core.alerts",
    "core.daily_risk",
    "core.daily_report_diagnostic_clarity",
    "core.retention",
    "core.calibration_preview",
    "core.bos_confirmation_quality_audit",
    "core.bos_pivot_trace_audit",
    "core.controlled_micro_adjustment_observability",
    "core.controlled_micro_adjustment_study",
    "core.fibonacci_alignment_audit",
    "core.feed_scope_reconciliation",
    "core.h1_confirmation_after_h4_bos_audit",
    "core.post_10d_calibration_plan",
    "core.provider_budget_visual_fallback",
    "core.market_structure_audit",
    "core.multi_timeframe_data_fetcher",
    "core.multi_timeframe_swing_audit",
    "core.no_setup_eligible_decomposition",
    "core.reversal_blocker_routing_audit",
    "core.setup_blocker_taxonomy_audit",
    "core.shadow_decision_simulator",
    "core.strategy_decision_bridge_trace",
    "core.strategy_bottleneck",
    "core.strategy_structure_audit",
    "core.signal_rejection_analysis",
    "core.swing_validation",
    "core.trader_profiles",
    "core.trader_reports",
    "core.auth.users_store",
    "paper_trading_engine",
    "engines.quant_bridge",
    "engines.trader_engine",
    "bot_engine",
]


def _clear_modules() -> None:
    for module_name in MODULES_TO_CLEAR:
        sys.modules.pop(module_name, None)


@pytest.fixture
def isolated_storage(tmp_path, monkeypatch):
    storage_dir = tmp_path / "storage"
    monkeypatch.setenv("ROBO_STORAGE_DIR", str(storage_dir))
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("AUTH_REQUIRED", "false")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("ADMIN_USERNAME", raising=False)
    monkeypatch.delenv("ADMIN_PASSWORD", raising=False)
    _clear_modules()
    yield storage_dir
    _clear_modules()


def load_module(name: str):
    if name in sys.modules:
        return importlib.reload(sys.modules[name])
    return importlib.import_module(name)


def fake_market_result(frames, requested_by="worker_cycle"):
    """Current feed contract, with source metadata derived from fixture frames."""
    counts = {"market": 0, "cached": 0, "fallback": 0, "unknown": 0}
    for frame in frames.values():
        source = str(frame.iloc[-1].get("data_source", "unknown"))
        counts[source if source in counts else "unknown"] += 1
    source = max(counts, key=counts.get)
    return {"frames": frames, "status": {
        "provider": "fixture", "status": "healthy" if source == "market" else "degraded",
        "last_sync_at": "2026-01-01T00:00:00+00:00", "last_source": source,
        "source_breakdown": counts, "symbols": list(frames),
        "requested_by": requested_by, "last_error": "",
    }}


@pytest.fixture(autouse=True)
def cleanup_owned_postgres_test_databases():
    """Bound local durable-volume growth; never connects without runner opt-in."""
    import os
    if os.getenv('ATLAS_INFRA12_LOCAL_RUN') != 'synthetic-disposable-only':
        yield
        return
    import psycopg
    from psycopg import sql
    from psycopg.conninfo import conninfo_to_dict
    dsn = os.environ['ATLAS_INFRA12_LOCAL_DSN']
    params = conninfo_to_dict(dsn)
    assert params['host'] == '127.0.0.1' and params['dbname'] == 'postgres' and params['user'] == 'atlas_fixture'
    with psycopg.connect(dsn, autocommit=True) as conn:
        before = {r[0] for r in conn.execute('SELECT datname FROM pg_database')}
    yield
    with psycopg.connect(dsn, autocommit=True) as conn:
        added = {r[0] for r in conn.execute('SELECT datname FROM pg_database')} - before
        for name in added:
            assert name.startswith(('atlas_infra12_', 'atlas_infra123_')), 'Unexpected database: refuse cleanup'
            conn.execute(sql.SQL('DROP DATABASE {} WITH (FORCE)').format(sql.Identifier(name)))
