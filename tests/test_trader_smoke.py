from __future__ import annotations

import numpy as np
import pandas as pd

from tests.conftest import load_module


def _fake_prices(rows: int = 120) -> pd.DataFrame:
    idx = np.arange(rows, dtype=float)
    close = 100 + idx * 0.05 + np.sin(idx / 4.0) * 1.8
    open_ = close + 0.1
    high = close + 0.6
    low = close - 0.6
    volume = 1000 + idx * 3
    return pd.DataFrame(
        {
            "datetime": pd.date_range("2025-01-01", periods=rows, freq="h"),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def test_run_trader_cycle_smoke_with_synthetic_prices(isolated_storage, monkeypatch):
    state_store = load_module("core.state_store")
    paper_engine = load_module("paper_trading_engine")
    trader_engine = load_module("engines.trader_engine")

    monkeypatch.setattr(paper_engine, "load_prices", lambda symbol, config: _fake_prices())
    monkeypatch.setattr(paper_engine, "_should_buy", lambda latest, config: (True, 0.95))
    monkeypatch.setattr(paper_engine, "_should_sell", lambda position, latest, holding_minutes, config: (False, ""))

    state = state_store.load_bot_state()
    state["bot_status"] = "RUNNING"
    state["trader"]["watchlist"] = ["AAPL", "MSFT"]
    state_store.save_bot_state(state)

    result = trader_engine.run_trader_cycle()
    paper_state = paper_engine.load_paper_state()

    assert result["status"] == "RUNNING"
    assert paper_state["run_count"] >= 1
    assert "cash" in paper_state
    assert "equity" in paper_state


def test_paper_engine_does_not_reset_db_state_when_local_file_is_missing(isolated_storage, monkeypatch):
    paper_engine = load_module("paper_trading_engine")

    load_calls = []
    cfg = paper_engine.PaperTradingConfig(initial_capital=25000.0)

    monkeypatch.setattr(paper_engine, "database_enabled", lambda: True)
    monkeypatch.setattr(
        paper_engine,
        "load_json_state",
        lambda namespace, default_factory, path: load_calls.append((namespace, path)) or {"cash": 123.0},
    )

    def fail_save(state: dict) -> None:
        raise AssertionError("save_paper_state should not be called when DATABASE_URL is active and local file is absent")

    monkeypatch.setattr(paper_engine, "save_paper_state", fail_save)

    paper_engine.ensure_paper_files(cfg)

    assert load_calls
