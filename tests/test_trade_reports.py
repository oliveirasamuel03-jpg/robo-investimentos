from __future__ import annotations

import numpy as np
import pandas as pd

from tests.conftest import load_module


def _fake_prices(rows: int = 120) -> pd.DataFrame:
    idx = np.arange(rows, dtype=float)
    close = 100 + idx * 0.05 + np.sin(idx / 5.0) * 1.4
    return pd.DataFrame(
        {
            "datetime": pd.date_range("2025-01-01", periods=rows, freq="h"),
            "open": close + 0.1,
            "high": close + 0.6,
            "low": close - 0.6,
            "close": close,
            "volume": 1000 + idx * 5,
        }
    )


def test_closed_trade_report_is_generated_after_position_exit(isolated_storage, monkeypatch):
    state_store = load_module("core.state_store")
    paper_engine = load_module("paper_trading_engine")
    trader_engine = load_module("engines.trader_engine")
    reports = load_module("core.trader_reports")

    monkeypatch.setattr(paper_engine, "load_prices", lambda symbol, config: _fake_prices())
    monkeypatch.setattr(paper_engine, "_should_buy", lambda latest, config: (True, 0.91))
    monkeypatch.setattr(paper_engine, "_should_sell", lambda position, latest, holding_minutes, config: (False, ""))

    state = state_store.load_bot_state()
    state["bot_status"] = "RUNNING"
    state["trader"]["profile"] = "Equilibrado"
    state["trader"]["watchlist"] = ["AAPL"]
    state_store.save_bot_state(state)

    trader_engine.run_trader_cycle()

    state = state_store.load_bot_state()
    state["bot_status"] = "PAUSED"
    state_store.save_bot_state(state)
    monkeypatch.setattr(
        paper_engine,
        "_should_sell",
        lambda position, latest, holding_minutes, config: (True, "teste de saida"),
    )

    trader_engine.run_trader_cycle()

    reports_df = reports.read_trade_reports()

    assert len(reports_df) == 1
    assert reports_df.iloc[0]["asset"] == "AAPL"
    assert reports_df.iloc[0]["profile"] == "Equilibrado"
    assert reports_df.iloc[0]["exit_reason"] == "teste de saida"
    assert pd.notna(reports_df.iloc[0]["opened_at"])
    assert pd.notna(reports_df.iloc[0]["closed_at"])
    assert pd.notna(reports_df.iloc[0]["realized_pnl"])
