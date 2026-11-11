"""Bounded replay guarantees, without pretending UUIDs provide exactly-once."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from tests.conftest import fake_market_result, load_module
from tests.test_trader_smoke import _fake_prices


def _paper_fixture(monkeypatch):
    paper = load_module("paper_trading_engine")
    now = datetime(2026, 4, 20, 12, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(paper, "_utc_now", lambda: now)
    monkeypatch.setattr(paper, "load_market_data_result", lambda symbols, config, requested_by="worker_cycle":
                        fake_market_result({symbol: _fake_prices() for symbol in symbols}, requested_by))
    evaluation = {
        "buy": True, "score": 0.95, "rejection_reasons": [],
        "trend_ok": True, "score_ok": True, "rsi_ok": True, "atr_ok": True,
        "structure_ok": True, "momentum_ok": True, "trend_alignment": "aligned",
    }
    monkeypatch.setattr(paper, "_evaluate_buy_signal", lambda *_args: dict(evaluation, rejection_reasons=[]))
    monkeypatch.setattr(paper, "_should_sell", lambda *_args: (False, ""))
    cfg = paper.PaperTradingConfig(custom_tickers=["AAPL"], allow_new_entries=True)
    return paper, cfg


def test_repeated_buy_cycle_keeps_one_trade_for_existing_position(isolated_storage, monkeypatch):
    paper, cfg = _paper_fixture(monkeypatch)
    first = paper.run_paper_cycle(cfg)
    initial_trades = paper.read_paper_trades()
    assert len(initial_trades) == 1 and initial_trades[0]["side"] == "BUY"
    initial_cash = first["state"]["cash"]
    initial_id = initial_trades[0]["trade_id"]
    for _ in range(3):
        result = paper.run_paper_cycle(cfg)
        assert result["trades_executed"] == 0
        assert result["state"]["cash"] == initial_cash
        assert result["state"]["positions"]["AAPL"]["trade_id"] == initial_id
    assert paper.read_paper_trades() == initial_trades


def test_repeated_sell_cycle_keeps_one_sell_and_one_closed_report(isolated_storage, monkeypatch):
    paper, cfg = _paper_fixture(monkeypatch)
    paper.run_paper_cycle(cfg)
    buy = paper.read_paper_trades()[0]
    monkeypatch.setattr(paper, "_should_sell", lambda *_args: (True, "synthetic-close"))
    closed = paper.run_paper_cycle(cfg)
    cash_after_close = closed["state"]["cash"]
    trades = paper.read_paper_trades()
    assert [row["side"] for row in trades] == ["BUY", "SELL"]
    assert trades[0]["trade_id"] == trades[1]["trade_id"] == buy["trade_id"]
    for _ in range(3):
        result = paper.run_paper_cycle(cfg)
        assert result["trades_executed"] == 0
        assert result["state"]["cash"] == cash_after_close
        assert result["state"]["positions"] == {}
    reports = load_module("core.trader_reports").read_trade_reports()
    assert len(reports) == 1
    assert reports.iloc[0]["trade_id"] == buy["trade_id"]
    assert paper.read_paper_trades() == trades


def test_failed_calculation_can_be_replayed_without_extra_trade(isolated_storage, monkeypatch):
    paper, cfg = _paper_fixture(monkeypatch)
    before = paper.load_paper_state(cfg)
    original_evaluate = paper._evaluate_buy_signal
    def fail_before_draft_commit(*_args):
        raise RuntimeError("synthetic calculation interruption")
    monkeypatch.setattr(paper, "_evaluate_buy_signal", fail_before_draft_commit)
    with pytest.raises(RuntimeError, match="synthetic calculation interruption"):
        paper.run_paper_cycle(cfg)
    assert paper.load_paper_state(cfg) == before
    assert paper.read_paper_trades() == []
    monkeypatch.setattr(paper, "_evaluate_buy_signal", original_evaluate)
    paper.run_paper_cycle(cfg)
    paper.run_paper_cycle(cfg)
    assert len(paper.read_paper_trades()) == 1
    assert paper.read_paper_trades()[0]["side"] == "BUY"


def test_stale_paper_draft_cannot_append_trade_or_report(isolated_storage, monkeypatch):
    paper, cfg = _paper_fixture(monkeypatch)
    stale = paper.load_paper_state(cfg)
    current = paper.load_paper_state(cfg)
    current["run_count"] += 1
    paper.save_paper_state(current)
    p = __import__("core.persistence", fromlist=["PersistenceConflictError"])
    calls = []
    monkeypatch.setattr(paper, "_append_paper_trade", lambda row: calls.append(("trade", row)))
    monkeypatch.setattr(paper, "append_trade_report", lambda row: calls.append(("report", row)))
    with pytest.raises(p.PersistenceConflictError):
        paper._commit_paper_draft(
            stale, [{"trade_id": "stale-synthetic", "side": "BUY"}],
            [{"trade_id": "stale-synthetic"}], cfg,
        )
    assert calls == []
    assert paper.load_paper_state(cfg)["run_count"] == current["run_count"]
    assert paper.read_paper_trades() == []
