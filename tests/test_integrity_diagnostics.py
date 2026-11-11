from copy import deepcopy

import pytest

from core.integrity_diagnostics import diagnose_paper_integrity


def snapshot():
    """Fictitious PAPER BUY at 10 and SELL at 12, with zero execution cost."""
    state = {"initial_capital": 1000.0, "cash": 1020.0, "equity": 1020.0,
             "realized_pnl": 20.0, "positions": {}, "last_prices": {},
             "history": [{"cash": 1020.0, "equity": 1020.0, "open_positions": 0}]}
    trades = [
        {"trade_id": "FAKE-1", "asset": "FAKE", "side": "BUY", "quantity": 10.0,
         "price": 10.0, "gross_value": 100.0, "cost": 0.0, "cash_after": 900.0, "realized_pnl": 0.0},
        {"trade_id": "FAKE-1", "asset": "FAKE", "side": "SELL", "quantity": 10.0,
         "price": 12.0, "gross_value": 120.0, "cost": 0.0, "cash_after": 1020.0, "realized_pnl": 20.0},
    ]
    return state, trades


def codes(state, trades, complete=True):
    return {item.code for item in diagnose_paper_integrity(state, trades, complete_trade_history=complete)}


def test_valid_round_trip_accepts_repeated_id_for_buy_then_sell_without_mutation():
    state, trades = snapshot()
    before = deepcopy((state, trades))
    assert diagnose_paper_integrity(state, trades, complete_trade_history=True) == []
    assert (state, trades) == before


def test_valid_open_position_reconciles_cash_equity_and_reference():
    state, trades = snapshot()
    state.update(cash=900.0, equity=1010.0, realized_pnl=0.0,
                 positions={"FAKE": {"trade_id": "FAKE-1", "quantity": 10.0,
                                     "avg_price": 10.0, "last_price": 11.0, "gross_entry_value": 100.0}},
                 history=[{"cash": 900.0, "equity": 1010.0, "open_positions": 1}])
    assert codes(state, trades[:1]) == set()


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), None, "oops", True])
def test_invalid_monetary_state_values_are_reported(bad):
    state, trades = snapshot()
    state["cash"] = bad
    assert "invalid_number" in codes(state, trades)


@pytest.mark.parametrize("field", ["quantity", "price", "gross_value", "cost", "cash_after", "realized_pnl"])
def test_nonfinite_trade_fields_are_reported(field):
    state, trades = snapshot()
    trades[0][field] = float("nan")
    assert "invalid_number" in codes(state, trades)


@pytest.mark.parametrize("field", ["quantity", "price", "gross_value", "cost", "cash_after"])
def test_negative_trade_amounts_are_reported(field):
    state, trades = snapshot()
    trades[0][field] = -1.0
    assert "invalid_amount" in codes(state, trades)


def test_negative_realized_pnl_is_valid():
    state, trades = snapshot()
    trades[1].update(price=8.0, gross_value=80.0, cash_after=980.0, realized_pnl=-20.0)
    state.update(cash=980.0, equity=980.0, realized_pnl=-20.0,
                 history=[{"cash": 980.0, "equity": 980.0, "open_positions": 0}])
    assert codes(state, trades) == set()


def test_duplicate_position_assets_and_ids_are_reported():
    state, trades = snapshot()
    position = {"trade_id": "FAKE-1", "quantity": 1.0, "avg_price": 10.0, "last_price": 10.0}
    state["positions"] = {"FAKE": position, " fake ": deepcopy(position)}
    assert "duplicate_position" in codes(state, trades)


def test_duplicate_history_trade_is_reported_but_not_removed():
    state, trades = snapshot()
    trades.append(deepcopy(trades[1]))
    assert "duplicate_trade" in codes(state, trades)
    assert len(trades) == 3


def test_sell_with_unknown_reference_is_reported():
    state, trades = snapshot()
    trades[1]["trade_id"] = "UNKNOWN"
    assert "invalid_reference" in codes(state, trades)


@pytest.mark.parametrize("field,value", [("asset", "OTHER"), ("quantity", 9.0)])
def test_sell_must_reference_buy_asset_and_quantity(field, value):
    state, trades = snapshot()
    trades[1][field] = value
    assert "invalid_reference" in codes(state, trades)


def test_trade_pnl_arithmetic_divergence_is_reported():
    state, trades = snapshot()
    trades[1]["realized_pnl"] = 999.0
    assert "trade_pnl_mismatch" in codes(state, trades)
    assert "ledger_pnl_mismatch" in codes(state, trades)


def test_failed_cash_update_is_detected():
    state, trades = snapshot()
    trades[0]["cash_after"] = 1000.0
    assert "cash_update_mismatch" in codes(state, trades)


def test_interruption_after_history_append_before_state_save_is_detected():
    state, trades = snapshot()
    # Trade persistence succeeded but state still reflects the preceding open BUY.
    state.update(cash=900.0, equity=1000.0, realized_pnl=0.0,
                 positions={"FAKE": {"trade_id": "FAKE-1", "quantity": 10.0,
                                     "avg_price": 10.0, "last_price": 10.0}},
                 history=[{"cash": 900.0, "equity": 1000.0, "open_positions": 1}])
    assert {"ledger_cash_mismatch", "ledger_pnl_mismatch", "ledger_positions_mismatch"} <= codes(state, trades)


def test_missing_trade_after_state_commit_is_detected():
    state, trades = snapshot()
    assert {"ledger_cash_mismatch", "ledger_pnl_mismatch", "ledger_positions_mismatch"} <= codes(state, trades[:1])


def test_limited_history_does_not_claim_global_reconciliation():
    state, trades = snapshot()
    assert diagnose_paper_integrity(state, trades[1:]) == []
    assert "invalid_reference" in codes(state, trades[1:], complete=True)


def test_equity_and_latest_snapshot_divergences_are_reported():
    state, trades = snapshot()
    state["equity"] = 5000.0
    assert {"equity_mismatch", "history_mismatch"} <= codes(state, trades)


def test_history_nonfinite_numbers_and_fractional_position_counts_are_reported():
    state, trades = snapshot()
    state["history"] = [{"cash": float("inf"), "equity": 1020.0, "open_positions": 0.5}]
    assert {"invalid_number", "invalid_position_count", "history_mismatch"} <= codes(state, trades)


@pytest.mark.parametrize("target", ["positions", "history", "last_prices"])
def test_malformed_state_collections_produce_findings(target):
    state, trades = snapshot()
    state[target] = None
    assert "invalid_structure" in codes(state, trades)


def test_malformed_rows_are_diagnosed_without_throwing():
    state, trades = snapshot()
    trades.append(None)
    state["history"].append(None)
    state["positions"] = {"FAKE": None}
    assert "invalid_structure" in codes(state, trades)


def test_fresh_empty_history_is_valid():
    state = {"initial_capital": 1000.0, "cash": 1000.0, "equity": 1000.0,
             "realized_pnl": 0.0, "positions": {}, "last_prices": {}, "history": []}
    assert codes(state, []) == set()


def test_six_decimal_price_rounding_does_not_create_false_pnl_finding():
    state, trades = snapshot()
    # Engine computes PnL using full prices but persists them to six decimals.
    trades[0].update(price=1.0, quantity=10000.0, gross_value=10000.0, cash_after=10000.0)
    trades[1].update(price=1.000001, quantity=10000.0, gross_value=10000.01,
                     cash_after=20000.01, realized_pnl=0.02)
    state.update(initial_capital=20000.0, cash=20000.01, equity=20000.01, realized_pnl=0.02,
                 history=[])
    assert "trade_pnl_mismatch" not in codes(state, trades)


def test_current_paper_engine_output_satisfies_diagnostic_contract(isolated_storage, monkeypatch):
    """Check the diagnostic against real engine output, with synthetic quotes only."""
    from tests.conftest import fake_market_result, load_module
    from tests.test_trade_reports import _fake_prices

    monkeypatch.setenv("BROKER_MODE", "paper")
    monkeypatch.setenv("BROKER_PROVIDER", "paper")
    paper = load_module("paper_trading_engine")
    monkeypatch.setattr(paper, "load_market_data_result",
                        lambda symbols, config, requested_by="worker_cycle": fake_market_result(
                            {symbol: _fake_prices() for symbol in symbols}, requested_by))
    monkeypatch.setattr(paper, "_evaluate_buy_signal", lambda latest, config: {
        "buy": True, "score": 0.91, "rejection_reasons": [],
        "trend_ok": True, "score_ok": True, "rsi_ok": True, "atr_ok": True,
        "structure_ok": True, "momentum_ok": True, "trend_alignment": "aligned",
    })
    cfg = paper.PaperTradingConfig(custom_tickers=["FAKE"], initial_capital=1000.0)
    first = paper.run_paper_cycle(cfg)
    assert first["trades_executed"] == 1
    assert codes(paper.load_paper_state(), paper.read_paper_trades()) == set()
    cfg.allow_new_entries = False
    monkeypatch.setattr(paper, "_should_sell", lambda *args: (True, "synthetic_exit"))
    second = paper.run_paper_cycle(cfg)
    assert second["trades_executed"] == 1
    assert codes(paper.load_paper_state(), paper.read_paper_trades()) == set()
