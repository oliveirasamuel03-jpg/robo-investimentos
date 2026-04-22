from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from tests.conftest import load_module


class _FakeUrlopenResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def read(self) -> bytes:
        import json

        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


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
            "data_source": "market",
        }
    )


def test_run_trader_cycle_smoke_with_synthetic_prices(isolated_storage, monkeypatch):
    state_store = load_module("core.state_store")
    paper_engine = load_module("paper_trading_engine")
    trader_engine = load_module("engines.trader_engine")

    monkeypatch.setattr(
        paper_engine,
        "load_market_data_map",
        lambda symbols, config: {str(symbol).upper(): _fake_prices() for symbol in symbols},
    )
    monkeypatch.setattr(
        paper_engine,
        "_evaluate_buy_signal",
        lambda latest, config: {
            "buy": True,
            "score": 0.95,
            "trend_ok": True,
            "score_ok": True,
            "rsi_ok": True,
            "atr_ok": True,
            "structure_ok": True,
            "momentum_ok": True,
            "trend_alignment": "aligned",
            "rejection_reasons": [],
        },
    )
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


def test_market_data_configures_yfinance_cache_in_writable_storage(isolated_storage, monkeypatch):
    configured_paths: list[str] = []

    monkeypatch.setattr(
        yf,
        "set_tz_cache_location",
        lambda path: configured_paths.append(str(path)),
    )

    market_data = load_module("core.market_data")

    assert configured_paths
    assert Path(configured_paths[-1]).name == "yfinance"
    assert Path(configured_paths[-1]).exists()
    assert market_data._YFINANCE_CACHE_DIR.exists()


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


def test_run_paper_cycle_skips_new_entries_when_prices_are_fallback(isolated_storage, monkeypatch):
    paper_engine = load_module("paper_trading_engine")

    def fallback_prices():
        df = _fake_prices().copy()
        df["data_source"] = "fallback"
        return df

    monkeypatch.setattr(
        paper_engine,
        "load_market_data_map",
        lambda symbols, config: {str(symbol).upper(): fallback_prices() for symbol in symbols},
    )
    monkeypatch.setattr(
        paper_engine,
        "_evaluate_buy_signal",
        lambda latest, config: {
            "buy": True,
            "score": 0.95,
            "trend_ok": True,
            "score_ok": True,
            "rsi_ok": True,
            "atr_ok": True,
            "structure_ok": True,
            "momentum_ok": True,
            "trend_alignment": "aligned",
            "rejection_reasons": [],
        },
    )

    result = paper_engine.run_paper_cycle(
        paper_engine.PaperTradingConfig(
            custom_tickers=["AAPL"],
            allow_new_entries=True,
            history_limit=50,
        )
    )

    assert result["trades_executed"] == 0
    assert result["state"]["positions"] == {}
    assert result["signals"]
    assert result["signals"][0]["data_source"] == "fallback"
    assert result["signals"][0]["buy"] is False


def test_run_paper_cycle_blocks_new_entries_when_daily_loss_guard_is_active(isolated_storage, monkeypatch):
    paper_engine = load_module("paper_trading_engine")

    monkeypatch.setattr(
        paper_engine,
        "load_market_data_map",
        lambda symbols, config: {str(symbol).upper(): _fake_prices() for symbol in symbols},
    )
    monkeypatch.setattr(
        paper_engine,
        "_evaluate_buy_signal",
        lambda latest, config: {
            "buy": True,
            "score": 0.96,
            "trend_ok": True,
            "score_ok": True,
            "rsi_ok": True,
            "atr_ok": True,
            "structure_ok": True,
            "momentum_ok": True,
            "trend_alignment": "aligned",
            "rejection_reasons": [],
        },
    )

    result = paper_engine.run_paper_cycle(
        paper_engine.PaperTradingConfig(
            custom_tickers=["AAPL"],
            allow_new_entries=True,
            daily_loss_limit_brl=100.0,
            daily_realized_pnl_brl=-120.0,
            daily_loss_day_key=paper_engine._utc_day_key(),
            history_limit=50,
        )
    )

    assert result["trades_executed"] == 0
    assert result["entries_blocked_by_daily_loss"] is True
    assert result["signals"]
    assert result["signals"][0]["buy"] is False
    assert "daily_loss_guard" in result["signals"][0]["rejection_reasons"]


def test_run_paper_cycle_does_not_close_open_position_when_market_data_is_not_live(isolated_storage, monkeypatch):
    paper_engine = load_module("paper_trading_engine")

    def cached_prices():
        df = _fake_prices().copy()
        df["data_source"] = "cached"
        return df

    monkeypatch.setattr(
        paper_engine,
        "load_market_data_map",
        lambda symbols, config: {str(symbol).upper(): cached_prices() for symbol in symbols},
    )
    monkeypatch.setattr(paper_engine, "_should_sell", lambda position, latest, holding_minutes, config: (True, "stop loss"))

    state = paper_engine.load_paper_state(paper_engine.PaperTradingConfig(initial_capital=10000.0))
    state["positions"] = {
        "AAPL": {
            "trade_id": "AAPL-test",
            "profile": "Equilibrado",
            "quantity": 1.0,
            "entry_price": 100.0,
            "avg_price": 100.0,
            "last_price": 100.0,
            "peak_price": 101.0,
            "atr_pct": 0.02,
            "trailing_stop": 98.0,
            "gross_entry_value": 100.0,
            "entry_score": 0.8,
            "rsi_entry": 55.0,
            "ma20_entry": 99.0,
            "ma50_entry": 98.0,
            "holding_minutes_limit": 60,
            "opened_at": "2026-04-19T20:00:00+00:00",
            "updated_at": "2026-04-19T20:00:00+00:00",
        }
    }
    paper_engine.save_paper_state(state)

    result = paper_engine.run_paper_cycle(
        paper_engine.PaperTradingConfig(
            custom_tickers=["AAPL"],
            allow_new_entries=True,
            history_limit=50,
        )
    )

    assert result["trades_executed"] == 0
    assert "AAPL" in result["state"]["positions"]


def test_run_paper_cycle_keeps_managing_open_position_outside_watchlist(isolated_storage, monkeypatch):
    paper_engine = load_module("paper_trading_engine")

    monkeypatch.setattr(
        paper_engine,
        "load_market_data_result",
        lambda symbols, config, requested_by="worker_cycle": {
            "frames": {str(symbol).upper(): _fake_prices() for symbol in symbols},
            "status": {
                "provider": "yahoo",
                "status": "healthy",
                "last_sync_at": "2026-04-21T00:00:00+00:00",
                "last_source": "market",
                "source_breakdown": {"market": len(symbols), "cached": 0, "fallback": 0, "unknown": 0},
                "symbols": [str(symbol).upper() for symbol in symbols],
                "requested_by": requested_by,
                "last_error": "",
            },
        },
    )
    monkeypatch.setattr(paper_engine, "_should_sell", lambda position, latest, holding_minutes, config: (False, ""))
    monkeypatch.setattr(
        paper_engine,
        "_evaluate_buy_signal",
        lambda latest, config: {
            "buy": False,
            "score": 0.5,
            "trend_ok": False,
            "score_ok": False,
            "rsi_ok": True,
            "atr_ok": True,
            "structure_ok": True,
            "momentum_ok": True,
            "trend_alignment": "aligned",
            "rejection_reasons": ["weak_score"],
        },
    )

    state = paper_engine.load_paper_state(paper_engine.PaperTradingConfig(initial_capital=10000.0))
    state["positions"] = {
        "AAPL": {
            "trade_id": "AAPL-test",
            "profile": "Equilibrado",
            "quantity": 1.0,
            "entry_price": 100.0,
            "avg_price": 100.0,
            "last_price": 100.0,
            "peak_price": 101.0,
            "atr_pct": 0.02,
            "trailing_stop": 98.0,
            "gross_entry_value": 100.0,
            "entry_score": 0.8,
            "rsi_entry": 55.0,
            "ma20_entry": 99.0,
            "ma50_entry": 98.0,
            "holding_minutes_limit": 60,
            "opened_at": "2026-04-19T20:00:00+00:00",
            "updated_at": "2026-04-19T20:00:00+00:00",
        }
    }
    paper_engine.save_paper_state(state)

    result = paper_engine.run_paper_cycle(
        paper_engine.PaperTradingConfig(
            custom_tickers=["BTC-USD"],
            allow_new_entries=True,
            history_limit=50,
        )
    )

    assert "AAPL" in result["state"]["positions"]
    assert result["state"]["positions"]["AAPL"]["updated_at"] != "2026-04-19T20:00:00+00:00"
    assert "AAPL" in result["state"]["last_prices"]


def test_load_market_data_map_reuses_cache_between_cycles(isolated_storage, monkeypatch):
    market_data = load_module("core.market_data")
    paper_engine = load_module("paper_trading_engine")

    calls = {"count": 0}

    def fake_download(*args, **kwargs):
        calls["count"] += 1
        return pd.DataFrame(
            {
                "Datetime": pd.date_range("2025-01-01", periods=120, freq="h"),
                "Open": _fake_prices()["open"].values,
                "High": _fake_prices()["high"].values,
                "Low": _fake_prices()["low"].values,
                "Close": _fake_prices()["close"].values,
                "Volume": _fake_prices()["volume"].values,
            }
        )

    monkeypatch.setattr(market_data.yf, "download", fake_download)

    config = paper_engine.PaperTradingConfig(custom_tickers=["AAPL"], history_limit=50)
    first = paper_engine.load_market_data_map(["AAPL"], config)
    second = paper_engine.load_market_data_map(["AAPL"], config)

    assert "AAPL" in first
    assert "AAPL" in second
    assert calls["count"] == 1


def test_fetch_market_data_map_uses_twelvedata_as_primary_provider(isolated_storage, monkeypatch):
    market_data = load_module("core.market_data")

    monkeypatch.setattr(market_data, "TWELVEDATA_API_KEY", "td-demo")
    monkeypatch.setattr(
        market_data.urllib_request,
        "urlopen",
        lambda request, timeout=20: _FakeUrlopenResponse(
            {
                "status": "ok",
                "meta": {"symbol": "BTC/USD", "interval": "1h", "type": "Digital Currency"},
                "values": [
                    {
                        "datetime": "2026-04-22 00:00:00",
                        "open": "100.0",
                        "high": "101.0",
                        "low": "99.5",
                        "close": "100.5",
                        "volume": "1200",
                    },
                    {
                        "datetime": "2026-04-22 01:00:00",
                        "open": "100.5",
                        "high": "102.0",
                        "low": "100.0",
                        "close": "101.5",
                        "volume": "1300",
                    },
                ],
            }
        ),
    )
    monkeypatch.setattr(market_data.yf, "download", lambda *args, **kwargs: pd.DataFrame())

    result = market_data.fetch_market_data_map(
        ["BTC-USD"],
        period="6mo",
        interval="1h",
        history_limit=50,
        provider="twelvedata",
        allow_stale=False,
        requested_by="worker_cycle",
    )

    assert result.status["provider"] == "twelvedata"
    assert result.status["configured_provider"] == "twelvedata"
    assert result.status["fallback_provider"] == "yahoo"
    assert result.status["feed_status"] == "LIVE"
    assert result.status["last_source"] == "market"
    assert result.status["provider_breakdown"]["twelvedata"] == 1
    assert result.status["provider_diagnostics"]["twelvedata"]["api_key_present"] is True
    assert result.status["provider_diagnostics"]["twelvedata"]["request_attempted"] is True
    assert result.status["provider_diagnostics"]["twelvedata"]["success_count"] == 1
    assert result.status["provider_diagnostics"]["twelvedata"]["last_stage"] == "success"
    assert result.frames["BTC-USD"]["provider_name"].iloc[-1] == "twelvedata"


def test_twelvedata_uses_provider_specific_fresh_cache_window(isolated_storage, monkeypatch):
    market_data = load_module("core.market_data")

    market_data._MARKET_DATA_CACHE.clear()
    market_data._store_cached_frames(
        ["BTC-USD"],
        period="6mo",
        interval="1h",
        history_limit=50,
        provider="twelvedata",
        frames={"BTC-USD": _fake_prices().assign(provider_name="twelvedata")},
    )

    fresh = market_data._cached_frames(
        ["BTC-USD"],
        period="6mo",
        interval="1h",
        history_limit=50,
        provider="twelvedata",
        allow_stale=False,
    )

    assert "BTC-USD" in fresh
    assert market_data._effective_cache_ttl_seconds("twelvedata") >= 900


def test_fetch_market_data_map_falls_back_to_yahoo_when_twelvedata_fails(isolated_storage, monkeypatch):
    market_data = load_module("core.market_data")

    monkeypatch.setattr(market_data, "TWELVEDATA_API_KEY", "td-demo")
    monkeypatch.setattr(
        market_data.urllib_request,
        "urlopen",
        lambda request, timeout=20: (_ for _ in ()).throw(RuntimeError("twelvedata offline")),
    )
    monkeypatch.setattr(
        market_data.yf,
        "download",
        lambda *args, **kwargs: pd.DataFrame(
            {
                "Datetime": pd.date_range("2025-01-01", periods=120, freq="h"),
                "Open": _fake_prices()["open"].values,
                "High": _fake_prices()["high"].values,
                "Low": _fake_prices()["low"].values,
                "Close": _fake_prices()["close"].values,
                "Volume": _fake_prices()["volume"].values,
            }
        ),
    )

    result = market_data.fetch_market_data_map(
        ["BTC-USD"],
        period="6mo",
        interval="1h",
        history_limit=50,
        provider="twelvedata",
        allow_stale=False,
        requested_by="worker_cycle",
    )

    assert result.status["provider"] == "yahoo"
    assert result.status["feed_status"] == "LIVE"
    assert "twelvedata:" in result.status["last_error"].lower()
    assert result.status["provider_breakdown"]["yahoo"] == 1
    assert result.frames["BTC-USD"]["provider_name"].iloc[-1] == "yahoo"


def test_fetch_market_data_map_reports_missing_twelvedata_key_before_request(isolated_storage, monkeypatch):
    market_data = load_module("core.market_data")

    monkeypatch.setattr(market_data, "TWELVEDATA_API_KEY", "")
    request_calls = {"count": 0}

    def fail_if_called(*args, **kwargs):
        request_calls["count"] += 1
        raise AssertionError("urlopen nao deveria ser chamado sem chave configurada")

    monkeypatch.setattr(market_data.urllib_request, "urlopen", fail_if_called)
    monkeypatch.setattr(market_data.yf, "download", lambda *args, **kwargs: pd.DataFrame())

    result = market_data.fetch_market_data_map(
        ["BTC-USD"],
        period="6mo",
        interval="1h",
        history_limit=50,
        provider="twelvedata",
        allow_stale=False,
        requested_by="worker_cycle",
    )

    td_diag = result.status["provider_diagnostics"]["twelvedata"]

    assert request_calls["count"] == 0
    assert td_diag["api_key_present"] is False
    assert td_diag["request_attempted"] is False
    assert td_diag["last_stage"] == "missing_api_key"
    assert "TWELVEDATA_API_KEY nao configurada" in td_diag["last_error"]


def test_fetch_market_data_map_uses_synthetic_fallback_when_all_providers_fail(isolated_storage, monkeypatch):
    market_data = load_module("core.market_data")

    monkeypatch.setattr(market_data, "TWELVEDATA_API_KEY", "td-demo")
    monkeypatch.setattr(
        market_data.urllib_request,
        "urlopen",
        lambda request, timeout=20: (_ for _ in ()).throw(RuntimeError("twelvedata offline")),
    )
    monkeypatch.setattr(market_data.yf, "download", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("yahoo offline")))

    result = market_data.fetch_market_data_map(
        ["BTC-USD"],
        period="6mo",
        interval="1h",
        history_limit=50,
        provider="twelvedata",
        allow_stale=False,
        requested_by="worker_cycle",
    )

    assert result.status["provider"] == "synthetic"
    assert result.status["feed_status"] == "FALLBACK"
    assert result.status["last_source"] == "fallback"
    assert result.status["provider_breakdown"]["synthetic"] == 1
    assert result.frames["BTC-USD"]["provider_name"].iloc[-1] == "synthetic"


def test_fetch_market_data_map_surfaces_twelvedata_credit_limit_errors(isolated_storage, monkeypatch):
    market_data = load_module("core.market_data")

    class _FakeHttpError(Exception):
        def __init__(self, code: int, payload: dict):
            self.code = code
            self._payload = payload

        def read(self):
            import json

            return json.dumps(self._payload).encode("utf-8")

    market_data.TWELVEDATA_API_KEY = "td-demo"
    monkeypatch.setattr(
        market_data.urllib_request,
        "urlopen",
        lambda request, timeout=20: (_ for _ in ()).throw(
            market_data.urllib_error.HTTPError(
                url="https://api.twelvedata.com/time_series",
                code=429,
                msg="Too Many Requests",
                hdrs=None,
                fp=io.BytesIO(b'{"code":429,"status":"error","message":"API request limit reached for your key."}'),
            )
        ),
    )
    monkeypatch.setattr(market_data.yf, "download", lambda *args, **kwargs: pd.DataFrame())

    result = market_data.fetch_market_data_map(
        ["BTC-USD"],
        period="6mo",
        interval="1h",
        history_limit=50,
        provider="twelvedata",
        allow_stale=False,
        requested_by="worker_cycle",
    )

    assert "429" in result.status["last_error"]
    assert "limit" in result.status["last_error"].lower()


def test_run_trader_cycle_updates_market_data_state(isolated_storage, monkeypatch):
    state_store = load_module("core.state_store")
    paper_engine = load_module("paper_trading_engine")
    trader_engine = load_module("engines.trader_engine")

    monkeypatch.setattr(
        paper_engine,
        "load_market_data_result",
        lambda symbols, config, requested_by="worker_cycle": {
            "frames": {str(symbol).upper(): _fake_prices() for symbol in symbols},
            "status": {
                "provider": "yahoo",
                "status": "healthy",
                "last_sync_at": "2026-04-20T01:00:00+00:00",
                "last_source": "market",
                "source_breakdown": {"market": len(symbols), "cached": 0, "fallback": 0, "unknown": 0},
                "symbols": [str(symbol).upper() for symbol in symbols],
                "requested_by": requested_by,
                "last_error": "",
            },
        },
    )
    monkeypatch.setattr(
        paper_engine,
        "_evaluate_buy_signal",
        lambda latest, config: {
            "buy": False,
            "score": 0.55,
            "trend_ok": False,
            "score_ok": False,
            "rsi_ok": True,
            "atr_ok": True,
            "structure_ok": True,
            "momentum_ok": True,
            "trend_alignment": "countertrend",
            "rejection_reasons": ["against_trend", "weak_score"],
        },
    )
    monkeypatch.setattr(paper_engine, "_should_sell", lambda position, latest, holding_minutes, config: (False, ""))

    state = state_store.load_bot_state()
    state["bot_status"] = "RUNNING"
    state["trader"]["watchlist"] = ["AAPL", "MSFT"]
    state_store.save_bot_state(state)

    trader_engine.run_trader_cycle()

    updated_state = state_store.load_bot_state()
    assert updated_state["market_data"]["status"] == "healthy"
    assert updated_state["market_data"]["feed_status"] == "LIVE"
    assert updated_state["market_data"]["last_source"] == "market"
