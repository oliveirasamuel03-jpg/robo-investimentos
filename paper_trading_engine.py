from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data_engine import load_data
from ml_engine import MLEngineConfig, EnsembleProbabilityModel, build_feature_panel
from risk_engine import RiskConfig, apply_portfolio_risk_overlay
from strategy_engine import StrategyConfig, combined_market_filter, generate_target_weights


APP_DIR = Path(__file__).resolve().parent
PAPER_STATE_PATH = APP_DIR / "paper_state.json"
PAPER_LOG_PATH = APP_DIR / "paper_trades.jsonl"
PAPER_EQUITY_PATH = APP_DIR / "paper_equity.jsonl"


@dataclass
class PaperTradingConfig:
    start_date: str = "2016-01-01"
    initial_capital: float = 10000.0

    probability_threshold: float = 0.50
    top_n: int = 2
    execution_slippage_rate: float = 0.0005
    execution_fee_rate: float = 0.0005

    use_regime_filter: bool = False
    use_volatility_filter: bool = False
    vol_threshold: float = 0.03
    cash_threshold: float = 0.50
    target_portfolio_vol: float = 0.08

    max_weight_per_asset: float = 0.35
    max_gross_exposure: float = 1.00
    min_trade_notional: float = 25.0

    benchmark: str = "SPY"
    history_limit: int = 2000


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _round_money(x: float) -> float:
    return float(round(float(x), 6))


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return default
        return float(x)
    except Exception:
        return default


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def ensure_paper_files(config: PaperTradingConfig | None = None) -> None:
    if config is None:
        config = PaperTradingConfig()

    if not PAPER_STATE_PATH.exists():
        initial_state = {
            "created_at": _utc_now_iso(),
            "updated_at": _utc_now_iso(),
            "cash": _round_money(config.initial_capital),
            "positions": {},
            "last_run_at": None,
            "last_signal_date": None,
            "last_prices": {},
            "run_count": 0,
            "equity": _round_money(config.initial_capital),
            "config": asdict(config),
        }
        _write_json(PAPER_STATE_PATH, initial_state)

    PAPER_LOG_PATH.touch(exist_ok=True)
    PAPER_EQUITY_PATH.touch(exist_ok=True)


def load_paper_state() -> dict:
    ensure_paper_files()
    return json.loads(PAPER_STATE_PATH.read_text(encoding="utf-8"))


def save_paper_state(state: dict) -> None:
    state["updated_at"] = _utc_now_iso()
    _write_json(PAPER_STATE_PATH, state)


def reset_paper_state(config: PaperTradingConfig | None = None) -> dict:
    if config is None:
        config = PaperTradingConfig()

    state = {
        "created_at": _utc_now_iso(),
        "updated_at": _utc_now_iso(),
        "cash": _round_money(config.initial_capital),
        "positions": {},
        "last_run_at": None,
        "last_signal_date": None,
        "last_prices": {},
        "run_count": 0,
        "equity": _round_money(config.initial_capital),
        "config": asdict(config),
    }
    _write_json(PAPER_STATE_PATH, state)
    return state


def get_current_equity(state: dict, latest_prices: dict[str, float]) -> float:
    cash = _safe_float(state.get("cash", 0.0))
    positions = state.get("positions", {}) or {}

    equity = cash
    for asset, pos in positions.items():
        qty = _safe_float(pos.get("quantity", 0.0))
        px = _safe_float(latest_prices.get(asset), _safe_float(pos.get("last_price", 0.0)))
        equity += qty * px

    return _round_money(equity)


def _build_live_signal_table(
    prices: pd.DataFrame,
    config: PaperTradingConfig,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    feature_panel = build_feature_panel(prices, MLEngineConfig())
    feature_panel["date"] = pd.to_datetime(feature_panel["date"])

    unique_dates = pd.Index(sorted(feature_panel["date"].unique()))
    if len(unique_dates) < 3:
        raise ValueError("Not enough dates to build live features")

    signal_date = pd.Timestamp(unique_dates[-1])
    train_df = feature_panel[feature_panel["date"] < signal_date].copy()
    live_df = feature_panel[feature_panel["date"] == signal_date].copy()

    if train_df.empty or live_df.empty:
        raise ValueError("Could not split train/live feature sets")

    feature_cols = [c for c in feature_panel.columns if c not in {"date", "asset", "target", "target_ret"}]
    if not feature_cols:
        raise ValueError("No feature columns found")

    X_train = train_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_train = train_df["target"].astype(int)

    if y_train.nunique() < 2:
        raise ValueError("Training target has only one class")

    X_live = live_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    model = EnsembleProbabilityModel(MLEngineConfig())
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_live)[:, 1]

    pred_df = live_df[["date", "asset"]].copy()
    pred_df["probability"] = proba
    pred_df = (
        pred_df.groupby(["date", "asset"], as_index=False)["probability"]
        .mean()
        .sort_values(["date", "probability"], ascending=[True, False])
        .reset_index(drop=True)
    )

    return pred_df, signal_date


def _build_target_weights_for_live_date(
    prices: pd.DataFrame,
    pred_df: pd.DataFrame,
    config: PaperTradingConfig,
) -> pd.DataFrame:
    strategy_config = StrategyConfig(
        probability_threshold=config.probability_threshold,
        top_n=config.top_n,
        max_weight_per_asset=config.max_weight_per_asset,
        cash_threshold=config.cash_threshold,
        target_portfolio_vol=config.target_portfolio_vol,
    )

    risk_config = RiskConfig(
        max_weight_per_asset=config.max_weight_per_asset,
        max_gross_exposure=config.max_gross_exposure,
        max_net_exposure=1.0,
        max_drawdown_circuit_breaker=0.12,
        recovery_exposure=0.35,
        min_nav_fraction=0.70,
    )

    filters = combined_market_filter(
        prices=prices,
        benchmark=config.benchmark,
        vol_threshold=config.vol_threshold,
        use_regime_filter=config.use_regime_filter,
        use_volatility_filter=config.use_volatility_filter,
    ).reset_index().rename(columns={"index": "date"})

    weighted = generate_target_weights(
        predictions=pred_df,
        prices=prices,
        config=strategy_config,
    ).copy()

    weighted["date"] = pd.to_datetime(weighted["date"])

    weighted = weighted.merge(
        filters[["date", "regime", "vol_filter", "trade_allowed"]],
        on="date",
        how="left",
    )
    weighted["trade_allowed"] = weighted["trade_allowed"].fillna(False)
    weighted.loc[~weighted["trade_allowed"], "weight"] = 0.0

    out_rows = []
    for dt, group in weighted.groupby("date", sort=True):
        base_weights = group.set_index("asset")["weight"]
        safe_weights = apply_portfolio_risk_overlay(
            base_weights,
            equity_curve=None,
            config=risk_config,
        )
        tmp = group.copy()
        tmp["weight"] = tmp["asset"].map(safe_weights).fillna(0.0)
        out_rows.append(tmp[["date", "asset", "weight"]])

    out = pd.concat(out_rows, ignore_index=True)
    out["weight"] = pd.to_numeric(out["weight"], errors="coerce").fillna(0.0)
    return out


def _latest_price_map(prices: pd.DataFrame, signal_date: pd.Timestamp) -> dict[str, float]:
    row = prices.loc[signal_date]
    return {str(asset): _safe_float(px) for asset, px in row.items() if pd.notna(px)}


def _current_position_values(state: dict, latest_prices: dict[str, float]) -> dict[str, float]:
    values: dict[str, float] = {}
    for asset, pos in (state.get("positions", {}) or {}).items():
        qty = _safe_float(pos.get("quantity", 0.0))
        px = _safe_float(latest_prices.get(asset), _safe_float(pos.get("last_price", 0.0)))
        values[asset] = qty * px
    return values


def _target_notional_map(
    equity: float,
    weight_table: pd.DataFrame,
) -> dict[str, float]:
    targets: dict[str, float] = {}
    for _, row in weight_table.iterrows():
        asset = str(row["asset"])
        w = _safe_float(row["weight"])
        if w > 0:
            targets[asset] = equity * w
    return targets


def _sell_first(
    state: dict,
    latest_prices: dict[str, float],
    target_notional: dict[str, float],
    config: PaperTradingConfig,
) -> list[dict]:
    logs: list[dict] = []
    cash = _safe_float(state.get("cash", 0.0))
    positions = state.get("positions", {}) or {}

    for asset in list(positions.keys()):
        pos = positions[asset]
        qty = _safe_float(pos.get("quantity", 0.0))
        price = _safe_float(latest_prices.get(asset), _safe_float(pos.get("last_price", 0.0)))
        current_value = qty * price
        desired_value = _safe_float(target_notional.get(asset, 0.0))

        delta_value = desired_value - current_value
        if delta_value >= -config.min_trade_notional:
            pos["last_price"] = price
            continue

        sell_value = min(current_value, abs(delta_value))
        sell_qty = sell_value / price if price > 0 else 0.0

        if sell_qty <= 0:
            pos["last_price"] = price
            continue

        gross = sell_qty * price
        cost = gross * (config.execution_fee_rate + config.execution_slippage_rate)
        net_cash = gross - cost

        new_qty = max(0.0, qty - sell_qty)
        cash += net_cash

        logs.append(
            {
                "timestamp": _utc_now_iso(),
                "asset": asset,
                "side": "SELL",
                "quantity": _round_money(sell_qty),
                "price": _round_money(price),
                "gross_value": _round_money(gross),
                "cost": _round_money(cost),
                "cash_after": _round_money(cash),
            }
        )

        if new_qty <= 1e-10:
            positions.pop(asset, None)
        else:
            pos["quantity"] = _round_money(new_qty)
            pos["last_price"] = _round_money(price)

    state["cash"] = _round_money(cash)
    state["positions"] = positions
    return logs


def _buy_second(
    state: dict,
    latest_prices: dict[str, float],
    target_notional: dict[str, float],
    config: PaperTradingConfig,
) -> list[dict]:
    logs: list[dict] = []
    cash = _safe_float(state.get("cash", 0.0))
    positions = state.get("positions", {}) or {}

    for asset, desired_value in target_notional.items():
        price = _safe_float(latest_prices.get(asset), 0.0)
        if price <= 0:
            continue

        pos = positions.get(asset, {"quantity": 0.0, "last_price": price})
        qty = _safe_float(pos.get("quantity", 0.0))
        current_value = qty * price

        delta_value = desired_value - current_value
        if delta_value <= config.min_trade_notional:
            pos["last_price"] = _round_money(price)
            positions[asset] = pos
            continue

        max_affordable_gross = cash / (1.0 + config.execution_fee_rate + config.execution_slippage_rate)
        gross_to_buy = min(delta_value, max_affordable_gross)

        if gross_to_buy < config.min_trade_notional:
            pos["last_price"] = _round_money(price)
            positions[asset] = pos
            continue

        buy_qty = gross_to_buy / price
        gross = buy_qty * price
        cost = gross * (config.execution_fee_rate + config.execution_slippage_rate)
        total_cash = gross + cost

        if total_cash > cash:
            continue

        new_qty = qty + buy_qty
        cash -= total_cash

        positions[asset] = {
            "quantity": _round_money(new_qty),
            "last_price": _round_money(price),
        }

        logs.append(
            {
                "timestamp": _utc_now_iso(),
                "asset": asset,
                "side": "BUY",
                "quantity": _round_money(buy_qty),
                "price": _round_money(price),
                "gross_value": _round_money(gross),
                "cost": _round_money(cost),
                "cash_after": _round_money(cash),
            }
        )

    state["cash"] = _round_money(cash)
    state["positions"] = positions
    return logs


def _mark_to_market(state: dict, latest_prices: dict[str, float]) -> None:
    for asset, pos in (state.get("positions", {}) or {}).items():
        if asset in latest_prices:
            pos["last_price"] = _round_money(latest_prices[asset])


def run_paper_cycle(config: PaperTradingConfig | None = None) -> dict:
    if config is None:
        config = PaperTradingConfig()

    ensure_paper_files(config)
    state = load_paper_state()

    prices = load_data(start=config.start_date, history_limit=config.history_limit)
    if prices.empty:
        raise ValueError("No prices loaded")

    pred_df, signal_date = _build_live_signal_table(prices, config)
    weight_table = _build_target_weights_for_live_date(prices, pred_df, config)

    latest_prices = _latest_price_map(prices, signal_date)
    _mark_to_market(state, latest_prices)

    equity_before = get_current_equity(state, latest_prices)
    target_notional = _target_notional_map(equity_before, weight_table)

    sell_logs = _sell_first(state, latest_prices, target_notional, config)
    buy_logs = _buy_second(state, latest_prices, target_notional, config)
    _mark_to_market(state, latest_prices)

    equity_after = get_current_equity(state, latest_prices)

    state["last_run_at"] = _utc_now_iso()
    state["last_signal_date"] = str(signal_date.date())
    state["last_prices"] = {k: _round_money(v) for k, v in latest_prices.items()}
    state["run_count"] = int(state.get("run_count", 0)) + 1
    state["equity"] = _round_money(equity_after)
    state["config"] = asdict(config)

    save_paper_state(state)

    for row in sell_logs + buy_logs:
        _append_jsonl(PAPER_LOG_PATH, row)

    _append_jsonl(
        PAPER_EQUITY_PATH,
        {
            "timestamp": _utc_now_iso(),
            "signal_date": str(signal_date.date()),
            "equity_before": _round_money(equity_before),
            "equity_after": _round_money(equity_after),
            "cash": _round_money(state.get("cash", 0.0)),
            "positions_count": len(state.get("positions", {})),
            "run_count": state["run_count"],
        },
    )

    current_positions = []
    for asset, pos in (state.get("positions", {}) or {}).items():
        px = _safe_float(latest_prices.get(asset), _safe_float(pos.get("last_price", 0.0)))
        qty = _safe_float(pos.get("quantity", 0.0))
        current_positions.append(
            {
                "asset": asset,
                "quantity": _round_money(qty),
                "last_price": _round_money(px),
                "market_value": _round_money(qty * px),
            }
        )

    return {
        "signal_date": str(signal_date.date()),
        "equity_before": _round_money(equity_before),
        "equity_after": _round_money(equity_after),
        "cash": _round_money(state.get("cash", 0.0)),
        "positions": current_positions,
        "trades_executed": len(sell_logs) + len(buy_logs),
        "run_count": state["run_count"],
        "weights": weight_table.sort_values(["date", "weight"], ascending=[True, False]).to_dict(orient="records"),
    }


def read_paper_trades(limit: int = 100) -> list[dict]:
    ensure_paper_files()
    lines = PAPER_LOG_PATH.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(x) for x in lines if x.strip()]
    return rows[-limit:]


def read_paper_equity(limit: int = 200) -> pd.DataFrame:
    ensure_paper_files()
    lines = PAPER_EQUITY_PATH.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(x) for x in lines if x.strip()]
    if not rows:
        return pd.DataFrame(columns=["timestamp", "signal_date", "equity_after", "cash", "positions_count", "run_count"])
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.tail(limit).reset_index(drop=True)