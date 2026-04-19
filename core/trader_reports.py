from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from core.config import TRADER_REPORTS_COLUMNS, TRADER_REPORTS_FILE
from core.persistence import append_table_row, read_table, replace_table


def _safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_iso(value: Any) -> str | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value)).isoformat()
    except ValueError:
        return str(value)


def _duration_minutes(opened_at: Any, closed_at: Any) -> int | None:
    if not opened_at or not closed_at:
        return None
    try:
        opened = datetime.fromisoformat(str(opened_at))
        closed = datetime.fromisoformat(str(closed_at))
    except ValueError:
        return None
    return max(0, int(round((closed - opened).total_seconds() / 60.0)))


def _final_status(realized_pnl: float | None) -> str | None:
    if realized_pnl is None:
        return None
    if realized_pnl > 0:
        return "gain"
    if realized_pnl < 0:
        return "loss"
    return "flat"


def empty_trade_reports_df() -> pd.DataFrame:
    return pd.DataFrame(columns=TRADER_REPORTS_COLUMNS)


def normalize_trade_reports_frame(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return empty_trade_reports_df()

    frame = df.copy()
    for column in TRADER_REPORTS_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.Series(dtype="object")

    for column in [
        "duration_minutes",
        "entry_price",
        "exit_price",
        "quantity",
        "gross_entry_value",
        "gross_exit_value",
        "realized_pnl",
        "realized_pnl_pct",
        "entry_score",
        "rsi_entry",
        "rsi_exit",
        "atr_pct_entry",
        "atr_pct_exit",
        "ma20_entry",
        "ma50_entry",
        "ma20_exit",
        "ma50_exit",
        "peak_price",
        "trailing_stop_final",
        "holding_minutes_limit",
    ]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    for column in ["opened_at", "closed_at"]:
        frame[column] = pd.to_datetime(frame[column], errors="coerce")

    for column in ["asset", "profile", "exit_reason", "status_final", "trade_id"]:
        frame[column] = frame[column].astype("string")

    return frame[TRADER_REPORTS_COLUMNS]


def read_trade_reports(limit: int | None = None) -> pd.DataFrame:
    df = read_table(TRADER_REPORTS_FILE, columns=TRADER_REPORTS_COLUMNS)
    df = normalize_trade_reports_frame(df)
    if limit is not None and not df.empty:
        return df.tail(int(limit)).reset_index(drop=True)
    return df


def append_trade_report(report: dict[str, Any]) -> None:
    append_table_row(TRADER_REPORTS_FILE, report, columns=TRADER_REPORTS_COLUMNS)


def replace_trade_reports(rows: list[dict[str, Any]]) -> None:
    replace_table(TRADER_REPORTS_FILE, rows, columns=TRADER_REPORTS_COLUMNS)


def build_closed_trade_report(
    *,
    position: dict[str, Any],
    closed_at: str,
    asset: str,
    quantity: float,
    exit_price: float,
    realized_pnl: float,
    exit_reason: str,
    exit_snapshot: dict[str, Any] | pd.Series | None,
    trailing_stop_final: float | None,
) -> dict[str, Any]:
    latest = exit_snapshot if isinstance(exit_snapshot, dict) else (
        exit_snapshot.to_dict() if hasattr(exit_snapshot, "to_dict") else {}
    )
    latest = latest or {}

    opened_at = position.get("opened_at")
    entry_price = _safe_float(position.get("avg_price") or position.get("entry_price"))
    gross_entry_value = _safe_float(position.get("gross_entry_value"))
    if gross_entry_value is None and entry_price is not None:
        gross_entry_value = round(float(quantity) * float(entry_price), 2)

    gross_exit_value = round(float(quantity) * float(exit_price), 2)
    realized_pnl_pct = None
    if gross_entry_value not in (None, 0):
        realized_pnl_pct = round(float(realized_pnl) / float(gross_entry_value), 6)

    report = {
        "trade_id": position.get("trade_id") or None,
        "opened_at": _safe_iso(opened_at),
        "closed_at": _safe_iso(closed_at),
        "duration_minutes": _duration_minutes(opened_at, closed_at),
        "asset": str(asset).upper(),
        "profile": position.get("profile") or None,
        "entry_price": entry_price,
        "exit_price": round(float(exit_price), 6),
        "quantity": round(float(quantity), 6),
        "gross_entry_value": gross_entry_value,
        "gross_exit_value": gross_exit_value,
        "realized_pnl": round(float(realized_pnl), 2),
        "realized_pnl_pct": realized_pnl_pct,
        "entry_score": _safe_float(position.get("entry_score")),
        "exit_reason": str(exit_reason) if exit_reason else None,
        "rsi_entry": _safe_float(position.get("rsi_entry")),
        "rsi_exit": _safe_float(latest.get("rsi")),
        "atr_pct_entry": _safe_float(position.get("atr_pct_entry", position.get("atr_pct"))),
        "atr_pct_exit": _safe_float(latest.get("atr_pct")),
        "ma20_entry": _safe_float(position.get("ma20_entry")),
        "ma50_entry": _safe_float(position.get("ma50_entry")),
        "ma20_exit": _safe_float(latest.get("ma20")),
        "ma50_exit": _safe_float(latest.get("ma50")),
        "peak_price": _safe_float(position.get("peak_price")),
        "trailing_stop_final": _safe_float(trailing_stop_final),
        "holding_minutes_limit": _safe_int(position.get("holding_minutes_limit")),
        "status_final": _final_status(_safe_float(realized_pnl)),
    }
    return report


def filter_trade_reports(
    reports_df: pd.DataFrame,
    *,
    profile: str | None = None,
    asset: str | None = None,
    status_final: str | None = None,
    start_at: datetime | None = None,
    end_at: datetime | None = None,
) -> pd.DataFrame:
    df = normalize_trade_reports_frame(reports_df)
    if df.empty:
        return df

    if profile and profile != "Todos":
        df = df[df["profile"] == profile]
    if asset and asset != "Todos":
        df = df[df["asset"] == asset]
    if status_final and status_final != "Todos":
        df = df[df["status_final"] == status_final]
    if start_at is not None:
        df = df[df["closed_at"] >= pd.Timestamp(start_at)]
    if end_at is not None:
        df = df[df["closed_at"] <= pd.Timestamp(end_at)]
    return df.reset_index(drop=True)


def calculate_trade_report_metrics(reports_df: pd.DataFrame) -> dict[str, float | int | None]:
    df = normalize_trade_reports_frame(reports_df)
    if df.empty:
        return {
            "total_trades": 0,
            "total_pnl": 0.0,
            "win_rate": 0.0,
            "avg_win": None,
            "avg_loss": None,
            "payoff": None,
            "avg_duration": None,
            "best_trade": None,
            "worst_trade": None,
        }

    pnl = pd.to_numeric(df["realized_pnl"], errors="coerce")
    winners = pnl[pnl > 0]
    losers = pnl[pnl < 0]
    avg_win = float(winners.mean()) if not winners.empty else None
    avg_loss = float(losers.mean()) if not losers.empty else None
    payoff = None
    if avg_win is not None and avg_loss not in (None, 0):
        payoff = float(avg_win / abs(avg_loss))

    durations = pd.to_numeric(df["duration_minutes"], errors="coerce").dropna()

    return {
        "total_trades": int(len(df)),
        "total_pnl": float(pnl.fillna(0.0).sum()),
        "win_rate": float((pnl > 0).mean()) if len(df) else 0.0,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "payoff": payoff,
        "avg_duration": float(durations.mean()) if not durations.empty else None,
        "best_trade": float(pnl.max()) if not pnl.dropna().empty else None,
        "worst_trade": float(pnl.min()) if not pnl.dropna().empty else None,
    }


def summarize_reports_by_profile(reports_df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_trade_reports_frame(reports_df)
    if df.empty:
        return pd.DataFrame(columns=["profile", "trades", "pnl", "win_rate", "avg_pnl", "avg_duration"])

    grouped = (
        df.groupby("profile", dropna=False)
        .agg(
            trades=("trade_id", "count"),
            pnl=("realized_pnl", "sum"),
            win_rate=("realized_pnl", lambda s: (pd.to_numeric(s, errors="coerce") > 0).mean()),
            avg_pnl=("realized_pnl", "mean"),
            avg_duration=("duration_minutes", "mean"),
        )
        .reset_index()
    )
    return grouped.sort_values("pnl", ascending=False).reset_index(drop=True)


def generate_trade_suggestions(reports_df: pd.DataFrame) -> list[dict[str, Any]]:
    df = normalize_trade_reports_frame(reports_df)
    if df.empty:
        return []

    suggestions: list[dict[str, Any]] = []
    profile_summary = summarize_reports_by_profile(df)

    for row in profile_summary.to_dict(orient="records"):
        profile = str(row.get("profile") or "Sem perfil")
        trades = int(row.get("trades", 0) or 0)
        win_rate = float(row.get("win_rate", 0.0) or 0.0)
        avg_pnl = float(row.get("avg_pnl", 0.0) or 0.0)

        if trades >= 4 and win_rate < 0.4:
            suggestions.append(
                {
                    "profile": profile,
                    "severity": "warning",
                    "message": f"Perfil {profile} teve win rate baixo; considerar elevar min_signal_score e reduzir max_open_positions.",
                    "proposed_adjustments": {
                        "min_signal_score_delta": 0.03,
                        "max_open_positions_delta": -1,
                    },
                }
            )
        if trades >= 4 and win_rate >= 0.6 and avg_pnl <= 0:
            suggestions.append(
                {
                    "profile": profile,
                    "severity": "info",
                    "message": f"Perfil {profile} acerta bastante, mas converte pouco PnL medio; revisar trailing_stop_atr_mult ou holding_minutes.",
                    "proposed_adjustments": {
                        "trailing_stop_atr_mult_delta": 0.2,
                        "holding_minutes_delta": 15,
                    },
                }
            )

    positive_profiles = profile_summary[profile_summary["pnl"] > 0].copy()
    if not positive_profiles.empty:
        best = positive_profiles.sort_values(["win_rate", "pnl"], ascending=False).iloc[0]
        suggestions.append(
            {
                "profile": str(best.get("profile") or "Equilibrado"),
                "severity": "success",
                "message": f"{best.get('profile')} apresentou a melhor combinacao recente de win rate e PnL; pode seguir como referencia.",
                "proposed_adjustments": {},
            }
        )

    return suggestions


def apply_suggested_adjustments(profile_config: dict[str, Any], suggestions: list[dict[str, Any]]) -> dict[str, Any]:
    updated = dict(profile_config)
    for suggestion in suggestions:
        adjustments = suggestion.get("proposed_adjustments", {}) or {}
        if not suggestion.get("apply", False):
            continue
        for key, value in adjustments.items():
            if key.endswith("_delta"):
                base_key = key[: -len("_delta")]
                current = updated.get(base_key)
                if isinstance(current, (int, float)) and isinstance(value, (int, float)):
                    updated[base_key] = current + value
            elif key in updated:
                updated[key] = value
    return updated
