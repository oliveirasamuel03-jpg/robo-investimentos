from __future__ import annotations

import pandas as pd


def _coerce_numeric_frame(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()

    out = frame.copy()
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.fillna(0.0)


def summarize_trades_by_asset(trades_df: pd.DataFrame | None) -> list[dict]:
    if trades_df is None or trades_df.empty or "asset" not in trades_df.columns:
        return []

    df = trades_df.copy()
    df["asset"] = df["asset"].astype(str).str.upper()

    if "realized_pnl" in df.columns:
        pnl = pd.to_numeric(df["realized_pnl"], errors="coerce").fillna(0.0)
    elif "pnl" in df.columns:
        pnl = pd.to_numeric(df["pnl"], errors="coerce").fillna(0.0)
    else:
        pnl = pd.Series(0.0, index=df.index, dtype=float)
    df["pnl"] = pnl

    if "gross_value" in df.columns:
        gross_value = pd.to_numeric(df["gross_value"], errors="coerce").fillna(0.0).abs()
    else:
        qty = pd.to_numeric(df["quantity"], errors="coerce").fillna(0.0) if "quantity" in df.columns else pd.Series(0.0, index=df.index, dtype=float)
        price = pd.to_numeric(df["price"], errors="coerce").fillna(0.0) if "price" in df.columns else pd.Series(0.0, index=df.index, dtype=float)
        gross_value = (qty * price).abs()
    df["gross_value"] = gross_value

    base = df["gross_value"].replace(0.0, pd.NA)
    df["trade_return"] = (df["pnl"] / base).fillna(0.0)

    rows: list[dict] = []
    for asset, group in df.groupby("asset", dropna=False):
        realized_mask = group["pnl"] != 0.0
        evaluable = group.loc[realized_mask] if realized_mask.any() else group
        returns = pd.to_numeric(evaluable["trade_return"], errors="coerce").fillna(0.0)
        rows.append(
            {
                "asset": str(asset),
                "trades": int(len(group)),
                "pnl": float(group["pnl"].sum()),
                "avg_trade_return": float(returns.mean()) if not returns.empty else 0.0,
                "win_rate": float((returns > 0).mean()) if not returns.empty else 0.0,
                "gross_volume": float(group["gross_value"].sum()),
            }
        )

    return sorted(rows, key=lambda row: row["pnl"], reverse=True)


def summarize_backtest_by_asset(
    asset_contribution_df: pd.DataFrame | None,
    effective_weights_df: pd.DataFrame | None = None,
) -> list[dict]:
    contributions = _coerce_numeric_frame(asset_contribution_df)
    if contributions.empty:
        return []

    weights = _coerce_numeric_frame(effective_weights_df)
    if weights.empty:
        weights = pd.DataFrame(0.0, index=contributions.index, columns=contributions.columns)

    common_cols = [col for col in contributions.columns if col in weights.columns]
    if common_cols:
        contributions = contributions[common_cols]
        weights = weights[common_cols]

    rows: list[dict] = []
    for asset in contributions.columns:
        contrib = pd.to_numeric(contributions[asset], errors="coerce").fillna(0.0)
        exposure = pd.to_numeric(weights.get(asset, 0.0), errors="coerce").fillna(0.0).abs()
        active = exposure > 1e-9
        active_returns = contrib.loc[active]
        rows.append(
            {
                "asset": str(asset),
                "trades": int(active.sum()),
                "pnl": float(contrib.sum()),
                "avg_trade_return": float(active_returns.mean()) if not active_returns.empty else 0.0,
                "win_rate": float((active_returns > 0).mean()) if not active_returns.empty else 0.0,
                "gross_volume": float(exposure.sum()),
            }
        )

    rows = [row for row in rows if row["trades"] > 0 or abs(row["pnl"]) > 1e-12 or row["gross_volume"] > 0.0]
    return sorted(rows, key=lambda row: row["pnl"], reverse=True)
