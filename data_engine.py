from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd
import yfinance as yf


@dataclass
class DataUniverse:
    brazil_stocks: list[str]
    us_stocks: list[str]
    etfs: list[str]
    fiis: list[str]
    crypto: list[str]
    grains: list[str]

    @property
    def all_assets(self) -> list[str]:
        seen = []
        for group in [
            self.brazil_stocks,
            self.us_stocks,
            self.etfs,
            self.fiis,
            self.crypto,
            self.grains,
        ]:
            for ticker in group:
                if ticker not in seen:
                    seen.append(ticker)
        return seen


DEFAULT_UNIVERSE = DataUniverse(
    brazil_stocks=[
        "VALE3.SA",
        "PETR4.SA",
        "ITUB4.SA",
        "BBDC4.SA",
        "BBAS3.SA",
        "WEGE3.SA",
        "ABEV3.SA",
        "RENT3.SA",
        "SUZB3.SA",
        "PRIO3.SA",
    ],
    us_stocks=[
        "AAPL",
        "MSFT",
        "NVDA",
        "AMZN",
        "GOOGL",
        "META",
        "TSLA",
        "AMD",
    ],
    etfs=[
        "SPY",
        "QQQ",
        "DIA",
        "IWM",
        "EEM",
        "XLF",
        "XLK",
        "GLD",
    ],
    fiis=[
        "HGLG11.SA",
        "KNRI11.SA",
        "XPLG11.SA",
        "MXRF11.SA",
        "VISC11.SA",
    ],
    crypto=[
        "BTC-USD",
        "ETH-USD",
    ],
    grains=[
        "KC=F",  # café
        "ZC=F",  # milho
        "ZS=F",  # soja
        "ZW=F",  # trigo
        "ZR=F",  # arroz
        "ZO=F",  # aveia
    ],
)


def _ensure_list(tickers: Iterable[str] | None) -> list[str]:
    if tickers is None:
        return []
    out = []
    for t in tickers:
        if t is None:
            continue
        s = str(t).strip().upper()
        if s and s not in out:
            out.append(s)
    return out


def _download_single_batch(
    tickers: list[str],
    start: str,
) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()

    raw = yf.download(
        tickers=tickers,
        start=start,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )

    if raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            closes = raw["Close"].copy()
        else:
            closes = raw.xs("Close", axis=1, level=0, drop_level=False)
            closes.columns = [c[-1] for c in closes.columns]
    else:
        closes = raw.rename(columns={"Close": tickers[0]})[[tickers[0]]]

    closes.index = pd.to_datetime(closes.index)
    closes = closes.sort_index()
    closes = closes.replace([float("inf"), float("-inf")], pd.NA)
    closes = closes.dropna(how="all")
    return closes


def _clean_prices(df: pd.DataFrame, min_non_na_ratio: float = 0.65) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy().sort_index()

    valid_ratio = out.notna().mean()
    keep_cols = valid_ratio[valid_ratio >= min_non_na_ratio].index.tolist()
    out = out[keep_cols]

    out = out.ffill()
    out = out.dropna(how="all")

    keep_cols = []
    for col in out.columns:
        s = out[col].dropna()
        if len(s) < 30:
            continue
        if s.nunique() <= 5:
            continue
        keep_cols.append(col)

    out = out[keep_cols]
    return out


def load_data(
    start: str = "2019-01-01",
    history_limit: int = 1500,
    include_brazil_stocks: bool = True,
    include_us_stocks: bool = True,
    include_etfs: bool = True,
    include_fiis: bool = True,
    include_crypto: bool = True,
    include_grains: bool = True,
    custom_tickers: list[str] | None = None,
) -> pd.DataFrame:
    universe_parts: list[str] = []

    if include_brazil_stocks:
        universe_parts.extend(DEFAULT_UNIVERSE.brazil_stocks)
    if include_us_stocks:
        universe_parts.extend(DEFAULT_UNIVERSE.us_stocks)
    if include_etfs:
        universe_parts.extend(DEFAULT_UNIVERSE.etfs)
    if include_fiis:
        universe_parts.extend(DEFAULT_UNIVERSE.fiis)
    if include_crypto:
        universe_parts.extend(DEFAULT_UNIVERSE.crypto)
    if include_grains:
        universe_parts.extend(DEFAULT_UNIVERSE.grains)

    universe_parts.extend(_ensure_list(custom_tickers))

    tickers = []
    for t in universe_parts:
        if t not in tickers:
            tickers.append(t)

    if not tickers:
        raise ValueError("No tickers selected for load_data")

    batch_size = 10
    batches = [tickers[i:i + batch_size] for i in range(0, len(tickers), batch_size)]

    frames = []
    for batch in batches:
        try:
            part = _download_single_batch(batch, start=start)
            if not part.empty:
                frames.append(part)
        except Exception:
            continue

    if not frames:
        raise ValueError("Could not download any market data")

    prices = pd.concat(frames, axis=1)
    prices = prices.loc[:, ~prices.columns.duplicated()].copy()
    prices = _clean_prices(prices)

    if prices.empty:
        raise ValueError("No usable price series after cleaning")

    if history_limit is not None and history_limit > 0:
        prices = prices.tail(history_limit)

    return prices
