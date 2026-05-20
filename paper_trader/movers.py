"""Intraday top-mover ranking for the ETF universe.

`compute_top_movers(top_k=10)` returns the largest % gainers from
today's open, ranked highest first. The agent's 10am intraday buy
picks the #1 entry on this list.

Trades use the agent's 10am-snapshot ranking, not "now" — see
`compute_top_movers_at(target_ts)` for the historical version used
when replaying missed trades.
"""
from __future__ import annotations

import logging
from datetime import datetime, time as dt_time, timezone, timedelta
from typing import Any

import pandas as pd
import yfinance as yf

from .universe import UNIVERSE

LOGGER = logging.getLogger("paper_trader.movers")


def _intraday_pct(ticker: str) -> dict[str, Any] | None:
    """Pull today's 5min bars + compute % change from open."""
    try:
        df = yf.download(ticker, period="1d", interval="5m",
                         auto_adjust=False, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty or len(df) < 2:
            return None
        op = float(df["Open"].iloc[0])
        cp = float(df["Close"].iloc[-1])
        if op <= 0:
            return None
        return {
            "ticker": ticker,
            "open": op,
            "current": cp,
            "change_pct": (cp - op) / op * 100,
        }
    except Exception as e:
        LOGGER.debug(f"intraday fetch {ticker} failed: {e}")
        return None


def compute_top_movers(top_k: int = 10) -> list[dict[str, Any]]:
    """Top % gainers from open across the universe, descending.

    Note: for the agent's BUY pick we want the top *gainer* (momentum
    continuation), not the biggest absolute move. So we sort by
    signed change_pct, not abs.
    """
    results = [r for r in (_intraday_pct(t) for t in UNIVERSE) if r]
    results.sort(key=lambda x: x["change_pct"], reverse=True)
    return results[:top_k]


def compute_top_mover_at(target_ts: datetime, top_k: int = 5) -> list[dict[str, Any]]:
    """Top % gainers as of a specific historical timestamp.

    Used when replaying missed scheduled trades — e.g. the page opens
    at 11am but the 10am buy slot wasn't fired yet, so we need the
    rankings AS OF 10am, not now.

    Resolves by pulling 1-min bars for `target_ts`'s date and finding
    the row at-or-just-before target_ts. If target_ts is today and the
    market hasn't closed, we still get useful data.
    """
    target_ts = target_ts.astimezone(timezone.utc).replace(tzinfo=None)
    date_str = target_ts.strftime("%Y-%m-%d")
    results = []
    for ticker in UNIVERSE:
        try:
            df = yf.download(ticker, start=date_str,
                             end=(target_ts + timedelta(days=1)).strftime("%Y-%m-%d"),
                             interval="1m", auto_adjust=False, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if df.empty:
                continue
            # Normalize index to naive UTC for comparison
            if df.index.tz is not None:
                df.index = df.index.tz_convert("UTC").tz_localize(None)
            # Open of the session = first row of the day
            op = float(df["Open"].iloc[0])
            # Price at-or-just-before target_ts
            before = df.loc[df.index <= target_ts]
            if before.empty:
                continue
            price_at = float(before["Close"].iloc[-1])
            if op <= 0:
                continue
            results.append({
                "ticker": ticker,
                "open": op,
                "price_at_ts": price_at,
                "change_pct": (price_at - op) / op * 100,
            })
        except Exception:
            continue
    results.sort(key=lambda x: x["change_pct"], reverse=True)
    return results[:top_k]


def get_price_at(ticker: str, target_ts: datetime) -> float | None:
    """Fetch ticker's price at a specific historical timestamp using
    1-min bars. Returns None if no data."""
    target_ts = target_ts.astimezone(timezone.utc).replace(tzinfo=None)
    date_str = target_ts.strftime("%Y-%m-%d")
    try:
        df = yf.download(ticker, start=date_str,
                         end=(target_ts + timedelta(days=2)).strftime("%Y-%m-%d"),
                         interval="1m", auto_adjust=False, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty:
            return None
        if df.index.tz is not None:
            df.index = df.index.tz_convert("UTC").tz_localize(None)
        before = df.loc[df.index <= target_ts]
        if before.empty:
            # Target_ts before first bar — use opening
            return float(df["Open"].iloc[0])
        return float(before["Close"].iloc[-1])
    except Exception as e:
        LOGGER.warning(f"get_price_at({ticker}, {target_ts}) failed: {e}")
        return None
