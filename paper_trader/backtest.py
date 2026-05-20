"""Historical backtest for the commodity paper-trader strategy.

Replays the two-slot schedule day-by-day over the past N days using
actual 5-min bar data from yfinance. Reports total return, equity
curve, per-trade P&L, and win rate so we can validate the strategy
BEFORE letting it run on real (paper) money going forward.

Why 5-min bars: yfinance free tier only retains intraday 1-min bars
for ~7 days but 5-min bars for ~60 days. 5-min granularity is fine
for our schedule (the closest two events are 15 min apart at 3:30 /
3:45 PM ET).

Data efficiency: pulls all 6 tickers' bars + the daily history for
predictor scoring ONCE up front, then iterates over the schedule in
memory.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, time as dt_time
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

from .universe import UNIVERSE
from .predictor import WEIGHTS, _rsi
from .agent import SCHEDULE, ALLOCATION_PCT, FILTERS, PAIR_UNDERLYING

LOGGER = logging.getLogger("paper_trader.backtest")
ET = ZoneInfo("America/Toronto")


def _fetch_universe_5min(days_back: int) -> dict[str, pd.DataFrame]:
    """Pull 5-min bars for every ticker in UNIVERSE, normalized to
    naive ET timestamps."""
    out: dict[str, pd.DataFrame] = {}
    period = f"{min(days_back + 5, 60)}d"  # yfinance caps at 60d
    for tkr in UNIVERSE:
        try:
            df = yf.download(tkr, period=period, interval="5m",
                             auto_adjust=False, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if df.empty:
                LOGGER.warning(f"{tkr}: no 5-min bars")
                continue
            # Normalize to ET-naive
            if df.index.tz is not None:
                df.index = df.index.tz_convert(ET).tz_localize(None)
            out[tkr] = df
        except Exception as e:
            LOGGER.warning(f"{tkr} fetch failed: {e}")
    return out


def _fetch_universe_daily(days_back: int) -> dict[str, pd.DataFrame]:
    """Daily bars (3 months) used by the bullish-opening predictor."""
    out: dict[str, pd.DataFrame] = {}
    for tkr in UNIVERSE:
        try:
            df = yf.download(tkr, period="3mo", interval="1d",
                             auto_adjust=False, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if df.empty:
                continue
            if df.index.tz is not None:
                df.index = df.index.tz_convert(ET).tz_localize(None)
            out[tkr] = df
        except Exception as e:
            LOGGER.warning(f"{tkr} daily fetch failed: {e}")
    return out


def _price_at(intraday: dict[str, pd.DataFrame], ticker: str,
              ts: pd.Timestamp) -> float | None:
    """Bar price at-or-just-before `ts` (close of that bar)."""
    df = intraday.get(ticker)
    if df is None or df.empty:
        return None
    rows = df.loc[df.index <= ts]
    if rows.empty:
        return None
    return float(rows["Close"].iloc[-1])


def _session_open(intraday: dict[str, pd.DataFrame], ticker: str,
                  session_date: pd.Timestamp.date) -> float | None:
    """First bar's open price on the given session date."""
    df = intraday.get(ticker)
    if df is None:
        return None
    day = df[df.index.date == session_date]
    if day.empty:
        return None
    return float(day["Open"].iloc[0])


def _pick_intraday_at(intraday: dict[str, pd.DataFrame],
                       ts: pd.Timestamp) -> tuple[str | None, dict[str, float]]:
    """Top % gainer from today's open at timestamp `ts`. Returns
    (ticker, debug_scores)."""
    scores = {}
    for tkr in UNIVERSE:
        op = _session_open(intraday, tkr, ts.date())
        px = _price_at(intraday, tkr, ts)
        if op is None or px is None or op <= 0:
            continue
        scores[tkr] = (px - op) / op * 100
    if not scores:
        return None, {}
    winner = max(scores.items(), key=lambda x: x[1])
    return winner[0], scores


def _pick_overnight_at(daily: dict[str, pd.DataFrame],
                        as_of_date: pd.Timestamp.date,
                        ) -> tuple[str | None, dict[str, float]]:
    """Top bullish-opening composite score using DAILY bars at the
    session ending on as_of_date. Mirrors predictor.py but indexed
    to a historical date."""
    scores = {}
    for tkr in UNIVERSE:
        df = daily.get(tkr)
        if df is None or len(df) < 25:
            continue
        # Truncate to as_of_date to avoid look-ahead
        df_trunc = df[df.index.date <= as_of_date]
        if len(df_trunc) < 25:
            continue
        close = df_trunc["Close"]
        high = df_trunc["High"]
        low = df_trunc["Low"]
        volume = df_trunc["Volume"]

        today_close = float(close.iloc[-1])
        today_high = float(high.iloc[-1])
        today_low = float(low.iloc[-1])

        rng = today_high - today_low
        close_pos = ((today_close - today_low) / rng) if rng > 0 else 0.5
        close_pos = float(np.clip(close_pos, 0, 1))

        ret_5d = float(close.iloc[-1] / close.iloc[-6] - 1) if len(close) >= 6 else 0
        momentum = float(np.clip((ret_5d + 0.05) / 0.10, 0, 1))

        rsi14 = _rsi(close, 14)
        if np.isnan(rsi14):
            rsi_zone = 0.5
        elif 55 <= rsi14 <= 70:
            rsi_zone = 1.0
        elif 50 <= rsi14 < 55 or 70 < rsi14 <= 75:
            rsi_zone = 0.7
        elif 40 <= rsi14 < 50:
            rsi_zone = 0.4
        elif 75 < rsi14:
            rsi_zone = 0.2
        else:
            rsi_zone = 0.1

        vol_ma = float(volume.iloc[-20:].mean()) if len(volume) >= 20 else float("nan")
        vol_ratio = float(volume.iloc[-1]) / vol_ma if vol_ma and vol_ma > 0 else 1.0
        vol_surge = float(np.clip((vol_ratio - 0.5) / 1.5, 0, 1))

        scores[tkr] = (
            WEIGHTS["close_position"] * close_pos
            + WEIGHTS["recent_momentum"] * momentum
            + WEIGHTS["rsi_zone"] * rsi_zone
            + WEIGHTS["volume_surge"] * vol_surge
        )

    if not scores:
        return None, {}
    winner = max(scores.items(), key=lambda x: x[1])
    return winner[0], scores


def _fetch_underlying_daily() -> dict[str, pd.DataFrame]:
    """Pull WTI + natgas daily bars for the trend-alignment filter."""
    out: dict[str, pd.DataFrame] = {}
    for sym in {u for u, _ in PAIR_UNDERLYING.values()}:
        try:
            df = yf.download(sym, period="6mo", interval="1d",
                             auto_adjust=False, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if not df.empty:
                if df.index.tz is not None:
                    df.index = df.index.tz_convert(ET).tz_localize(None)
                out[sym] = df
        except Exception:
            continue
    return out


def _trend_aligned_at(ticker: str,
                       as_of: pd.Timestamp.date,
                       underlying_daily: dict[str, pd.DataFrame],
                       threshold_pct: float) -> bool:
    """Trend alignment check using ALREADY-FETCHED underlying daily
    history truncated to `as_of`. No yfinance calls during backtest
    iteration."""
    if ticker not in PAIR_UNDERLYING:
        return True
    sym, side = PAIR_UNDERLYING[ticker]
    df = underlying_daily.get(sym)
    if df is None or df.empty:
        return True
    trunc = df[df.index.date <= as_of]
    if len(trunc) < 30:
        return True
    sma20 = trunc["Close"].rolling(20).mean().dropna()
    if len(sma20) < 11:
        return True
    slope_pct = float((sma20.iloc[-1] - sma20.iloc[-11]) / sma20.iloc[-11] * 100)
    if side == "bull":
        return slope_pct > threshold_pct
    return slope_pct < -threshold_pct


def run_backtest(days_back: int = 30,
                  initial_capital: float = 10_000.0,
                  apply_filters: bool = True) -> dict[str, Any]:
    """Replay the agent's schedule over the past `days_back` trading
    days (capped by yfinance 60-day 5-min limit).

    Returns:
      summary    : totals (return%, sharpe-lite, win rate, max DD)
      trades     : list of every buy + sell
      equity_curve: timestamp + equity value at each event
    """
    LOGGER.info(f"Fetching universe data for {days_back}d backtest…")
    intraday = _fetch_universe_5min(days_back)
    daily = _fetch_universe_daily(days_back)
    underlying = _fetch_underlying_daily() if apply_filters else {}
    if not intraday or not daily:
        return {"error": "no_data"}

    # Build list of trading-day dates from any ticker that has data
    sample_df = next(iter(intraday.values()))
    all_dates = sorted({d.date() for d in sample_df.index})
    # Keep only weekdays (Mon-Fri = 0-4). Last `days_back` of these.
    all_dates = [d for d in all_dates if d.weekday() < 5][-days_back:]
    if not all_dates:
        return {"error": "no_trading_days"}

    # Build chronological schedule of events
    events = []
    for d in all_dates:
        for hour, minute, action, slot in SCHEDULE:
            ts = pd.Timestamp.combine(d, dt_time(hour, minute))
            events.append({"ts": ts, "action": action, "slot": slot})
    events.sort(key=lambda e: e["ts"])

    # Replay
    cash = initial_capital
    positions: dict[str, dict[str, Any]] = {}  # slot -> {ticker, shares, entry_price, cost}
    trades: list[dict[str, Any]] = []
    skipped_log: list[dict[str, Any]] = []  # filter-rejected events
    equity_curve: list[dict[str, Any]] = []
    peak_equity = initial_capital
    max_dd = 0.0

    for evt in events:
        ts = evt["ts"]
        slot = evt["slot"]
        action = evt["action"]

        if action == "BUY":
            if slot in positions:
                continue  # slot already filled
            if slot == "intraday":
                ticker, scores = _pick_intraday_at(intraday, ts)
                if not ticker:
                    continue
                top_pct = scores[ticker]
                # Filter: minimum momentum
                if apply_filters and top_pct < FILTERS["min_intraday_pct"]:
                    skipped_log.append({"ts": ts, "slot": slot,
                        "reason": f"momentum {top_pct:.2f}% < threshold"})
                    continue
                # Filter: trend alignment
                if apply_filters and not _trend_aligned_at(
                    ticker, ts.date(), underlying,
                    FILTERS["trend_slope_threshold_pct"],
                ):
                    skipped_log.append({"ts": ts, "slot": slot,
                        "reason": f"{ticker} fails trend alignment"})
                    continue
                rationale = f"top intraday {top_pct:+.2f}% from open"
            else:
                ticker, scores = _pick_overnight_at(daily, ts.date())
                if not ticker:
                    continue
                top_score = scores[ticker]
                # Filter: minimum bullish score
                if apply_filters and top_score < FILTERS["min_overnight_score"]:
                    skipped_log.append({"ts": ts, "slot": slot,
                        "reason": f"score {top_score:.2f} < threshold"})
                    continue
                # Filter: trend alignment
                if apply_filters and not _trend_aligned_at(
                    ticker, ts.date(), underlying,
                    FILTERS["trend_slope_threshold_pct"],
                ):
                    skipped_log.append({"ts": ts, "slot": slot,
                        "reason": f"{ticker} fails trend alignment"})
                    continue
                rationale = f"top bullish score {top_score:.3f}"
            px = _price_at(intraday, ticker, ts)
            if not px:
                continue
            # MTM total equity for sizing
            mtm = cash
            for p in positions.values():
                cur = _price_at(intraday, p["ticker"], ts) or p["entry_price"]
                mtm += p["shares"] * cur
            notional = mtm * ALLOCATION_PCT
            if notional > cash:
                notional = cash * 0.99
            if notional < 50:
                continue
            shares = notional / px
            cash -= notional
            positions[slot] = {
                "ticker": ticker, "shares": shares,
                "entry_price": px, "cost": notional, "entry_ts": ts,
            }
            trades.append({
                "ts": ts, "slot": slot, "ticker": ticker,
                "side": "BUY", "shares": shares, "price": px,
                "notional": notional, "rationale": rationale,
            })

        else:  # SELL
            pos = positions.get(slot)
            if not pos:
                continue
            px = _price_at(intraday, pos["ticker"], ts) or pos["entry_price"]
            proceeds = pos["shares"] * px
            pnl = proceeds - pos["cost"]
            pnl_pct = (px - pos["entry_price"]) / pos["entry_price"] * 100
            cash += proceeds
            trades.append({
                "ts": ts, "slot": slot, "ticker": pos["ticker"],
                "side": "SELL", "shares": pos["shares"], "price": px,
                "notional": proceeds, "pnl": pnl, "pnl_pct": pnl_pct,
                "hold_minutes": (ts - pos["entry_ts"]).total_seconds() / 60,
            })
            del positions[slot]

        # Snapshot equity after this event
        mtm = cash
        for p in positions.values():
            cur = _price_at(intraday, p["ticker"], ts) or p["entry_price"]
            mtm += p["shares"] * cur
        equity_curve.append({"ts": ts, "equity": mtm,
                             "cash": cash, "open_positions": len(positions)})
        if mtm > peak_equity:
            peak_equity = mtm
        dd = (mtm - peak_equity) / peak_equity if peak_equity else 0
        if dd < max_dd:
            max_dd = dd

    # Force-close any still-open positions at the last available price
    last_ts = events[-1]["ts"]
    for slot, pos in list(positions.items()):
        px = _price_at(intraday, pos["ticker"], last_ts) or pos["entry_price"]
        proceeds = pos["shares"] * px
        pnl = proceeds - pos["cost"]
        cash += proceeds
        trades.append({
            "ts": last_ts, "slot": slot, "ticker": pos["ticker"],
            "side": "SELL", "shares": pos["shares"], "price": px,
            "notional": proceeds, "pnl": pnl,
            "pnl_pct": (px - pos["entry_price"]) / pos["entry_price"] * 100,
            "hold_minutes": (last_ts - pos["entry_ts"]).total_seconds() / 60,
            "note": "force-closed at backtest end",
        })

    final_equity = cash
    completed = [t for t in trades if t["side"] == "SELL" and "pnl" in t]
    wins = sum(1 for t in completed if t["pnl"] > 0)
    total_pnl = sum(t["pnl"] for t in completed)
    avg_pnl = total_pnl / len(completed) if completed else 0
    win_rate = wins / len(completed) if completed else 0
    intraday_trades = [t for t in completed if t["slot"] == "intraday"]
    overnight_trades = [t for t in completed if t["slot"] == "overnight"]

    def _slot_stats(lst):
        if not lst:
            return {"n": 0, "win_rate": 0, "avg_pnl": 0, "total_pnl": 0}
        w = sum(1 for x in lst if x["pnl"] > 0)
        return {
            "n": len(lst),
            "win_rate": w / len(lst),
            "avg_pnl": sum(x["pnl"] for x in lst) / len(lst),
            "total_pnl": sum(x["pnl"] for x in lst),
        }

    return {
        "initial_capital": initial_capital,
        "final_equity": final_equity,
        "total_return_pct": (final_equity / initial_capital - 1) * 100,
        "total_pnl": total_pnl,
        "n_trades": len(completed),
        "win_rate": win_rate,
        "avg_pnl_per_trade": avg_pnl,
        "max_drawdown_pct": max_dd * 100,
        "trading_days": len(all_dates),
        "intraday": _slot_stats(intraday_trades),
        "overnight": _slot_stats(overnight_trades),
        "trades": trades,
        "skipped": skipped_log,
        "equity_curve": equity_curve,
        "date_range": (str(all_dates[0]), str(all_dates[-1])),
        "filters_applied": apply_filters,
    }
