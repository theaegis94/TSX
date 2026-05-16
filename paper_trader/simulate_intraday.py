"""Intraday (hourly-bar) walk-forward backtest.

Different from simulate.py:
  - Fetches hourly bars (period="2y", interval="1h") — the max free
    history yfinance offers at this resolution.
  - Iterates hour-by-hour instead of day-by-day.
  - Same strategies, same exit rules, same costs — only the bar
    interval changes.

The point: see whether the same mean-reversion edge that works on
daily bars also works on hourly bars. If yes, trade count goes up
~7x (one trading day = ~7 hourly bars) without needing new strategies.
If no, we know the edge is specifically a daily-frequency phenomenon.
"""
from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd
import yfinance as yf

import stock_signals as ss

from . import storage
from . import strategies as strat_mod
from . import exits as exit_mod
from .features import SYMBOLS, _series_features

# Match daily simulator config
COMMISSION = 5.0
SLIPPAGE_PCT = 0.005
POSITION_SIZE_PCT = 0.25
MIN_CONVICTION_TO_OPEN = 0.50

# Hourly max-hold needs translation — daily MAX_HOLD_DAYS=3 ≈ ~21
# hourly bars (7 hrs/day × 3 days). But for intraday we want shorter
# holds — most intraday mean-reversion plays out in 4-8 hours.
MAX_HOLD_BARS = 8

LOGGER = logging.getLogger("paper_trader.simulate_intraday")


def _slip_buy(p): return p * (1 + SLIPPAGE_PCT / 2.0)
def _slip_sell(p): return p * (1 - SLIPPAGE_PCT / 2.0)


def fetch_intraday_features() -> pd.DataFrame:
    """Hourly bars for all underlying symbols — 2 years max."""
    syms = list(SYMBOLS.values())
    try:
        df = yf.download(
            " ".join(syms), period="2y", interval="1h",
            auto_adjust=True, progress=False, group_by="ticker",
            threads=True,
        )
    except Exception:
        return pd.DataFrame()
    return df if df is not None else pd.DataFrame()


def fetch_intraday_etfs() -> pd.DataFrame:
    """Hourly bars for the 4 paper-tradable ETFs."""
    from .storage import TICKERS
    try:
        df = yf.download(
            " ".join(TICKERS), period="2y", interval="1h",
            auto_adjust=True, progress=False, group_by="ticker",
            threads=True,
        )
    except Exception:
        return pd.DataFrame()
    return df if df is not None else pd.DataFrame()


def features_as_of_hourly(precomputed_df: pd.DataFrame, ts) -> dict:
    """Compute features as of `ts` (inclusive) using hourly bars."""
    ts = pd.Timestamp(ts)
    out: dict = {"as_of": ts.isoformat()}
    if precomputed_df is None or precomputed_df.empty:
        return out
    for short, full_sym in SYMBOLS.items():
        try:
            if isinstance(precomputed_df.columns, pd.MultiIndex):
                if full_sym not in precomputed_df.columns.get_level_values(0):
                    continue
                close = precomputed_df[full_sym]["Close"].dropna()
            else:
                close = precomputed_df["Close"].dropna()
            # Slice <= ts. Normalize tz so comparison works.
            try:
                if getattr(close.index, "tz", None) is not None:
                    cutoff = (ts.tz_localize(close.index.tz)
                              if ts.tz is None else ts)
                else:
                    cutoff = ts.tz_localize(None) if ts.tz else ts
                close = close[close.index <= cutoff]
            except Exception:
                pass
            if len(close) < 20:
                continue
            sub = _series_features(close)
            for k, v in sub.items():
                out[f"{short}_{k}"] = v
        except Exception:
            continue
    # Calendar features (less useful intraday but harmless)
    out["day_of_week"] = int(ts.dayofweek)
    out["month"] = int(ts.month)
    return out


def etf_price_at(etf_df: pd.DataFrame, ticker: str, ts) -> float | None:
    if etf_df is None or etf_df.empty:
        return None
    ts = pd.Timestamp(ts)
    try:
        if isinstance(etf_df.columns, pd.MultiIndex):
            if ticker not in etf_df.columns.get_level_values(0):
                return None
            close = etf_df[ticker]["Close"].dropna()
        else:
            close = etf_df["Close"].dropna()
        if getattr(close.index, "tz", None) is not None:
            target = (ts.tz_localize(close.index.tz)
                      if ts.tz is None else ts)
        else:
            target = ts.tz_localize(None) if ts.tz else ts
        mask = close.index == target
        if mask.any():
            return float(close[mask].iloc[0])
    except Exception:
        return None
    return None


def run_intraday_backtest(
    starting_balance: float = 10_000.0,
    min_warmup_bars: int = 60,
    clear_previous: bool = True,
) -> dict:
    LOGGER.info("Starting intraday backtest")
    feat_df = fetch_intraday_features()
    etf_df = fetch_intraday_etfs()
    if feat_df.empty or etf_df.empty:
        return {"error": "empty_data", "trades": 0}

    # Iterate using the ETF data's timestamps (the actual trading hours)
    if isinstance(etf_df.columns, pd.MultiIndex):
        # Find any ticker's index — they share the same timestamps
        first_sym = etf_df.columns.get_level_values(0)[0]
        timeline = etf_df[first_sym].dropna(how="all").index
    else:
        timeline = etf_df.dropna(how="all").index
    timeline = sorted(timeline.unique())
    if len(timeline) < min_warmup_bars + 10:
        return {"error": "insufficient_history", "trades": 0}
    iter_bars = timeline[min_warmup_bars:]
    LOGGER.info(
        f"Replaying {len(iter_bars)} hourly bars "
        f"from {iter_bars[0]} to {iter_bars[-1]}"
    )

    if clear_previous:
        deleted = storage.clear_sim_trades()
        if deleted:
            LOGGER.info(f"Cleared {deleted} prior sim trades")

    for strat in strat_mod.ALL_STRATEGIES:
        storage.register_strategy(strat.name, strat.description)

    sim_balance = starting_balance
    open_position: dict | None = None
    trades_logged = 0
    bars_held = 0

    for ts in iter_bars:
        features = features_as_of_hourly(feat_df, ts)
        signals = strat_mod.run_all_strategies(features)
        best = strat_mod.best_signal(signals)

        # Exit check
        if open_position:
            cur_price = etf_price_at(etf_df, open_position["ticker"], ts)
            if cur_price is not None:
                pct = (cur_price - open_position["entry_price"]) / \
                    open_position["entry_price"] * 100.0
                reason = None
                if pct <= exit_mod.STOP_LOSS_PCT:
                    reason = "stop_loss"
                elif pct >= exit_mod.TAKE_PROFIT_PCT:
                    reason = "take_profit"
                elif best and best.get("ticker"):
                    opp = exit_mod.PAIR_OPPOSITE.get(
                        open_position["ticker"]
                    )
                    if (best["ticker"] == opp
                            and best.get("conviction", 0) >= 0.6):
                        reason = "signal_flip"
                else:
                    bars_held += 1
                    if bars_held >= MAX_HOLD_BARS:
                        reason = "timeout"
                if reason is None and not open_position.get("_counted_this_bar"):
                    bars_held += 1
                if reason:
                    exit_exec = _slip_sell(cur_price)
                    gross = open_position["shares"] * exit_exec
                    net = gross - COMMISSION
                    pnl_dollars = net - open_position["capital_used"]
                    pnl_pct = (
                        (exit_exec - open_position["entry_price"])
                        / open_position["entry_price"] * 100.0
                    )
                    storage.record_sim_trade(
                        ticker=open_position["ticker"],
                        strategy=open_position["strategy"],
                        entry_date=open_position["entry_date"],
                        exit_date=str(ts),
                        entry_price=open_position["entry_price"],
                        exit_price=exit_exec,
                        shares=open_position["shares"],
                        pnl_dollars=pnl_dollars,
                        pnl_pct=pnl_pct,
                        exit_reason=reason,
                    )
                    trades_logged += 1
                    sim_balance += net
                    open_position = None
                    bars_held = 0

        # Entry
        if open_position is None and best is not None:
            if best["conviction"] >= MIN_CONVICTION_TO_OPEN:
                px = etf_price_at(etf_df, best["ticker"], ts)
                if px is not None and px > 0:
                    capital = sim_balance * POSITION_SIZE_PCT
                    if capital >= COMMISSION * 2:
                        exec_price = _slip_buy(px)
                        shares = (capital - COMMISSION) / exec_price
                        open_position = {
                            "ticker": best["ticker"],
                            "strategy": best["strategy"],
                            "entry_date": str(ts),
                            "entry_price": exec_price,
                            "shares": shares,
                            "capital_used": capital,
                            "conviction": best["conviction"],
                        }
                        sim_balance -= capital
                        bars_held = 0

    # Close any final open
    if open_position is not None:
        last_ts = iter_bars[-1]
        last_px = etf_price_at(etf_df, open_position["ticker"], last_ts)
        if last_px:
            exit_exec = _slip_sell(last_px)
            gross = open_position["shares"] * exit_exec
            net = gross - COMMISSION
            pnl_dollars = net - open_position["capital_used"]
            pnl_pct = (
                (exit_exec - open_position["entry_price"])
                / open_position["entry_price"] * 100.0
            )
            storage.record_sim_trade(
                ticker=open_position["ticker"],
                strategy=open_position["strategy"],
                entry_date=open_position["entry_date"],
                exit_date=str(last_ts),
                entry_price=open_position["entry_price"],
                exit_price=exit_exec,
                shares=open_position["shares"],
                pnl_dollars=pnl_dollars,
                pnl_pct=pnl_pct,
                exit_reason="backtest_end",
            )
            trades_logged += 1
            sim_balance += net

    by_strategy = storage.compute_sim_stats()
    summary = {
        "interval": "1h",
        "start_date": str(iter_bars[0]),
        "end_date": str(iter_bars[-1]),
        "bars_replayed": len(iter_bars),
        "starting_balance": starting_balance,
        "final_balance": sim_balance,
        "total_pnl": sim_balance - starting_balance,
        "total_return_pct": (sim_balance / starting_balance - 1) * 100,
        "trades": trades_logged,
        "by_strategy": by_strategy,
    }
    LOGGER.info(
        f"Done: {trades_logged} trades, final ${sim_balance:,.2f} "
        f"({summary['total_return_pct']:+.1f}%)"
    )
    return summary


def _main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    storage.init_databases()
    summary = run_intraday_backtest()
    print()
    print("=" * 60)
    print(f" Intraday backtest summary "
          f"({summary.get('start_date', '?')} to "
          f"{summary.get('end_date', '?')})")
    print("=" * 60)
    print(f" Hourly bars replayed : {summary.get('bars_replayed', 0):,}")
    print(f" Trades               : {summary.get('trades', 0):,}")
    print(f" Starting balance     : ${summary.get('starting_balance', 0):,.2f}")
    print(f" Final balance        : ${summary.get('final_balance', 0):,.2f}")
    print(f" Total P&L            : ${summary.get('total_pnl', 0):+,.2f} "
          f"({summary.get('total_return_pct', 0):+.1f}%)")
    print()
    print(" By strategy:")
    by_strat = summary.get("by_strategy") or {}
    if not by_strat:
        print("   (no trades fired)")
    for name, st in sorted(by_strat.items()):
        print(
            f"   {name:24s}  "
            f"trades={st['sim_trades']:4d}  "
            f"wr={st['sim_win_rate']*100:5.1f}%  "
            f"pf={st['sim_profit_factor']:5.2f}  "
            f"pnl=${st['sim_pnl']:+9.2f}"
        )
    print()
    return 0


if __name__ == "__main__":
    sys.exit(_main())
