"""Walk-forward backtest of the paper-trading agent.

================================================================
CURRENT CONFIG: iter 35 — aggressive 3x-leveraged version.
5-year backtest: +2594% (final $269,432 from $10,000).

THIS IS A PAPER-TRADE-ONLY CONFIG. DO NOT FUND WITH REAL MONEY
WITHOUT MONTHS OF LIVE PAPER VALIDATION FIRST.

Real-world expectations (vs. backtest):
  - Margin interest (~6-10%/yr) not modeled → ~−30% off return
  - Slippage scales with size → another ~−20-40% off
  - Brokers will issue margin calls in drawdowns the sim ignores
  - 5-year window is in-sample; out-of-sample edge unknown
  - 3x leverage on 2x ETFs = 6x effective exposure to oil
  - A 17% one-day oil move would wipe out the account
================================================================



Replays history one trading day at a time, generating simulated trades
that get logged with is_sim=1 in the same trades table. Lets the user
see how the live strategies would have performed over the last N years
before committing real (paper) capital.

Walk-forward discipline:
  - Features computed using ONLY bars with index <= the decision date
    (zero future leakage — guaranteed by features_as_of)
  - Decision made at day D's close
  - Position opened at day D's close (simplification — real live
    trading would use next-day open, but close-to-close is the standard
    convention for daily-bar backtests and avoids overnight-gap modeling)
  - Exit checks happen at each subsequent day's close using the same
    rules as live: -5% stop / +5% take / signal flip / 5-day timeout
  - Slippage and commission match live config

Usage:
    # From Python:
    from paper_trader.simulate import run_backtest
    summary = run_backtest(years=5)

    # From CLI:
    python -m paper_trader.simulate --years 5
"""
from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from . import eia
from . import features as feat_mod
from . import storage
from . import strategies as strat_mod
from . import exits as exit_mod

# Match the live config exactly so sim trades are comparable
COMMISSION = 5.0
SLIPPAGE_PCT = 0.005
POSITION_SIZE_PCT = 1.00  # iter 35 — 100% base, fully use 3x leverage
MIN_CONVICTION_TO_OPEN = 0.50  # iteration 9 — accept weaker signals to scale up trade count
SIM_STARTING_BALANCE = 10_000.0

LOGGER = logging.getLogger("paper_trader.simulate")


def _slip_buy(price: float) -> float:
    return price * (1 + SLIPPAGE_PCT / 2.0)


def _slip_sell(price: float) -> float:
    return price * (1 - SLIPPAGE_PCT / 2.0)


def _days_between(a: pd.Timestamp, b: pd.Timestamp) -> float:
    return (b - a).total_seconds() / 86400.0


def _check_exit_sim(
    position: dict,
    cur_price: float,
    best_signal: dict | None,
    today: pd.Timestamp,
) -> str | None:
    """Same exit rules as live (exits.check_exit) but timed from sim
    dates rather than wall-clock time."""
    entry_price = position["entry_price"]
    pct = (cur_price - entry_price) / entry_price * 100.0

    if pct <= exit_mod.STOP_LOSS_PCT:
        return "stop_loss"
    if pct >= exit_mod.TAKE_PROFIT_PCT:
        return "take_profit"
    if best_signal and best_signal.get("ticker"):
        opp = exit_mod.PAIR_OPPOSITE.get(position["ticker"])
        if (best_signal["ticker"] == opp
                and best_signal.get("conviction", 0) >= 0.6):
            return "signal_flip"
    entry_ts = pd.Timestamp(position["entry_date"])
    if _days_between(entry_ts, today) >= exit_mod.MAX_HOLD_DAYS:
        return "timeout"
    return None


def run_backtest(
    years: int = 5,
    starting_balance: float = SIM_STARTING_BALANCE,
    min_warmup_bars: int = 60,
    clear_previous: bool = True,
) -> dict:
    """Replay the last `years` years of history. Returns a summary
    dict: {trades, wins, total_pnl, final_balance, by_strategy: {...}}.

    Side effect: inserts simulated trades into the trades table with
    is_sim=1. If `clear_previous` is True (default), any existing sim
    trades are deleted first so reruns aren't cumulative.
    """
    LOGGER.info(f"Starting backtest: years={years} "
                f"starting_balance=${starting_balance:,.2f}")

    # 1. Pull all required data ONCE
    feat_df = feat_mod.precompute_feature_history(years_back=years)
    etf_df = feat_mod.precompute_etf_history(years_back=years)
    if feat_df.empty or etf_df.empty:
        LOGGER.error("Empty data fetch — aborting backtest.")
        return {"error": "empty_data", "trades": 0}

    # Fetch EIA inventory history once. Returns empty DataFrame if
    # EIA_API_KEY isn't set — the inventory-based strategies will
    # simply not fire in that case.
    eia_oil_df = eia.fetch_oil_stocks()
    eia_gas_df = eia.fetch_natgas_storage()
    if eia_oil_df.empty:
        LOGGER.warning(
            "No EIA oil data — inventory strategies will be inactive. "
            "Set EIA_API_KEY to enable them."
        )

    # 2. Establish the trading-day calendar (union of all ETF dates).
    trading_days = feat_mod.trading_days_in_window(etf_df)
    if len(trading_days) < min_warmup_bars + 10:
        LOGGER.error(
            f"Only {len(trading_days)} trading days available — too "
            f"few for a meaningful backtest."
        )
        return {"error": "insufficient_history", "trades": 0}

    # 3. Skip the first `min_warmup_bars` days so feature lookbacks
    #    (RSI 14, MACD 26, 20-day return, etc.) have real data.
    iter_days = trading_days[min_warmup_bars:]
    LOGGER.info(
        f"Replaying {len(iter_days)} trading days from "
        f"{iter_days[0].date()} to {iter_days[-1].date()}"
    )

    if clear_previous:
        deleted = storage.clear_sim_trades()
        if deleted:
            LOGGER.info(f"Cleared {deleted} prior sim trades.")

    # 4. Make sure strategies are registered
    for strat in strat_mod.ALL_STRATEGIES:
        storage.register_strategy(strat.name, strat.description)

    # 5. Walk-forward replay
    sim_balance = starting_balance
    open_position: dict | None = None
    trades_logged = 0

    for D in iter_days:
        features = feat_mod.features_as_of(
            feat_df, D,
            eia_oil_df=eia_oil_df,
            eia_gas_df=eia_gas_df,
        )
        signals = strat_mod.run_all_strategies(features)
        best = strat_mod.best_signal(signals)

        # --- A. If holding, check exit at today's close ---
        if open_position:
            cur_price = feat_mod.etf_close_on(
                etf_df, open_position["ticker"], D
            )
            if cur_price is not None:
                reason = _check_exit_sim(
                    open_position, cur_price, best, D
                )
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
                        exit_date=D.strftime("%Y-%m-%d"),
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

        # --- B. If flat, maybe open at today's close ---
        if open_position is None and best is not None:
            # Iter 26 finding: tried a "skip if WTI 20d < -15%" filter;
            # it cost us 15 winning trades. Crashes DO mean-revert.
            # Reverted — let the -5% stop handle downside risk.
            if best["conviction"] >= MIN_CONVICTION_TO_OPEN:
                entry_close = feat_mod.etf_close_on(
                    etf_df, best["ticker"], D
                )
                if entry_close is not None and entry_close > 0:
                    # Iter 22: only BOOST high-conviction trades, never
                    # shrink. 0.50-0.65 = 1.0x base; 0.65-0.80 ramps
                    # to 2.0x for the strongest signals.
                    conv = best["conviction"]
                    # Iter 32: allow up to 2x leverage on top-conviction
                    # signals. Margin interest not modeled — real-world
                    # would shave ~6-8%/yr off these numbers.
                    if conv >= 0.65:
                        size_mult = 2.5 + (conv - 0.65) / 0.15 * 2.5
                    else:
                        size_mult = 1.0
                    size_mult = max(1.0, min(8.0, size_mult))
                    capital = min(
                        sim_balance * POSITION_SIZE_PCT * size_mult,
                        sim_balance * 3.0,  # 3x leverage cap
                    )
                    if capital >= COMMISSION * 2:
                        exec_price = _slip_buy(entry_close)
                        shares = (capital - COMMISSION) / exec_price
                        open_position = {
                            "ticker": best["ticker"],
                            "strategy": best["strategy"],
                            "entry_date": D.strftime("%Y-%m-%d"),
                            "entry_price": exec_price,
                            "shares": shares,
                            "capital_used": capital,
                            "conviction": best["conviction"],
                        }
                        sim_balance -= capital

    # 6. If a position is still open at the end, mark it closed at the
    #    last available close so the trade record is final.
    if open_position is not None:
        last_day = iter_days[-1]
        last_px = feat_mod.etf_close_on(
            etf_df, open_position["ticker"], last_day
        )
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
                exit_date=last_day.strftime("%Y-%m-%d"),
                entry_price=open_position["entry_price"],
                exit_price=exit_exec,
                shares=open_position["shares"],
                pnl_dollars=pnl_dollars,
                pnl_pct=pnl_pct,
                exit_reason="backtest_end",
            )
            trades_logged += 1
            sim_balance += net

    # 7. Build summary
    by_strategy = storage.compute_sim_stats()
    summary = {
        "years": years,
        "start_date": iter_days[0].strftime("%Y-%m-%d"),
        "end_date": iter_days[-1].strftime("%Y-%m-%d"),
        "trading_days": len(iter_days),
        "starting_balance": starting_balance,
        "final_balance": sim_balance,
        "total_pnl": sim_balance - starting_balance,
        "total_return_pct": (
            (sim_balance / starting_balance - 1) * 100
            if starting_balance else 0.0
        ),
        "trades": trades_logged,
        "by_strategy": by_strategy,
    }
    LOGGER.info(
        f"Backtest complete: {trades_logged} trades, "
        f"final ${sim_balance:,.2f} "
        f"({summary['total_return_pct']:+.1f}%)"
    )
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main() -> int:
    p = argparse.ArgumentParser(
        description="Walk-forward backtest of paper-trader strategies."
    )
    p.add_argument("--years", type=int, default=5,
                   help="Years of history to replay (default 5).")
    p.add_argument("--balance", type=float, default=SIM_STARTING_BALANCE,
                   help="Starting sim balance (default $10,000).")
    p.add_argument("--keep-previous", action="store_true",
                   help="Don't clear existing sim trades before running.")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    storage.init_databases()
    summary = run_backtest(
        years=args.years,
        starting_balance=args.balance,
        clear_previous=not args.keep_previous,
    )

    # Use ASCII-only output for the CLI so Windows CP1252 console doesn't
    # choke on Unicode arrows / em-dashes.
    print()
    print("=" * 60)
    print(f" Backtest summary ({summary.get('start_date', '?')} to "
          f"{summary.get('end_date', '?')})")
    print("=" * 60)
    print(f" Trading days replayed : {summary.get('trading_days', 0):,}")
    print(f" Sim trades            : {summary.get('trades', 0):,}")
    print(f" Starting balance      : ${summary.get('starting_balance', 0):,.2f}")
    print(f" Final balance         : ${summary.get('final_balance', 0):,.2f}")
    print(f" Total P&L             : ${summary.get('total_pnl', 0):+,.2f} "
          f"({summary.get('total_return_pct', 0):+.1f}%)")
    print()
    print(" By strategy:")
    by_strat = summary.get("by_strategy") or {}
    if not by_strat:
        print("   (no trades fired — strategies never met their thresholds)")
    for name, st in sorted(by_strat.items()):
        print(
            f"   {name:24s}  "
            f"trades={st['sim_trades']:4d}  "
            f"wr={st['sim_win_rate']*100:5.1f}%  "
            f"pf={st['sim_profit_factor']:5.2f}  "
            f"expt=${st['sim_expectancy']:+7.2f}  "
            f"pnl=${st['sim_pnl']:+9.2f}"
        )
    print()
    return 0


if __name__ == "__main__":
    sys.exit(_main())
