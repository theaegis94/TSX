"""Decay-capture backtest — short both sides of each 2x leveraged ETF pair.

The thesis (math, not prediction):
  - 2x leveraged ETFs reset daily and accumulate volatility drag.
  - Holding either HOU or HOD for many days loses to "rebalancing
    decay" — this is a documented, repeatable structural cost.
  - By shorting BOTH HOU and HOD in equal dollar amounts (and likewise
    HNU + HND), the pair is roughly delta-neutral to oil price moves
    but harvests the decay from both sides.

Why this works mathematically:
  If oil moves r% on day 1 and -r% on day 2, oil ends flat — but
  HOU = (1 + 2r)(1 - 2r) - 1 = -4r² (down) and HOD similarly
  down ~4r². Both decay. Short both: ~+8r² profit per round trip.

Why this doesn't work mathematically:
  If oil trends strongly in one direction, the bull-ETF leg gains
  faster than the bear-ETF leg loses (or vice versa), and the short
  pair takes a loss. Trending markets ≈ bad. Choppy markets ≈ good.

Risks not in this model:
  - Borrow cost: I'm using 5% annual as a baseline (HOU/HOD typical
    short-borrow rate). Real cost can be 2-15% depending on broker
    and squeeze conditions.
  - Tail risk: a 30% one-day move in oil would dislocate the pair
    badly. Real risk management requires wider stops than I model.
  - Liquidity: closing both shorts at the same time during a crisis
    can mean unfavorable fills.

This is NOT a signal-based strategy — it doesn't predict anything.
It's a structural arbitrage that runs continuously, rebalancing every
N days to keep the dollar exposures equal.
"""
from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd
import yfinance as yf

LOGGER = logging.getLogger("paper_trader.decay_capture")

# Config — borrow rate is the biggest cost driver
COMMISSION_PER_LEG = 5.0          # $5 to open or close each short
SLIPPAGE_PCT = 0.005              # 0.5% round-trip per leg
BORROW_RATE_ANNUAL = 0.05         # 5% annual borrow cost
DEFAULT_REBALANCE_DAYS = 5         # weekly rebalance
DEFAULT_NOTIONAL_PER_PAIR = 5_000  # $5k short on each leg = $10k notional per pair

PAIRS = [
    ("HOU.TO", "HOD.TO"),   # 2x bull/bear WTI oil
    ("HNU.TO", "HND.TO"),   # 2x bull/bear Henry Hub natgas
]


def _fetch_pair_prices(pair: tuple, years: int) -> pd.DataFrame:
    """Get daily closes for both legs of a pair, aligned."""
    try:
        df = yf.download(
            " ".join(pair), period=f"{years}y", interval="1d",
            auto_adjust=True, progress=False, group_by="ticker",
            threads=True,
        )
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    out = pd.DataFrame()
    for t in pair:
        try:
            if isinstance(df.columns, pd.MultiIndex):
                if t not in df.columns.get_level_values(0):
                    return pd.DataFrame()
                out[t] = df[t]["Close"]
            else:
                out[t] = df["Close"]
        except KeyError:
            return pd.DataFrame()
    out = out.dropna()
    return out


def backtest_pair(
    pair: tuple,
    years: int,
    notional_per_leg: float,
    rebalance_days: int,
    borrow_rate: float = BORROW_RATE_ANNUAL,
) -> dict:
    """Run the decay-capture backtest on one pair. Returns summary."""
    bull, bear = pair
    df = _fetch_pair_prices(pair, years)
    if df.empty or len(df) < rebalance_days * 4:
        return {
            "pair": f"{bull}/{bear}",
            "error": "no_data",
            "trades": 0,
        }

    LOGGER.info(
        f"{bull}/{bear}: {len(df)} trading days from "
        f"{df.index[0].date()} to {df.index[-1].date()}"
    )

    total_pnl = 0.0
    trades_logged = 0   # each "trade" = one open + one close of one leg
    cycles = 0
    wins = 0
    losses = 0
    cycle_pnls: list[float] = []

    # Iterate in chunks of rebalance_days
    i = 0
    while i + rebalance_days < len(df):
        start_row = df.iloc[i]
        end_row = df.iloc[i + rebalance_days]
        # Open: short notional_per_leg of each
        bull_entry = float(start_row[bull])
        bear_entry = float(start_row[bear])
        if bull_entry <= 0 or bear_entry <= 0:
            i += rebalance_days
            continue
        bull_shares = notional_per_leg / bull_entry
        bear_shares = notional_per_leg / bear_entry
        # Slippage on the short sale (we sell at a slightly lower price)
        bull_proceeds = bull_shares * bull_entry * (1 - SLIPPAGE_PCT / 2)
        bear_proceeds = bear_shares * bear_entry * (1 - SLIPPAGE_PCT / 2)
        # Subtract opening commissions
        open_costs = COMMISSION_PER_LEG * 2

        # Close: buy back at end-row prices, with slippage
        bull_exit = float(end_row[bull])
        bear_exit = float(end_row[bear])
        if bull_exit <= 0 or bear_exit <= 0:
            i += rebalance_days
            continue
        bull_buyback = bull_shares * bull_exit * (1 + SLIPPAGE_PCT / 2)
        bear_buyback = bear_shares * bear_exit * (1 + SLIPPAGE_PCT / 2)
        close_costs = COMMISSION_PER_LEG * 2

        # Borrow cost — pro-rated daily on the notional
        days_held = rebalance_days
        borrow_cost = (
            (notional_per_leg * 2) * borrow_rate * days_held / 365.0
        )

        # Cycle P&L
        cycle_pnl = (
            (bull_proceeds - bull_buyback)
            + (bear_proceeds - bear_buyback)
            - open_costs - close_costs - borrow_cost
        )
        total_pnl += cycle_pnl
        cycle_pnls.append(cycle_pnl)
        trades_logged += 4  # 2 opens + 2 closes
        cycles += 1
        if cycle_pnl > 0:
            wins += 1
        else:
            losses += 1
        i += rebalance_days

    # Stats
    win_rate = wins / cycles if cycles else 0.0
    gross_wins = sum(p for p in cycle_pnls if p > 0)
    gross_losses = sum(-p for p in cycle_pnls if p < 0)
    pf = (gross_wins / gross_losses) if gross_losses > 0 else 0.0
    avg_pnl = (total_pnl / cycles) if cycles else 0.0

    return {
        "pair": f"{bull}/{bear}",
        "trading_days": len(df),
        "rebalance_days": rebalance_days,
        "notional_per_leg": notional_per_leg,
        "cycles": cycles,
        "trades": trades_logged,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "profit_factor": pf,
        "avg_pnl_per_cycle": avg_pnl,
        "total_pnl": total_pnl,
        "max_cycle_win": max(cycle_pnls) if cycle_pnls else 0,
        "max_cycle_loss": min(cycle_pnls) if cycle_pnls else 0,
    }


def run_decay_backtest(
    years: int = 5,
    rebalance_days: int = DEFAULT_REBALANCE_DAYS,
    notional_per_leg: float = DEFAULT_NOTIONAL_PER_PAIR,
    borrow_rate: float = BORROW_RATE_ANNUAL,
) -> dict:
    """Run decay-capture on all pairs and return combined summary."""
    LOGGER.info(
        f"Decay-capture backtest: years={years}, "
        f"rebalance={rebalance_days}d, "
        f"notional=${notional_per_leg:,.0f} per leg "
        f"(${notional_per_leg*2:,.0f} per pair), "
        f"borrow={borrow_rate*100:.1f}%"
    )

    by_pair = []
    total_pnl = 0.0
    total_trades = 0
    total_cycles = 0
    for pair in PAIRS:
        result = backtest_pair(
            pair, years, notional_per_leg, rebalance_days, borrow_rate,
        )
        by_pair.append(result)
        if "total_pnl" in result:
            total_pnl += result["total_pnl"]
            total_trades += result["trades"]
            total_cycles += result["cycles"]

    # Effective starting capital = sum of all notionals (max we'd need)
    total_capital_used = notional_per_leg * 2 * len(PAIRS)
    return_pct = (total_pnl / total_capital_used * 100
                  if total_capital_used else 0.0)
    annualized = return_pct / years if years else 0.0

    return {
        "years": years,
        "rebalance_days": rebalance_days,
        "capital_required": total_capital_used,
        "total_pnl": total_pnl,
        "return_pct": return_pct,
        "annualized_return_pct": annualized,
        "total_trades": total_trades,
        "trades_per_week": (
            total_trades / years / 52 if years else 0.0
        ),
        "total_cycles": total_cycles,
        "by_pair": by_pair,
    }


def _main():
    p = argparse.ArgumentParser(description="Decay-capture backtest.")
    p.add_argument("--years", type=int, default=5)
    p.add_argument("--rebalance", type=int, default=DEFAULT_REBALANCE_DAYS,
                   help="Days between rebalances (default 5 = weekly).")
    p.add_argument("--notional", type=float,
                   default=DEFAULT_NOTIONAL_PER_PAIR,
                   help="Dollars short on each leg.")
    p.add_argument("--borrow", type=float, default=BORROW_RATE_ANNUAL,
                   help="Annual borrow rate (default 0.05 = 5%%).")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    summary = run_decay_backtest(
        years=args.years,
        rebalance_days=args.rebalance,
        notional_per_leg=args.notional,
        borrow_rate=args.borrow,
    )

    print()
    print("=" * 60)
    print(f" Decay-capture backtest summary ({args.years} years, "
          f"rebalance every {args.rebalance} days)")
    print("=" * 60)
    print(f" Borrow rate          : {args.borrow*100:.1f}% annual")
    print(f" Capital required     : ${summary['capital_required']:,.0f}")
    print(f" Total P&L            : ${summary['total_pnl']:+,.2f}")
    print(f" Total return         : {summary['return_pct']:+.1f}%")
    print(f" Annualized           : {summary['annualized_return_pct']:+.1f}%/yr")
    print(f" Trades (legs traded) : {summary['total_trades']:,}")
    print(f" Trades per week      : {summary['trades_per_week']:.1f}")
    print(f" Total cycles         : {summary['total_cycles']:,}")
    print()
    print(" By pair:")
    for r in summary["by_pair"]:
        if "error" in r:
            print(f"   {r['pair']:18s} ERROR: {r['error']}")
            continue
        print(
            f"   {r['pair']:18s}  "
            f"cycles={r['cycles']:4d}  "
            f"wr={r['win_rate']*100:5.1f}%  "
            f"pf={r['profit_factor']:5.2f}  "
            f"avg=${r['avg_pnl_per_cycle']:+7.2f}  "
            f"pnl=${r['total_pnl']:+9.2f}"
        )
        print(
            f"   {'':18s}  "
            f"max win=${r['max_cycle_win']:+.0f}  "
            f"max loss=${r['max_cycle_loss']:+.0f}"
        )
    print()
    return 0


if __name__ == "__main__":
    sys.exit(_main())
